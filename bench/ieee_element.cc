#include "benchmark.h"
#include <fp16.h>
#include <cstdint>

#ifdef FP16_COMPARATIVE_BENCHMARKS
	#include <third-party/THHalf.h>
	#include <third-party/npy-halffloat.h>
	#include <third-party/eigen-half.h>
	#include <third-party/float16-compressor.h>
	#include <third-party/half.hpp>
#endif


// XorShift Algorithm for avoiding compiler optimization.
static inline uint16_t next_xorshift16(uint16_t x) {
    x ^= x >> 8;
    x ^= x << 9;
    x ^= x >> 5;
    return x;
}

static inline uint32_t next_xorshift32(uint32_t x) {
    x ^= x >> 13;
    x ^= x << 17;
    x ^= x >> 5;
    return x;
}
typedef uint16_t float16;
typedef uint32_t float32_b;


float32_b benchmark_func_fp16_ieee_to_fp32_bits(float16 fp16) {
    float32_b fp32 = fp16_ieee_to_fp32_bits(fp16);
    fp16 = next_xorshift16(fp16);
    return fp32;
}

float benchmark_func_fp16_ieee_to_fp32_value(float16 fp16) {
    float fp32 = fp16_ieee_to_fp32_value(fp16);
    fp16 = next_xorshift16(fp16);
    return fp32;
}

#ifdef FP16_COMPARATIVE_BENCHMARKS
    static float TH_fp16_to_f32_bits(float16 fp16) {
        float fp32;
        TH_halfbits2float(&fp16, &fp32);
        fp16 = next_xorshift16(fp16);
        return fp32;
    }
    
	static float32_b npy_fp16_to_f32_bits(float16 fp16) {
        uint32_t fp32 = npy_halfbits_to_floatbits(fp16);
		fp16 = next_xorshift16(fp16);
        return fp32;
	}

	static float Eigen_fp16_to_f32_value(float16 fp16) {
		float fp32 = Eigen::half_impl::half_to_float(Eigen::half_impl::raw_uint16_to_half(fp16));
		fp16 = next_xorshift16(fp16);
        return fp32;
	}

	static float Float16Compressor_fp16_to_f32_value(float16 fp16) {
		float fp32 = Float16Compressor::decompress(fp16);
		fp16 = next_xorshift16(fp16);
        return fp32;
	}

	static float half_float_detail_fp16_to_f32_table(float16 fp16) {
		float fp32 =
				half_float::detail::half2float_impl(fp16,
					half_float::detail::true_type());

			fp16 = next_xorshift16(fp16);
        return fp32;
	}
	
	static float half_float_detail_fp16_to_f32_branch(float16 fp16) {
		float fp32 =
				half_float::detail::half2float_impl(fp16,
					half_float::detail::false_type());
			fp16 = next_xorshift16(fp16);
        return fp32;
	}
#endif

/* Conversion from IEEE FP32 to IEEE FP16 */
float16 benchmark_func_fp32_ieee_to_fp16_value(float32_b fp32) {
    float16 fp16 = fp32_ieee_to_fp16_value(fp32);
    fp32 = next_xorshift32(fp32);
    return fp16;
}

#ifdef FP16_COMPARATIVE_BENCHMARKS

static float16 TH_f32_to_fp16_bits(float32_b fp32) {
    uint16_t fp16;
    float fp32_value = fp32b_to_fp32v(fp32);
    TH_float2halfbits(&fp32_value, &fp16);
    fp32 = next_xorshift32(fp32);
    return fp16;
}

static float16 npy_f32_to_fp16_bits(float32_b fp32) {
    uint16_t fp16 = npy_floatbits_to_halfbits(fp32);
    fp32 = next_xorshift32(fp32);
    return fp16;
}

static float16 Eigen_f32b_to_fp16_rtne(float32_b fp32) {
    Eigen::half_impl::__half fp16 = Eigen::half_impl::float_to_half_rtne((float)fp32b_to_fp32v(fp32));
    fp32 = next_xorshift32(fp32);
    return fp16.x;
}

static float16 Float16Compressor_f32b_to_fp16_(float32_b fp32) {
    uint16_t fp16 = Float16Compressor::compress(fp32b_to_fp32v(fp32));
    fp32 = next_xorshift32(fp32);
    return fp16;
}

static float16 half_float_detail_f32b_to_fp16_table(float32_b fp32) {
    uint16_t fp16 = half_float::detail::float2half_impl<std::round_to_nearest>(fp32b_to_fp32v(fp32), half_float::detail::true_type());
    fp32 = next_xorshift32(fp32);
    return fp16;
}

static float16 half_float_detail_f32b_to_fp16_branch(float32_b fp32) {
    uint16_t fp16 = half_float::detail::float2half_impl<std::round_to_nearest>(fp32b_to_fp32v(fp32), half_float::detail::false_type());
    fp32 = next_xorshift32(fp32);
    return fp16;
}

#endif

int main() {
    // Global variables for benchmarking
    float16 fp16 = UINT16_C(0x7C00);
    float32_b fp32 = UINT32_C(0x7F800000);
    float32_b result_bits = 0;
    float result_value = 0.0f;
    float16 result_fp16 = 0;
    printf("Running FP16 IEEE element benchmarks...\n");
    
    result_bits = benchmark_func_fp16_ieee_to_fp32_bits(fp16);
    printf("Bit-Op fp16: %x -> fp32: %x\n\n", fp16, result_bits);

    result_value = benchmark_func_fp16_ieee_to_fp32_value(fp16);
    printf("FP-Op fp16: %x -> fp32: %x\n\n", fp16, (float32_b)result_value);
    #ifdef FP16_COMPARATIVE_BENCHMARKS
    fp16 = UINT16_C(0x7C00);
    result_bits = TH_fp16_to_f32_bits(fp16);
    printf("Torch fp16: %x -> fp32: %x\n\n", fp16, result_bits);
    result_bits = npy_fp16_to_f32_bits(fp16);
    printf("Numpy fp16: %x -> fp32: %x\n\n", fp16, result_bits);
    result_value = Eigen_fp16_to_f32_value(fp16);
    printf("Eigen fp16: %x -> fp32: %x\n\n", fp16, (float32_b)result_value);
    result_value = Float16Compressor_fp16_to_f32_value(fp16);
    printf("Float16Compressor fp16: %x -> fp32: %x\n\n", fp16, (float32_b)result_value);
    result_value = half_float_detail_fp16_to_f32_table(fp16);
    printf("half_float_detail(table) fp16: %x -> fp32: %x\n\n", fp16, (float32_b)result_value);
    result_value = half_float_detail_fp16_to_f32_branch(fp16);
    printf("half_float_detail(branch) fp16: %x -> fp32: %x\n\n", fp16, (float32_b)result_value);
    #endif
    printf("--------------------------------\n");
    result_fp16 = benchmark_func_fp32_ieee_to_fp16_value(fp32);
    printf("Bit-Op fp32: %x -> fp16: %x\n\n", fp32, result_fp16);
    #ifdef FP16_COMPARATIVE_BENCHMARKS
    fp32 = UINT32_C(0x7F800000);
    result_fp16 = TH_f32_to_fp16_bits(fp32);
    printf("Torch fp32: %x -> fp16: %x\n\n", fp32, result_fp16);
    result_fp16 = npy_f32_to_fp16_bits(fp32);
    printf("Numpy fp32: %x -> fp16: %x\n\n", fp32, result_fp16);
    result_fp16 = Eigen_f32b_to_fp16_rtne(fp32);
    printf("Eigen fp32: %x -> fp16: %x\n\n", fp32, result_fp16);
    result_fp16 = Float16Compressor_f32b_to_fp16_(fp32);
    printf("Float16Compressor fp32: %x -> fp16: %x\n\n", fp32, result_fp16);
    result_fp16 = half_float_detail_f32b_to_fp16_table(fp32);
    printf("half_float_detail(table) fp32: %x -> fp16: %x\n\n", fp32, result_fp16);
    result_fp16 = half_float_detail_f32b_to_fp16_branch(fp32);
    printf("half_float_detail(branch) fp32: %x -> fp16: %x\n\n", fp32, result_fp16);
    #endif

    printf("All IEEE element benchmarks completed!\n");
    return 0;
} 