#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// FP16 헤더 포함
#include "benchmark.h"
#include <fp16.h>

typedef uint16_t float16;
typedef uint32_t float32_b;

#ifdef FP16_COMPARATIVE_BENCHMARKS
	#include <third-party/THHalf.h>
	#include <third-party/npy-halffloat.h>
	#include <third-party/eigen-half.h>
	#include <third-party/float16-compressor.h>
	#include <third-party/half.hpp>
#endif

// 테스트 데이터 생성 함수
std::vector<float> generate_test_data(size_t size) {
    const uint_fast32_t seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::mt19937(seed));
    std::vector<float> fp32(size);
    std::generate(fp32.begin(), fp32.end(), std::ref(rng));

    return fp32;
}

static void benchmark_fp32v_to_fp16_ieee_value_array(std::vector<float> &fp32, std::vector<float16> &fp16, size_t size) 
{
    auto result = run_benchmark("fp32v_to_fp16_ieee_array", size, sizeof(uint16_t),[&]() {
        float* input = fp32.data();
        float16* output = fp16.data();
        const size_t n = size;
        for (size_t i = 0; i < n; i++) {
            output[i] = fp32_ieee_to_fp16_value(input[i]);
        }
    });
    print_result(result);

}

#ifdef FP16_COMPARATIVE_BENCHMARKS
	static void TH_float2halfbits(std::vector<float> &fp32, std::vector<float16> &fp16, size_t size) {
        float* input = fp32.data();
        float16* output = fp16.data();
        const size_t n = size;
        auto result = run_benchmark("TH_float2halfbits", size, sizeof(float16),[&]() {
            for (size_t i = 0; i < n; i++) {
                TH_float2halfbits(&input[i], &output[i]);
            }
        });
        print_result(result);
    }

	static void npy_floatbits_to_halfbits(std::vector<float> &fp32, std::vector<float16> &fp16, size_t size) {
        float* input = fp32.data();
        float16* output = fp16.data();
        const size_t n = size;
        auto result = run_benchmark("npy_floatbits_to_halfbits", size, sizeof(float16),[&]() {
			for (size_t i = 0; i < n; i++) {
				output[i] = npy_floatbits_to_halfbits(fp32v_to_fp32b(input[i]));
			}
        });
        print_result(result);
    }

	static void Eigen_float_to_half_rtne(std::vector<float> &fp32, std::vector<float16> &fp16, size_t size) {
        float* input = fp32.data();
        float16* output = fp16.data();
        const size_t n = size;
        auto result = run_benchmark("Eigen_float_to_half_rtne", size, sizeof(float16),[&]() {
			for (size_t i = 0; i < n; i++) {
				output[i] = Eigen::half_impl::float_to_half_rtne(input[i]).x;
			}
        });
        print_result(result);
    }

	static void Float16Compressor_compress(std::vector<float> &fp32, std::vector<float16> &fp16, size_t size) {
        float* input = fp32.data();
        float16* output = fp16.data();
        const size_t n = size;
        auto result = run_benchmark("Float16Compressor_compress", size, sizeof(float16),[&]() {
			for (size_t i = 0; i < n; i++) {
				output[i] = Float16Compressor::compress(input[i]);
			}
        });
        print_result(result);
    }


	static void half_float_detail_float2half_table(std::vector<float> &fp32, std::vector<float16> &fp16, size_t size) {
        float* input = fp32.data();
        float16* output = fp16.data();
        const size_t n = size;
        auto result = run_benchmark("half_float_detail_float2half_table", size, sizeof(float16),[&]() {
			for (size_t i = 0; i < n; i++) {
				output[i] = half_float::detail::float2half_impl<std::round_to_nearest>(input[i], half_float::detail::true_type());
			}
        });
        print_result(result);
    }

	static void half_float_detail_float2half_branch(std::vector<float> &fp32, std::vector<float16> &fp16, size_t size) {
        float* input = fp32.data();
        float16* output = fp16.data();
        const size_t n = size;
        auto result = run_benchmark("half_float_detail_float2half_branch", size, sizeof(float16),[&]() {
			for (size_t i = 0; i < n; i++) {
				output[i] = half_float::detail::float2half_impl<std::round_to_nearest>(input[i], half_float::detail::false_type());
			}
        });
        print_result(result);
    }
#endif

int main() {
    std::cout << "FP32 to FP16 Alternative Format Conversion Benchmarks" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << std::left << std::setw(25) << "Function" << std::right
            << std::setw(10) << "Items" << std::setw(15) << "Avg Time"
            << std::setw(15) << "Throughput" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    std::vector<size_t> sizes;
    for (size_t size = 1 << 10; size <= 64 << 20; size *= 2) {
        sizes.push_back(size);
    }
    
    for (size_t size : sizes) {
        std::vector<float> fp32 = generate_test_data(size);
        std::vector<float16> fp16(size);

        benchmark_fp32v_to_fp16_ieee_value_array(fp32, fp16, size);

#ifdef FP16_COMPARATIVE_BENCHMARKS
    TH_float2halfbits(fp32, fp16, size);
    npy_floatbits_to_halfbits(fp32, fp16, size);
    Eigen_float_to_half_rtne(fp32, fp16, size);
    Float16Compressor_compress(fp32, fp16, size);
    half_float_detail_float2half_table(fp32, fp16, size);
    half_float_detail_float2half_branch(fp32, fp16, size);
#endif
        std::cout << std::endl;
    }
    return 0;
}