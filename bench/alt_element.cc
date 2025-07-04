#include "benchmark.h"
#include <fp16.h>
#include <cstdint>

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

float32_b benchmark_fp16_alt_to_fp32_bits(float16 fp16) {
    float32_b fp32 = fp16_alt_to_fp32_bits(fp16);
    fp16 = next_xorshift16(fp16);
    return fp32;
}

float benchmark_fp16_alt_to_fp32_value(float16 fp16) {
    float fp32 = fp16_alt_to_fp32_value(fp16);
    fp16 = next_xorshift16(fp16);
    return fp32;
}

float16 benchmark_fp32b_alt_to_fp16_value(float32_b fp32) {
    // f32b -> f32v -> f16
    float16 fp16 = fp32_alt_to_fp16_value(fp32b_to_fp32v(fp32));
    fp32 = next_xorshift32(fp32);
    return fp16;
}


int main() {
    // Global variables for benchmarking
    float16 fp16 = UINT16_C(0x7C00);
    float32_b fp32b = UINT32_C(0x7F800000);
    float32_b result_f32b = 0;
    float result_f32v = 0.0f;
    float16 result_f16b = 0;
    printf("Running FP16 alt-element benchmarks...\n");
    
    result_f32b = benchmark_fp16_alt_to_fp32_bits(fp16);
    printf("fp16: %x -> fp32: %x\n\n", fp16, result_f32b);

    result_f32v = benchmark_fp16_alt_to_fp32_value(fp16);
    printf("fp16: %x -> fp32: %x\n\n", fp16, (uint32_t)result_f32v);

    result_f16b = benchmark_fp32b_alt_to_fp16_value(fp32b);
    printf("fp32: %x -> fp16: %x\n\n", fp32b, result_f16b);
    
    printf("All alt-element benchmarks completed!\n");
    return 0;
} 