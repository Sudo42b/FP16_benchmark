#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>
#include <iomanip>
#include <string>
#include <cstdint>

// FP16 헤더 포함
#include <fp16.h>
#include "benchmark.h"

typedef uint16_t float16;
typedef uint32_t float32_b;

// 테스트 데이터 생성 함수
std::vector<float16> generate_test_data(size_t size) {
    const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto rng = std::bind(std::uniform_int_distribution<float16>(0, 0x7BFF), std::mt19937(seed));
    
    std::vector<float16> fp16(size);
    std::generate(fp16.begin(), fp16.end(), std::ref(rng));
    
    return fp16;
}

// fp16_alt_to_fp32_bits 벤치마크 함수
static void benchmark_fp16_alt_to_fp32_bits(std::vector<float16>& fp16, 
                                                              std::vector<float32_b>& fp32, size_t size) 
{
    auto result = run_benchmark("fp16_alt_to_fp32_bits", size, sizeof(float32_b), [&]() {
        float16* input = fp16.data();
        float32_b* output = fp32.data();
        const size_t n = size;
        for (size_t i = 0; i < n; i++) {
            output[i] = fp16_alt_to_fp32_bits(input[i]);
        }
    });
    
    print_result(result);
}

// fp16_alt_to_fp32_value 벤치마크 함수
static void benchmark_fp16_alt_to_fp32_value(std::vector<float16>& fp16, 
    std::vector<float>& fp32, size_t size) {

    auto result = run_benchmark("fp16_alt_to_fp32_value", size, sizeof(float), [&]() {
        float16* input = fp16.data();
        float* output = fp32.data();
        const size_t n = size;
        for (size_t i = 0; i < n; i++) {
            output[i] = fp16_alt_to_fp32_value(input[i]);
        }
    });
    
    print_result(result);
}

int main() {
    std::cout << "FP16 to FP32 Alternative Format Conversion Benchmarks" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << std::left << std::setw(25) << "Function"
              << std::right << std::setw(10) << "Items"
              << std::setw(15) << "Avg Time"
              << std::setw(15) << "Throughput"
              << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    // 2의 거듭제곱으로 1<<10부터(1MB) 64<<20까지(64MB)
    std::vector<size_t> sizes;
    for (size_t size = 1 << 10; size <= 64 << 20; size *= 2) {
        sizes.push_back(size);
    }
    
    for (size_t size : sizes) {
        std::vector<float16> fp16 = generate_test_data(size);
        std::vector<float32_b> fp32_b(size);
        std::vector<float> fp32(size);

        // fp16_alt_to_fp32_bits 벤치마크
        benchmark_fp16_alt_to_fp32_bits(fp16, fp32_b, size);
        
        // fp16_alt_to_fp32_value 벤치마크
        benchmark_fp16_alt_to_fp32_value(fp16, fp32, size);
        
        std::cout << std::endl;
    }
    
    return 0;
}