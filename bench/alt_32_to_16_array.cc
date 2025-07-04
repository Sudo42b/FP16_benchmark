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

// 테스트 데이터 생성 함수
std::vector<float> generate_test_data(size_t size) {
  const uint_fast32_t seed =
      std::chrono::system_clock::now().time_since_epoch().count();
  auto rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
                       std::mt19937(seed));

  std::vector<float> fp32(size);

  std::generate(fp32.begin(), fp32.end(), std::ref(rng));

  return fp32;
}
// fp16_alt_from_fp32_value 벤치마크 함수
static void benchmark_fp32v_to_fp16_alt_value_array(std::vector<float> &fp32,
    std::vector<float16> &fp16,
    size_t size) {
    auto result = run_benchmark("fp32v_to_fp16_alt_array", size, sizeof(float16),[&]() {
                float *input = fp32.data();
                float16 *output = fp16.data();
                const size_t n = size;
                for (size_t i = 0; i < n; i++) 
                {
                    output[i] = fp32_alt_to_fp16_value(input[i]);
                }
            });

  print_result(result);
}

int main() {
  std::cout << "FP32 to FP16 Alternative Format Conversion Benchmarks" << std::endl;
  std::cout << "=====================================" << std::endl;
  std::cout << std::left << std::setw(25) << "Function" << std::right
            << std::setw(10) << "Items" << std::setw(15) << "Avg Time"
            << std::setw(15) << "Throughput" << std::endl;
  std::cout << std::string(65, '-') << std::endl;

  // 2의 거듭제곱으로 1<<10부터(1MB) 64<<20까지(64MB)
  std::vector<size_t> sizes;
  for (size_t size = 1 << 10; size <= 64 << 20; size *= 2) {
    sizes.push_back(size);
  }

  for (size_t size : sizes) {
    std::vector<float> fp32 = generate_test_data(size);
    std::vector<float16> fp16(size);

    // fp16_alt_to_fp32_bits 벤치마크
    benchmark_fp32v_to_fp16_alt_value_array(fp32, fp16, size);

    std::cout << std::endl;
  }

  return 0;
}