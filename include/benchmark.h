#ifndef SIMPLE_BENCHMARK_H
#define SIMPLE_BENCHMARK_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdint>


// 벤치마크 결과를 저장하는 구조체
struct BenchmarkResult {
    std::string name;
    size_t iterations;
    double total_time_sec;
    double avg_time_per_iteration_sec;
    double throughput_mbps;
};

// 벤치마크 실행 함수
template<typename Func>
static BenchmarkResult run_benchmark(const std::string& name, size_t iterations, size_t type_size, Func func) {

    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        func();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    BenchmarkResult result;
    result.name = name;
    result.iterations = iterations;
    result.total_time_sec = duration.count();
    result.avg_time_per_iteration_sec = result.total_time_sec / iterations;
    result.throughput_mbps = (iterations * type_size) / result.total_time_sec;
    
    return result;
}

// 결과 출력 함수
static void print_result(const BenchmarkResult& result) {
    std::cout << std::left << std::setw(25) << result.name
              << std::right << std::setw(10) << result.iterations
              << std::setw(15) << std::fixed << std::setprecision(3) << result.avg_time_per_iteration_sec << " sec"
              << std::setw(15) << std::fixed << std::setprecision(2) << result.throughput_mbps << " Bytes/s"
              << std::endl;
}


#endif // SIMPLE_BENCHMARK_H 