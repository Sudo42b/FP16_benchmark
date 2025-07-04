#include <iostream>
#include <iomanip>
#include <cstdint>
#include <fp16.h>
#include "simple_test.h"
#include <regex>
#include <string>
#include <sstream>
#include <cmath>

// 진행률 표시줄 함수
void print_progress(uint32_t current, uint32_t total, const std::string& test_name) {
    const int bar_width = 50;
    float progress = (float)current / total;
    int pos = bar_width * progress;
    
    std::cout << "\r" << test_name << " [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% (" 
              << current << "/" << total << ")" << std::flush;
}

void test_fp32_to_bits_positive() {
	for (uint32_t bits = UINT32_C(0x00000000); bits <= UINT32_C(0x7F800000); bits++) {
		float value;
		memcpy(&value, &bits, sizeof(value));
        std::stringstream ss;
        ss << std::hex << std::uppercase << std::setfill('0') <<
            "BITS = 0x" << std::setw(8) << bits << ", " <<
            "BITCAST(VALUE) = 0x" << std::setw(8) << fp32v_to_fp32b(value);
        std::string message = ss.str();
		
		ASSERT_EQ(bits, fp32v_to_fp32b(value), message);
		
		if (bits % 10 == 0) { // 10번마다 진행률 업데이트
			print_progress(bits, UINT32_C(0x7F800000), "test_fp32_to_bits_positive");
		}
	}
	std::cout << std::endl;
}

void test_fp32_to_bits_negative() {
	for (uint32_t bits = UINT32_C(0xFF800000); bits >= UINT32_C(0x80000000); bits--) {
		float value;
		memcpy(&value, &bits, sizeof(value));
		std::stringstream ss;
		ss << std::hex << std::uppercase << std::setfill('0') <<
			"BITS = 0x" << std::setw(8) << bits << ", " <<
			"BITCAST(VALUE) = 0x" << std::setw(8) << fp32v_to_fp32b(value);
		std::string message = ss.str();
		ASSERT_EQ(bits, fp32v_to_fp32b(value), message);

		if (bits % 10 == 0) {
			print_progress(bits, UINT32_C(0x80000000), "test_fp32_to_bits_negative");
		}
	}
	std::cout << std::endl;
}

void test_fp32_to_bits_nan() {
	for (uint32_t bits = UINT32_C(0x7F800001); bits <= UINT32_C(0x7FFFFFFF); bits++) {
		float value;
		memcpy(&value, &bits, sizeof(value));
		std::stringstream ss;
		ss << std::hex << std::uppercase << std::setfill('0') <<
			"BITS = 0x" << std::setw(8) << bits << ", " <<
			"BITCAST(VALUE) = 0x" << std::setw(8) << fp32v_to_fp32b(value);
		std::string message = ss.str();
		ASSERT_GT(fp32v_to_fp32b(value) & UINT32_C(0x7FFFFFFF), UINT32_C(0x7F800000), message);
		

		if (bits % 10 == 0) {
			print_progress(bits, UINT32_C(0x7FFFFFFF), "test_fp32_to_bits_nan (pos)");
		}
	}
	std::cout << std::endl;

	for (uint32_t bits = UINT32_C(0xFFFFFFFF); bits >= UINT32_C(0xFF800001); bits--) {
		float value;
		memcpy(&value, &bits, sizeof(value));
		std::stringstream ss;
		ss << std::hex << std::uppercase << std::setfill('0') <<
			"BITS = 0x" << std::setw(8) << bits << ", " <<
			"BITCAST(VALUE) = 0x" << std::setw(8) << fp32v_to_fp32b(value);
		std::string message = ss.str();
		ASSERT_GT(fp32v_to_fp32b(value) & UINT32_C(0x7FFFFFFF), UINT32_C(0x7F800000), message);
		
		if (bits % 10 == 0) {
			print_progress(bits, UINT32_C(0xFF800001), "test_fp32_to_bits_nan (neg)");
		}
	}
	std::cout << std::endl;
}

void test_fp32_from_bits_positive() {
	for (uint32_t bits = UINT32_C(0x00000000); bits <= UINT32_C(0x7F800000); bits++) {
		const float value = fp32b_to_fp32v(bits);
		uint32_t bitcast;
		memcpy(&bitcast, &value, sizeof(bitcast));

		std::stringstream ss;
		ss << std::hex << std::uppercase << std::setfill('0') <<
			"BITS = 0x" << std::setw(8) << bits << ", " <<
			"VALUE = 0x" << std::setw(8) << bitcast;
		std::string message = ss.str();
		ASSERT_EQ(bits, bitcast, message);
		
		if (bits % 10 == 0) {
			print_progress(bits, UINT32_C(0x7F800000), "test_fp32_from_bits_positive");
		}
	}
	std::cout << std::endl;
}

void test_fp32_from_bits_negative() {
	for (uint32_t bits = UINT32_C(0xFF800000); bits >= UINT32_C(0x80000000); bits--) {
		const float value = fp32b_to_fp32v(bits);
		uint32_t bitcast;
		memcpy(&bitcast, &value, sizeof(bitcast));

		std::stringstream ss;
		ss << std::hex << std::uppercase << std::setfill('0') <<
			"BITS = 0x" << std::setw(8) << bits << ", " <<
			"VALUE = 0x" << std::setw(8) << bitcast;
		std::string message = ss.str();
		ASSERT_EQ(bits, bitcast, message);
		
		if (bits % 10 == 0) {
			print_progress(bits, UINT32_C(0x80000000), "test_fp32_from_bits_negative");
		}
	}
	std::cout << std::endl;
}

void test_fp32_from_bits_nan() {
	for (uint32_t bits = UINT32_C(0x7F800001); bits <= UINT32_C(0x7FFFFFFF); bits++) {
		const float value = fp32b_to_fp32v(bits);

		std::stringstream ss;
		ss << std::hex << std::uppercase << std::setfill('0') <<
			"BITS = 0x" << std::setw(8) << bits;
		std::string message = ss.str();
		ASSERT_TRUE(std::isnan(value), message);

		if (bits % 10 == 0) {
			print_progress(bits, UINT32_C(0x7FFFFFFF), "test_fp32_from_bits_nan (pos)");
		}
	}

	for (uint32_t bits = UINT32_C(0xFFFFFFFF); bits >= UINT32_C(0xFF800001); bits--) {
		const float value = fp32b_to_fp32v(bits);

		std::stringstream ss;
		ss << std::hex << std::uppercase << std::setfill('0') <<
			"BITS = 0x" << std::setw(8) << bits;
		std::string message = ss.str();
		ASSERT_TRUE(std::isnan(value), message);
		
		if (bits % 10 == 0) {
			print_progress(bits, UINT32_C(0xFF800001), "test_fp32_from_bits_nan (neg)");
		}
	}
	std::cout << std::endl;
}


int main() {
    printf("Running FP16 bitcasts tests...\n");
    printf("Note: Tests now use sampling with progress bars to avoid infinite loops.\n\n");
    
    RUN_TEST(test_fp32_to_bits_positive);
    RUN_TEST(test_fp32_to_bits_negative);
    RUN_TEST(test_fp32_to_bits_nan);
    RUN_TEST(test_fp32_from_bits_positive);
    RUN_TEST(test_fp32_from_bits_negative);
    RUN_TEST(test_fp32_from_bits_nan);
    
    printf("All bitcasts tests passed!\n");
    return 0;
} 