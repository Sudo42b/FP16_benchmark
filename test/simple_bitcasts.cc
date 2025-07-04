#include "simple_test.h"
#include <fp16.h>
#include <cstdint>
#include <cmath>

void test_fp32_to_bits_positive() {
    for (uint32_t bits = UINT32_C(0x00000000); bits <= UINT32_C(0x7F800000); bits++) {
        float value;
        memcpy(&value, &bits, sizeof(value));
        
        uint32_t result = fp32v_to_fp32b(value);
        TEST_ASSERT_BITS_EQ(bits, result, "fp32_to_bits positive test failed");
    }
}

void test_fp32_to_bits_negative() {
    for (uint32_t bits = UINT32_C(0xFF800000); bits >= UINT32_C(0x80000000); bits--) {
        float value;
        memcpy(&value, &bits, sizeof(value));
        
        uint32_t result = fp32v_to_fp32b(value);
        TEST_ASSERT_BITS_EQ(bits, result, "fp32_to_bits negative test failed");
    }
}

void test_fp32_to_bits_nan() {
    for (uint32_t bits = UINT32_C(0x7F800001); bits <= UINT32_C(0x7FFFFFFF); bits++) {
        float value;
        memcpy(&value, &bits, sizeof(value));
        
        uint32_t result = fp32v_to_fp32b(value);
        TEST_ASSERT((result & UINT32_C(0x7FFFFFFF)) > UINT32_C(0x7F800000), 
                   "fp32_to_bits nan test failed");
    }
    
    for (uint32_t bits = UINT32_C(0xFFFFFFFF); bits >= UINT32_C(0xFF800001); bits--) {
        float value;
        memcpy(&value, &bits, sizeof(value));
        
        uint32_t result = fp32v_to_fp32b(value);
        TEST_ASSERT((result & UINT32_C(0x7FFFFFFF)) > UINT32_C(0x7F800000), 
                   "fp32_to_bits nan test failed");
    }
}

void test_fp32_from_bits_positive() {
    for (uint32_t bits = UINT32_C(0x00000000); bits <= UINT32_C(0x7F800000); bits++) {
        const float value = fp32b_to_fp32v(bits);
        uint32_t bitcast;
        memcpy(&bitcast, &value, sizeof(bitcast));
        
        TEST_ASSERT_BITS_EQ(bits, bitcast, "fp32_from_bits positive test failed");
    }
}

void test_fp32_from_bits_negative() {
    for (uint32_t bits = UINT32_C(0xFF800000); bits >= UINT32_C(0x80000000); bits--) {
        const float value = fp32b_to_fp32v(bits);
        uint32_t bitcast;
        memcpy(&bitcast, &value, sizeof(bitcast));
        
        TEST_ASSERT_BITS_EQ(bits, bitcast, "fp32_from_bits negative test failed");
    }
}

void test_fp32_from_bits_nan() {
    for (uint32_t bits = UINT32_C(0x7F800001); bits <= UINT32_C(0x7FFFFFFF); bits++) {
        const float value = fp32b_to_fp32v(bits);
        TEST_ASSERT(isnan(value), "fp32_from_bits nan test failed");
    }
    
    for (uint32_t bits = UINT32_C(0xFFFFFFFF); bits >= UINT32_C(0xFF800001); bits--) {
        const float value = fp32b_to_fp32v(bits);
        TEST_ASSERT(isnan(value), "fp32_from_bits nan test failed");
    }
}

int main() {
    printf("Running FP16 bitcasts tests...\n");
    
    RUN_TEST(test_fp32_to_bits_positive);
    RUN_TEST(test_fp32_to_bits_negative);
    RUN_TEST(test_fp32_to_bits_nan);
    RUN_TEST(test_fp32_from_bits_positive);
    RUN_TEST(test_fp32_from_bits_negative);
    RUN_TEST(test_fp32_from_bits_nan);
    
    printf("All bitcasts tests passed!\n");
    return 0;
} 