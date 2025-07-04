#include "simple_test.h"
#include <fp16.h>
#include "tables.h"
#include <cstdint>
#include <cmath>

void test_ieee_normalized_powers_of_2() {
    const uint16_t min_po2_f16   = UINT16_C(0x0400);
    const uint16_t eighths_f16   = UINT16_C(0x3000);
    const uint16_t quarter_f16   = UINT16_C(0x3400);
    const uint16_t half_f16      = UINT16_C(0x3800);
    const uint16_t one_f16       = UINT16_C(0x3C00);
    const uint16_t two_f16       = UINT16_C(0x4000);
    const uint16_t four_f16      = UINT16_C(0x4400);
    const uint16_t eight_f16     = UINT16_C(0x4800);
    const uint16_t sixteen_f16   = UINT16_C(0x4C00);
    const uint16_t thirtytwo_f16 = UINT16_C(0x5000);
    const uint16_t sixtyfour_f16 = UINT16_C(0x5400);
    const uint16_t max_po2_f16   = UINT16_C(0x7800);

    const uint32_t min_po2_f32   = UINT32_C(0x38800000);
    const uint32_t eighths_f32   = UINT32_C(0x3E000000);
    const uint32_t quarter_f32   = UINT32_C(0x3E800000);
    const uint32_t half_f32      = UINT32_C(0x3F000000);
    const uint32_t one_f32       = UINT32_C(0x3F800000);
    const uint32_t two_f32       = UINT32_C(0x40000000);
    const uint32_t four_f32      = UINT32_C(0x40800000);
    const uint32_t eight_f32     = UINT32_C(0x41000000);
    const uint32_t sixteen_f32   = UINT32_C(0x41800000);
    const uint32_t thirtytwo_f32 = UINT32_C(0x42000000);
    const uint32_t sixtyfour_f32 = UINT32_C(0x42800000);
    const uint32_t max_po2_f32   = UINT32_C(0x47000000);

    // Test each power of 2
    const float min_po2_value = fp16_ieee_to_fp32_value(min_po2_f16);
    uint32_t min_po2_bits;
    memcpy(&min_po2_bits, &min_po2_value, sizeof(min_po2_bits));
    TEST_ASSERT_BITS_EQ(min_po2_f32, min_po2_bits, "min_po2 test failed");

    const float eighths_value = fp16_ieee_to_fp32_value(eighths_f16);
    uint32_t eighths_bits;
    memcpy(&eighths_bits, &eighths_value, sizeof(eighths_bits));
    TEST_ASSERT_BITS_EQ(eighths_f32, eighths_bits, "eighths test failed");

    const float quarter_value = fp16_ieee_to_fp32_value(quarter_f16);
    uint32_t quarter_bits;
    memcpy(&quarter_bits, &quarter_value, sizeof(quarter_bits));
    TEST_ASSERT_BITS_EQ(quarter_f32, quarter_bits, "quarter test failed");

    const float half_value = fp16_ieee_to_fp32_value(half_f16);
    uint32_t half_bits;
    memcpy(&half_bits, &half_value, sizeof(half_bits));
    TEST_ASSERT_BITS_EQ(half_f32, half_bits, "half test failed");

    const float one_value = fp16_ieee_to_fp32_value(one_f16);
    uint32_t one_bits;
    memcpy(&one_bits, &one_value, sizeof(one_bits));
    TEST_ASSERT_BITS_EQ(one_f32, one_bits, "one test failed");

    const float two_value = fp16_ieee_to_fp32_value(two_f16);
    uint32_t two_bits;
    memcpy(&two_bits, &two_value, sizeof(two_bits));
    TEST_ASSERT_BITS_EQ(two_f32, two_bits, "two test failed");

    const float four_value = fp16_ieee_to_fp32_value(four_f16);
    uint32_t four_bits;
    memcpy(&four_bits, &four_value, sizeof(four_bits));
    TEST_ASSERT_BITS_EQ(four_f32, four_bits, "four test failed");

    const float eight_value = fp16_ieee_to_fp32_value(eight_f16);
    uint32_t eight_bits;
    memcpy(&eight_bits, &eight_value, sizeof(eight_bits));
    TEST_ASSERT_BITS_EQ(eight_f32, eight_bits, "eight test failed");

    const float sixteen_value = fp16_ieee_to_fp32_value(sixteen_f16);
    uint32_t sixteen_bits;
    memcpy(&sixteen_bits, &sixteen_value, sizeof(sixteen_bits));
    TEST_ASSERT_BITS_EQ(sixteen_f32, sixteen_bits, "sixteen test failed");

    const float thirtytwo_value = fp16_ieee_to_fp32_value(thirtytwo_f16);
    uint32_t thirtytwo_bits;
    memcpy(&thirtytwo_bits, &thirtytwo_value, sizeof(thirtytwo_bits));
    TEST_ASSERT_BITS_EQ(thirtytwo_f32, thirtytwo_bits, "thirtytwo test failed");

    const float sixtyfour_value = fp16_ieee_to_fp32_value(sixtyfour_f16);
    uint32_t sixtyfour_bits;
    memcpy(&sixtyfour_bits, &sixtyfour_value, sizeof(sixtyfour_bits));
    TEST_ASSERT_BITS_EQ(sixtyfour_f32, sixtyfour_bits, "sixtyfour test failed");

    const float max_po2_value = fp16_ieee_to_fp32_value(max_po2_f16);
    uint32_t max_po2_bits;
    memcpy(&max_po2_bits, &max_po2_value, sizeof(max_po2_bits));
    TEST_ASSERT_BITS_EQ(max_po2_f32, max_po2_bits, "max_po2 test failed");
}

void test_ieee_denormalized_powers_of_2() {
    const uint16_t exp2_minus_15_f16 = UINT16_C(0x0200);
    const uint16_t exp2_minus_16_f16 = UINT16_C(0x0100);
    const uint16_t exp2_minus_17_f16 = UINT16_C(0x0080);
    const uint16_t exp2_minus_18_f16 = UINT16_C(0x0040);
    const uint16_t exp2_minus_19_f16 = UINT16_C(0x0020);
    const uint16_t exp2_minus_20_f16 = UINT16_C(0x0010);
    const uint16_t exp2_minus_21_f16 = UINT16_C(0x0008);
    const uint16_t exp2_minus_22_f16 = UINT16_C(0x0004);
    const uint16_t exp2_minus_23_f16 = UINT16_C(0x0002);
    const uint16_t exp2_minus_24_f16 = UINT16_C(0x0001);

    const uint32_t exp2_minus_15_f32 = UINT32_C(0x38000000);
    const uint32_t exp2_minus_16_f32 = UINT32_C(0x37800000);
    const uint32_t exp2_minus_17_f32 = UINT32_C(0x37000000);
    const uint32_t exp2_minus_18_f32 = UINT32_C(0x36800000);
    const uint32_t exp2_minus_19_f32 = UINT32_C(0x36000000);
    const uint32_t exp2_minus_20_f32 = UINT32_C(0x35800000);
    const uint32_t exp2_minus_21_f32 = UINT32_C(0x35000000);
    const uint32_t exp2_minus_22_f32 = UINT32_C(0x34800000);
    const uint32_t exp2_minus_23_f32 = UINT32_C(0x34000000);
    const uint32_t exp2_minus_24_f32 = UINT32_C(0x33800000);

    // Test each denormalized power of 2
    const float exp2_minus_15_value = fp16_ieee_to_fp32_value(exp2_minus_15_f16);
    uint32_t exp2_minus_15_bits;
    memcpy(&exp2_minus_15_bits, &exp2_minus_15_value, sizeof(exp2_minus_15_bits));
    TEST_ASSERT_BITS_EQ(exp2_minus_15_f32, exp2_minus_15_bits, "exp2_minus_15 test failed");

    const float exp2_minus_16_value = fp16_ieee_to_fp32_value(exp2_minus_16_f16);
    uint32_t exp2_minus_16_bits;
    memcpy(&exp2_minus_16_bits, &exp2_minus_16_value, sizeof(exp2_minus_16_bits));
    TEST_ASSERT_BITS_EQ(exp2_minus_16_f32, exp2_minus_16_bits, "exp2_minus_16 test failed");

    const float exp2_minus_17_value = fp16_ieee_to_fp32_value(exp2_minus_17_f16);
    uint32_t exp2_minus_17_bits;
    memcpy(&exp2_minus_17_bits, &exp2_minus_17_value, sizeof(exp2_minus_17_bits));
    TEST_ASSERT_BITS_EQ(exp2_minus_17_f32, exp2_minus_17_bits, "exp2_minus_17 test failed");

    const float exp2_minus_18_value = fp16_ieee_to_fp32_value(exp2_minus_18_f16);
    uint32_t exp2_minus_18_bits;
    memcpy(&exp2_minus_18_bits, &exp2_minus_18_value, sizeof(exp2_minus_18_bits));
    TEST_ASSERT_BITS_EQ(exp2_minus_18_f32, exp2_minus_18_bits, "exp2_minus_18 test failed");

    const float exp2_minus_19_value = fp16_ieee_to_fp32_value(exp2_minus_19_f16);
    uint32_t exp2_minus_19_bits;
    memcpy(&exp2_minus_19_bits, &exp2_minus_19_value, sizeof(exp2_minus_19_bits));
    TEST_ASSERT_BITS_EQ(exp2_minus_19_f32, exp2_minus_19_bits, "exp2_minus_19 test failed");

    const float exp2_minus_20_value = fp16_ieee_to_fp32_value(exp2_minus_20_f16);
    uint32_t exp2_minus_20_bits;
    memcpy(&exp2_minus_20_bits, &exp2_minus_20_value, sizeof(exp2_minus_20_bits));
    TEST_ASSERT_BITS_EQ(exp2_minus_20_f32, exp2_minus_20_bits, "exp2_minus_20 test failed");

    const float exp2_minus_21_value = fp16_ieee_to_fp32_value(exp2_minus_21_f16);
    uint32_t exp2_minus_21_bits;
    memcpy(&exp2_minus_21_bits, &exp2_minus_21_value, sizeof(exp2_minus_21_bits));
    TEST_ASSERT_BITS_EQ(exp2_minus_21_f32, exp2_minus_21_bits, "exp2_minus_21 test failed");

    const float exp2_minus_22_value = fp16_ieee_to_fp32_value(exp2_minus_22_f16);
    uint32_t exp2_minus_22_bits;
    memcpy(&exp2_minus_22_bits, &exp2_minus_22_value, sizeof(exp2_minus_22_bits));
    TEST_ASSERT_BITS_EQ(exp2_minus_22_f32, exp2_minus_22_bits, "exp2_minus_22 test failed");

    const float exp2_minus_23_value = fp16_ieee_to_fp32_value(exp2_minus_23_f16);
    uint32_t exp2_minus_23_bits;
    memcpy(&exp2_minus_23_bits, &exp2_minus_23_value, sizeof(exp2_minus_23_bits));
    TEST_ASSERT_BITS_EQ(exp2_minus_23_f32, exp2_minus_23_bits, "exp2_minus_23 test failed");

    const float exp2_minus_24_value = fp16_ieee_to_fp32_value(exp2_minus_24_f16);
    uint32_t exp2_minus_24_bits;
    memcpy(&exp2_minus_24_bits, &exp2_minus_24_value, sizeof(exp2_minus_24_bits));
    TEST_ASSERT_BITS_EQ(exp2_minus_24_f32, exp2_minus_24_bits, "exp2_minus_24 test failed");
}

void test_ieee_special_values() {
    // Test special values: zero, infinity, NaN
    const uint16_t zero_f16 = UINT16_C(0x0000);
    const uint16_t neg_zero_f16 = UINT16_C(0x8000);
    const uint16_t inf_f16 = UINT16_C(0x7C00);
    const uint16_t neg_inf_f16 = UINT16_C(0xFC00);
    const uint16_t nan_f16 = UINT16_C(0x7C01);

    const uint32_t zero_f32 = UINT32_C(0x00000000);
    const uint32_t neg_zero_f32 = UINT32_C(0x80000000);
    const uint32_t inf_f32 = UINT32_C(0x7F800000);
    const uint32_t neg_inf_f32 = UINT32_C(0xFF800000);

    // Test zero
    const float zero_value = fp16_ieee_to_fp32_value(zero_f16);
    uint32_t zero_bits;
    memcpy(&zero_bits, &zero_value, sizeof(zero_bits));
    TEST_ASSERT_BITS_EQ(zero_f32, zero_bits, "zero test failed");

    // Test negative zero
    const float neg_zero_value = fp16_ieee_to_fp32_value(neg_zero_f16);
    uint32_t neg_zero_bits;
    memcpy(&neg_zero_bits, &neg_zero_value, sizeof(neg_zero_bits));
    TEST_ASSERT_BITS_EQ(neg_zero_f32, neg_zero_bits, "negative zero test failed");

    // Test infinity
    const float inf_value = fp16_ieee_to_fp32_value(inf_f16);
    uint32_t inf_bits;
    memcpy(&inf_bits, &inf_value, sizeof(inf_bits));
    TEST_ASSERT_BITS_EQ(inf_f32, inf_bits, "infinity test failed");

    // Test negative infinity
    const float neg_inf_value = fp16_ieee_to_fp32_value(neg_inf_f16);
    uint32_t neg_inf_bits;
    memcpy(&neg_inf_bits, &neg_inf_value, sizeof(neg_inf_bits));
    TEST_ASSERT_BITS_EQ(neg_inf_f32, neg_inf_bits, "negative infinity test failed");

    // Test NaN
    const float nan_value = fp16_ieee_to_fp32_value(nan_f16);
    TEST_ASSERT_BITS_EQ(isnan(nan_value), "NaN test failed");
}

int main() {
    printf("Running FP16 IEEE tests...\n");
    
    RUN_TEST(test_ieee_normalized_powers_of_2);
    RUN_TEST(test_ieee_denormalized_powers_of_2);
    RUN_TEST(test_ieee_special_values);
    
    printf("All IEEE tests passed!\n");
    return 0;
} 