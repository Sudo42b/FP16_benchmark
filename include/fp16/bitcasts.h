#pragma once
#ifndef FP16_BITCASTS_H
#define FP16_BITCASTS_H

#include <cstdint>
#include <stdint.h>

static inline float fp32b_to_fp32v(uint32_t w) {
	union {
		uint32_t as_bits;
		float as_value;
	} fp32 = { w };
	return fp32.as_value;
}

static inline uint32_t fp32v_to_fp32b(float f) {
	union {
		float as_value;
		uint32_t as_bits;
	} fp32 = { f };
	return fp32.as_bits;
}

static inline double fp64b_to_fp64v(uint64_t w) {
	union {
		uint64_t as_bits;
		double as_value;
	} fp64 = { w };
	return fp64.as_value;
}

static inline uint64_t fp64v_to_fp64b(double f) {
	union {
		double as_value;
		uint64_t as_bits;
	} fp64 = { f };
	return fp64.as_bits;
}

#endif /* FP16_BITCASTS_H */
