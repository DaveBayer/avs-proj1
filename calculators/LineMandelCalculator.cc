/**
 * @file LineMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>
#include <immintrin.h>

#include "LineMandelCalculator.h"

//	#define USE_INTRINSICS

#define SIMD_512_ALIGNMENT 64
#define SIMD_512_PSIZE_32BIT 16

LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	//	extend for aligned loading without mask
	int xs_len = width;
	if (xs_len % SIMD_512_PSIZE_32BIT > 0)
		xs_len += SIMD_512_PSIZE_32BIT - (xs_len % SIMD_512_PSIZE_32BIT);

	data = (int *) _mm_malloc(height * width * sizeof(int), SIMD_512_ALIGNMENT);
	xs = (float *) _mm_malloc(xs_len * sizeof(float), SIMD_512_ALIGNMENT);
	zReal = (float *) _mm_malloc(height * width * sizeof(float), SIMD_512_ALIGNMENT);
	zImag = (float *) _mm_malloc(height * width * sizeof(float), SIMD_512_ALIGNMENT);

	if (data == nullptr || xs == nullptr || zReal == nullptr || zImag == nullptr) {
		throw std::bad_alloc();
	}

	for (int i = 0; i < xs_len; i++) {
		xs[i] = x_start + i * dx;
	}

	for (int i = 0; i < height; i++) {
		float y = y_start + i * dy;

		for (int j = 0; j < width; j++) {
			int index = i * width + j;

			data[index] = limit;
			zReal[index] = xs[j];
			zImag[index] = y;
		}
	}
}

LineMandelCalculator::~LineMandelCalculator()
{
	_mm_free(data);
	_mm_free(xs);
	_mm_free(zReal);
	_mm_free(zImag);

	data = nullptr;
	xs = nullptr;
	zReal = nullptr;
	zImag = nullptr;
}

#ifndef USE_INTRINSICS

int * LineMandelCalculator::calculateMandelbrot()
{
	for (int i = 0; i < height; i++) {
		float y = y_start + i * dy;

		int *d = data + i * width;
		float *zR = zReal + i * width;
		float *zI = zImag + i * width;

		int done = 0;

		for (int k = 0; k < limit && done < width; k++) {
			
#			pragma omp simd simdlen(SIMD_512_ALIGNMENT) reduction(+: done)
			for (int j = 0; j < width; j++) {
				float r2 = zR[j] * zR[j];
				float i2 = zI[j] * zI[j];

				d[j] = d[j] == limit && r2 + i2 > 4.0f ? done++, k : d[j];

				zI[j] = 2.0f * zR[j] * zI[j] + y;
				zR[j] = r2 - i2 + xs[j];
			}
		}
	}

	return data;
}

#else

#	define MM_PSIZE_32BIT 16

static inline __attribute__((always_inline))
__m512i mandelbrot(__m512 real, __m512 imag, int limit, __mmask16 mask)
{
	__m512i result = _mm512_set1_epi32(limit);
	__mmask16 result_mask = mask;

	const __m512 two = _mm512_set1_ps(2.f);
	const __m512 four = _mm512_set1_ps(4.f);

	__m512 zReal = real;
	__m512 zImag = imag;

	for (int i = 0; i < limit; i++) {
	//	r2 = zReal * zReal
		const __m512 r2 = _mm512_mul_ps(zReal, zReal);

	//	i2 = zImag * zImag
		const __m512 i2 = _mm512_mul_ps(zImag, zImag);

	//	if (r2 + i2 > 4.0f) then write i to result and update result_mask
		__mmask16 test_mask = _mm512_cmp_ps_mask(_mm512_add_ps(r2, i2), four, _CMP_GT_OS);

		result = _mm512_mask_mov_epi32(result, test_mask & result_mask, _mm512_set1_epi32(i));
		result_mask &= ~test_mask;

		if (result_mask == 0x0000U)
			break;
		
	//	zImag = 2.0f * zReal * zImag + imag;
		zImag = _mm512_fmadd_ps(_mm512_mul_ps(two, zReal), zImag, imag);

	//	zReal = r2 - i2 + real;
		zReal = _mm512_add_ps(_mm512_sub_ps(r2, i2), real);
	}

	return result;
}

int *LineMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers
	int *pdata = data;
	
	for (int i = 0; i < height; i++) {
	//	y = y_start + i * dy
		const __m512 y_ps = _mm512_set1_ps(y_start + i * dy);

		for (int j = 0; j < width; j += MM_PSIZE_32BIT) {

		//	get mask for avx instructions and data pointer increment
			__mmask16 mask = 0xffffU;
			int inc = MM_PSIZE_32BIT;

			int diff = width - j;
			
			if (diff < MM_PSIZE_32BIT) {
				mask >>= MM_PSIZE_32BIT - diff;
				inc = diff;
			}

			__m512 x_ps = _mm512_load_ps(xs + j * MM_PSIZE_32BIT);

			__m512i values = mandelbrot(x_ps, y_ps, limit, mask);

		//	store values in memory pointed by pdata using mask
			_mm512_mask_storeu_epi32(pdata, mask, values);

			pdata += inc;
		}
	}

	return data;
}

#endif	//	USE_INTRINSICS
