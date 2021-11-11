/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>
#include <immintrin.h>

#include "BatchMandelCalculator.h"

//	#define USE_INTRINSICS

#define SIMD_512_ALIGNMENT 64
#define BATCH_SIZE 128

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	data = (int *) _mm_malloc(height * width * sizeof(int), SIMD_512_ALIGNMENT);
	xs = (float *) _mm_malloc(width * sizeof(float), SIMD_512_ALIGNMENT);
	zReal = (float *) _mm_malloc(height * width * sizeof(float), SIMD_512_ALIGNMENT);
	zImag = (float *) _mm_malloc(height * width * sizeof(float), SIMD_512_ALIGNMENT);

	if (data == nullptr || xs == nullptr || zReal == nullptr || zImag == nullptr) {
		throw std::bad_alloc();
	}

	for (int i = 0; i < width; i++) {
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

BatchMandelCalculator::~BatchMandelCalculator()
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

int * BatchMandelCalculator::calculateMandelbrot()
{
	float dx_f = dx, dy_f = dy, y_start_f = y_start, x_start_f = x_start;

	for (int i = 0; i < height; i += BATCH_SIZE) {
		int h_limit = i + BATCH_SIZE > height ? height : i + BATCH_SIZE;

		for (int j = 0; j < width; j += BATCH_SIZE) {
			int w_limit = j + BATCH_SIZE > width ? width : j + BATCH_SIZE;

			for (int l = i; l < h_limit; l++) {
				float y = y_start_f + l * dy_f;

				int *d = data + l * width;
				float *zR = zReal + l * width;
				float *zI = zImag + l * width;

				int done = 0;

				for (int k = 0; k < limit && done < BATCH_SIZE; k++) {
					
#					pragma omp simd reduction(+: done) simdlen(SIMD_512_ALIGNMENT)
					for (int m = j; m < w_limit; m++) {
						float r2 = zR[m] * zR[m];
						float i2 = zI[m] * zI[m];

						d[m] = d[m] == limit && r2 + i2 > 4.0f ? done++, k : d[m];

						zI[m] = 2.0f * zR[m] * zI[m] + y;
						zR[m] = r2 - i2 + xs[m];
					}
				}
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
	__m512i result = _mm512_setzero_epi32();
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
			return result;
		
	//	zImag = 2.0f * zReal * zImag + imag;
		zImag = _mm512_fmadd_ps(_mm512_mul_ps(two, zReal), zImag, imag);

	//	zReal = r2 - i2 + real;
		zReal = _mm512_add_ps(_mm512_sub_ps(r2, i2), real);
	}

	result = _mm512_mask_mov_epi32(result, result_mask, _mm512_set1_epi32(limit));

	return result;
}

int * BatchMandelCalculator::calculateMandelbrot()
{
	for (int i = 0; i < height; i += MM_PSIZE_32BIT) {
		for (int j = 0; j < width; j += MM_PSIZE_32BIT) {
			int h_limit = i + MM_PSIZE_32BIT;

			if (h_limit >= height)
				h_limit = height;

			__mmask16 mask = 0xffffU;
			int diff = width - j;
			if (diff < MM_PSIZE_32BIT)
				mask >>= MM_PSIZE_32BIT - diff;

			int *pdata = data + i * width + i;

			__m512 dx_ps, x_start_ps, inc_ps;

			dx_ps = _mm512_set1_ps(static_cast<float>(dx));
			x_start_ps = _mm512_set1_ps(static_cast<float>(x_start));

			inc_ps = _mm512_set_ps(15.f, 14.f, 13.f, 12.f, 11.f, 10.f, 9.f, 8.f,
						   		   7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);

			for (int k = i; k < h_limit; k++) {
				const __m512 y_ps = _mm512_set1_ps(y_start + k * dy);

			//	prepare j
				__m512 j_ps = _mm512_add_ps(_mm512_set1_ps(j), inc_ps);

			//	x = x_start + j * dx
				__m512 x_ps = _mm512_add_ps(x_start_ps, _mm512_mul_ps(j_ps, dx_ps));

				__m512i values = mandelbrot(x_ps, y_ps, limit, mask);

			//	store values in memory pointed by pdata using mask
				_mm512_mask_storeu_epi32(pdata, mask, values);

				pdata += width;
			}
		}
	}

	return data;
}

#endif	//	USE_INTRINSICS