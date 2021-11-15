/**
 * @file LineMandelCalculator.cc
 * @author David Bayer <xbayer09@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 2021/11/14
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

constexpr int SIMD_512_PSIZE_32BIT = 16;

LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	int xs_len;

	if (matrixBaseSize % SIMD_512_PSIZE_32BIT == 0)
		xs_len = width;
	else
		xs_len = width + SIMD_512_PSIZE_32BIT - (width % SIMD_512_PSIZE_32BIT);

	data = (int *) _mm_malloc(height * width * sizeof(int), SIMD_512_ALIGNMENT);
	xs = (float *) _mm_malloc(xs_len * sizeof(float), SIMD_512_ALIGNMENT);
	ys = (float *) _mm_malloc(height * sizeof(float), SIMD_512_ALIGNMENT);

	if (data == nullptr || xs == nullptr || ys == nullptr) {
		throw std::bad_alloc();
	}

	for (int i = 0; i < xs_len; i++) {
		xs[i] = x_start + i * dx;
	}

	for (int i = 0; i < height; i++) {
		ys[i] = y_start + i * dy;
	}

#ifndef USE_INTRINSICS
	zReal = (float *) _mm_malloc(height * width * sizeof(float), SIMD_512_ALIGNMENT);
	zImag = (float *) _mm_malloc(height * width * sizeof(float), SIMD_512_ALIGNMENT);

	if (zReal == nullptr || zImag == nullptr) {
		throw std::bad_alloc();
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int index = i * width + j;

			data[index] = limit;
			zReal[index] = xs[j];
			zImag[index] = ys[i];
		}
	}
#endif	//	USE_INTRINSICS
}

LineMandelCalculator::~LineMandelCalculator()
{
	_mm_free(data);
	_mm_free(xs);
	_mm_free(ys);

	data = nullptr;
	xs = nullptr;
	ys = nullptr;

#ifndef USE_INTRINSICS
	_mm_free(zReal);
	_mm_free(zImag);

	zReal = nullptr;
	zImag = nullptr;
#endif	//	USE_INTRINSICS
}

#ifndef USE_INTRINSICS

int *LineMandelCalculator::calculateMandelbrot()
{
	for (int i = 0; i < height; i++) {
		int *d = data + i * width;
		float *zR = zReal + i * width;
		float *zI = zImag + i * width;

		int done = 0;

		for (int k = 0; k < limit && done < width; k++) {
			
#			pragma omp simd simdlen(SIMD_512_ALIGNMENT) reduction(+: done)
			for (int j = 0; j < width; j++) {
				float x = x_start + j * dx;

				float r2 = zR[j] * zR[j];
				float i2 = zI[j] * zI[j];

				d[j] = d[j] == limit && r2 + i2 > 4.0f ? done++, k : d[j];

				zI[j] = 2.0f * zR[j] * zI[j] + ys[i];
				zR[j] = r2 - i2 + xs[j];
			}
		}
	}

	return data;
}

#else

static inline __attribute__((always_inline))
__m512i mandelbrot(__m512 real, __m512 imag, int limit, __mmask16 mask = 0xffffU)
{
	__m512i result = _mm512_set1_epi32(limit);
	__mmask16 result_mask = mask;

	const __m512 two = _mm512_set1_ps(2.f);
	const __m512 four = _mm512_set1_ps(4.f);

	__m512 zReal = real;
	__m512 zImag = imag;

	for (int i = 0; i < limit && result_mask != 0x0000U; i++) {
		const __m512 r2 = _mm512_mul_ps(zReal, zReal);
		const __m512 i2 = _mm512_mul_ps(zImag, zImag);

		__mmask16 test_mask = _mm512_cmp_ps_mask(_mm512_add_ps(r2, i2), four, _CMP_GT_OS);
		result = _mm512_mask_mov_epi32(result, test_mask & result_mask, _mm512_set1_epi32(i));
		
		zImag = _mm512_fmadd_ps(_mm512_mul_ps(two, zReal), zImag, imag);
		zReal = _mm512_add_ps(_mm512_sub_ps(r2, i2), real);

		result_mask &= ~test_mask;
	}

	return result;
}

int *LineMandelCalculator::calculateMandelbrot()
{
	if (height % SIMD_512_PSIZE_32BIT == 0 && width % SIMD_512_PSIZE_32BIT == 0) {
		for (int i = 0; i < height; i++) {
			const __m512 y_ps = _mm512_set1_ps(ys[i]);
			
			for (int j = 0; j < width; j += SIMD_512_PSIZE_32BIT) {
				int *pdata = data + i * width + j;

				__m512 x_ps = _mm512_load_ps(xs + j);
				__m512i values = mandelbrot(x_ps, y_ps, limit);

				_mm512_store_epi32(pdata, values);
			}
		}
	} else {
		for (int i = 0; i < height; i++) {
			const __m512 y_ps = _mm512_set1_ps(ys[i]);
			
			for (int j = 0; j < width; j += SIMD_512_PSIZE_32BIT) {
				int *pdata = data + i * width + j;
				__mmask16 mask = 0xffffU;

				int diff = width - j;
				if (diff < SIMD_512_PSIZE_32BIT)
					mask >>= SIMD_512_PSIZE_32BIT - diff;

				__m512 x_ps = _mm512_load_ps(xs + j);
				__m512i values = mandelbrot(x_ps, y_ps, limit, mask);

				_mm512_mask_storeu_epi32(pdata, mask, values);
			}
		}
	}

	return data;
}

#endif	//	USE_INTRINSICS
