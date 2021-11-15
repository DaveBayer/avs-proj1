/**
 * @file BatchMandelCalculator.cc
 * @author David Bayer <xbayer09@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 2021/11/14
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

constexpr int SIMD_512_PSIZE_32BIT = 16;
constexpr int L3_batch_size = 512;
constexpr int L2_batch_size = 128;

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
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

	if (matrixBaseSize % L3_batch_size == 0)
		chosen_calculator = &BatchMandelCalculator::calculateMandelbrot_aligned;
	else
		chosen_calculator = &BatchMandelCalculator::calculateMandelbrot_unaligned;
}

BatchMandelCalculator::~BatchMandelCalculator()
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

void BatchMandelCalculator::calculateMandelbrot_aligned()
{
	for (int i3 = 0; i3 < height / L3_batch_size; i3++) {
		for (int j3 = 0; j3 < width / L3_batch_size; j3++) {
			for (int i2 = 0; i2 < L3_batch_size / L2_batch_size; i2++) {
				for (int j2 = 0; j2 < L3_batch_size / L2_batch_size; j2++) {

					int done = 0;
					const int todo = L2_batch_size * L2_batch_size;

					for (int k = 0; k < limit && done < todo; k++) {
						for (int i1 = 0; i1 < L2_batch_size; i1++) {

#							pragma omp simd reduction(+: done) simdlen(SIMD_512_ALIGNMENT)
							for (int j1 = 0; j1 < L2_batch_size; j1++) {
								const int row = i3 * L3_batch_size + i2 * L2_batch_size + i1;
								const int col = j3 * L3_batch_size + j2 * L2_batch_size + j1;

								const int index = row * width + col;

								float r2 = zReal[index] * zReal[index];
								float i2 = zImag[index] * zImag[index];

								data[index] = data[index] == limit && r2 + i2 > 4.0f ? done++, k : data[index];

								zImag[index] = 2.0f * zReal[index] * zImag[index] + ys[row];
								zReal[index] = r2 - i2 + xs[col];
							}
						}
					}
				}
			}
		}
	}
}

void BatchMandelCalculator::calculateMandelbrot_unaligned()
{
	for (int i3 = 0; i3 < height; i3 += L3_batch_size) {
		int h2_limit = i3 + L3_batch_size > height ? height - i3 : L3_batch_size;

		for (int j3 = 0; j3 < width; j3 += L3_batch_size) {
			int w2_limit = j3 + L3_batch_size > width ? width - j3 : L3_batch_size;

			for (int i2 = 0; i2 < h2_limit; i2 += L2_batch_size) {
				int h1_limit = i3 + i2 + L2_batch_size > height ? height - i3 - i2 : L2_batch_size;

				for (int j2 = 0; j2 < w2_limit; j2 += L2_batch_size) {
					int w1_limit = j3 + j2 + L2_batch_size > width ? width - j3 - j2 : L2_batch_size;

					int done = 0;
					const int todo = h1_limit * w1_limit;

					for (int k = 0; k < limit && done < todo; k++) {
						for (int i1 = 0; i1 < h1_limit; i1++) {

#							pragma omp simd reduction(+: done) simdlen(SIMD_512_ALIGNMENT)
							for (int j1 = 0; j1 < w1_limit; j1++) {
								const int row = i3 + i2 + i1;
								const int col = j3 + j2 + j1;

								const int index = row * width + col;

								float r2 = zReal[index] * zReal[index];
								float i2 = zImag[index] * zImag[index];

								data[index] = data[index] == limit && r2 + i2 > 4.0f ? done++, k : data[index];

								zImag[index] = 2.0f * zReal[index] * zImag[index] + ys[row];
								zReal[index] = r2 - i2 + xs[col];
							}
						}
					}
				}
			}
		}
	}
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

void BatchMandelCalculator::calculateMandelbrot_aligned()
{
	for (int i3 = 0; i3 < height / L3_batch_size; i3++) {
		for (int j3 = 0; j3 < width / L3_batch_size; j3++) {
			for (int i2 = 0; i2 < L3_batch_size / L2_batch_size; i2++) {
				for (int j2 = 0; j2 < L3_batch_size / L2_batch_size; j2++) {
					for (int i1 = 0; i1 < L2_batch_size; i1++) {
						for (int j1 = 0; j1 < L2_batch_size; j1 += SIMD_512_PSIZE_32BIT) {
							const int row = i3 * L3_batch_size + i2 * L2_batch_size + i1;
							const int col = j3 * L3_batch_size + j2 * L2_batch_size + j1;

							int *pdata = data + row * width + col;

							__m512 x_ps = _mm512_load_ps(xs + col);
							__m512 y_ps = _mm512_set1_ps(ys[row]);

							__m512i values = mandelbrot(x_ps, y_ps, limit);

							_mm512_store_epi32(pdata, values);
						}
					}
				}
			}
		}
	}
}

void BatchMandelCalculator::calculateMandelbrot_unaligned()
{
	for (int i3 = 0; i3 < height; i3 += L3_batch_size) {
		int h2_limit = i3 + L3_batch_size > height ? height - i3 : L3_batch_size;

		for (int j3 = 0; j3 < width; j3 += L3_batch_size) {
			int w2_limit = j3 + L3_batch_size > width ? width - j3 : L3_batch_size;

			for (int i2 = 0; i2 < h2_limit; i2 += L2_batch_size) {
				int h1_limit = i3 + i2 + L2_batch_size > height ? height - i3 - i2 : L2_batch_size;

				for (int j2 = 0; j2 < w2_limit; j2 += L2_batch_size) {
					int w1_limit = j3 + j2 + L2_batch_size > width ? width - j3 - j2 : L2_batch_size;

					for (int i1 = 0; i1 < h1_limit; i1++) {
						for (int j1 = 0; j1 < w1_limit; j1 += SIMD_512_PSIZE_32BIT) {
							const int row = i3 + i2 + i1;
							const int col = j3 + j2 + j1;

							int *pdata = data + row * width + col;

							__mmask16 mask = 0xffffU;
							int diff = width - col;
							if (diff < SIMD_512_PSIZE_32BIT)
								mask >>= SIMD_512_PSIZE_32BIT - diff;

							__m512 x_ps = _mm512_load_ps(xs + col);
							__m512 y_ps = _mm512_set1_ps(ys[row]);

							__m512i values = mandelbrot(x_ps, y_ps, limit, mask);

							_mm512_mask_storeu_epi32(pdata, mask, values);
						}
					}
				}
			}
		}
	}
}

#endif	//	USE_INTRINSICS

int *BatchMandelCalculator::calculateMandelbrot()
{
	(this->*chosen_calculator)();

	return data;
}
