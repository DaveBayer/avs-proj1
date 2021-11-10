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

#define SIMD_512_ALIGNMENT 64
#define BATCH_SIZE 128

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	data = (int *) _mm_malloc(height * width * sizeof(int), SIMD_512_ALIGNMENT);
	zReal = (float *) _mm_malloc(height * width * sizeof(float), SIMD_512_ALIGNMENT);
	zImag = (float *) _mm_malloc(height * width * sizeof(float), SIMD_512_ALIGNMENT);

	if (data == nullptr || zReal == nullptr || zImag == nullptr) {
		throw std::bad_alloc();
	}

	for (int i = 0; i < height; i++) {
		float y = y_start + i * dy;

		for (int j = 0; j < width; j++) {
			float x = x_start + j * dx;

			int index = i * width + j;

			data[index] = limit;
			zReal[index] = x;
			zImag[index] = y;
		}
	}
}

BatchMandelCalculator::~BatchMandelCalculator()
{
	_mm_free(data);
	_mm_free(zReal);
	_mm_free(zImag);

	data = nullptr;
	zReal = nullptr;
	zImag = nullptr;
}



int * BatchMandelCalculator::calculateMandelbrot()
{
	for (int i = 0; i < height; i += BATCH_SIZE) {
		int h_limit = i + BATCH_SIZE > height ? height : i + BATCH_SIZE;

		for (int j = 0; j < width; j += BATCH_SIZE) {
			int w_limit = j + BATCH_SIZE > width ? width : j + BATCH_SIZE;

			for (int l = i; l < h_limit; l++) {
				float y = y_start + l * dy;

				int *d = data + l * width;
				float *zR = zReal + l * width;
				float *zI = zImag + l * width;

				int done = 0;

				for (int k = 0; k < limit && done < BATCH_SIZE; k++) {
					
#					pragma omp simd reduction(+: done) simdlen(SIMD_512_ALIGNMENT)
					for (int m = j; m < w_limit; m++) {
						float x = x_start + m * dx;

						float r2 = zR[m] * zR[m];
						float i2 = zI[m] * zI[m];

						d[m] = d[m] == limit && r2 + i2 > 4.0f ? done++, k : d[m];

						zI[m] = 2.0f * zR[m] * zI[m] + y;
						zR[m] = r2 - i2 + x;
					}
				}
			}
		}
	}
/*
	for (int i = 0; i < height; i++) {
		std::cout << i << ":\t";

		for (int j = 0; j < width; j++) {
			std::cout << data[i * width + j] << "\t";
		}

		std::cout << std::endl;
	}

	std::cout << std::endl;
*/
	return data;
}