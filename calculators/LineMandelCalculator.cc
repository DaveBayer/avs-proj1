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
#include <immintrin.h>

#include "LineMandelCalculator.h"

#define SIMD_512_ALIGNMENT 64

LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
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

LineMandelCalculator::~LineMandelCalculator()
{
	_mm_free(data);
	_mm_free(zReal);
	_mm_free(zImag);

	data = nullptr;
	zReal = nullptr;
	zImag = nullptr;
}

int * LineMandelCalculator::calculateMandelbrot()
{
	for (int i = 0; i < height; i++) {
		float y = y_start + i * dy; // current imaginary value

		int *pdata = data + i * width;
		float *pzReal = zReal + i * width;
		float *pzImag = zImag + i * width;

		for (int k = 0; k < limit; k++) {
			
#			pragma omp simd simdlen(64) lastprivate(k)
			for (int j = 0; j < width; j++) {

				float x = x_start + j * dx; // current real value

				float r2 = pzReal[j] * zReal[j];
				float i2 = zImag[j] * zImag[j];

				(pdata[j] == limit && r2 + i2 > 4.0f) ? pdata[j] = k : 0;

				zImag[j] = 2.0f * zReal[j] * zImag[j] + y;
				zReal[j] = r2 - i2 + x;
			}
		}
	}

	return data;
}
