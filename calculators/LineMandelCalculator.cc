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

				zI[j] = 2.0f * zR[j] * zI[j] + y;
				zR[j] = r2 - i2 + x;
			}
		}
	}

	for (int i = 0; i < height; i++) {
		std::cout << i << ":\t";

		for (int j = 0; j < width; j++) {
			std::cout << data[i * width + j] << "\t";
		}

		std::cout << std::endl;
	}

	std::cout << std::endl;

	return data;
}
