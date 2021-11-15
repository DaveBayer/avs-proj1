/**
 * @file BatchMandelCalculator.h
 * @author David Bayer <xbayer09@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 2021/11/14
 */
#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int *calculateMandelbrot();
    void calculateMandelbrot_aligned();
    void calculateMandelbrot_unaligned();

private:
    int *data;
    float *xs;
    float *ys;
    float *zReal;
    float *zImag;

    void (BatchMandelCalculator::*chosen_calculator)();
};

#endif