#ifndef GLFM_CUDA_ACC_GPUACC_H
#define GLFM_CUDA_ACC_GPUACC_H

#include <gsl/gsl_matrix.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")

#include <math.h>
#include <gsl/gsl_cblas.h>

void gpuInverseMethod1(gsl_matrix *original, gsl_matrix *inverseM);

void gpuInverseMethod2(gsl_matrix *original, gsl_matrix *inverseM);

void
gpuMatrixMultiply(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *C, double scale1, double scale2, CBLAS_TRANSPOSE_t transA,
                  CBLAS_TRANSPOSE_t transB);
#endif //GLFM_CUDA_ACC_GPUACC_H
