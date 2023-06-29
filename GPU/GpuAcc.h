#ifndef GLFM_CUDA_ACC_GPUACC_H
#define GLFM_CUDA_ACC_GPUACC_H

#include <gsl/gsl_matrix.h>
#include <iostream>
#include <gsl/gsl_cblas.h>

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")


#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>



void gpuInverseMethod1(gsl_matrix *original, gsl_matrix *inverseM);

void
gpuMatrixMultiply(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *C, double scale1, double scale2,
                  CBLAS_TRANSPOSE_t transA,
                  CBLAS_TRANSPOSE_t transB);

void
gpuKron(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *Res);

void symmetricAndPDMatrixInverse(gsl_matrix * matrix);

void gpuBoostedComputeFullEta(const gsl_matrix * Z, const gsl_matrix * Rho, gsl_matrix * etaKK);

#endif //GLFM_CUDA_ACC_GPUACC_H
