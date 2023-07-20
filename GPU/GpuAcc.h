#ifndef GLFM_CUDA_ACC_GPUACC_H
#define GLFM_CUDA_ACC_GPUACC_H

#include <gsl/gsl_matrix.h>
#include <iostream>

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")



void gpuInverseMethod1(gsl_matrix *original, gsl_matrix *inverseM);

void
gpuMatrixMultiply(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *C, double scale1, double scale2,
                  CBLAS_TRANSPOSE_t transA,
                  CBLAS_TRANSPOSE_t transB);

void
gpuKron(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *Res);

//void symmetricAndPDMatrixInverse(gsl_matrix * matrix);

void gpuBoostedComputeFullEta(const gsl_matrix * Z, const gsl_matrix * Rho, gsl_matrix * etaKK);

void
gpuBoostedEtaUpdate(int N, int K, const double *znkZ, const double *Zkzn, const double *znkzn,
                    const gsl_matrix *rho_col, const gsl_matrix *rho_row,
                    double rho_nn, const gsl_matrix *fullEta, gsl_matrix *etanon);

#endif //GLFM_CUDA_ACC_GPUACC_H
