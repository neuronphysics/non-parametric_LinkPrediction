//
// Created by su999 on 2023/6/22.
//

#ifndef GLFM_CUDA_ACC_DECRYPTED_UTILS_H
#define GLFM_CUDA_ACC_DECRYPTED_UTILS_H

#include "Utils.h"

double fint_1(double x, double w, double theta_L, double theta_H);

int
rank_one_update_eta(gsl_matrix *Z, gsl_matrix *zn, gsl_matrix *Rho, gsl_matrix *Eta, int index, int K, int N, int add);


int rank_one_update_covariance_rho(gsl_matrix *Z, gsl_matrix *Zn, gsl_matrix *Q, gsl_matrix *m, double s2Rho, int index,
                                   int K, int N, int add);

gsl_matrix *inverse_sigma_rho(gsl_matrix *Znon, gsl_matrix *Zn, gsl_matrix *Q, gsl_matrix *S, int index, int K, int N,
                              double s2Rho);

int inverse_matrix_Q(double alpha, gsl_matrix *Z, gsl_matrix *X, int N, int K, double *ldet_X);


double recursive_inverse(int K, gsl_matrix *X, gsl_matrix *E, gsl_matrix *F, int add);

double gsl_trace(gsl_matrix *A);

int poissrnd(double lambda);

double trace(gsl_matrix *Amat, int Asize);

double det_get(gsl_matrix *Amat, int Arows, int Acols, int inPlace);

double *column_to_row_major_order(double *A, int nRows, int nCols);

double compute_matrix_max(double missing, gsl_matrix *v);

#endif //GLFM_CUDA_ACC_DECRYPTED_UTILS_H
