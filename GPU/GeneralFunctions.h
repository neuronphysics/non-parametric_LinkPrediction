#ifndef GENERAL_FUNCTION
#define GENERAL_FUNCTION


#include <math.h>
#include <stdio.h>
#include <stdlib.h>     /* abs */
#include <iostream>
#include <time.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>

// if run in non linux environment, please add this line
#include "gpuAcc.h"
#include <chrono>
#include "Log.h"

//Transformations
double fre_1(double x, double func, double mu, double w);

double f_1(double x, double func, double mu, double w);

double f_w(double x, double func, double mu, double w);

double fint_1(double x, double w, double theta_L, double theta_H);
// double f(double x, double w);


// General functions
double compute_vector_mean(int N, double missing, gsl_vector *v);

double compute_vector_var(int N, double missing, gsl_vector *v);

double compute_vector_max(int N, double missing, gsl_vector *v);

double compute_vector_min(int N, double missing, gsl_vector *v);

double compute_matrix_max(double missing, gsl_matrix *v);

int factorial(int N);

int poissrnd(double lambda);

//gsl_matrix *double2gsl(double *Amat, int nRows, int nCols);
void matrix_multiply(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C, double alpha, double beta, CBLAS_TRANSPOSE_t TransA,
                     CBLAS_TRANSPOSE_t TransB);

double *column_to_row_major_order(double *A, int nRows, int nCols);

//double *row_to_column_major_order(double *A,int nRows,int nCols);
double det_get(gsl_matrix *Amat, int Arows, int Acols, int inPlace);

double lndet_get(gsl_matrix *Amat, int Arows, int Acols, int inPlace);

// gsl_matrix *inverse(gsl_matrix *Amat, int Asize);
void inverse(gsl_matrix *Amat, int Asize);

double trace(gsl_matrix *Amat, int Asize);

double logFun(double x);

double expFun(double x);

//Sampling functions
int mnrnd(double *p, int nK);

void mvnrnd(gsl_vector *X, gsl_matrix *Sigma, gsl_vector *Mu, int K, const gsl_rng *seed);

double truncnormrnd(double mu, double sigma, double xlo, double xhi);

//****New for Network analysis
int gsl_Kronecker_product(gsl_matrix *prod, gsl_matrix * a, gsl_matrix *b);

int gsl_matrix2vector(gsl_matrix *vect, gsl_matrix *matrix);

int gsl_vector2matrix(gsl_matrix *vect, gsl_matrix *matrix);

double gsl_trace(gsl_matrix *A);

int remove_col(int K, int N, int i, gsl_matrix *Sn, gsl_matrix *Z);//remove a row from matrix Z
double recursive_inverse(int K, gsl_matrix *X, gsl_matrix *E, gsl_matrix *F, int add);

int
rank_one_update_eta(gsl_matrix *Z, gsl_matrix *zn, gsl_matrix *Rho, gsl_matrix *Eta, int index, int K, int N, int add);

int rank_one_update_Kronecker_ldet(gsl_matrix *Z, gsl_matrix *Q, double *ldet_Q, int index, int K, int N, int add);

int rank_one_update_Kronecker(gsl_matrix *Z, gsl_matrix *Zn, gsl_matrix *Q, int index, int K, int N, int add);

int rank_one_update_covariance_rho(gsl_matrix *Z, gsl_matrix *Zn, gsl_matrix *Q, gsl_matrix *m, double s2Rho, int index,
                                   int K, int N, int add);

int Update_Q_after_removing(int N, int K, int index, double beta, gsl_matrix *Z, gsl_matrix *Qnon, int add);

gsl_matrix *inverse_sigma_rho(gsl_matrix *Znon, gsl_matrix *Zn, gsl_matrix *Q, gsl_matrix *S, int index, int K, int N,
                              double s2Rho);

int inverse_matrix_Q(double alpha, gsl_matrix *Z, gsl_matrix *X, int N, int K, double *ldet_X);

void compute_inverse_Q_directly(int N, int K, gsl_matrix *Z, double beta, gsl_matrix *Q);

void normal_update_eta(gsl_matrix * Znon, gsl_matrix *Rho, int n, gsl_matrix * Enon);
#endif