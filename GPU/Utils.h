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
#include "GpuAcc.h"
#include <chrono>
#include "Log.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "random"
#include "string"
#include "ostream"
#include "fstream"
#include <thread>

extern std::ofstream matrixOut;
extern uint64_t timeSeed;

// transformations
double fre_1(double x, double func, double mu, double w);

double f_1(double x, double func, double mu, double w);

double f_w(double x, double func, double mu, double w);



// vector computation
double compute_vector_mean(int N, double missing, const gsl_vector *v);

double compute_vector_var(int N, double missing, const gsl_vector *v);

double compute_vector_max(int N, double missing, const gsl_vector *v);

double compute_vector_min(int N, double missing, const gsl_vector *v);



// sampling functions
int mnrnd(double *p, int nK);

void mvnrnd(gsl_vector *X, gsl_matrix *Sigma, gsl_vector *Mu, int K, const gsl_rng *seed);

double truncnormrnd(double mu, double sigma, double xlo, double xhi, const gsl_rng *rng);


// matrix operations
void matrix_multiply(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *C, double alpha, double beta,
                     CBLAS_TRANSPOSE_t TransA,
                     CBLAS_TRANSPOSE_t TransB);

void inverse(gsl_matrix *matrix);

void gsl_Kronecker_product(gsl_matrix *prod, const gsl_matrix *a, const gsl_matrix *b);

void gsl_matrix2vector(gsl_matrix *vect, gsl_matrix *matrix);

void gsl_vector2matrix(gsl_matrix *vect, gsl_matrix *matrix);

void remove_col(int K, int N, int i, gsl_matrix *out, gsl_matrix *in);

void compute_inverse_Q_directly(int N, int K, const gsl_matrix *Z, double beta, gsl_matrix *Q);

void normal_update_eta(const gsl_matrix *Znon, const gsl_matrix *Rho, int n, gsl_matrix *Enon);

void compute_full_eta(const gsl_matrix * Z, const gsl_matrix * Rho, gsl_matrix * eta);

void rank_one_update_eta(int K, int N, int n, gsl_matrix *Z, gsl_matrix *zn, gsl_matrix *Rho, gsl_matrix *Eta,
                         gsl_matrix *Etanon);


void init_util_functions(const std::string &exeName, const std::string &detail);

int factorial(int N);

double expFun(double x);

double lndet_get(const gsl_matrix *Amat, int Arows, int Acols);

double rand01();

void print_iteration_num(int iterationNum);

void print_matrix(const gsl_matrix **matrix, const std::string &name, int rowNum, int columnNum);

void print_matrix(const gsl_matrix *matrix, const std::string &name, size_t entryPerRow = 0);

void matrix_compare(const gsl_matrix *A, const gsl_matrix *B);
#endif