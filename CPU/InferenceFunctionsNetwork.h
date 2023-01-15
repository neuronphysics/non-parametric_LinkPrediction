#include "GeneralFunctions.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>

int log_likelihood_Rho(int N, int K, int r, gsl_matrix *Znon, gsl_matrix *zn, gsl_matrix *Rho, gsl_matrix *Qnon,
                       gsl_matrix *Eta, double s2Rho, double *lik);

int AcceleratedGibbs(int maxK, int bias, int N, int D, int K, char *C, int *R, double alpha, double s2B, double *s2Y,
                     double s2H, double s2Rho, gsl_matrix **Y, gsl_matrix *Rho, gsl_matrix *vecRho, gsl_matrix *Z,
                     int *nest, gsl_matrix *P, gsl_matrix *Pnon, gsl_matrix **lambda, gsl_matrix **lambdanon,
                     gsl_matrix *Q, gsl_matrix *Qnon, gsl_matrix *eta, gsl_matrix *etanon, double *ldet_Q,
                     double *ldet_Q_n);

void
SampleY(double missing, int N, int D, int K, char Cd, int Rd, double fd, double mud, double wd, double s2Y, double s2u,
        double s2theta, gsl_matrix *X, gsl_matrix *Z, gsl_matrix *Yd, gsl_matrix *Bd, gsl_vector *thetad,
        const gsl_rng *seed);

void SampleRho(double missing, int N, int K, char Ca, double fa, double s2Rho, double s2u, gsl_matrix *A, gsl_matrix *Z,
               gsl_matrix *vecRho, gsl_matrix *H, const gsl_rng *seed);

double SampleAlpha(int K, int N, const gsl_rng *seed);

double Samples2Y(double missing, int N, int d, int K, char Cd, int Rd, double fd, double mud, double wd, double s2u,
                 double s2theta, gsl_matrix *X, gsl_matrix *Z, gsl_matrix *Yd, gsl_matrix *Bd, gsl_vector *thetad,
                 const gsl_rng *seed);

double
Samples2Rho(int N, int K, gsl_matrix *A, gsl_matrix *Z, gsl_matrix *vecRho, gsl_matrix *vecH, const gsl_rng *seed);

double Samples2H(int K, gsl_matrix *vecH, const gsl_rng *seed);

int
IBPsampler_func(double missing, gsl_matrix *X, char *C, char *Net, gsl_matrix *Z, gsl_matrix **B, gsl_vector **theta,
                gsl_matrix *H, gsl_matrix *A, int *R, double *f, double fa, double *mu, double *w, int maxR, int bias,
                int N, int D, int K, double alpha, double s2B, double *s2Y, double s2Rho, double s2H, double s2u,
                int maxK, int Nsim);

int initialize_func(int N, int D, int maxK, double missing, gsl_matrix *X, char *C, gsl_matrix **B, gsl_vector **theta,
                    int *R, double *f, double *mu, double *w, double *s2Y);
