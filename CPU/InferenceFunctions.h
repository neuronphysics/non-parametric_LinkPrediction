#include "GeneralFunctions.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <ctime>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include "gsl/gsl_cdf.h"
#include "gsl/gsl_randist.h"

int AcceleratedGibbs(int maxK, int bias, int N, int D, int K, char *C, int *R, double alpha, double s2B, double s2Y,
                     gsl_matrix **Y, gsl_matrix *Z, int *nest, gsl_matrix *P, gsl_matrix *Pnon, gsl_matrix **lambda,
                     gsl_matrix **lambdanon);

void
SampleY(double missing, int N, int D, int K, char Cd, int Rd, double fd, double mud, double wd, double s2Y, double s2u,
        double s2theta, gsl_matrix *X, gsl_matrix *Z, gsl_matrix *Yd, gsl_matrix *Bd, gsl_vector *thetad,
        const gsl_rng *seed);

double Samples2Y(double missing, int N, int d, int K, char Cd, int Rd, double fd, double mud, double wd, double s2u,
                 double s2theta, gsl_matrix *X, gsl_matrix *Z, gsl_matrix *Yd, gsl_matrix *Bd, gsl_vector *thetad,
                 const gsl_rng *seed);

int IBPsampler_func(double missing, gsl_matrix *X, char *C, gsl_matrix *Z, gsl_matrix **B, gsl_vector **theta, int *R,
                    double *f, double *mu, double *w, int maxR, int bias, int N, int D, int K, double alpha, double s2B,
                    double *s2Y, double s2u, int maxK, int Nsim);

int initialize_func(int N, int D, int maxK, double missing, gsl_matrix *X, char *C, gsl_matrix **B, gsl_vector **theta,
                    int *R, double *f, double *mu, double *w, double *s2Y);