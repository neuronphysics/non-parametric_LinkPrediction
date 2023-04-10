#ifndef GLFM_INFERENCE_FUNCTION
#define GLFM_INFERENCE_FUNCTION

#include "Utils.h"
#include "GibbsSampler.cuh"

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

int initialize_func(int N, int D, int maxK, double missing, gsl_matrix *X, const char *C, gsl_matrix **B, gsl_vector **theta,
                    int *R, double *f, double *mu, double *w, double *s2Y);

#endif