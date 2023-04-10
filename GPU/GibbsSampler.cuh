//
// Created by su999 on 2023/4/10.
//

#ifndef GLFM_CUDA_ACC_GIBBSSAMPLER_CUH
#define GLFM_CUDA_ACC_GIBBSSAMPLER_CUH

#include "Utils.h"
#include "unordered_map"

void sample_znk(int N,
                int n,
                int K,
                int k,
                int D,
                int nCount,
                double s2Rho,
                double *s2Y,
                char *C,
                int *R,
                double *p,
                gsl_matrix *Zn,
                gsl_matrix *Qnon,
                gsl_matrix *Enon,
                gsl_matrix *Snon,
                gsl_matrix *Znon,
                gsl_matrix *Rho,
                gsl_matrix *s2y_p,
                gsl_matrix **Y,
                gsl_matrix **lambdanon);

double
compute_likelihood_given_znk(int D, int K, int n, double *s2Y, char *C, int *R, gsl_matrix *s2y_p, gsl_matrix *aux,
                             gsl_matrix *Zn, gsl_matrix **Y, gsl_matrix **lambdanon);

int log_likelihood_Rho(int N,
                       int K,
                       int r,
                       gsl_matrix *Znon,// Z_{-n} N-1 x K matrix
                       gsl_matrix *zn,
                       gsl_matrix *Rho,
                       gsl_matrix *Qnon,
                       gsl_matrix *Eta,// Snon^T vec(Rho -n, -n)
                       double s2Rho,
                       double &lik);

int AcceleratedGibbs(int maxK,          //Maximum number of latent features
                     int bias,          //An extra latent feature
                     int N,             //Number of objects
                     int D,             //Number of attributes
                     int K,             //Number of latent features
                     char *C,           //data type
                     int *R,            //The number of categories in each discrete attribute
                     double alpha,      //The concentration parameter
                     double s2B,        //variance of the weighting matrix??
                     double *s2Y,       //noise variance of the pseudo-observation of the attribute matrix
                     double s2H,        //**** variance of the affinity matrix
                     double s2Rho,      //**** noise variance of the pseudo-observation of the adjacency matrix
                     gsl_matrix **Y,    //The pseudo-observation matrix of the affinity matrix (the auxiliary Gaussian variable)
                     gsl_matrix *Rho,//**** The pseudo-observation matrix of the adjacency matrix,
                     gsl_matrix *vecRho,
                     gsl_matrix *Z,     // The IBP latent matrix
                     int *nest,         //m_{-n,k}
                     gsl_matrix *P,     //P = Z^T Z + 1./s2B
                     gsl_matrix *Pnon,  //P_{-n} = P - z_{n}^T z_{n}
                     gsl_matrix **lambda,//Lambda_r^d=Z^T y_r^d
                     gsl_matrix **lambdanon,//Lambdanon_r^d{-n}=Lambda_r^d-Z_{n}^T y_{nr}^d
                     gsl_matrix *Q,      //**** Q_{K2xK2}=[(S^T S) + s2Rho/s2H]^{-1}]
                     gsl_matrix *Qnon,   //**** Q_{-n} = [Q^{-1} - (S_{n}^T S_{n})]^{-1}
                     gsl_matrix *eta, //***eta_{K2x1}=(Z kron Z)^T vec(Rho)
                     gsl_matrix *etanon,//***etanon_{K2x1}=(Znon kron Znon) vec(Rhonon)
                     double *ldet_Q,
                     double *ldet_Q_n
);

#endif //GLFM_CUDA_ACC_GIBBSSAMPLER_CUH
