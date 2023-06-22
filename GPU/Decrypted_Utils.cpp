//
// Created by su999 on 2023/6/22.
//

#include "Decrypted_Utils.h"

double fint_1(double x, double w, double theta_L, double theta_H) {
    return -1 / w * log((theta_H - x) / (x - theta_L));
}

// todo function contain bug
int rank_one_update_eta(gsl_matrix *Z,
                        gsl_matrix *zn,
                        gsl_matrix *Rho,
                        gsl_matrix *Eta,
                        int index,
                        int K,
                        int N,
                        int add) {
    gsl_matrix *S1 = gsl_matrix_calloc(K * K, N);
    gsl_matrix *S2 = gsl_matrix_calloc(K * K, N);
    gsl_matrix *S3 = gsl_matrix_calloc(K * K, 1);
    gsl_Kronecker_product(S1, zn, Z);
    gsl_Kronecker_product(S2, Z, zn);
    gsl_Kronecker_product(S3, zn, zn);
    // whole column
    gsl_matrix_view Rho_view = gsl_matrix_submatrix(Rho, 0, index, N, 1);
    // whole row
    gsl_matrix_view RhoT_view = gsl_matrix_submatrix(Rho, index, 0, 1, N);
    gsl_matrix_view Rho_nn = gsl_matrix_submatrix(Rho, index, index, 1, 1);
    if (add == 0) {//remove
        // Eta = Eta - Z * zn * rho
        matrix_multiply(S2, &Rho_view.matrix, Eta, -1, 1, CblasNoTrans, CblasNoTrans);
        // Eta = Eta - Z * zn * rho - zn * Z * rho^T
        matrix_multiply(S1, &RhoT_view.matrix, Eta, -1, 1, CblasNoTrans, CblasTrans);
        // Eta = Eta - Z * zn * rho - zn * Z * rho^T + zn * zn * rho_nn
        matrix_multiply(S3, &Rho_nn.matrix, Eta, 1, 1, CblasNoTrans, CblasNoTrans);
    } else {//add
        matrix_multiply(S2, &Rho_view.matrix, Eta, 1, 1, CblasNoTrans, CblasNoTrans);
        matrix_multiply(S1, &RhoT_view.matrix, Eta, 1, 1, CblasNoTrans, CblasTrans);
        matrix_multiply(S3, &Rho_nn.matrix, Eta, -1, 1, CblasNoTrans, CblasNoTrans);
    }

    gsl_matrix_free(S1);
    gsl_matrix_free(S2);
    gsl_matrix_free(S3);
    return 0;
}


int rank_one_update_covariance_rho(gsl_matrix *Z,
                                   gsl_matrix *Zn,
                                   gsl_matrix *Q,
                                   gsl_matrix *m, //the inverse covariance of rho_non N-1xN-1
                                   double s2Rho,
                                   int index,
                                   int K,
                                   int N,
                                   int add) {
    //removing a row from kronnecker_product (z_n,z) add=0
    double nu;
    gsl_matrix_view zn_view;
    gsl_matrix *Znon = gsl_matrix_calloc(K, N - 1);
    gsl_matrix *Snon = gsl_matrix_calloc(K * K, N - 1);
    gsl_matrix *T = gsl_matrix_calloc(K * K, K * K);
    remove_col(K, N, index, Znon, Z);
    gsl_Kronecker_product(Snon, Znon, Zn);
    gsl_matrix *base = gsl_matrix_calloc(K, K);
    gsl_matrix *aux = gsl_matrix_calloc(K, K);
    gsl_matrix_memcpy(T, Q);
    matrix_multiply(Zn, Zn, base, 1, 0, CblasNoTrans, CblasTrans);
    //compute (Znon kron Z)^T (Znon kron Z)
    for (int i = 0; i < N; i++) {
        zn_view = gsl_matrix_submatrix(Z, 0, i, K, 1);
        matrix_multiply(&zn_view.matrix, &zn_view.matrix, aux, 1, 0, CblasNoTrans, CblasTrans);
        nu = recursive_inverse(K, T, base, aux, add);
    }
    gsl_matrix_set_identity(m);
    gsl_matrix *a = gsl_matrix_calloc(N - 1, K * K);
    matrix_multiply(Snon, T, a, 1, 0, CblasTrans, CblasNoTrans);
    double alpha = 1. / s2Rho;
    matrix_multiply(a, Snon, m, -alpha, alpha, CblasNoTrans, CblasNoTrans);//equation 21
    gsl_matrix_free(aux);
    gsl_matrix_free(a);
    gsl_matrix_free(T);
    gsl_matrix_free(Snon);
    gsl_matrix_free(Znon);
    gsl_matrix_free(base);
    return 0;
}

gsl_matrix *inverse_sigma_rho(gsl_matrix *Znon,//Kx(N-1)
                              gsl_matrix *Zn,
                              gsl_matrix *Q,//This is Qnon
                              gsl_matrix *S,//S=Znon kronnecker_product S
                              int index,
                              int K,
                              int N,
                              double s2Rho) {
    //In implementation of equation 21 but just using Qnon instead of Q
    double nu;
    int add = 1;
    gsl_matrix_view zn_view;
    gsl_matrix *Qnon = gsl_matrix_calloc(K * K, K * K);
    gsl_matrix *base = gsl_matrix_calloc(K, K);
    gsl_matrix *aux = gsl_matrix_calloc(K, K);
    gsl_matrix_memcpy(Qnon, Q);
    matrix_multiply(Zn, Zn, base, 1, 0, CblasNoTrans, CblasTrans);
    for (int i = 0; i < (N - 1); i++) {
        zn_view = gsl_matrix_submatrix(Znon, 0, i, K, 1);
        matrix_multiply(&zn_view.matrix, &zn_view.matrix, aux, 1, 0, CblasNoTrans, CblasTrans);
        nu = recursive_inverse(K, Qnon, aux, base, add);
    }
    gsl_matrix *sigma = gsl_matrix_calloc(N - 1, N - 1);
    gsl_matrix_set_identity(sigma); //NxN matrix
    gsl_matrix *F = gsl_matrix_calloc(N - 1, K * K);
    matrix_multiply(S, Qnon, F, 1, 0, CblasTrans, CblasNoTrans);
    matrix_multiply(F, S, sigma, -1, 1, CblasNoTrans, CblasNoTrans);
    gsl_matrix_scale(sigma, 1. / s2Rho);
    gsl_matrix_free(aux);
    gsl_matrix_free(base);
    gsl_matrix_free(F);

    return sigma;
}


int inverse_matrix_Q(double alpha,
                     gsl_matrix *Z,
                     gsl_matrix *X, //K^2xK^2 ***THE OUTPUT***
                     int N,
                     int K,
                     double *ldet_X) {
    int add = 1;
    ldet_X[0] = -pow(K, 2) * gsl_sf_log(alpha);
    gsl_matrix_view Z_view;
    gsl_matrix_view X_view = gsl_matrix_submatrix(X, 0, 0, K * K, K * K);
    gsl_matrix **Y = (gsl_matrix **) calloc(N, sizeof(gsl_matrix *));
    for (int d = 0; d < N; d++) {
        Y[d] = gsl_matrix_calloc(K, K);
        Z_view = gsl_matrix_submatrix(Z, 0, d, K, 1);
        matrix_multiply(&Z_view.matrix, &Z_view.matrix, Y[d], 1, 0, CblasNoTrans, CblasTrans);
    }
    //*****
    gsl_matrix *E = gsl_matrix_calloc(K * K, K * K);
    double coef = 0.;
    double det_X = 1;
    double nu;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0 && j == 0) {
                // implementing Eq. 46
                gsl_matrix_set_identity(&X_view.matrix);
                gsl_Kronecker_product(E, Y[i], Y[j]);
                coef = pow(gsl_trace(Y[i]), 2);
                ldet_X[0] += gsl_sf_log(1 / (1 + 1 / alpha * coef));
                gsl_matrix_scale(E, 1. / (alpha + coef));
                gsl_matrix_sub(&X_view.matrix, E);
                gsl_matrix_scale(&X_view.matrix, 1. / alpha);
            } else {
                nu = recursive_inverse(K, X, Y[i], Y[j], add);
                det_X *= nu;
            }
        }
    }
    ldet_X[0] += gsl_sf_log(det_X);

    gsl_matrix_free(E);
    for (int d = 0; d < N; d++) {
        gsl_matrix_free(Y[d]);
    }
    free(Y);
    return 0;
}

double recursive_inverse(int K,
                         gsl_matrix *X,
                         gsl_matrix *E,
                         gsl_matrix *F,
                         int add) {
    //implementing Miller 1980 C_{k+1}^{-1}=C_{k}^{-1}-C_{k+1}^{-1}E_{k}C_{k+1}^{-1}/(1+tr(C_{k+1}^{-1}E_{k}))

    gsl_matrix *aux = gsl_matrix_calloc(K * K, K * K);
    gsl_matrix *inv = gsl_matrix_calloc(K * K, K * K);
    gsl_matrix_view X_view = gsl_matrix_submatrix(X, 0, 0, K * K, K * K);
    gsl_Kronecker_product(aux, E, F);
    double t;
    matrix_multiply(&X_view.matrix, aux, inv, 1., 0, CblasNoTrans, CblasNoTrans);
    if (add) {
        t = 1. / (trace(inv, K * K) + 1.);
    } else {
        t = 1. / (trace(inv, K * K) - 1.);
    }
    matrix_multiply(inv, &X_view.matrix, aux, t, 0, CblasNoTrans, CblasNoTrans);
    gsl_matrix_sub(&X_view.matrix, aux);
    gsl_matrix_free(aux);
    gsl_matrix_free(inv);
    return t;
}

double gsl_trace(gsl_matrix *A) {
    double sum = 0.0;
    int r = A->size1;
    int c = A->size2;
    if (r != c) {
        LOG(OUTPUT_NORMAL, "error: cannot calculate trace of non-square matrix.")
        return 0;
    }
    // calculate sum of diagonal elements
    for (int i = 0; i < r; i++) {
        sum += gsl_matrix_get(A, i, i);
    }
    return sum;
}

// Sampling functions
int poissrnd(double lambda) {
    double L = gsl_sf_exp(-lambda);
    int k = 0;
    double p = 1;
    do {
        k++;
        p *= rand01();
    } while (p > L);
    return (k - 1);
}

double trace(gsl_matrix *Amat, int Asize) {
    // Assume Amat is square
    double resul = 0;

    double *p = Amat->data;

    for (int i = 0; i < Asize; i++) {
        resul += p[i * Asize + i];
    }

    return resul;
}

double det_get(gsl_matrix *Amat, int Arows, int Acols, int inPlace) {
/*
  inPlace = 1 => A is replaced with the LU decomposed copy.
  inPlace = 0 => A is retained, and a copy is used for LU.
*/

    double det;
    int signum;
    gsl_permutation *p = gsl_permutation_alloc(Arows);
    gsl_matrix *tmpA;

    if (inPlace)
        tmpA = Amat;
    else {
        tmpA = gsl_matrix_calloc(Arows, Acols);
        gsl_matrix_memcpy(tmpA, Amat);
    }


    gsl_linalg_LU_decomp(tmpA, p, &signum);
    det = gsl_linalg_LU_det(tmpA, signum);
    gsl_permutation_free(p);
    if (!inPlace)
        gsl_matrix_free(tmpA);


    return det;
}

double *column_to_row_major_order(double *A, int nRows, int nCols) {
    auto *copy = (double *) malloc(nRows * nCols * sizeof(double));
    if (nullptr == copy) {
        LOG(OUTPUT_NORMAL, "Not enough memory (in column_to_row_major_order)")
    }

    for (int i = 0; i < nRows * nCols; i++) {
        int cociente = i / nRows;
        int x = cociente + nCols * (i - nRows * cociente);
        copy[x] = A[i];
    }

    return copy;
}

double compute_matrix_max(double missing, gsl_matrix *v) {
    double maxX = -1e100;
    for (int i = 0; i < v->size1; i++) {
        for (int j = 0; j < v->size2; ++j) {
            double xnd = gsl_matrix_get(v, i, j);
            if (xnd != missing) {
                if (xnd > maxX) { maxX = xnd; }
            }
        }
    }
    return maxX;
}