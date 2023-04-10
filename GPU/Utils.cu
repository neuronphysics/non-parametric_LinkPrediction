#include "Utils.h"


using namespace std;

double fre_1(double x, double func, double mu, double w) {
    return w * (x - mu);
}

double f_1(double x, double func, double mu, double w) {

    if (func == 1) {
        return logFun(gsl_sf_exp(w * (x - mu)) - 1);
    } else if (func == 2) {
        return sqrt(w * (x - mu));
    } else {
        LOG(OUTPUT_DEBUG,"error: unknown transformation function. Used default transformation log(exp(y)-1)");
        return logFun(gsl_sf_exp(w * (x - mu)) - 1);
    }
}

double f_w(double x, double func, double mu, double w) {
    //     replace function for f_1
    //author:Zahra???
    if (func == 1) {
        if (x != 0) {
            return (1 / w) * (logFun(gsl_sf_exp(x) - 1) - mu);
        } else {
            return 0;
        }
    } else if (func == 2) {
        return sqrt(w * (x - mu));
    } else {
        LOG(OUTPUT_DEBUG,"error: unknown transformation function. Used default transformation log(exp(y)-1)");
        if (x != 0) {
            return (1 / w) * (logFun(gsl_sf_exp(x) - 1) - mu);
        } else {
            return 0;
        }
    }
}

double fint_1(double x, double w, double theta_L, double theta_H) {
    return -1 / w * logFun((theta_H - x) / (x - theta_L));
}

// Functions
double compute_vector_mean(int N, double missing, gsl_vector *v) {
    double sumX = 0;
    double countX = 0;
    for (int nn = 0; nn < N; nn++) {
        double xnd = gsl_vector_get(v, nn);
        if (xnd != missing && !(gsl_isnan(xnd))) {
            sumX += xnd;
            countX += 1;
        }
    }
    return sumX / countX;
}

double compute_vector_var(int N, double missing, gsl_vector *v) {
    double meanX = compute_vector_mean(N, missing, v);
    double sumX = 0;
    double countX = 0;
    for (int nn = 0; nn < N; nn++) {
        double xnd = gsl_vector_get(v, nn);
        if (xnd != missing && !(gsl_isnan(xnd))) {
            sumX += pow(xnd - meanX, 2);
            countX += 1;
        }
    }
    return sumX / countX;
}

double compute_vector_max(int N, double missing, gsl_vector *v) {
    double maxX = -1e100;
    for (int nn = 0; nn < N; nn++) {
        double xnd = gsl_vector_get(v, nn);
        if (xnd != missing) {
            if (xnd > maxX) { maxX = xnd; }
        }
    }
    return maxX;
}

double compute_vector_min(int N, double missing, gsl_vector *v) {
    double minX = 1e100;
    for (int nn = 0; nn < N; nn++) {
        double xnd = gsl_vector_get(v, nn);
        if (xnd != missing) {
            if (xnd < minX) { minX = xnd; }
        }
    }
    return minX;
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

double logFun(double x) {
    if (x == 0) {
        LOG(OUTPUT_DEBUG,"logarithm of 0 is -inf \n");
        return GSL_NEGINF;
    } else if (x < 0) {
        LOG(OUTPUT_DEBUG,"Error: logarithm is not defined for negative numbers\n");
        return -1;
    } else { return gsl_sf_log(x); }
}

double expFun(double x) {
    if (x > 300) { return GSL_POSINF; }
    else if (x < -300) { return 0; }
    else { return gsl_sf_exp(x); }
}

int factorial(int N) {
    int fact = 1;
    for (int c = 1; c <= N; c++) {
        fact = fact * c;
    }
    return fact;
}

void matrix_multiply(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C, double alpha, double beta, CBLAS_TRANSPOSE_t TransA,
                     CBLAS_TRANSPOSE_t TransB) {
    // C = alpha * AB + beta * C
    gpuMatrixMultiply(A, B, C, alpha, beta, TransA, TransB);
}

double *column_to_row_major_order(double *A, int nRows, int nCols) {
    auto *copy = (double *) malloc(nRows * nCols * sizeof(double));
    if (nullptr == copy) {
        LOG(OUTPUT_NORMAL,"Not enough memory (in column_to_row_major_order)");
    }

    for (int i = 0; i < nRows * nCols; i++) {
        int cociente = i / nRows;
        int x = cociente + nCols * (i - nRows * cociente);
        copy[x] = A[i];
    }

    return copy;
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
        tmpA = gsl_matrix_alloc(Arows, Acols);
        gsl_matrix_memcpy(tmpA, Amat);
    }


    gsl_linalg_LU_decomp(tmpA, p, &signum);
    det = gsl_linalg_LU_det(tmpA, signum);
    gsl_permutation_free(p);
    if (!inPlace)
        gsl_matrix_free(tmpA);


    return det;
}

double lndet_get(gsl_matrix *Amat, int Arows, int Acols, int inPlace) {
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
        tmpA = gsl_matrix_alloc(Arows, Acols);
        gsl_matrix_memcpy(tmpA, Amat);
    }


    gsl_linalg_LU_decomp(tmpA, p, &signum);
    det = gsl_linalg_LU_lndet(tmpA);
    gsl_permutation_free(p);
    if (!inPlace)
        gsl_matrix_free(tmpA);


    return det;
}


void inverse(gsl_matrix *Amat, int Asize) {
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    gpuInverseMethod1(Amat, Amat);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    LOG(OUTPUT_DEBUG, "Inverse cost = %lld [ms]", chrono::duration_cast<chrono::milliseconds>(end - begin).count())
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

// Sampling functions
int poissrnd(double lambda) {
    double L = gsl_sf_exp(-lambda);
    int k = 0;
    double p = 1;
    do {
        k++;
        p *= drand48();
    } while (p > L);
    return (k - 1);
}

//  k will not increase unless p[0] is small
int mnrnd(double *p, int nK) {
    double pMin = 0;
    double pMax = p[0];
    double s = drand48();
    int k = 0;
    int flag = 1;
    int Knew;
    while (flag) {

        if ((s > pMin) && (s <= pMax)) {
            flag = 0;
            Knew = k;
        } else {
            pMin += p[k];
            pMax += p[k + 1];
        }
        k++;

    }
    return Knew;
}

void mvnrnd(gsl_vector *x, gsl_matrix *Sigma, gsl_vector *Mu, int K, const gsl_rng *seed) {

    gsl_matrix *A = gsl_matrix_alloc(K, K);
    gsl_matrix_memcpy(A, Sigma);
    gsl_linalg_cholesky_decomp(A);

    for (int k = 0; k < K; k++) {
        gsl_vector_set(x, k, gsl_ran_ugaussian(seed));
    }
    // x = op(A)x
    gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, A, x);

    // a = a + b
    gsl_vector_add(x, Mu);

}

double truncnormrnd(double mu, double sigma, double xlo, double xhi) {

    if (xlo > xhi) {
        LOG(OUTPUT_NORMAL,"error: xlo<xhi");
    }

    // when (xlo - mu) / sigma greater than 5, the result will be 1, resulting z = inf
    double plo = gsl_cdf_ugaussian_P((xlo - mu) / sigma);
    if (plo == 1) {
        LOG(OUTPUT_NORMAL,"plo too large")
        plo = 0.99999;
    }
    double phi = gsl_cdf_ugaussian_P((xhi - mu) / sigma);
    if (phi == 0) {
        LOG(OUTPUT_NORMAL,"phi too small")
        phi = 0.00001;
    }
    double r = drand48();
    double res = plo + (phi - plo) * r;
    if(res == 1){
        LOG(OUTPUT_NORMAL,"res too large")
        res = 0.99999999;
    }
    if(res == 0){
        LOG(OUTPUT_NORMAL,"res too small")
        res = 0.00000001;
    }

    double z = gsl_cdf_ugaussian_Pinv(res);
    return mu + z * sigma;
}


// Extra utilities for the Link Prediction paper (author: Zahra)
int gsl_Kronecker_product(gsl_matrix *prod,
                          gsl_matrix *a,
                          gsl_matrix *b) {
    //https://github.com/SDerrode/PLGM2/blob/4eeeb34d3f957e03091b0cafcae5b22e734adade/tkalman_c/PKF/gsl/source/gsl_Kronecker_product.cpp
    if (prod->size1 != a->size1 * b->size1 || prod->size2 != a->size2 * b->size2) { return 1; }
    for (unsigned int i = 0; i < a->size1; ++i) {
        for (unsigned int j = 0; j < a->size2; ++j) {
            gsl_matrix toto = gsl_matrix_submatrix(prod, i * b->size1, j * b->size2, b->size1, b->size2).matrix;
            gsl_matrix_memcpy(&toto, b);
            gsl_matrix_scale(&toto, a->data[i * a->tda + j]);
        }
    }
    return 0;
}


int gsl_matrix2vector(gsl_matrix *vect, gsl_matrix *matrix) {
    //flatten a matrix
    if (vect->size1 != matrix->size1 * matrix->size2 || vect->size2 != 1) { return 1; }
    for (unsigned int i = 0; i < matrix->size1; ++i) {
        for (unsigned int j = 0; j < matrix->size2; ++j) {
            vect->data[(i * matrix->size2 + j) * vect->tda] = matrix->data[i * matrix->tda + j];
        }
    }
    return 0;
}

int gsl_vector2matrix(gsl_matrix *vect, gsl_matrix *matrix) {
    if (vect->size1 != matrix->size1 * matrix->size2 || vect->size2 != 1) { return 1; }
    for (unsigned int i = 0; i < matrix->size1; ++i) {
        for (unsigned int j = 0; j < matrix->size2; ++j) {
            matrix->data[i * matrix->tda + j] = vect->data[(i * matrix->size2 + j) * vect->tda];
        }
    }
    return 0;
}

double gsl_trace(gsl_matrix *A) {
    double sum = 0.0;
    int r = A->size1;
    int c = A->size2;
    if (r != c) {
        LOG(OUTPUT_NORMAL,"error: cannot calculate trace of non-square matrix.\n");
        return 0;
    }
    // calculate sum of diagonal elements
    for (int i = 0; i < r; i++) {
        sum += gsl_matrix_get(A, i, i);
    }
    return sum;
}

int remove_col(int K,
               int N,
               int i,//between range 1 to N
               gsl_matrix *Sn, //Kx(N-1)
               gsl_matrix *Z) {
    //remove a row of matrix (tested and approved)
    int j;
    gsl_matrix_view Z_view;

    gsl_matrix_view Sn_view;
    if (i == 0) {
        Z_view = gsl_matrix_submatrix(Z, 0, 1, K, N - 1);
        gsl_matrix_memcpy(Sn, &Z_view.matrix);
    } else if (i == (N - 1)) {
        Z_view = gsl_matrix_submatrix(Z, 0, 0, K, N - 1);
        gsl_matrix_memcpy(Sn, &Z_view.matrix);
    } else {
        j = i + 1;
        Z_view = gsl_matrix_submatrix(Z, 0, 0, K, j);
        Sn_view = gsl_matrix_submatrix(Sn, 0, 0, K, j);
        gsl_matrix_memcpy(&Sn_view.matrix, &Z_view.matrix);
        Z_view = gsl_matrix_submatrix(Z, 0, j, K, N - j);
        Sn_view = gsl_matrix_submatrix(Sn, 0, j - 1, K, N - j);
        gsl_matrix_memcpy(&Sn_view.matrix, &Z_view.matrix);
    }
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

int rank_one_update_Kronecker_ldet(gsl_matrix *Z,
                                   gsl_matrix *Q,
                                   double *ldet_Q,
                                   int index,
                                   int K,
                                   int N,
                                   int add) {
    //removing a row from kronnecker_product (z,z_n) & (z_n,z) add=0
    double nu;
    gsl_matrix *base = gsl_matrix_calloc(K, K);
    gsl_matrix *aux = gsl_matrix_calloc(K, K);
    gsl_matrix_view Zn = gsl_matrix_submatrix(Z, 0, index, K, 1);
    matrix_multiply(&Zn.matrix, &Zn.matrix, base, 1, 0, CblasNoTrans, CblasTrans);
    double coeff = 1.;
    for (int i = 0; i < N; i++) {
        Zn = gsl_matrix_submatrix(Z, 0, i, K, 1);
        matrix_multiply(&Zn.matrix, &Zn.matrix, aux, 1, 0, CblasNoTrans, CblasTrans);
        if (i != index) {
            nu = recursive_inverse(K, Q, base, aux, add);
            if (add == 0) {
                coeff *= -nu;//remove
            } else {
                coeff *= nu;//add
            }
        }
        nu = recursive_inverse(K, Q, aux, base, add);
        if (add == 0) {
            coeff *= -nu;
        } else {
            coeff *= nu;
        }
    }
    
    ldet_Q[0] += gsl_sf_log(coeff);
    LOG(OUTPUT_DEBUG,"log(det(Q))=%f....\n", ldet_Q[0]);
    gsl_matrix_free(aux);
    gsl_matrix_free(base);
    return 0;
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

int rank_one_update_Kronecker(gsl_matrix *Z,
                              gsl_matrix *Zn,
                              gsl_matrix *Q,
                              int index,
                              int K,
                              int N,
                              int add) {
    //removing a row from kronnecker_product (z,z_n) & (z_n,z) add=0
    double nu;
    gsl_matrix_view Z_view;
    gsl_matrix **Y = (gsl_matrix **) calloc(N, sizeof(gsl_matrix *));
    for (int i = 0; i < N; i++) {
        Y[i] = gsl_matrix_alloc(K, K);
        Z_view = gsl_matrix_submatrix(Z, 0, i, K, 1);
        matrix_multiply(&Z_view.matrix, &Z_view.matrix, Y[i], 1, 0, CblasNoTrans, CblasTrans);
    }

    for(int i = 0; i < N; i++){
        if(i == index){
            continue;
        }
        for(int j = 0; j < N; j++){
            if(j == index){
                continue;
            }

            recursive_inverse(K, Q, Y[i], Y[j], add);
        }
    }

    for (int d = 0; d < N; d++) {
        gsl_matrix_free(Y[d]);
    }
    free(Y);

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

int Update_Q_after_removing(int N,
                            int K,
                            int index,
                            double beta,
                            gsl_matrix *Z,
                            gsl_matrix *Qnon,
                            int add) {
    //compute the Q_non using its standard formulation without any accelerating tricks
    gsl_matrix *Znon = gsl_matrix_calloc(K, N - 1);
    gsl_matrix *Snon;
    if (add == 0) {
        remove_col(K, N, index, Znon, Z);
        Snon = gsl_matrix_calloc(K * K, (N - 1) * (N - 1));
        gsl_Kronecker_product(Snon, Znon, Znon);
    } else {
        Snon = gsl_matrix_calloc(K * K, N * N);
        gsl_Kronecker_product(Snon, Z, Z);
    }
    gsl_matrix *invQnon = gsl_matrix_calloc(K * K, K * K);
    gsl_matrix_set_identity(invQnon);
    matrix_multiply(Snon, Snon, invQnon, 1, beta, CblasNoTrans, CblasTrans);
    gsl_matrix_memcpy(Qnon, invQnon);
    inverse(Qnon, K * K);
    gsl_matrix_free(Znon);
    gsl_matrix_free(Snon);
    gsl_matrix_free(invQnon);
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
        Y[d] = gsl_matrix_alloc(K, K);
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

void compute_inverse_Q_directly(int N,
                               int K,
                               gsl_matrix *Z,
                               double beta,
                               gsl_matrix *Q) {
    // it cleans Q to identity matrix
    gsl_matrix_set_identity(Q);
    gsl_matrix *S = gsl_matrix_calloc(K * K, N * N);

    gsl_Kronecker_product(S, Z, Z);

    // Q = s * s + beta * I
    matrix_multiply(S, S, Q, 1, beta, CblasNoTrans, CblasTrans);

    inverse(Q, K * K);
    gsl_matrix_free(S);
}

// will only update Enon, Znon and Rho will not be touched
void normal_update_eta(gsl_matrix * Znon, gsl_matrix *Rho, int n, gsl_matrix * Enon){
    gsl_matrix *ZnonOZnon = gsl_matrix_alloc(Znon->size1 * Znon->size1, Znon->size2 * Znon->size2);
    gsl_Kronecker_product(ZnonOZnon, Znon, Znon);
    gsl_matrix *rhocy = gsl_matrix_alloc(Rho->size1, Rho->size2);
    gsl_matrix_memcpy(rhocy, Rho);

    for (int i = n; i < Rho->size1 - 1; i++) {
        gsl_matrix_swap_rows(rhocy, i, i + 1);
        gsl_matrix_swap_columns(rhocy, i, i + 1);
    }
    gsl_matrix_view rho_n_n = gsl_matrix_submatrix(rhocy, 0, 0, Rho->size1 - 1, Rho->size2 - 1);
    gsl_matrix *vecRho_n_n = gsl_matrix_alloc((Rho->size1 - 1) * (Rho->size2 - 1), 1);
    gsl_matrix2vector(vecRho_n_n, &rho_n_n.matrix);

    matrix_multiply(ZnonOZnon, vecRho_n_n, Enon, 1, 0, CblasNoTrans, CblasNoTrans);

    gsl_matrix_free(ZnonOZnon);
    gsl_matrix_free(rhocy);
    gsl_matrix_free(vecRho_n_n);
}

void print_Zn(gsl_matrix * Zn, int K){
    for(int i = 0; i < K; i++){
        cout << gsl_matrix_get(Zn, i, 0) << ", ";
    }
    cout << "\n";
}