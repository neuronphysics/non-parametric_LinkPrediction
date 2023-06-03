#include "InferenceFunctionsNetwork.h"

using namespace std;
//**********************************************************************************************************************//
//**********************************************************************************************************************//
//
//Authors: This code is originally developed by Isabel Valera for the "General latent feature models for heterogeneous dataset" paper
//It has been modified by Zahra Sheikhbahaee to take into account the network data. In new version, there are two pseudo likelihood, one is related to the heterogeneous attribute data and the other belongs to the adjacency matrix and we posit that there is an affinity matrix which takes into account the degree of homophily/heterophily between the latent communities (features)......
//
//**********************************************************************************************************************//
//**********************************************************************************************************************//

void
SampleY(double missing, int N, int d, int K, char Cd, int Rd, double fd, double mud, double wd, double s2Y, double s2u,
        double s2theta, gsl_matrix *X, gsl_matrix *Z, gsl_matrix *Yd, gsl_matrix *Bd, gsl_vector *thetad,
        const gsl_rng *seed) {
    double sYd = sqrt(s2Y);
    double stheta = sqrt(s2theta);
    gsl_matrix_view Zn;
    gsl_matrix_view Bd_view;
    gsl_matrix *muy;
    double xnd;
    switch (Cd) {
        case 'g':
            //real-valued observations Eq. (14)
            muy = gsl_matrix_calloc(1, 1);
            for (int n = 0; n < N; n++) {
                xnd = gsl_matrix_get(X, d, n);
                Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
                Bd_view = gsl_matrix_submatrix(Bd, 0, 0, K, 1);
                //muy=Zn*Bd_view
                matrix_multiply(&Zn.matrix, &Bd_view.matrix, muy, 1, 0, CblasTrans, CblasNoTrans);
                if (xnd == missing || gsl_isnan(xnd)) {
                    gsl_matrix_set(Yd, 0, n, gsl_matrix_get(muy, 0, 0) + gsl_ran_gaussian(seed, sYd));
                } else {
                    gsl_matrix_set(Yd, 0, n, (fre_1(xnd, fd, mud, wd) / s2u + gsl_matrix_get(muy, 0, 0) / s2Y) /
                                             (1 / s2Y + 1 / s2u) +
                                             gsl_ran_gaussian(seed, sqrt(1 / (1 / s2Y + 1 / s2u))));
                }
            }
            gsl_matrix_free(muy);
            break;

        case 'p':
            //positive real-valued observations Eq. (15)
            muy = gsl_matrix_calloc(1, 1);
            for (int n = 0; n < N; n++) {
                xnd = gsl_matrix_get(X, d, n);
                Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
                Bd_view = gsl_matrix_submatrix(Bd, 0, 0, K, 1);
                matrix_multiply(&Zn.matrix, &Bd_view.matrix, muy, 1, 0, CblasTrans, CblasNoTrans);
                if (xnd == missing || gsl_isnan(xnd)) {
                    gsl_matrix_set(Yd, 0, n, gsl_matrix_get(muy, 0, 0) + gsl_ran_gaussian(seed, sYd));
                } else {
                    gsl_matrix_set(Yd, 0, n, (f_1(xnd, fd, mud, wd) / s2u + gsl_matrix_get(muy, 0, 0) / s2Y) /
                                             (1 / s2Y + 1 / s2u) +
                                             gsl_ran_gaussian(seed, sqrt(1 / (1 / s2Y + 1 / s2u))));
                }
            }
            gsl_matrix_free(muy);
            break;

        case 'n':
            //count observations
            muy = gsl_matrix_calloc(1, 1);
            for (int n = 0; n < N; n++) {
                xnd = gsl_matrix_get(X, d, n);
                Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
                Bd_view = gsl_matrix_submatrix(Bd, 0, 0, K, 1);
                matrix_multiply(&Zn.matrix, &Bd_view.matrix, muy, 1, 0, CblasTrans, CblasNoTrans);
                if (xnd == missing || gsl_isnan(xnd)) {
                    gsl_matrix_set(Yd, 0, n, gsl_matrix_get(muy, 0, 0) + gsl_ran_gaussian(seed, sYd));
                } else {
                    gsl_matrix_set(Yd, 0, n, truncnormrnd(gsl_matrix_get(muy, 0, 0), sYd, f_1(xnd, fd, mud, wd),
                                                          f_1(xnd + 1, fd, mud, wd), seed));
                }
                if (isinf(gsl_matrix_get(Yd, 0, n)) || isnan(gsl_matrix_get(Yd, 0, n))) {
                    LOG(OUTPUT_NORMAL,
                        "EXECUTION STOPPED: the distribution of attribute %d (%d in Matlab) leads to numerical errors at the sampler. \n                   Have you considered applying a pre-processing transformation to this attribute? \n",
                        d, d + 1);
                    break;
                }
            }
            gsl_matrix_free(muy);
            break;

        case 'c':
            //categorical observations
            muy = gsl_matrix_calloc(1, Rd);
            for (int n = 0; n < N; n++) {
                xnd = gsl_matrix_get(X, d, n);
                Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
                Bd_view = gsl_matrix_submatrix(Bd, 0, 0, K, Rd);
                //muy=Zn * Bd_view
                matrix_multiply(&Zn.matrix, &Bd_view.matrix, muy, 1, 0, CblasTrans, CblasNoTrans);
                if (xnd == missing || gsl_isnan(xnd)) {
                    for (int r = 0; r < Rd; r++) {
                        gsl_matrix_set(Yd, r, n, gsl_matrix_get(muy, 0, r) + gsl_ran_gaussian(seed, sYd));
                    }
                } else {
                    double maxY = 0;
                    double ytrue = gsl_matrix_get(Yd, xnd - 1, n);
                    for (int r = 0; r < Rd; r++) {
                        double ydr = gsl_matrix_get(Yd, r, n);
                        if ((ydr != ytrue) & (ydr > maxY)) { maxY = ydr; }
                    }
                    gsl_matrix_set(Yd, xnd - 1, n,
                                   truncnormrnd(gsl_matrix_get(muy, 0, xnd - 1), sYd, maxY, GSL_POSINF, seed));
                    for (int r = 0; r < Rd; r++) {
                        if (r != xnd - 1) {
                            gsl_matrix_set(Yd, r, n, truncnormrnd(gsl_matrix_get(muy, 0, r), sYd, GSL_NEGINF,
                                                                  gsl_matrix_get(Yd, xnd - 1, n), seed));
                        }
                    }
                }
            }
            gsl_matrix_free(muy);
            break;

        case 'b':
            //binary observations
            muy = gsl_matrix_calloc(1, 1);
            for (int n = 0; n < N; n++) {
                xnd = (int) gsl_matrix_get(X, d, n);
                Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
                Bd_view = gsl_matrix_submatrix(Bd, 0, 0, K, 1);
                matrix_multiply(&Zn.matrix, &Bd_view.matrix, muy, 1, 0, CblasTrans, CblasNoTrans);
                if (xnd == -1 || gsl_isnan(xnd)) {//missing data
                    gsl_matrix_set(Yd, 0, n, gsl_matrix_get(muy, 0, 0) + gsl_ran_gaussian(seed, sYd));
                } else if (xnd == 0) {
                    gsl_matrix_set(Yd, 0, n, truncnormrnd(gsl_matrix_get(muy, 0, 0), sYd, GSL_NEGINF, 0, seed));
                } else if (xnd == 1) {
                    gsl_matrix_set(Yd, 0, n, truncnormrnd(gsl_matrix_get(muy, 0, 0), sYd, 0, GSL_POSINF, seed));
                } else {
                    LOG(OUTPUT_NORMAL, "Error! xnd for binary is not 0, 1, -1 (for missing data)");
                }
            }
            gsl_matrix_free(muy);
            break;

        case 'o':
            //ordinal observations
            gsl_vector *Ymax = gsl_vector_calloc(Rd);
            gsl_vector *Ymin = gsl_vector_alloc(Rd);
            gsl_vector_set_all(Ymin, GSL_POSINF);
            muy = gsl_matrix_calloc(1, 1);
            for (int n = 0; n < N; n++) {
                xnd = gsl_matrix_get(X, d, n);
                Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
                Bd_view = gsl_matrix_submatrix(Bd, 0, 0, K, 1);
                //muy=Zn * Bd_view
                matrix_multiply(&Zn.matrix, &Bd_view.matrix, muy, 1, 0, CblasTrans, CblasNoTrans);
                if (xnd == missing || gsl_isnan(xnd)) {
                    gsl_matrix_set(Yd, 0, n, gsl_matrix_get(muy, 0, 0) + gsl_ran_gaussian(seed, sYd));
                } else if (xnd == 1) {
                    gsl_matrix_set(Yd, 0, n, truncnormrnd(gsl_matrix_get(muy, 0, 0), sYd, GSL_NEGINF,
                                                          gsl_vector_get(thetad, xnd - 1), seed));
                    if (gsl_matrix_get(Yd, 0, n) > gsl_vector_get(Ymax, xnd - 1)) {
                        gsl_vector_set(Ymax, xnd - 1, gsl_matrix_get(Yd, 0, n));
                    }
                    if (gsl_matrix_get(Yd, 0, n) < gsl_vector_get(Ymin, xnd - 1)) {
                        gsl_vector_set(Ymin, xnd - 1, gsl_matrix_get(Yd, 0, n));
                    }
                } else {
                    gsl_matrix_set(Yd, 0, n,
                                   truncnormrnd(gsl_matrix_get(muy, 0, 0), sYd, gsl_vector_get(thetad, xnd - 2),
                                                gsl_vector_get(thetad, xnd - 1), seed));
                    if (gsl_matrix_get(Yd, 0, n) > gsl_vector_get(Ymax, xnd - 1)) {
                        gsl_vector_set(Ymax, xnd - 1, gsl_matrix_get(Yd, 0, n));
                    }
                    if (gsl_matrix_get(Yd, 0, n) < gsl_vector_get(Ymin, xnd - 1)) {
                        gsl_vector_set(Ymin, xnd - 1, gsl_matrix_get(Yd, 0, n));
                    }
                }
            }
            gsl_matrix_free(muy);
            //Sample Theta
            for (int r = 1; r < Rd - 1; r++) {
                double xlo;
                double xhi;
                if (gsl_vector_get(thetad, r) > gsl_vector_get(Ymax, r)) { xlo = gsl_vector_get(thetad, r); }
                else { xlo = gsl_vector_get(Ymax, r); }
                if (gsl_vector_get(thetad, r + 1) < gsl_vector_get(Ymin, r + 1)) {
                    xhi = gsl_vector_get(thetad, r + 1);
                } else { xhi = gsl_vector_get(Ymin, r + 1); }
                //theta_r^d=Gaussian(theta_r^d|0,sigma_{theta}^2)I(theta_r^d>theta_{r-1}^d)
                gsl_vector_set(thetad, r, truncnormrnd(0, stheta, xlo, xhi, seed));
            }
            break;
    }
}

// Sample Rho : pseudo-observation of the adjacency matrix
void SampleRho(double missing,
               int N,
               int K,
               char Ca,
               double fa,
               double s2Rho,
               double s2u,
               gsl_matrix *A,
               gsl_matrix *Z,
               gsl_matrix *vecRho,
               gsl_matrix *H,
               const gsl_rng *seed) {
    double sRho = sqrt(s2Rho);
    gsl_matrix_view Zview = gsl_matrix_submatrix(Z, 0, 0, K, N);
    gsl_matrix_view H_view = gsl_matrix_submatrix(H, 0, 0, K, K);
    gsl_matrix *vecH = gsl_matrix_calloc(K * K, 1);
    gsl_matrix2vector(vecH, &H_view.matrix);
    gsl_matrix *mu_rho;
    gsl_matrix *aux = gsl_matrix_calloc(1, K * K);
    gsl_matrix *ZmT = gsl_matrix_calloc(1, K);
    gsl_matrix *ZnT = gsl_matrix_calloc(1, K);
    gsl_matrix_view Zn;
    gsl_matrix_view Zm;
    int a_nm;
    // Sample pseudo adjacency matrix
    if (Ca == 'w') {
        double mud;
        double wd;
        //https://gist.github.com/microo8/4065693
        gsl_vector_view An_view;

        for (int m = 0; m < N; m++) {
            Zm = gsl_matrix_submatrix(Z, 0, m, K, 1);
            An_view = gsl_matrix_row(A, m);
            mud = compute_vector_mean(N, missing, &An_view.vector);
            wd = 1. / sqrt(compute_vector_var(N, missing, &An_view.vector));
            gsl_matrix_transpose_memcpy(ZmT, &Zm.matrix);
            for (int n = 0; n < m; n++) {//try to keep Rho matrix symmetric
                mu_rho = gsl_matrix_calloc(1, 1);
                a_nm = gsl_matrix_get(A, m, n);
                Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
                gsl_matrix_transpose_memcpy(ZnT, &Zn.matrix);
                gsl_matrix_transpose_memcpy(ZmT, &Zm.matrix);
                //gsl_Kronecker_product(aux, &z_n_column.matrix, &z_m_column.matrix);//???
                gsl_Kronecker_product(aux, ZnT, ZmT);
                matrix_multiply(aux, vecH, mu_rho, 1, 0, CblasNoTrans, CblasNoTrans);
                if (gsl_isnan(a_nm) || a_nm == missing) {
                    gsl_matrix_set(vecRho, m * N + n, 0, gsl_matrix_get(mu_rho, 0, 0) + gsl_ran_gaussian(seed, sRho));
                } else {
                    gsl_matrix_set(vecRho, m * N + n, 0,
                                   (f_w(a_nm, fa, mud, wd) / s2u + gsl_matrix_get(mu_rho, 0, 0) / s2Rho) /
                                   (1 / s2Rho + 1 / s2u) + gsl_ran_gaussian(seed, sqrt(1 / (1 / s2Rho + 1 / s2u))));
                }
                gsl_matrix_set(vecRho, n * N + m, 0,
                               gsl_matrix_get(vecRho, m * N + n, 0)); //extend symmetric matrix Rho to its vector

            }
        }

    } else if (Ca == 'b') {
        // binary values
        for (int m = 0; m < N; m++) {
            Zm = gsl_matrix_submatrix(Z, 0, m, K, 1);
            for (int n = 0; n < m; n++) {
                mu_rho = gsl_matrix_calloc(1, 1);
                Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
                gsl_matrix_transpose_memcpy(ZnT, &Zn.matrix);
                gsl_matrix_transpose_memcpy(ZmT, &Zm.matrix);
                gsl_Kronecker_product(aux, ZnT, ZmT);
                matrix_multiply(aux, vecH, mu_rho, 1, 0, CblasNoTrans, CblasNoTrans);

                a_nm = (int) gsl_matrix_get(A, m, n);
                if (gsl_isnan(a_nm || a_nm == missing)) {
                    gsl_matrix_set(vecRho, m * N + n, 0, gsl_matrix_get(mu_rho, 0, 0) + gsl_ran_gaussian(seed, sRho));
                } else if (a_nm == 0) {

                    gsl_matrix_set(vecRho, m * N + n, 0,
                                   truncnormrnd(gsl_matrix_get(mu_rho, 0, 0), sRho, GSL_NEGINF, 0, seed));
                } else if (a_nm == 1) {

                    gsl_matrix_set(vecRho, m * N + n, 0,
                                   truncnormrnd(gsl_matrix_get(mu_rho, 0, 0), sRho, 0, GSL_POSINF, seed));
                }
                gsl_matrix_set(vecRho, n * N + m, 0, gsl_matrix_get(vecRho, m * N + n, 0));

                //print the problematic part of the code
                if (isinf(gsl_matrix_get(vecRho, m * N + n, 0))) {
                    LOG(OUTPUT_DEBUG, "mu: %3.2f\n", gsl_matrix_get(mu_rho, 0, 0));
                    LOG(OUTPUT_DEBUG, "\n vec(H):\n");
                    for (int row = 0; row < K * K; ++row)
                        LOG(OUTPUT_DEBUG, "%6.5f\t", gsl_matrix_get(vecH, row, 0));
                    LOG(OUTPUT_DEBUG, "\n Z x Z:\n");
                    for (int col = 0; col < K * K; ++col)
                        LOG(OUTPUT_DEBUG, "%6.5f\t", gsl_matrix_get(aux, 0, col));
                    LOG(OUTPUT_DEBUG, "\n---\n---\n");
                    LOG(OUTPUT_DEBUG, "m:%d , n:%d , A_{mn}: %.2f, A_{mn}: %d, Rho: %.3f\n", m, n,
                        gsl_matrix_get(A, m, n), a_nm,
                        gsl_matrix_get(vecRho, m * N + n, 0));
                }
            }
        }
    }
    gsl_matrix_free(ZmT);
    gsl_matrix_free(ZnT);
    gsl_matrix_free(vecH);
    gsl_matrix_free(aux);
    gsl_matrix_free(mu_rho);
}

double SampleAlpha(int Kplus, int N, const gsl_rng *seed) {
    double Harmonic_N = 0.;
    double i = 1.;
    while (i < N + 1) {
        Harmonic_N += 1.0 / i;
        i++;
    }

    // equation 21 https://arxiv.org/pdf/1011.6293.pdf
    double alpha = gsl_ran_gamma(seed, 1 + Kplus,
                                 1 / (1 + Harmonic_N));
    return alpha;
}

double Samples2Y(double missing, int N, int d, int K, char Cd, int Rd, double fd, double mud, double wd, double s2u,
                 double s2theta, gsl_matrix *X, gsl_matrix *Z, gsl_matrix *Yd, gsl_matrix *Bd, gsl_vector *thetad,
                 const gsl_rng *seed) {
    double a = 2;
    double b = 2;
    gsl_matrix_view Zn;
    gsl_matrix_view Bd_view;
    gsl_matrix *muy;
    double sumY = 0;
    double xnd;

    muy = gsl_matrix_calloc(1, 1);

    for (int n = 0; n < N; n++) {
        for (int r = 0; r < Rd; r++) {
            xnd = gsl_matrix_get(X, d, n);
            Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
            Bd_view = gsl_matrix_submatrix(Bd, 0, 0, K, 1);
            //muy=Zn*Bd_view
            matrix_multiply(&Zn.matrix, &Bd_view.matrix, muy, 1, 0, CblasTrans, CblasNoTrans);
            sumY += pow(gsl_matrix_get(Yd, 0, n) - gsl_matrix_get(muy, 0, 0), 2);
        }
    }
    gsl_matrix_free(muy);
    //Eq. (18) of Infinite Sparse Factor Analysis and Infinite Independent Components Analysis
    //IG(s2Y|a+ND/2,b/(1+b/2*tr(E^T E)))
    double precision = gsl_ran_gamma(seed, a + N / 2., 1 / (b + sumY / 2));//inverse Gamma prior
    return 1. / precision;
}

// sample noise variance of the pseudo-observation of the adjacency matrix
double
Samples2Rho(int N, int K, gsl_matrix *A, gsl_matrix *Z, gsl_matrix *vecRho, gsl_matrix *vecH, const gsl_rng *seed) {
    double a = 1;
    double b = 1;

    gsl_matrix *aux = gsl_matrix_calloc(N * N, 1);
    gsl_matrix *S = gsl_matrix_calloc(K * K, N * N);
    gsl_matrix *D = gsl_matrix_calloc(1, 1);

    gsl_matrix_view Z_view = gsl_matrix_submatrix(Z, 0, 0, K, N);


    gsl_Kronecker_product(S, &Z_view.matrix, &Z_view.matrix);
    matrix_multiply(S, vecH, aux, 1, 0, CblasTrans, CblasNoTrans);
    gsl_matrix_sub(aux, vecRho);
    matrix_multiply(aux, aux, D, 1, 0, CblasTrans, CblasNoTrans);

    for (int n = 0; n < N * N; n++) {
        if (isinf(gsl_matrix_get(vecRho, n, 0))) {
            LOG(OUTPUT_DEBUG, "%d , %.3f \n", n, gsl_matrix_get(vecRho, n, 0));
        }
    }
    LOG(OUTPUT_DEBUG, "sample s2rho: %.4f\n", gsl_matrix_get(D, 0, 0));
    double precision = gsl_ran_gamma(seed, a + N * N / 2., 1 / (b + gsl_matrix_get(D, 0, 0) / 2.));//???????
    gsl_matrix_free(aux);
    gsl_matrix_free(S);
    gsl_matrix_free(D);
    return 1. / precision;
}


double Samples2H(int K, gsl_matrix *vecH, const gsl_rng *seed) {
    double a = 2;
    double b = 1;
    gsl_matrix *var = gsl_matrix_calloc(1, 1);

    matrix_multiply(vecH, vecH, var, 1, 0, CblasTrans, CblasNoTrans);
    LOG(OUTPUT_DEBUG, "sample s2H: %.4f\n", gsl_matrix_get(var, 0, 0));
    double precision = gsl_ran_gamma(seed, a + K * K / 2., b / (1 + b * gsl_matrix_get(var, 0, 0) / 2));
    gsl_matrix_free(var);
    return 1. / precision;
}


int IBPsampler_func(double missing,
                    gsl_matrix *X,
                    char *C,
                    char *Net,//*** the type of network
                    gsl_matrix *Z, //The binary feature vector
                    gsl_matrix **B, //the weighting vectors D * maxK * 1, initially all 0
                    gsl_vector **theta,
                    gsl_matrix *H,// The homophily matrix
                    gsl_matrix *A,// The adjacency matrix
                    int *R, //unordered index set of the categorical data
                    double *f, //mapping function from the real space R into the observation space
                    double fa,
                    double *mu, // mean, mu[d] = mean(X[d]) the mean value of an attribute to all nodes
                    double *w, //variance
                    int maxR,
                    int bias,
                    int N,
                    int D,
                    int K,
                    double alpha,
                    double s2B,
                    double *s2Y,
                    double s2Rho,
                    double s2H,
                    double s2u,
                    int maxK,
                    int Nsim) {

    LOG(OUTPUT_NORMAL, "N=%d, D=%d, K=%d", N, D, K);
    LOG(OUTPUT_INFO, "Running inference algorithm (currently inside C++ routine...)");

    double s2theta = 2;

    gsl_rng *seed = gsl_rng_alloc(gsl_rng_taus);
    time_t clck = time(nullptr);
    gsl_rng_set(seed, clck);

    // auxiliary variables
    int Kest = K;
    gsl_matrix *P = gsl_matrix_calloc(maxK, maxK);
    gsl_matrix_set_identity(P);
    gsl_matrix_view P_view = gsl_matrix_submatrix(P, 0, 0, Kest, Kest);

    gsl_matrix_view Z_view = gsl_matrix_submatrix(Z, 0, 0, Kest, N);
    matrix_multiply(&Z_view.matrix, &Z_view.matrix, &P_view.matrix, 1, 1 / s2B, CblasNoTrans, CblasTrans);
    gsl_matrix *Pnon = gsl_matrix_calloc(maxK, maxK);


    // Initialize Q and Qnon
    gsl_matrix *Q = gsl_matrix_calloc(maxK * maxK, maxK * maxK);
    double ldet_Q = 0;


    double coeff = s2Rho / s2H;
    gsl_matrix_view Q_view_init = gsl_matrix_submatrix(Q, 0, 0, Kest * Kest, Kest * Kest);
    compute_inverse_Q_directly(N, Kest, &Z_view.matrix, coeff, &Q_view_init.matrix);

    gsl_matrix *Qnon = gsl_matrix_calloc(maxK * maxK, maxK * maxK);
    double ldet_Q_n = 0;



    // initialize counts
    int *nest = new int[maxK];
    for (int i = 0; i < maxK; i++) {
        nest[i] = 0;
    }

    for (int k = 0; k < Kest; k++) {
        int ncount = 0;
        for (int n = 0; n < N; n++) {
            if (gsl_matrix_get(Z, k, n) == 1) { ncount++; }
        }
        nest[k] = ncount;
    }

    auto **Y = (gsl_matrix **) calloc(D, sizeof(gsl_matrix *));
    auto **lambda = (gsl_matrix **) calloc(D, sizeof(gsl_matrix *));
    auto **lambdanon = (gsl_matrix **) calloc(D, sizeof(gsl_matrix *));

    //Initialize Y
    for (int d = 0; d < D; d++) {
        switch (C[d]) {
            double xnd;
            case 'g':
                Y[d] = gsl_matrix_calloc(1, N);
                for (int n = 0; n < N; n++) {
                    xnd = gsl_matrix_get(X, d, n);
                    if (xnd == missing || gsl_isnan(xnd)) {
                        //  if the real observation is missing, use random number follow Gaussian distribution with mean 0 and std sY
                        //  mean 0 because Bd is initially all 0
                        gsl_matrix_set(Y[d], 0, n, gsl_ran_gaussian(seed, sqrt(s2Y[d])));
                    } else {
                        //  f[d] is meaningless in this function call, it set pseudo ob to w * (x - mu)
                        gsl_matrix_set(Y[d], 0, n, fre_1(xnd, f[d], mu[d], w[d]));
                    }
                }

                break;

            case 'p':
                Y[d] = gsl_matrix_calloc(1, N);
                for (int n = 0; n < N; n++) {
                    xnd = gsl_matrix_get(X, d, n);

                    if (xnd == missing || gsl_isnan(xnd)) {
                        gsl_matrix_set(Y[d], 0, n, gsl_ran_gaussian(seed, sqrt(s2Y[d])));
                    } else {
                        gsl_matrix_set(Y[d], 0, n, f_1(xnd, f[d], mu[d], w[d])); //+gsl_ran_gaussian (seed, s2Y)
                    }
                }
                break;

            case 'n':
                Y[d] = gsl_matrix_calloc(1, N);
                lambda[d] = gsl_matrix_calloc(maxK, 1);
                matrix_multiply(Z, Y[d], lambda[d], 1, 0, CblasNoTrans, CblasTrans);
                for (int n = 0; n < N; n++) {
                    xnd = gsl_matrix_get(X, d, n);

                    if (xnd == missing || gsl_isnan(xnd)) {
                        gsl_matrix_set(Y[d], 0, n, gsl_ran_gaussian(seed, sqrt(s2Y[d])));
                    } else {
                        gsl_matrix_set(Y[d], 0, n, f_1(xnd, f[d], mu[d],
                                                       w[d]));// +gsl_ran_beta (seed, 5,1)????? shouldn't this be floor (f_1(xnd,f[d], mu[d], w[d]))
                    }
                }
                break;

            case 'c':
                Y[d] = gsl_matrix_calloc(R[d], N);
                for (int n = 0; n < N; n++) {
                    xnd = gsl_matrix_get(X, d, n);
                    if (xnd == missing || gsl_isnan(xnd)) {
                        for (int r = 0; r < R[d]; r++) {
                            gsl_matrix_set(Y[d], r, n, gsl_ran_gaussian(seed, sqrt(s2Y[d])));
                        }
                    } else {
                        gsl_matrix_set(Y[d], xnd - 1, n, truncnormrnd(0, sqrt(s2Y[d]), 0, GSL_POSINF, seed));
                        for (int r = 0; r < R[d]; r++) {
                            if (r != xnd - 1) {
                                gsl_matrix_set(Y[d], r, n, truncnormrnd(0, sqrt(s2Y[d]), GSL_NEGINF,
                                                                        gsl_matrix_get(Y[d], xnd - 1, n), seed));
                            }
                        }
                    }
                }
                break;

            case 'b':
                Y[d] = gsl_matrix_calloc(1, N);
                for (int n = 0; n < N; n++) {
                    xnd = (int) gsl_matrix_get(X, d, n);
                    if (xnd == -1 || gsl_isnan(xnd)) {
                        // it is a missing binary value
                        gsl_matrix_set(Y[d], 0, n, gsl_ran_gaussian(seed, sqrt(s2Y[d])));

                    } else if (xnd == 0) {
                        // it just gives it a negative number follows normal distribution with mean 0
                        gsl_matrix_set(Y[d], 0, n, truncnormrnd(0, sqrt(s2Y[d]), GSL_NEGINF, 0, seed));
                    } else if (xnd == 1) {
                        gsl_matrix_set(Y[d], 0, n, truncnormrnd(0, sqrt(s2Y[d]), 0, GSL_POSINF, seed));
                    }
                }
                break;

            case 'o':
                Y[d] = gsl_matrix_calloc(R[d], N);
                gsl_vector_view Xd_view = gsl_matrix_row(X, d);
                double maxX = compute_vector_max(N, missing, &Xd_view.vector);
                gsl_vector_set(theta[d], 0, -sqrt(s2Y[d]));
                for (int r = 1; r < R[d] - 1; r++) {
                    gsl_vector_set(theta[d], r,
                                   gsl_vector_get(theta[d], r - 1) + (4 * sqrt(s2Y[d]) / maxX) * rand01());
                }
                gsl_vector_set(theta[d], R[d] - 1, GSL_POSINF);
                for (int n = 0; n < N; n++) {
                    xnd = gsl_matrix_get(X, d, n);

                    if (xnd == missing || gsl_isnan(xnd)) {
                        gsl_matrix_set(Y[d], 0, n, gsl_ran_gaussian(seed, sqrt(s2Y[d])));
                    } else if (xnd == 1) {
                        gsl_matrix_set(Y[d], 0, n,
                                       truncnormrnd(0, sqrt(s2Y[d]), GSL_NEGINF, gsl_vector_get(theta[d], xnd - 1), seed));
                    } else {
                        gsl_matrix_set(Y[d], 0, n, truncnormrnd(0, sqrt(s2Y[d]), gsl_vector_get(theta[d], xnd - 2),
                                                                gsl_vector_get(theta[d], xnd - 1), seed));
                    }
                }
                break;
        }
        // R[d] is always 1
        lambda[d] = gsl_matrix_calloc(maxK, R[d]);
        matrix_multiply(Z, Y[d], lambda[d], 1, 0, CblasNoTrans, CblasTrans);
        lambdanon[d] = gsl_matrix_calloc(maxK, R[d]);
    }


    LOG(OUTPUT_INFO, "Initialize Rho using pseudo-observation of the adjacency matrix ..... !\n");
    gsl_matrix *Rho = gsl_matrix_calloc(N, N);
    gsl_matrix *vecRho = gsl_matrix_calloc(N * N, 1);


    if (Net[0] == 'w') {
        double a_mn;
        double mu_a;
        double wa;
        gsl_vector_view Am_view;
        for (int m = 0; m < N; m++) {
            Am_view = gsl_matrix_row(A, m);
            //mu_a  = compute_vector_mean(N, missing, &Am_view.vector);
            mu_a = compute_vector_min(N, missing, &Am_view.vector) - 1e-6;
            wa = 1. / sqrt(compute_vector_var(N, missing, &Am_view.vector));
            for (int n = 0; n < m; n++) {
                a_mn = gsl_matrix_get(A, m, n);
                if ((a_mn == missing) || gsl_isnan(a_mn)) {
                    gsl_matrix_set(vecRho, m * N + n, 0, gsl_ran_gaussian(seed, sqrt(s2Rho)));
                } else {
                    gsl_matrix_set(vecRho, m * N + n, 0, f_w(a_mn, fa, mu_a, wa));
                }
                gsl_matrix_set(vecRho, n * N + m, 0, gsl_matrix_get(vecRho, m * N + n, 0));
            }
        }
    } else if (Net[0] == 'b') {
        // adjacency matrix is binary
        int a_mn;
        for (int m = 0; m < N; m++) {
            for (int n = 0; n < m; n++) {
                a_mn = (int) gsl_matrix_get(A, m, n);
                if ((a_mn == missing) || gsl_isnan(a_mn)) {
                    // use 0 as the mean here because H is currently all 0, thus, mean is 0
                    gsl_matrix_set(vecRho, m * N + n, 0, gsl_ran_gaussian(seed, sqrt(s2Rho)));
                } else if (a_mn == 0) {
                    // it just give it a negative number follows normal distribution with mean 0
                    gsl_matrix_set(vecRho, m * N + n, 0, truncnormrnd(0, sqrt(s2Rho), GSL_NEGINF, 0, seed));
                } else if (a_mn == 1) {
                    gsl_matrix_set(vecRho, m * N + n, 0, truncnormrnd(0, sqrt(s2Rho), 0, GSL_POSINF, seed));
                }
                gsl_matrix_set(vecRho, n * N + m, 0, gsl_matrix_get(vecRho, m * N + n, 0));
            }
        }
    }

    // compute full Eta
    gsl_matrix *Eta = gsl_matrix_calloc(maxK * maxK, 1);
    gsl_matrix_view Eta_init_view = gsl_matrix_submatrix(Eta, 0, 0, Kest * Kest, 1);

    gsl_matrix *ZoZ = gsl_matrix_calloc(Kest * Kest, N * N);

    gsl_Kronecker_product(ZoZ, &Z_view.matrix, &Z_view.matrix);
    matrix_multiply(ZoZ, vecRho, &Eta_init_view.matrix, 1, 0, CblasNoTrans, CblasNoTrans);
    gsl_matrix_free(ZoZ);

    // todo debug only
    print_matrix(Eta, "init Eta", maxK);


    gsl_matrix *Etanon = gsl_matrix_calloc(maxK * maxK, 1);


    LOG(OUTPUT_DEBUG, "Before IT loop...");
    LOG(OUTPUT_DEBUG, "Nsim = %d", Nsim);

    // main loop
    for (int it = 0; it < Nsim; it++) {
        LOG(OUTPUT_NORMAL, "Start iteration %d", it);
        gsl_vector2matrix(vecRho, Rho);
        int Kaux = AcceleratedGibbs(maxK, bias, N, D, Kest, C, R, alpha, s2B, s2Y, s2H, s2Rho, Y, Rho, vecRho, Z, nest,
                                    P, Pnon, lambda, lambdanon, Q, Qnon, Eta, Etanon, &ldet_Q, &ldet_Q_n);

        LOG(OUTPUT_NORMAL, "iteration %d, K= %d\n", it, Kaux);


        if (Kaux == 0) { return Kest; } else { Kest = Kaux; }

        P_view = gsl_matrix_submatrix(P, 0, 0, Kest, Kest);
        gsl_matrix *S = gsl_matrix_calloc(Kest, Kest);
        gsl_matrix_memcpy(S, &P_view.matrix);
        inverse(S, Kest);
        gsl_matrix *MuB = gsl_matrix_calloc(Kest, 1);

        for (int d = 0; d < D; d++) {
            //Sample Bs
            if (C[d] == 'c') {
                gsl_vector_view Bd_view;
                for (int r = 0; r < R[d] - 1; r++) {
                    gsl_matrix_view L_view = gsl_matrix_submatrix(lambda[d], 0, r, Kest, 1);
                    matrix_multiply(S, &L_view.matrix, MuB, 1, 0, CblasNoTrans, CblasNoTrans);
                    Bd_view = gsl_matrix_subcolumn(B[d], r, 0, Kest);
                    gsl_vector_view MuB_view = gsl_matrix_column(MuB, 0);
                    mvnrnd(&Bd_view.vector, S, &MuB_view.vector, Kest, seed);
                }
                Bd_view = gsl_matrix_subcolumn(B[d], R[d] - 1, 0, Kest);
                gsl_vector_set_zero(&Bd_view.vector);
            } else {
                gsl_matrix_view Lnon_view = gsl_matrix_submatrix(lambda[d], 0, 0, Kest, 1);
                matrix_multiply(S, &Lnon_view.matrix, MuB, 1, 0, CblasNoTrans, CblasNoTrans);

                gsl_vector_view Bd_view = gsl_matrix_subcolumn(B[d], 0, 0, Kest);
                gsl_vector_view MuB_view = gsl_matrix_subcolumn(MuB, 0, 0, Kest);
                mvnrnd(&Bd_view.vector, S, &MuB_view.vector, Kest, seed);
            }

            //Sample Y
            SampleY(missing, N, d, Kest, C[d], R[d], f[d], mu[d], w[d], s2Y[d], s2u, s2theta, X, Z, Y[d], B[d],
                    theta[d], seed);
            if (C[d] != 'c' && C[d] != 'o') {
                double aux = Samples2Y(missing, N, d, Kest, C[d], R[d], f[d], mu[d], w[d], s2u, s2theta, X, Z, Y[d],
                                       B[d], theta[d], seed);
                if (aux != 0 && !isinf(aux) && !isnan(aux)) {
                    s2Y[d] = aux;
                } else {
                    return Kest;
                }
            }

            //Update lambda
            matrix_multiply(Z, Y[d], lambda[d], 1, 0, CblasNoTrans, CblasTrans);

        }

        // Sample Hs
        gsl_matrix_view H_view = gsl_matrix_submatrix(H, 0, 0, Kest, Kest);

        LOG(OUTPUT_DEBUG, "old H");
        for (int i = 0; i < Kest; i++) {
            for (int j = 0; j < Kest; j++) {
                LOG(OUTPUT_DEBUG, "%3.2f", gsl_matrix_get(&H_view.matrix, i, j));
            }
            LOG(OUTPUT_DEBUG, "")
        }
        gsl_matrix *MuH = gsl_matrix_calloc(Kest * Kest, 1);

        gsl_matrix *vecH = gsl_matrix_calloc(Kest * Kest, 1);
        gsl_matrix2vector(vecH, &H_view.matrix);
        gsl_vector_view vecH_view = gsl_matrix_subcolumn(vecH, 0, 0, Kest * Kest);
        LOG(OUTPUT_DEBUG, "vecH_view size = %zd", (&vecH_view.vector)->size);


        gsl_matrix_view Q_view = gsl_matrix_submatrix(Q, 0, 0, Kest * Kest, Kest * Kest);
        gsl_matrix_view Eta_view = gsl_matrix_submatrix(Eta, 0, 0, Kest * Kest, 1);

        //  MuH sometime is very large causing new H become too large
        //  MuH = Q * S^T * vec(rho) = Q * Eta  (see equation 14)
        matrix_multiply(&Q_view.matrix, &Eta_view.matrix, MuH, 1, 0, CblasNoTrans, CblasNoTrans);
        gsl_vector_view MuH_view = gsl_matrix_column(MuH, 0);

        mvnrnd(&vecH_view.vector, &Q_view.matrix, &MuH_view.vector, Kest * Kest, seed);

        gsl_vector2matrix(vecH, &H_view.matrix);

        print_matrix(&Eta_view.matrix, "Eta matrix", Kest);
        print_matrix(&H_view.matrix, "new H");
        print_matrix(MuH, "Mu H");
        print_matrix(&Q_view.matrix, "Q matrix");


        // *****End Sampling Hs
        // sampleRho

        SampleRho(missing, N, Kest, Net[0], fa, s2Rho, s2u, A, Z, vecRho, &H_view.matrix, seed);
        // sample the variance of Rho and H
        s2Rho = Samples2Rho(N, Kest, A, Z, vecRho, vecH, seed);
        s2H = Samples2H(Kest, vecH, seed);

        alpha = SampleAlpha(Kest, N, seed);

        LOG(OUTPUT_INFO, "\n");
        LOG(OUTPUT_INFO, "s2_rho --> %.3f", s2Rho);
        LOG(OUTPUT_INFO, "s2_h   --> %.3f", s2H);
        LOG(OUTPUT_INFO, "alpha  --> %.3f", alpha);


        LOG(OUTPUT_INFO, "\n\nB matrix");
        for (int i = 0; i < D; i++) {
            gsl_matrix *Brow = B[i];
            for (int j = 0; j < Kest; j++) {
                if (OUTPUT_LEVEL >= OUTPUT_INFO) {
                    cout << gsl_matrix_get(Brow, j, 0) << " , ";
                }
            }
            LOG(OUTPUT_INFO, "");
        }
        LOG(OUTPUT_INFO, "\n");


        gsl_matrix_free(vecH);
        gsl_matrix_free(MuH);
        gsl_matrix_free(S);
    }
    LOG(OUTPUT_DEBUG, "After IT loop...\n");

    for (int d = 0; d < D; d++) {
        gsl_matrix_free(Y[d]);
        gsl_matrix_free(lambda[d]);
        gsl_matrix_free(lambdanon[d]);
    }
    free(lambda);
    free(lambdanon);
    free(Y);

    gsl_matrix_free(P);
    gsl_matrix_free(Pnon);
    gsl_matrix_free(Q);
    gsl_matrix_free(Qnon);
    gsl_matrix_free(Rho);
    gsl_matrix_free(vecRho);
    gsl_matrix_free(Eta);
    gsl_matrix_free(Etanon);

    delete[] nest;
    return Kest;
}


int initialize_func(int N,
                    int D,
                    int maxK,
                    double missing,
                    gsl_matrix *X,
                    const char *C,
                    gsl_matrix **B,
                    gsl_vector **theta,
                    int *R,
                    double *f,
                    double *mu,
                    double *w,
                    double *s2Y) {

    int maxR = 1;
    auto *maxX = new double[D];
    auto *minX = new double[D];
    auto *meanX = new double[D];
    auto *varX = new double[D];
    gsl_vector_view Xd_view;
    for (int d = 0; d < D; d++) {
        Xd_view = gsl_matrix_row(X, d);
        maxX[d] = compute_vector_max(N, missing, &Xd_view.vector);
        minX[d] = compute_vector_min(N, missing, &Xd_view.vector);
        meanX[d] = compute_vector_mean(N, missing, &Xd_view.vector);
        varX[d] = compute_vector_var(N, missing, &Xd_view.vector);
        mu[d] = 1;
        R[d] = 1;
        w[d] = 1;
        switch (C[d]) {
            case 'g':
                s2Y[d] = 2;
                B[d] = gsl_matrix_calloc(maxK, 1);
                mu[d] = meanX[d];
                if (varX[d] > 0) { w[d] = 1 / sqrt(varX[d]); }
                else { w[d] = 1; }
                break;
            case 'p':
                s2Y[d] = 2;
                B[d] = gsl_matrix_calloc(maxK, 1);
                mu[d] = minX[d] - 1e-6;
                if (varX[d] > 0) { w[d] = 1 / sqrt(varX[d]); }
                else { w[d] = 1; }
                break;
            case 'n':
                s2Y[d] = 2;
                B[d] = gsl_matrix_calloc(maxK, 1);
                mu[d] = minX[d] - 1;
                if (varX[d] > 0) { w[d] = 1 / sqrt(varX[d]); }
                else { w[d] = 1; }
                break;
            case 'c':
                s2Y[d] = 1;
                R[d] = (int) maxX[d];
                B[d] = gsl_matrix_calloc(maxK, R[d]);
                if (R[d] > maxR) { maxR = R[d]; }
                break;
            case 'o':
                s2Y[d] = 1;
                R[d] = (int) maxX[d];
                B[d] = gsl_matrix_calloc(maxK, 1);
                theta[d] = gsl_vector_alloc(R[d]);
                if (R[d] > maxR) { maxR = R[d]; }
                break;
            case 'b':
                s2Y[d] = 1;
                R[d] = 1;
                B[d] = gsl_matrix_calloc(maxK, 1);
                break;
        }
    }

    delete[] maxX;
    delete[] minX;
    delete[] meanX;
    delete[] varX;
    return maxR;
}
