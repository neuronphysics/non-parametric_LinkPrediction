#include "InferenceFunctionsNetwork.h"

using namespace std;


void
sample_Y(double missing, int N, int d, int K, char Cd, int Rd, double fd, double mud, double wd, double s2Y, double s2u,
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

void parallel_sample_binary_rho(int N, int K, int start, int end, gsl_matrix *A, gsl_matrix *Z, gsl_matrix *Rho,
                             gsl_matrix *vecH, const gsl_rng *seed,
                             double sRho, double missing) {
    gsl_matrix *ZmT = gsl_matrix_calloc(1, K);
    gsl_matrix *ZnT = gsl_matrix_calloc(1, K);
    gsl_matrix *aux = gsl_matrix_calloc(1, K * K);
    gsl_matrix *mu_rho = gsl_matrix_calloc(1, 1);
    for (int row = start; row < end; row++) {
        gsl_matrix_view Zm = gsl_matrix_submatrix(Z, 0, row, K, 1);
        for (int col = row; col < N; col++) {
            gsl_matrix_view Zn = gsl_matrix_submatrix(Z, 0, col, K, 1);
            gsl_matrix_transpose_memcpy(ZnT, &Zn.matrix);
            gsl_matrix_transpose_memcpy(ZmT, &Zm.matrix);
            gsl_Kronecker_product(aux, ZnT, ZmT);
            matrix_multiply(aux, vecH, mu_rho, 1, 0, CblasNoTrans, CblasNoTrans);

            int a_nm = (int) gsl_matrix_get(A, row, col);
            if (gsl_isnan(a_nm || a_nm == missing)) {
                gsl_matrix_set(Rho, row, col, gsl_matrix_get(mu_rho, 0, 0) + gsl_ran_gaussian(seed, sRho));
            } else if (a_nm == 0) {

                gsl_matrix_set(Rho, row, col,
                               truncnormrnd(gsl_matrix_get(mu_rho, 0, 0), sRho, GSL_NEGINF, 0, seed));
            } else if (a_nm == 1) {

                gsl_matrix_set(Rho, row, col,
                               truncnormrnd(gsl_matrix_get(mu_rho, 0, 0), sRho, 0, GSL_POSINF, seed));
            }
            gsl_matrix_set(Rho, col, row, gsl_matrix_get(Rho, row, col));

        }
    }
    gsl_matrix_free(mu_rho);
    gsl_matrix_free(aux);
    gsl_matrix_free(ZmT);
    gsl_matrix_free(ZnT);
}


// Sample Rho : pseudo-observation of the adjacency matrix
void sample_rho(double missing,
               int N,
               int K,
               char Ca,
               double fa,
               double s2Rho,
               double s2u,
               gsl_matrix *A,
               gsl_matrix *Z,
               gsl_matrix *Rho,
               gsl_matrix *H,
               const gsl_rng *seed) {
    double sRho = sqrt(s2Rho);
    gsl_matrix_view Z_view = gsl_matrix_submatrix(Z, 0, 0, K, N);
    gsl_matrix_view H_view = gsl_matrix_submatrix(H, 0, 0, K, K);
    gsl_matrix *vecH = gsl_matrix_calloc(K * K, 1);
    gsl_matrix2vector(vecH, &H_view.matrix);


    // Sample pseudo adjacency matrix
    if (Ca == 'w') {
        gsl_matrix *mu_rho = gsl_matrix_calloc(1, 1);
        gsl_matrix *aux = gsl_matrix_calloc(1, K * K);
        gsl_matrix *ZmT = gsl_matrix_calloc(1, K);
        gsl_matrix *ZnT = gsl_matrix_calloc(1, K);
        gsl_matrix_view Zn;
        gsl_matrix_view Zm;
        int a_nm;
        double mud;
        double wd;
        gsl_vector_view An_view;

        for (int m = 0; m < N; m++) {
            Zm = gsl_matrix_submatrix(&Z_view.matrix, 0, m, K, 1);
            An_view = gsl_matrix_row(A, m);
            mud = compute_vector_mean(N, missing, &An_view.vector);
            wd = 1. / sqrt(compute_vector_var(N, missing, &An_view.vector));
            gsl_matrix_transpose_memcpy(ZmT, &Zm.matrix);
            for (int n = 0; n < m; n++) {//try to keep Rho matrix symmetric
                a_nm = gsl_matrix_get(A, m, n);
                Zn = gsl_matrix_submatrix(&Z_view.matrix, 0, n, K, 1);
                gsl_matrix_transpose_memcpy(ZnT, &Zn.matrix);
                gsl_matrix_transpose_memcpy(ZmT, &Zm.matrix);
                //gsl_Kronecker_product(aux, &z_n_column.matrix, &z_m_column.matrix);//???
                gsl_Kronecker_product(aux, ZnT, ZmT);
                matrix_multiply(aux, vecH, mu_rho, 1, 0, CblasNoTrans, CblasNoTrans);
                if (gsl_isnan(a_nm) || a_nm == missing) {
                    gsl_matrix_set(Rho, m, n, gsl_matrix_get(mu_rho, 0, 0) + gsl_ran_gaussian(seed, sRho));
                } else {
                    gsl_matrix_set(Rho, m, n,
                                   (f_w(a_nm, fa, mud, wd) / s2u + gsl_matrix_get(mu_rho, 0, 0) / s2Rho) /
                                   (1 / s2Rho + 1 / s2u) + gsl_ran_gaussian(seed, sqrt(1 / (1 / s2Rho + 1 / s2u))));
                }
                gsl_matrix_set(Rho, n, m,
                               gsl_matrix_get(Rho, m, n)); //extend symmetric matrix Rho to its vector

            }
        }
        gsl_matrix_free(ZmT);
        gsl_matrix_free(ZnT);
        gsl_matrix_free(aux);
        gsl_matrix_free(mu_rho);
    } else if (Ca == 'b') {
        vector<thread*> threads;
        int start = 0;
        int range = 30;
        while(start < N){
            int end = min(start + range, N);
            auto * t = new thread(parallel_sample_binary_rho, N, K, start, end, A, &Z_view.matrix, Rho, vecH, seed, sRho, missing);
            threads.emplace_back(t);
            start = end;
            range++;
        }

        for(auto t : threads){
            t->join();
            delete t;
        }
    }
    gsl_matrix_free(vecH);

}

double sample_alpha(int Kplus, int N, const gsl_rng *seed) {
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

double sample_s2Y(double missing, int N, int d, int K, char Cd, int Rd, double fd, double mud, double wd, double s2u,
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
sample_s2Rho(int N, int K, gsl_matrix *A, gsl_matrix *Z, gsl_matrix *vecRho, gsl_matrix *vecH, const gsl_rng *seed) {
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


    double precision = gsl_ran_gamma(seed, a + N * N / 2., 1 / (b + gsl_matrix_get(D, 0, 0) / 2.));//???????
    gsl_matrix_free(aux);
    gsl_matrix_free(S);
    gsl_matrix_free(D);
    return 1. / precision;
}


double sample_s2H(int K, gsl_matrix *vecH, const gsl_rng *seed) {
    double a = 2;
    double b = 1;
    gsl_matrix *var = gsl_matrix_calloc(1, 1);

    matrix_multiply(vecH, vecH, var, 1, 0, CblasTrans, CblasNoTrans);
    double precision = gsl_ran_gamma(seed, a + K * K / 2., b / (1 + b * gsl_matrix_get(var, 0, 0) / 2));
    gsl_matrix_free(var);
    return 1. / precision;
}


int IBP_sampler_func(double missing,     // how the missing data is defined
                    gsl_matrix *X,      // user-attribute matrix, real observation of the users
                    char *C,            // define the data type of each attribute
                    char *Net,          // the type of network
                    gsl_matrix *Z,      // the IBP latent feature matrix
                    gsl_matrix **B,     // the weighting vectors D * maxK * 1, initially all 0
                    gsl_vector **theta, // ?
                    gsl_matrix *H,      // the homophilic matrix
                    gsl_matrix *A,      // the adjacency matrix
                    int *R,             // the number of categories in each discrete attribute
                    double *f,          // mapping function from the real space R into the observation space
                    double fa,          // ? used in weighted network
                    double *mu,         // mean, mu[d] = mean(X[d]) the mean value of an attribute to all nodes
                    double *w,          // ?
                    int maxR,           // ? always 1
                    int bias,           // the index of actual feature, also you can treat it as the number of random feature
                    int N,              // number of users
                    int D,              // number of attributes
                    int K,              // initial number of features
                    double alpha,       // the concentration parameter
                    double s2B,         // variance of the weighting matrix??
                    double *s2Y,        // noise variance of the pseudo-observation of the attribute matrix
                    double s2Rho,       // noise variance of the pseudo-observation of the adjacency matrix
                    double s2H,         // variance of the affinity matrix
                    double s2u,         // auxiliary variance noise
                    int maxK,           // max number of features
                    int Nsim            // numbers of iterations for training
) {
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

    double coeff = s2Rho / s2H;
    gsl_matrix_view Q_view_init = gsl_matrix_submatrix(Q, 0, 0, Kest * Kest, Kest * Kest);
    compute_inverse_Q_directly(N, Kest, &Z_view.matrix, coeff, &Q_view_init.matrix);

    gsl_matrix *Qnon = gsl_matrix_calloc(maxK * maxK, maxK * maxK);



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
                                       truncnormrnd(0, sqrt(s2Y[d]), GSL_NEGINF, gsl_vector_get(theta[d], xnd - 1),
                                                    seed));
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
                    gsl_matrix_set(Rho, m, n, gsl_ran_gaussian(seed, sqrt(s2Rho)));
                } else {
                    gsl_matrix_set(Rho, m, n, f_w(a_mn, fa, mu_a, wa));
                }
                gsl_matrix_set(Rho, n, m, gsl_matrix_get(Rho, m, n));
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
                    gsl_matrix_set(Rho, m, n, gsl_ran_gaussian(seed, sqrt(s2Rho)));
                } else if (a_mn == 0) {
                    // it just give it a negative number follows normal distribution with mean 0
                    gsl_matrix_set(Rho, m, n, truncnormrnd(0, sqrt(s2Rho), GSL_NEGINF, 0, seed));
                } else if (a_mn == 1) {
                    gsl_matrix_set(Rho, m, n, truncnormrnd(0, sqrt(s2Rho), 0, GSL_POSINF, seed));
                }
                gsl_matrix_set(Rho, n, m, gsl_matrix_get(Rho, m, n));
            }
        }
    }

    // compute full Eta
    gsl_matrix *Eta = gsl_matrix_calloc(maxK * maxK, 1);
    gsl_matrix *Etanon = gsl_matrix_calloc(maxK * maxK, 1);


    LOG(OUTPUT_DEBUG, "Before IT loop...");
    LOG(OUTPUT_DEBUG, "Nsim = %d", Nsim);

    // main loop
    for (int it = 0; it < Nsim; it++) {
        print_iteration_num(it);
        int Kaux = accelerated_gibbs(maxK, bias, N, D, Kest, C, R, alpha, s2B, s2Y, s2H, s2Rho, Y, Rho, Z, nest,
                                    P, Pnon, lambda, lambdanon, Q, Qnon, Eta, Etanon);

        LOG(OUTPUT_NORMAL, "new K = %d", Kaux);


        if (Kaux == 0) { return Kest; } else { Kest = Kaux; }

        P_view = gsl_matrix_submatrix(P, 0, 0, Kest, Kest);
        gsl_matrix *S = gsl_matrix_calloc(Kest, Kest);
        gsl_matrix_memcpy(S, &P_view.matrix);
        inverse(S);
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
            sample_Y(missing, N, d, Kest, C[d], R[d], f[d], mu[d], w[d], s2Y[d], s2u, s2theta, X, Z, Y[d], B[d],
                    theta[d], seed);
            if (C[d] != 'c' && C[d] != 'o') {
                double aux = sample_s2Y(missing, N, d, Kest, C[d], R[d], f[d], mu[d], w[d], s2u, s2theta, X, Z, Y[d],
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

        gsl_matrix_free(MuB);

        // Sample Hs
        gsl_matrix_view H_view = gsl_matrix_submatrix(H, 0, 0, Kest, Kest);
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


        // make H matrix symmetrical
        print_matrix(&H_view.matrix, "H matrix before sym");
        gsl_matrix *H_lower_triangular = gsl_matrix_calloc(Kest, Kest);
        gsl_matrix_transpose_memcpy(H_lower_triangular, &H_view.matrix);
        gsl_matrix_add(&H_view.matrix, H_lower_triangular);
        gsl_matrix_scale(&H_view.matrix, 0.5);
        gsl_matrix_free(H_lower_triangular);


        print_matrix(&Eta_view.matrix, "Eta matrix", Kest);
        print_matrix(&H_view.matrix, "new H matrix");
        print_matrix(MuH, "Mu H matrix");
        print_matrix(&Q_view.matrix, "Q matrix");
        print_matrix((const gsl_matrix **) B, "B matrix", D, Kest);


        // sample_rho
        sample_rho(missing, N, Kest, Net[0], fa, s2Rho, s2u, A, Z, Rho, &H_view.matrix, seed);

        // sample the variance of Rho and H
        gsl_matrix *vecRho = gsl_matrix_calloc(N * N, 1);
        gsl_matrix2vector(vecRho, Rho);
        s2Rho = sample_s2Rho(N, Kest, A, Z, vecRho, vecH, seed);
        gsl_matrix_free(vecRho);

        s2H = sample_s2H(Kest, vecH, seed);

        alpha = sample_alpha(Kest, N, seed);

        LOG(OUTPUT_INFO, "");
        LOG(OUTPUT_INFO, "s2_rho --> %.3f", s2Rho);
        LOG(OUTPUT_INFO, "s2_h   --> %.3f", s2H);
        LOG(OUTPUT_INFO, "alpha  --> %.3f", alpha);
        LOG(OUTPUT_INFO, "\n\n");

        print_matrix(Rho, "Rho matrix");

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
