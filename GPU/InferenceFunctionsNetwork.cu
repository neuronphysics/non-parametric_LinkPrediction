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

double compute_likelihood_given_znk(int D,
                                    int K,
                                    int n,
                                    double *s2Y,
                                    char *C,
                                    int *R,
                                    gsl_matrix *s2y_p,
                                    gsl_matrix *aux,
                                    gsl_matrix *Zn,
                                    gsl_matrix **Y,
                                    gsl_matrix **lambdanon) {
    double likelihood = 0;
    for (int d = 0; d < D; d++) {
        gsl_matrix_set(s2y_p, 0, 0, s2Y[d]);
        matrix_multiply(aux, Zn, s2y_p, 1, 1, CblasNoTrans, CblasNoTrans);
        double s2y_num = gsl_matrix_get(s2y_p, 0, 0);
        gsl_matrix_view Ydn = gsl_matrix_submatrix(Y[d], 0, n, R[d], 1);
        gsl_matrix_view Lnon_view = gsl_matrix_submatrix(lambdanon[d], 0, 0, K, R[d]);
        gsl_matrix *muy = gsl_matrix_alloc(1, R[d]);
        matrix_multiply(aux, &Lnon_view.matrix, muy, 1, 0, CblasNoTrans, CblasNoTrans);
        if (C[d] == 'c') {
            for (int r = 0; r < R[d] - 1; r++) {
                likelihood -= 0.5 / s2y_num *
                              pow((gsl_matrix_get(&Ydn.matrix, r, 0) - gsl_matrix_get(muy, 0, r)), 2) +
                              0.5 * gsl_sf_log(2 * M_PI * s2y_num);
            }
        } else {
            likelihood -= 0.5 / s2y_num *
                          pow((gsl_matrix_get(&Ydn.matrix, 0, 0) - gsl_matrix_get(muy, 0, 0)), 2) +
                          0.5 * gsl_sf_log(2 * M_PI * s2y_num);
        }
        gsl_matrix_free(muy);
    }
    return likelihood;
}


int log_likelihood_Rho(int N,
                       int K,
                       int r,
                       gsl_matrix *Znon,// Z_{-n} N-1 x K matrix
                       gsl_matrix *zn,
                       gsl_matrix *Rho,
                       gsl_matrix *Qnon,
                       gsl_matrix *Eta,// Snon^T vec(Rho -n, -n)
                       double s2Rho,
                       double &lik
) {
    //*******
    gsl_matrix *mu = gsl_matrix_calloc(N - 1, 1);// (Z_{-n} Kronecker_Product Zn ) Qnon . Snon. vec(Rho_n)
    gsl_matrix_view Q_view = gsl_matrix_submatrix(Qnon, 0, 0, K * K, K * K);
    gsl_matrix *SQnon = gsl_matrix_calloc(N - 1, K * K);//(Znon Kron Zn)(Snon^T Snon+beta I)^{-1}
    gsl_matrix *S = gsl_matrix_calloc(K * K, N - 1);//S=(Z_{-n} Kronecker_Product Zn )

    gsl_Kronecker_product(S, zn, Znon);

    // SQnon = s{-n} * Qnon
    matrix_multiply(S, &Q_view.matrix, SQnon, 1, 0, CblasTrans, CblasNoTrans);

    //compute the covariance
    //gsl_matrix *invSigma   = inverse_sigma_rho(Znon, zn, &Q_view.matrix, S, r, K, N, s2Rho);
    gsl_matrix *invSigma = gsl_matrix_calloc(N - 1, N - 1);
    gsl_matrix_set_identity(invSigma);

    // invSigma = s{-n} * Qnon * s{-n} + I (see equation 20)
    matrix_multiply(SQnon, S, invSigma, 1, 1, CblasNoTrans, CblasNoTrans);
    gsl_matrix_scale(invSigma, s2Rho);
    double s2rho_p = lndet_get(invSigma, N - 1, N - 1, 0);//logdet(X)=log(detX)
    inverse(invSigma, N - 1);

    //compute the mean
    matrix_multiply(SQnon, Eta, mu, 1, 0, CblasNoTrans, CblasNoTrans);
    //compute the likelihood
    gsl_matrix *aux = gsl_matrix_calloc(1, N - 1);
    gsl_matrix *Rho_non = gsl_matrix_calloc(1, N - 1);//vector rho_{n,-n}
    int m = 0;
    for (int n = 0; n < N; n++) {
        if (n != r) {
            gsl_matrix_set(Rho_non, 0, m, gsl_matrix_get(Rho, r, n));
            m += 1;
        }
    }

    gsl_matrix *mu_tran = gsl_matrix_calloc(1, N - 1);
    gsl_matrix_transpose_memcpy(mu_tran, mu);
    // res = rho_{n,-n} - h^T
    gsl_matrix_sub(Rho_non, mu_tran);

    // aux = (rho_{n,-n} - h^T) * P
    matrix_multiply(Rho_non, invSigma, aux, 1, 0, CblasNoTrans, CblasNoTrans);
    gsl_matrix *Val = gsl_matrix_calloc(1, 1);
    // Val = (rho_{n,-n} - h^T) * P * (rho_{n,-n} - h^T)
    matrix_multiply(aux, Rho_non, Val, 1, 0, CblasNoTrans, CblasTrans);

    // (N - 1) * gsl_sf_log(2 * M_PI) = 113.46
    lik -= 0.5 * (gsl_matrix_get(Val, 0, 0) + (N - 1) * gsl_sf_log(2 * M_PI) + s2rho_p);

    //free the pointers
    gsl_matrix_free(mu);
    gsl_matrix_free(SQnon);
    gsl_matrix_free(S);
    gsl_matrix_free(aux);
    gsl_matrix_free(Rho_non);
    gsl_matrix_free(mu_tran);
    gsl_matrix_free(invSigma);
    gsl_matrix_free(Val);
    return 0;
}

// Functions section 3.1
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

) {
    int TK = 2;
    gsl_matrix_view Zn;
    gsl_matrix_view Pnon_view;
    gsl_matrix_view Lnon_view;
    gsl_matrix_view Qnon_view;//****
    gsl_matrix_view Enon_view;//***
    gsl_matrix_view Ydn;
    gsl_matrix_view Z_view; //temperory Z
    gsl_matrix *muy;//muy=Zn * muB ===> muB=P^{-1}*lambda
    gsl_matrix *s2y_p = gsl_matrix_alloc(1, 1);
    gsl_matrix *aux;//aux=Zn^T_{1xK}*P^{-1}_{KxK}
    gsl_matrix *Snon;//Snon= P^{-1}
    double beta = s2Rho / s2H; //temperory variable
    double s2y_num;//sigma=sigma_{y}
    gsl_matrix *ZoZ = gsl_matrix_calloc(maxK * maxK, N * N);

    gsl_matrix *Iden;//Identity matrix
    gsl_matrix *Qexp;
    gsl_matrix *Qmatrix;
    gsl_matrix_view Qmatrix_view;
    gsl_matrix_view Qexp_view;
    gsl_matrix_view Q_view;

    gsl_matrix *Znon;
    gsl_matrix_memcpy(Pnon, P);

    for (int d = 0; d < D; d++) {
        gsl_matrix_memcpy(lambdanon[d], lambda[d]);
    }
    //**** Delta_{K^2x1}=Kronecker-product(Z, Z) vecRho=S vec(Rho)
    gsl_matrix_memcpy(Qnon, Q);//****
    gsl_matrix_memcpy(etanon, eta);
    memcpy(ldet_Q_n, ldet_Q, sizeof(double));
    //test parameters


    //sample every user
    for (int n = 0; n < N; n++) {
        auto *p = new double[TK];
        for (int i = 0; i < TK; i++) {
            p[i] = 0.0;
        }

        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        chrono::steady_clock::time_point middle = chrono::steady_clock::now();
        chrono::steady_clock::time_point end;

        Pnon_view = gsl_matrix_submatrix(Pnon, 0, 0, K, K);
        // Pnon_view=P - Zn*Zn
        matrix_multiply(&Zn.matrix, &Zn.matrix, &Pnon_view.matrix, -1, 1, CblasNoTrans, CblasTrans);

        //The upper-left element of the submatrix is the element (0,n) of the original matrix. The submatrix has K rows and one column.  Zn_{Kx1}
        Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
        // Sigma_B   This function allocates memory for a matrix of length n and initializes all the elements of the matrix to zero.
        Snon = gsl_matrix_calloc(K, K);
        gsl_matrix_memcpy(Snon, &Pnon_view.matrix);
        // Snon=Pnon_view^{-1}
        inverse(Snon, K);

        for (int d = 0; d < D; d++) {
            Lnon_view = gsl_matrix_submatrix(lambdanon[d], 0, 0, K, R[d]);
            Ydn = gsl_matrix_submatrix(Y[d], 0, n, R[d], 1);
            //Lnon_view=-1*Zn*Ydn+1*Lnon_view-- Section 3.1 in "General Latent Feature Models for heterogeneous dataset"
            matrix_multiply(&Zn.matrix, &Ydn.matrix, &Lnon_view.matrix, -1, 1, CblasNoTrans, CblasTrans);
        }

        //****************** Update Qnon & etanon ************
        Qnon_view = gsl_matrix_submatrix(Qnon, 0, 0, K * K, K * K);
        //compute both Qnon and log(det(Qnon)) by removing the n-th row
        Z_view = gsl_matrix_submatrix(Z, 0, 0, K, N);

        // compute Znon
        Znon = gsl_matrix_calloc(K, N - 1);
        remove_col(K, N, n, Znon, &Z_view.matrix);

        // compute Qnon inverse
        compute_inverse_Q_directly(N - 1, K, Znon, beta, &Qnon_view.matrix); // original


        end = chrono::steady_clock::now();
        LOG(OUTPUT_DEBUG, "Prepare Qnon cost =  %lld [ms]",
            chrono::duration_cast<chrono::milliseconds>(end - middle).count());
        middle = end;

        //similar component as Lambda for the network part

        // by this point etanon is just a copy of eta
        Enon_view = gsl_matrix_submatrix(etanon, 0, 0, K * K, 1);

        // temp replace eta update
        normal_update_eta(Znon, Rho, n, &Enon_view.matrix);

        end = chrono::steady_clock::now();
        LOG(OUTPUT_DEBUG, "Prepare Enon and rho cost =  %lld [ms]",
            chrono::duration_cast<chrono::milliseconds>(end - middle).count());
        middle = end;


        // Sampling znk for k=1...K
        for (int k = bias; k < K; k++) {

            if (gsl_matrix_get(&Zn.matrix, k, 0) == 1) {
                nest[k]--;
            }
            if (nest[k] > 0) {
                aux = gsl_matrix_alloc(1, K);
                // z_nk=0
                gsl_matrix_set(&Zn.matrix, k, 0, 0);
                matrix_multiply(&Zn.matrix, Snon, aux, 1, 0, CblasTrans, CblasNoTrans);
                double lik0 = compute_likelihood_given_znk(D, K, n, s2Y, C, R, s2y_p, aux, &Zn.matrix, Y, lambdanon);
                LOG(OUTPUT_DEBUG, "-- lik0=%f\n", lik0);
                //compute the pseudo-likelihood given Znk=0
                log_likelihood_Rho(N, K, n, Znon, &Zn.matrix, Rho, &Qnon_view.matrix, &Enon_view.matrix, s2Rho, lik0);

                // z_nk=1
                gsl_matrix_set(&Zn.matrix, k, 0, 1);
                matrix_multiply(&Zn.matrix, Snon, aux, 1, 0, CblasTrans, CblasNoTrans);
                double lik1 = compute_likelihood_given_znk(D, K, n, s2Y, C, R, s2y_p, aux, &Zn.matrix, Y, lambdanon);
                LOG(OUTPUT_DEBUG, "-- lik1=%f\n", lik1);
                //Pseudo-likelihood for H when z_nk=1 (marginalised)
                log_likelihood_Rho(N, K, n, Znon, &Zn.matrix, Rho, &Qnon_view.matrix, &Enon_view.matrix, s2Rho, lik1);

                LOG(OUTPUT_DEBUG, "lik0=%f , lik1=%f \n", lik0, lik1);
                double p0 = gsl_sf_log(N - nest[k]) + lik0;
                double p1 = gsl_sf_log(nest[k]) + lik1;
                double p1_n, p0_n;
                LOG(OUTPUT_DEBUG, "p1=%f, p0=%f \n", p1, p0);

                if (p0 > p1) {
                    p1_n = expFun(p1 - p0);
                    p0_n = 1;
                } else {
                    p0_n = expFun(p0 - p1);
                    p1_n = 1;
                }
                p1_n = p1_n / (p1_n + p0_n);
                if (isinf(p1_n) || isnan(p1_n)) {
                    LOG(OUTPUT_NORMAL, "nest[%d]=%d \n", k, nest[k]);
                    LOG(OUTPUT_NORMAL, "lik0=%f , lik1=%f \n", lik0, lik1);
                    LOG(OUTPUT_NORMAL,
                        "EXECUTION STOPPED: numerical error at the sampler.\n Please restart the sampler and if error persists check hyperparameters. \n");
                    return 0;
                }
                //sampling znk
                if (drand48() > p1_n) {
                    gsl_matrix_set(&Zn.matrix, k, 0, 0);
                    p[0] = lik0;
                } else {
                    nest[k] += 1;
                    p[0] = lik1;
                }
                LOG(OUTPUT_DEBUG, "after sampling : p[0]=%f, nest[%d]=%d\n", p[0], k, nest[k]);
                gsl_matrix_free(aux);
            } else {
                gsl_matrix_set(&Zn.matrix, k, 0, 0);
            }
        }


        end = chrono::steady_clock::now();
        LOG(OUTPUT_INFO, "Sample all K cost =  %lld [ms]",
            chrono::duration_cast<chrono::milliseconds>(end - middle).count());
        middle = end;


        gsl_matrix_free(Snon);
        gsl_matrix_free(Znon);
        // remove empty features
        int flagDel = 0;
        int Kdel = 0;
        for (int k = 0; k < K; k++) {
            if (nest[k] == 0 && K - Kdel > 1) {
                LOG(OUTPUT_INFO, "remove empty features: K= %d .......\n", K);
                Kdel++;
                flagDel = 1;
                for (int kk = k; kk < K - 1; kk++) {
                    gsl_vector_view Zrow = gsl_matrix_row(Z, kk + 1);
                    gsl_matrix_set_row(Z, kk, &Zrow.vector);
                    nest[kk] = nest[kk + 1];
                }
            }
        }
        for (int k = K - Kdel; k < K; k++) {
            gsl_vector_view Zrow = gsl_matrix_row(Z, k);
            gsl_vector_set_zero(&Zrow.vector);
            nest[k] = 0;
        }
        K -= Kdel;


        if (flagDel) {
            LOG(OUTPUT_INFO, "Update P and Q after removing a zero feature column ......\n");
            gsl_matrix_set_identity(P);
            matrix_multiply(Z, Z, P, 1, 1 / s2B, CblasNoTrans, CblasTrans);
            gsl_matrix_memcpy(Pnon, P);
            Pnon_view = gsl_matrix_submatrix(Pnon, 0, 0, K, K);
            Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
            matrix_multiply(&Zn.matrix, &Zn.matrix, &Pnon_view.matrix, -1, 1, CblasNoTrans, CblasTrans);
            for (int d = 0; d < D; d++) {
                matrix_multiply(Z, Y[d], lambda[d], 1, 0, CblasNoTrans, CblasTrans);
                gsl_matrix_memcpy(lambdanon[d], lambda[d]);
                Lnon_view = gsl_matrix_submatrix(lambdanon[d], 0, 0, K, R[d]);
                Ydn = gsl_matrix_submatrix(Y[d], 0, n, R[d], 1);
                matrix_multiply(&Zn.matrix, &Ydn.matrix, &Lnon_view.matrix, -1, 1, CblasNoTrans, CblasTrans);
            }
            //****************** Update Q & Qnon *********************
            //remove the empty column in matrix Z and its corresponding features in Q
            Z_view = gsl_matrix_submatrix(Z, 0, 0, K, N);

            gsl_matrix_set_identity(Q);
            gsl_Kronecker_product(ZoZ, Z, Z);
            matrix_multiply(ZoZ, ZoZ, Q, 1, beta, CblasNoTrans, CblasTrans);
            Q_view = gsl_matrix_submatrix(Q, 0, 0, K * K, K * K);
            inverse(&Q_view.matrix, K * K);

            ldet_Q[0] = lndet_get(&Q_view.matrix, K * K, K * K, 0);

            gsl_matrix_memcpy(Qnon, Q);
            memcpy(ldet_Q_n, ldet_Q, sizeof(double));
            //Update both Qnon, log(det(Qnon)) and etanon
            Qnon_view = gsl_matrix_submatrix(Qnon, 0, 0, K * K, K * K);
            //rank_one_update_Kronecker(&Z_view.matrix, &Zn.matrix, &Qnon_view.matrix, n, K, N, 0); // removing Zn ****????
            //test covariance Q by removing the effect of n-th row
            Znon = gsl_matrix_calloc(K, N - 1);
            remove_col(K, N, n, Znon, &Z_view.matrix);
            compute_inverse_Q_directly(N - 1, K, Znon, beta, &Qnon_view.matrix);
            LOG(OUTPUT_DEBUG, "Removing a feature column ldet_Q=%f, ldet_Qnon = %f\n", ldet_Q[0],
                lndet_get(&Qnon_view.matrix, K * K, K * K, 0));

            matrix_multiply(ZoZ, vecRho, eta, 1, 0, CblasNoTrans, CblasNoTrans);
            gsl_matrix_memcpy(etanon, eta);
            Enon_view = gsl_matrix_submatrix(etanon, 0, 0, K * K, 1);

            // todo, compare the result with normal update
            rank_one_update_eta(&Z_view.matrix, &Zn.matrix, Rho, &Enon_view.matrix, n, K, N, 0);//removing Zn
            gsl_matrix_free(Znon);
            LOG(OUTPUT_DEBUG, "End of updating Eta after removing a feature column.......\n");
        }


        end = chrono::steady_clock::now();
        LOG(OUTPUT_DEBUG, "Remove feature cost = %lld [ms]",
            chrono::duration_cast<chrono::milliseconds>(end - middle).count());
        middle = end;



        // Adding new features
        gsl_matrix_view Znew = gsl_matrix_submatrix(Z, K, 0, TK, N);
        gsl_matrix_set_zero(&Znew.matrix);

        if (K + TK < maxK) {
            double pmax = p[0];
            double kk = 0;
            // since TK is fixed to 2, thus this for loop will only run once
            for (int j = 1; j < TK; j++) {
                gsl_vector_view Pnon_colum = gsl_matrix_column(Pnon, K + j - 1);
                gsl_vector_set_zero(&Pnon_colum.vector);
                Pnon_colum = gsl_matrix_row(Pnon, K + j - 1);
                gsl_vector_set_basis(&Pnon_colum.vector, K + j - 1);
                //because we have P=Z^TZ+1/s2B I (modified this compared to Isabel's code)?
                //gsl_vector_scale(&Pnon_colum.vector, 1/s2B);//**?Is it correct???? It is me adding this not in the Isabel's code
                aux = gsl_matrix_alloc(1, K + j);
                Pnon_view = gsl_matrix_submatrix(Pnon, 0, 0, K + j, K + j);
                Snon = gsl_matrix_calloc(K + j, K + j);
                gsl_matrix_memcpy(Snon, &Pnon_view.matrix);
                inverse(Snon, K + j);
                Zn = gsl_matrix_submatrix(Z, 0, n, K + j, 1);

                gsl_matrix_set(&Zn.matrix, K + j - 1, 0, 1);
                matrix_multiply(&Zn.matrix, Snon, aux, 1, 0, CblasTrans, CblasNoTrans);

                double lik = compute_likelihood_given_znk(D, K + j, n, s2Y, C, R, s2y_p, aux, &Zn.matrix, Y, lambdanon);

                Iden = gsl_matrix_calloc(K, K);
                gsl_matrix_set_identity(Iden);
                gsl_matrix_scale(Iden, 1. / beta);
                //make a copy of old Qnon matrix in Qexp
                Qnon_view = gsl_matrix_submatrix(Qnon, 0, 0, K * K, K * K);
                Qexp = gsl_matrix_calloc((K + j) * (K + j), (K + j) * (K + j));
                Qmatrix = gsl_matrix_calloc(K * K, K * K);
                gsl_matrix_memcpy(Qmatrix, &Qnon_view.matrix);
                //gsl_matrix_memcpy (Qexp, &Qmatrix.matrix);
                //Now set all its elements to zero
                gsl_matrix_set_zero(Qnon);
                for (int row = 0; row < K; ++row) {
                    for (int col = 0; col < K; ++col) {
                        //get the blocks of KxK size of it
                        Qmatrix_view = gsl_matrix_submatrix(Qmatrix, row * K, col * K, K, K);
                        //pick the corresponding blocks in the new expanded Q
                        Qexp_view = gsl_matrix_submatrix(Qexp, row * (K + j), col * (K + j), K, K);
                        if (row == col) {
                            //set the elements on the diagonal equal to 1./beta value
                            gsl_matrix_set(Qexp, row * (K + j) + K, col * (K + j) + K, 1. / beta);
                        }
                        gsl_matrix_memcpy(&Qexp_view.matrix, &Qmatrix_view.matrix);
                    }
                }
                gsl_matrix_set(Qexp, (K + j) * K, (K + j) * K, 1. / beta);

                Qexp_view = gsl_matrix_submatrix(Qexp, (K + j) * K + j, (K + j) * K + j, K, K);
                gsl_matrix_memcpy(&Qexp_view.matrix, Iden);
                Qnon_view = gsl_matrix_submatrix(Qnon, 0, 0, (K + j) * (K + j), (K + j) * (K + j));
                gsl_matrix_memcpy(&Qnon_view.matrix, Qexp);
                gsl_matrix_free(Iden);
                gsl_matrix_free(Qexp);
                gsl_matrix_free(Qmatrix);


                //Setting up the calculation of the log likelihood of adding a new feature

                Z_view = gsl_matrix_submatrix(Z, 0, 0, K + j, N);
                Znon = gsl_matrix_calloc(K + j, N - 1);
                remove_col(K + j, N, n, Znon, &Z_view.matrix);

                Enon_view = gsl_matrix_submatrix(etanon, 0, 0, (K + j) * (K + j), 1);

                log_likelihood_Rho(N, K + j, n, Znon, &Zn.matrix, Rho, &Qnon_view.matrix, &Enon_view.matrix, s2Rho,
                                   lik);

                p[j] = lik + j * gsl_sf_log(alpha / N) - gsl_sf_log(factorial(j));
                gsl_matrix_free(aux);
                gsl_matrix_free(Snon);
                gsl_matrix_free(Znon);
                if (pmax < p[j]) {
                    pmax = p[j];
                }
                kk = j;
            }
            // kk is always 1, p[1] need to be larger so that k will increase
            double den = 0;
            for (int k = 0; k <= kk; k++) {
                p[k] -= pmax;
                p[k] = expFun(p[k]);
                den += p[k];
            }

            for (int k = 0; k <= kk; k++) {
                p[k] = p[k] / den;

            }
            int Knew = mnrnd(p, kk);
            if (Knew > 0) {
                LOG(OUTPUT_INFO, "add new feature ....\n");
                for (int k = K; k < K + Knew; k++) { nest[k] = 1; }
            }
            K += Knew;
        }

        //Adding Zn
        Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
        Pnon_view = gsl_matrix_submatrix(Pnon, 0, 0, K, K);
        matrix_multiply(&Zn.matrix, &Zn.matrix, &Pnon_view.matrix, 1, 1, CblasNoTrans, CblasTrans);

        gsl_matrix_memcpy(P, Pnon);

        for (int d = 0; d < D; d++) {
            Ydn = gsl_matrix_submatrix(Y[d], 0, n, R[d], 1);
            Lnon_view = gsl_matrix_submatrix(lambdanon[d], 0, 0, K, R[d]);
            matrix_multiply(&Zn.matrix, &Ydn.matrix, &Lnon_view.matrix, 1, 1, CblasNoTrans, CblasTrans);
            gsl_matrix_memcpy(lambda[d], lambdanon[d]);
        }


        end = chrono::steady_clock::now();
        LOG(OUTPUT_DEBUG, "Add new feature cost = %lld [ms]",
            chrono::duration_cast<chrono::milliseconds>(end - middle).count());

        middle = end;


        //****************Update Q and log(det(Q))
        //update Qnon by adding Zn
        Qnon_view = gsl_matrix_submatrix(Qnon, 0, 0, K * K, K * K);
        Z_view = gsl_matrix_submatrix(Z, 0, 0, K, N);
        // compute full Q with new Z
        compute_inverse_Q_directly(N, K, &Z_view.matrix, beta, &Qnon_view.matrix);
        gsl_matrix_memcpy(Q, Qnon);

        Enon_view = gsl_matrix_submatrix(etanon, 0, 0, K * K, 1);

        gsl_Kronecker_product(ZoZ, Z, Z);
        matrix_multiply(ZoZ, vecRho, &Enon_view.matrix, 1, 0, CblasNoTrans, CblasNoTrans);

        gsl_matrix_memcpy(eta, etanon);

        delete[] p;


        end = chrono::steady_clock::now();
        LOG(OUTPUT_DEBUG, "Update Z cost = %lld [ms]",
            chrono::duration_cast<chrono::milliseconds>(end - middle).count());
        middle = end;

        LOG(OUTPUT_INFO, "Total cost = %lld [ms]", chrono::duration_cast<chrono::milliseconds>(end - begin).count());
    }

    //end testing
    gsl_matrix_free(s2y_p);
    gsl_matrix_free(ZoZ);

    return K;
}


//Sample Y
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
        case 'g': //real-valued observations Eq. (14)
            muy = gsl_matrix_alloc(1, 1);
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

        case 'p': //positive real-valued observations Eq. (15)
            muy = gsl_matrix_alloc(1, 1);
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

        case 'n': //count observations
            muy = gsl_matrix_alloc(1, 1);
            for (int n = 0; n < N; n++) {
                xnd = gsl_matrix_get(X, d, n);
                Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
                Bd_view = gsl_matrix_submatrix(Bd, 0, 0, K, 1);
                matrix_multiply(&Zn.matrix, &Bd_view.matrix, muy, 1, 0, CblasTrans, CblasNoTrans);
                if (xnd == missing || gsl_isnan(xnd)) {
                    gsl_matrix_set(Yd, 0, n, gsl_matrix_get(muy, 0, 0) + gsl_ran_gaussian(seed, sYd));
                } else {
                    gsl_matrix_set(Yd, 0, n, truncnormrnd(gsl_matrix_get(muy, 0, 0), sYd, f_1(xnd, fd, mud, wd),
                                                          f_1(xnd + 1, fd, mud, wd)));
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

        case 'c': //categorical observations
            muy = gsl_matrix_alloc(1, Rd);
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
                                   truncnormrnd(gsl_matrix_get(muy, 0, xnd - 1), sYd, maxY, GSL_POSINF));//?
                    for (int r = 0; r < Rd; r++) {
                        if (r != xnd - 1) {
                            gsl_matrix_set(Yd, r, n, truncnormrnd(gsl_matrix_get(muy, 0, r), sYd, GSL_NEGINF,
                                                                  gsl_matrix_get(Yd, xnd - 1, n)));//?
                        }
                    }
                }
            }
            gsl_matrix_free(muy);
            break;

        case 'b': //binary observations
            muy = gsl_matrix_alloc(1, 1);
            for (int n = 0; n < N; n++) {
                xnd = (int)gsl_matrix_get(X, d, n);
                Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
                Bd_view = gsl_matrix_submatrix(Bd, 0, 0, K, 1);
                matrix_multiply(&Zn.matrix, &Bd_view.matrix, muy, 1, 0, CblasTrans, CblasNoTrans);
                if (xnd == -1|| gsl_isnan(xnd)) {//missing data
                    gsl_matrix_set(Yd, 0, n, gsl_matrix_get(muy, 0, 0) + gsl_ran_gaussian(seed, sYd));
                } else if (xnd == 0) {
                    gsl_matrix_set(Yd, 0, n, truncnormrnd(gsl_matrix_get(muy, 0, 0), sYd, GSL_NEGINF, 0));
                } else if (xnd == 1) {
                    gsl_matrix_set(Yd, 0, n, truncnormrnd(gsl_matrix_get(muy, 0, 0), sYd, 0, GSL_POSINF));
                } else {
                    printf("Error! xnd for binary is not 0, 1, -1 (for missing data)\n");
                }
            }
            gsl_matrix_free(muy);
            break;

        case 'o': //ordinal observations
            // Sample Y
            gsl_vector *Ymax = gsl_vector_calloc(Rd);
            gsl_vector *Ymin = gsl_vector_alloc(Rd);
            gsl_vector_set_all(Ymin, GSL_POSINF);
            muy = gsl_matrix_alloc(1, 1);
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
                                                          gsl_vector_get(thetad, xnd - 1)));
                    if (gsl_matrix_get(Yd, 0, n) > gsl_vector_get(Ymax, xnd - 1)) {
                        gsl_vector_set(Ymax, xnd - 1, gsl_matrix_get(Yd, 0, n));
                    }
                    if (gsl_matrix_get(Yd, 0, n) < gsl_vector_get(Ymin, xnd - 1)) {
                        gsl_vector_set(Ymin, xnd - 1, gsl_matrix_get(Yd, 0, n));
                    }
                } else {
                    gsl_matrix_set(Yd, 0, n,
                                   truncnormrnd(gsl_matrix_get(muy, 0, 0), sYd, gsl_vector_get(thetad, xnd - 2),
                                                gsl_vector_get(thetad, xnd - 1)));
                    if (gsl_matrix_get(Yd, 0, n) > gsl_vector_get(Ymax, xnd - 1)) {
                        gsl_vector_set(Ymax, xnd - 1, gsl_matrix_get(Yd, 0, n));
                    }
                    if (gsl_matrix_get(Yd, 0, n) < gsl_vector_get(Ymin, xnd - 1)) {
                        gsl_vector_set(Ymin, xnd - 1, gsl_matrix_get(Yd, 0, n));
                    }
                }
            }
            break;
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
                gsl_vector_set(thetad, r, truncnormrnd(0, stheta, xlo,
                                                       xhi));//theta_r^d=Gaussian(theta_r^d|0,sigma_{theta}^2)I(theta_r^d>theta_{r-1}^d)
            }
    }
}

//****Sample Rho : pseudo-observation of the adjacency matrix
void SampleRho(double missing,
               int N,
               int K,
               char Ca,
               double fa,
               double s2Rho,
               double s2u, //??
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
    gsl_matrix *aux = gsl_matrix_alloc(1, K * K);
    gsl_matrix *ZmT = gsl_matrix_calloc(1, K);
    gsl_matrix *ZnT = gsl_matrix_calloc(1, K);
    gsl_matrix_view Zn;
    gsl_matrix_view Zm;
    int a_nm;
    //
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
        // Sample pseudo adjacency matrix
        //binary values
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
                                   truncnormrnd(gsl_matrix_get(mu_rho, 0, 0), sRho, GSL_NEGINF, 0));
                } else if (a_nm == 1) {

                    gsl_matrix_set(vecRho, m * N + n, 0,
                                   truncnormrnd(gsl_matrix_get(mu_rho, 0, 0), sRho, 0, GSL_POSINF));
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
    double alpha = gsl_ran_gamma(seed, 1 + Kplus,
                                 1 / (1 + Harmonic_N));// equation 21 https://arxiv.org/pdf/1011.6293.pdf
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

    muy = gsl_matrix_alloc(1, 1);

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
    double precision = gsl_ran_gamma(seed, a + N / 2, 1 / (b + sumY / 2));//inverse Gamma prior
    return 1. / precision;
}

//**** sample noise variance of the pseudo-observation of the adjacency matrix
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
    double precision = gsl_ran_gamma(seed, a + N * N / 2, 1 / (b + gsl_matrix_get(D, 0, 0) / 2.));//???????
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
    double precision = gsl_ran_gamma(seed, a + K * K / 2, b / (1 + b * gsl_matrix_get(var, 0, 0) / 2));
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
                    double fa,//****?????
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

    //.....INIZIALIZATION........//
    double s2theta = 2;
    // random numbers
    srand48(time(NULL));

    gsl_rng *seed = gsl_rng_alloc(gsl_rng_taus);
    time_t clck = time(NULL);
    gsl_rng_set(seed, clck);

    // auxiliary variables
    int Kest = K;
    gsl_matrix *P = gsl_matrix_alloc(maxK, maxK);
    gsl_matrix_set_identity(P);
    matrix_multiply(Z, Z, P, 1, 1 / s2B, CblasNoTrans, CblasTrans);
    gsl_matrix *Pnon = gsl_matrix_alloc(maxK, maxK);


    ///************ Initialize Q and Qnon *******
    gsl_matrix *Q = gsl_matrix_alloc(maxK * maxK, maxK * maxK);
    double ldet_Q = 0;


    double coeff = s2Rho / s2H;
//    inverse_matrix_Q(coeff, Z, Q, N, Kest, &ldet_Q);
    gsl_matrix_view Q_view_init = gsl_matrix_submatrix(Q, 0, 0, Kest * Kest, Kest * Kest);
    compute_inverse_Q_directly(N, Kest, Z, coeff, &Q_view_init.matrix);

    gsl_matrix *Qnon = gsl_matrix_calloc(maxK * maxK, maxK * maxK);
    double ldet_Q_n = 0;
    ///************


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

    gsl_matrix **Y = (gsl_matrix **) calloc(D, sizeof(gsl_matrix *));
    gsl_matrix **lambda = (gsl_matrix **) calloc(D, sizeof(gsl_matrix *));
    gsl_matrix **lambdanon = (gsl_matrix **) calloc(D, sizeof(gsl_matrix *));
    for (int d = 0; d < D; d++) {
        //Initialize Y !!!!!!
        switch (C[d]) {
            double xnd;
            case 'g':
                Y[d] = gsl_matrix_alloc(1, N);
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
                Y[d] = gsl_matrix_alloc(1, N);
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
                Y[d] = gsl_matrix_alloc(1, N);
                lambda[d] = gsl_matrix_calloc(maxK, 1);
                //lambda[d]=Z*Y[d]
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
                Y[d] = gsl_matrix_alloc(R[d], N);
                for (int n = 0; n < N; n++) {
                    xnd = gsl_matrix_get(X, d, n);
                    if (xnd == missing || gsl_isnan(xnd)) {
                        for (int r = 0; r < R[d]; r++) {
                            gsl_matrix_set(Y[d], r, n, gsl_ran_gaussian(seed, sqrt(s2Y[d])));
                        }
                    } else {
                        gsl_matrix_set(Y[d], xnd - 1, n, truncnormrnd(0, sqrt(s2Y[d]), 0, GSL_POSINF));
                        for (int r = 0; r < R[d]; r++) {
                            if (r != xnd - 1) {
                                gsl_matrix_set(Y[d], r, n, truncnormrnd(0, sqrt(s2Y[d]), GSL_NEGINF,
                                                                        gsl_matrix_get(Y[d], xnd - 1, n)));
                            }
                        }
                    }
                }
                break;

                // todo add binary here
            case 'b':
                Y[d] = gsl_matrix_alloc(1, N);
                for (int n = 0; n < N; n++) {
                    xnd = (int)gsl_matrix_get(X, d, n);
                    if (xnd ==-1|| gsl_isnan(xnd)) {
                        // it is a missing binary value
                        gsl_matrix_set(Y[d], 0, n, gsl_ran_gaussian(seed, sqrt(s2Y[d])));

                    } else if (xnd == 0) {
                        // it just gives it a negative number follows normal distribution with mean 0
                        gsl_matrix_set(Y[d], 0, n, truncnormrnd(0, sqrt(s2Y[d]), GSL_NEGINF, 0));
                    } else if (xnd == 1) {
                        gsl_matrix_set(Y[d], 0, n, truncnormrnd(0, sqrt(s2Y[d]), 0, GSL_POSINF));
                    }
                }
                break;

            case 'o':
                Y[d] = gsl_matrix_alloc(R[d], N);
                //gsl_vector_set (theta[d], 0, -2*stheta);
                gsl_vector_view Xd_view = gsl_matrix_row(X, d);
                double maxX = compute_vector_max(N, missing, &Xd_view.vector);//gsl_vector_max(&Xd_view.vector);
                gsl_vector_set(theta[d], 0, -sqrt(s2Y[d]));
                for (int r = 1; r < R[d] - 1; r++) {
                    //gsl_vector_set (theta[d], r, truncnormrnd(0, stheta, gsl_vector_get (theta[d], r-1), GSL_POSINF));
                    gsl_vector_set(theta[d], r,
                                   gsl_vector_get(theta[d], r - 1) + (4 * sqrt(s2Y[d]) / maxX) * drand48());
                }
                gsl_vector_set(theta[d], R[d] - 1, GSL_POSINF);
                for (int n = 0; n < N; n++) {
                    xnd = gsl_matrix_get(X, d, n);

                    if (xnd == missing || gsl_isnan(xnd)) {
                        gsl_matrix_set(Y[d], 0, n, gsl_ran_gaussian(seed, sqrt(s2Y[d])));
                    } else if (xnd == 1) {
                        gsl_matrix_set(Y[d], 0, n,
                                       truncnormrnd(0, sqrt(s2Y[d]), GSL_NEGINF, gsl_vector_get(theta[d], xnd - 1)));
                    } else {
                        gsl_matrix_set(Y[d], 0, n, truncnormrnd(0, sqrt(s2Y[d]), gsl_vector_get(theta[d], xnd - 2),
                                                                gsl_vector_get(theta[d], xnd - 1)));
                    }
                }
                break;
        }
        // R[d] is always 1
        lambda[d] = gsl_matrix_calloc(maxK, R[d]);
        matrix_multiply(Z, Y[d], lambda[d], 1, 0, CblasNoTrans, CblasTrans);
        lambdanon[d] = gsl_matrix_calloc(maxK, R[d]);
    }



    ///  ********************
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
    } else if (Net[0] == 'b') {//If the adjacency matrix is binary
        int a_mn;
        for (int m = 0; m < N; m++) {
            for (int n = 0; n < m; n++) {
                a_mn = (int) gsl_matrix_get(A, m, n);
                if ((a_mn == missing) || gsl_isnan(a_mn)) {
                    // use 0 as the mean here because H is currently all 0, thus, mean is 0
                    gsl_matrix_set(vecRho, m * N + n, 0, gsl_ran_gaussian(seed, sqrt(s2Rho)));
                } else if (a_mn == 0) {
                    // it just give it a negative number follows normal distribution with mean 0
                    gsl_matrix_set(vecRho, m * N + n, 0, truncnormrnd(0, sqrt(s2Rho), GSL_NEGINF, 0));
                } else if (a_mn == 1) {
                    gsl_matrix_set(vecRho, m * N + n, 0, truncnormrnd(0, sqrt(s2Rho), 0, GSL_POSINF));
                }
                gsl_matrix_set(vecRho, n * N + m, 0, gsl_matrix_get(vecRho, m * N + n, 0));
            }
        }
    }

    //Initialize Eta= (Z kron. Prod. Z)^Tvec(Rho)
    gsl_matrix *Eta = gsl_matrix_calloc(maxK * maxK, 1);
    gsl_matrix *ZoZ = gsl_matrix_calloc(maxK * maxK, N * N);
    gsl_Kronecker_product(ZoZ, Z, Z);
    matrix_multiply(ZoZ, vecRho, Eta, 1, 0, CblasNoTrans, CblasNoTrans);
    gsl_matrix *Etanon = gsl_matrix_calloc(maxK * maxK, 1);
    gsl_matrix_free(ZoZ);
    ///*************End of addition




    LOG(OUTPUT_DEBUG, "Before IT loop...\n");
    LOG(OUTPUT_DEBUG, "Nsim = %d\n", Nsim);

    //....Body functions....//
    for (int it = 0; it < Nsim; it++) {

        /// different from the old version, because additional features
        LOG(OUTPUT_NORMAL, "Start iteration %d", it);
        gsl_vector2matrix(vecRho, Rho);
        int Kaux = AcceleratedGibbs(maxK, bias, N, D, Kest, C, R, alpha, s2B, s2Y, s2H, s2Rho, Y, Rho, vecRho, Z, nest,
                                    P, Pnon, lambda, lambdanon, Q, Qnon, Eta, Etanon, &ldet_Q, &ldet_Q_n);

        LOG(OUTPUT_NORMAL, "iteration %d, K= %d\n", it, Kaux);


        if (Kaux == 0) { return Kest; } else { Kest = Kaux; }

        gsl_matrix_view P_view = gsl_matrix_submatrix(P, 0, 0, Kest, Kest);
        gsl_matrix *S = gsl_matrix_calloc(Kest, Kest);
        gsl_matrix_memcpy(S, &P_view.matrix);
        inverse(S, Kest);
        gsl_matrix *MuB = gsl_matrix_alloc(Kest, 1);

        for (int d = 0; d < D; d++) {
            //Sample Bs
            if (C[d] == 'c') {
                gsl_vector_view Bd_view;
                for (int r = 0; r < R[d] - 1; r++) {
                    gsl_matrix_view L_view = gsl_matrix_submatrix(lambda[d], 0, r, Kest, 1);
                    matrix_multiply(S, &L_view.matrix, MuB, 1, 0, CblasNoTrans, CblasNoTrans);
                    // gsl_vector_view gsl_matrix_subcolumn (gsl_matrix * m,
                    // size_t j, size_t offset, size_t n)
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




        /// *****Sample Hs Troublesome part of the code
        gsl_matrix_view H_view = gsl_matrix_submatrix(H, 0, 0, Kest, Kest);

        LOG(OUTPUT_DEBUG, "old H\n");
        for (int i = 0; i < Kest; i++) {
            for (int j = 0; j < Kest; j++) {
                LOG(OUTPUT_DEBUG, j == Kest - 1 ? "%3.2f\n" : "%3.2f ", gsl_matrix_get(&H_view.matrix, i, j));
            }
        }
        gsl_matrix *MuH = gsl_matrix_calloc(Kest * Kest, 1);

        gsl_matrix *vecH = gsl_matrix_calloc(Kest * Kest, 1);
//        gsl_matrix *SigmaH = gsl_matrix_calloc(Kest * Kest, Kest * Kest);
        gsl_matrix2vector(vecH, &H_view.matrix);
        gsl_vector_view vecH_view = gsl_matrix_subcolumn(vecH, 0, 0, Kest * Kest);
        LOG(OUTPUT_DEBUG, "vecH_view size = %zd\n", (&vecH_view.vector)->size);


        gsl_matrix_view Q_view = gsl_matrix_submatrix(Q, 0, 0, Kest * Kest, Kest * Kest);

        gsl_matrix_view Eta_view = gsl_matrix_submatrix(Eta, 0, 0, Kest * Kest, 1);

        //  MuH sometime is very large causing new H become too large
        //  MuH = Q * S^T * vec(rho) = Q * Eta  (see equation 14)
        matrix_multiply(&Q_view.matrix, &Eta_view.matrix, MuH, 1, 0, CblasNoTrans, CblasNoTrans);
        gsl_vector_view MuH_view = gsl_matrix_column(MuH, 0);

        mvnrnd(&vecH_view.vector, &Q_view.matrix, &MuH_view.vector, Kest * Kest, seed);

        gsl_vector2matrix(vecH, &H_view.matrix);
        LOG(OUTPUT_INFO, "new H\n");

        for (int i = 0; i < Kest; i++) {
            for (int j = 0; j < Kest; j++) {
                if (OUTPUT_LEVEL >= OUTPUT_INFO) {
                    cout << gsl_matrix_get(&H_view.matrix, i, j) << " , ";
                }
            }
            LOG(OUTPUT_INFO, "");
        }

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


        gsl_matrix_free(vecH);//****
        gsl_matrix_free(MuH);
        ///  ********End



        gsl_matrix_free(S);
    }
    LOG(OUTPUT_DEBUG, "After IT loop...\n");

    for (int d = 0; d < D; d++) {
        gsl_matrix_free(Y[d]);
        gsl_matrix_free(lambda[d]);
        gsl_matrix_free(lambdanon[d]);
        //gsl_vector_free(theta[d]);
    }
    gsl_matrix_free(P);
    gsl_matrix_free(Pnon);
    free(lambda);
    free(lambdanon);
    gsl_matrix_free(Q);///****
    gsl_matrix_free(Qnon);///****

    free(Y);
    gsl_matrix_free(Rho);///****
    gsl_matrix_free(vecRho);///****
    gsl_matrix_free(Eta);///**
    gsl_matrix_free(Etanon);///

    delete[] nest;
    return Kest;
}


int initialize_func(int N,
                    int D,
                    int maxK,
                    double missing,
                    gsl_matrix *X,
                    char *C,
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
                // todo add binary type here
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
