//
// Created by su999 on 2023/4/10.
//

#include <thread>
#include "GibbsSampler.cuh"

using namespace std;

void
generate_zn_combination(int count, int startK, unordered_map<string, gsl_matrix *> *records, const string &combination,
                        gsl_matrix *ZnOrigin) {
    if (count == 0) {
        // read combination and create corresponding Zn copy
        gsl_matrix *ZnCopy = gsl_matrix_calloc(ZnOrigin->size1, ZnOrigin->size2);
        gsl_matrix_memcpy(ZnCopy, ZnOrigin);

        int pos = 0;
        for (char c: combination) {
            if (c == '0') {
                gsl_matrix_set(ZnCopy, startK + pos, 0, 0);
            } else {
                gsl_matrix_set(ZnCopy, startK + pos, 0, 1);
            }
            pos++;
        }
        records->insert(pair<string, gsl_matrix *>(combination, ZnCopy));
        return;
    }

    generate_zn_combination(count - 1, startK, records, combination + "0", ZnOrigin);
    generate_zn_combination(count - 1, startK, records, combination + "1", ZnOrigin);
}


int parallel_sample_znk(int N,
                        int n,
                        int K,
                        int startK,
                        int D,
                        int *nest,
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
                        gsl_matrix **lambdanon) {

    unordered_map<string, gsl_matrix *> records;

    for (int i = 0; i < 4 && startK + i < K - 1; i++) {
        generate_zn_combination(i, startK, &records, "", Zn);
    }

    // start all sampling thread
    auto **threads = new thread *[records.size()];
    int pos = 0;
    for (auto &v: records) {
        int currentK = startK + v.first.size();
        auto *temp = new thread(sample_znk, N, n, K, currentK, D, nest[currentK], s2Rho, s2Y, C, R, p, v.second, Qnon, Enon,
                                Snon, Znon, Rho, s2y_p, Y, lambdanon);
        threads[pos] = temp;
        pos++;
    }

    // wait for all threads to finish
    for(int i = 0; i < records.size(); i++){
        threads[i]->join();
        delete threads[i];
    }
    delete[] threads;

    // start to determine real Znk
    string realRes;
    for (int i = 0; i < 4 && startK + i < K - 1; i++) {
        gsl_matrix * ZnRes = records[realRes];
        double res = gsl_matrix_get(ZnRes, startK + i, 0);

        cout << "Zn" << startK + i << " = " << res << endl;
        // update the data to original Zn
        gsl_matrix_set(Zn, startK + i, 0, res);
        res == 0 ? realRes += "0" : realRes += "1";
    }

    // free all the Zn copy
    for (auto &v: records) {
        gsl_matrix_free(v.second);
    }

    // return next unsampled index
    return startK + 4 < K - 1 ? startK + 4 : K - 1;
}


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
                gsl_matrix **lambdanon) {
    if (nCount > 0) {
        gsl_matrix *aux = gsl_matrix_alloc(1, K);
        // z_nk=0
        gsl_matrix_set(Zn, k, 0, 0);
        matrix_multiply(Zn, Snon, aux, 1, 0, CblasTrans, CblasNoTrans);
        double lik0 = compute_likelihood_given_znk(D, K, n, s2Y, C, R, s2y_p, aux, Zn, Y, lambdanon);
        LOG(OUTPUT_DEBUG, "-- lik0=%f\n", lik0);
        //compute the pseudo-likelihood given Znk=0
        log_likelihood_Rho(N, K, n, Znon, Zn, Rho, Qnon, Enon, s2Rho, lik0);

        // z_nk=1
        gsl_matrix_set(Zn, k, 0, 1);
        matrix_multiply(Zn, Snon, aux, 1, 0, CblasTrans, CblasNoTrans);
        double lik1 = compute_likelihood_given_znk(D, K, n, s2Y, C, R, s2y_p, aux, Zn, Y, lambdanon);
        LOG(OUTPUT_DEBUG, "-- lik1=%f\n", lik1);
        //Pseudo-likelihood for H when z_nk=1 (marginalised)
        log_likelihood_Rho(N, K, n, Znon, Zn, Rho, Qnon, Enon, s2Rho, lik1);

        LOG(OUTPUT_DEBUG, "lik0=%f , lik1=%f \n", lik0, lik1);
        double p0 = gsl_sf_log(N - nCount) + lik0;
        double p1 = gsl_sf_log(nCount) + lik1;
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
            LOG(OUTPUT_NORMAL, "nest[%d]=%d \n", k, nCount);
            LOG(OUTPUT_NORMAL, "lik0=%f , lik1=%f \n", lik0, lik1);
            LOG(OUTPUT_NORMAL,
                "EXECUTION STOPPED: numerical error at the sampler.\n Please restart the sampler and if error persists check hyperparameters. \n");
            return;
        }
        //sampling znk
        if (drand48() > p1_n) {
            gsl_matrix_set(Zn, k, 0, 0);
            p[0] = lik0;
        } else {
            p[0] = lik1;
        }
        gsl_matrix_free(aux);
    } else {
        gsl_matrix_set(Zn, k, 0, 0);
    }
}

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
                       double &lik) {
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

    //logdet(X)=log(detX)
    double s2rho_p = lndet_get(invSigma, N - 1, N - 1, 0);
    inverse(invSigma, N - 1);

    //compute the mean
    matrix_multiply(SQnon, Eta, mu, 1, 0, CblasNoTrans, CblasNoTrans);
    //compute the likelihood
    gsl_matrix *aux = gsl_matrix_calloc(1, N - 1);

    //vector rho_{n,-n}
    gsl_matrix *Rho_non = gsl_matrix_calloc(1, N - 1);
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

    lik -= 0.5 * (gsl_matrix_get(Val, 0, 0) + (N - 1) * gsl_sf_log(2 * M_PI) + s2rho_p);

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
                     double *ldet_Q_n) {
    int TK = 2;
    gsl_matrix_view Zn;
    gsl_matrix_view Pnon_view;
    gsl_matrix_view Lnon_view;
    gsl_matrix_view Qnon_view;
    gsl_matrix_view Enon_view;
    gsl_matrix_view Ydn;
    gsl_matrix_view Z_view;
    gsl_matrix *muy;//muy=Zn * muB ===> muB=P^{-1}*lambda
    gsl_matrix *s2y_p = gsl_matrix_alloc(1, 1);
    gsl_matrix *aux;//aux=Zn^T_{1xK}*P^{-1}_{KxK}
    gsl_matrix *Snon;//Snon= P^{-1}
    double beta = s2Rho / s2H;
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
    gsl_matrix_memcpy(Qnon, Q);
    gsl_matrix_memcpy(etanon, eta);
    memcpy(ldet_Q_n, ldet_Q, sizeof(double));

    //sample every user
    for (int n = 0; n < N; n++) {
        auto *p = new double[TK];
        for (int i = 0; i < TK; i++) {
            p[i] = 0.0;
        }

        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        chrono::steady_clock::time_point middle = chrono::steady_clock::now();
        chrono::steady_clock::time_point end;

        //The upper-left element of the submatrix is the element (0,n) of the original matrix. The submatrix has K rows and one column.  Zn_{Kx1}
        Zn = gsl_matrix_submatrix(Z, 0, n, K, 1);
        Pnon_view = gsl_matrix_submatrix(Pnon, 0, 0, K, K);
        // Pnon_view = P - Zn*Zn
        matrix_multiply(&Zn.matrix, &Zn.matrix, &Pnon_view.matrix, -1, 1, CblasNoTrans, CblasTrans);

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

        Qnon_view = gsl_matrix_submatrix(Qnon, 0, 0, K * K, K * K);
        Z_view = gsl_matrix_submatrix(Z, 0, 0, K, N);

        // compute Znon
        Znon = gsl_matrix_calloc(K, N - 1);
        remove_col(K, N, n, Znon, &Z_view.matrix);

        // compute Qnon inverse
        compute_inverse_Q_directly(N - 1, K, Znon, beta, &Qnon_view.matrix); // original


        end = chrono::steady_clock::now();
        LOG(OUTPUT_DEBUG, "Prepare Qnon cost = %lld [ms]",
            chrono::duration_cast<chrono::milliseconds>(end - middle).count());
        middle = end;


        // by this point etanon is full eta
        Enon_view = gsl_matrix_submatrix(etanon, 0, 0, K * K, 1);

        // compute etanon
        normal_update_eta(Znon, Rho, n, &Enon_view.matrix);

        end = chrono::steady_clock::now();
        LOG(OUTPUT_DEBUG, "Prepare Enon and rho cost =  %lld [ms]",
            chrono::duration_cast<chrono::milliseconds>(end - middle).count());
        middle = end;


        // compute nest array without Zn
        for (int k = bias; k < K; k++) {
            if (gsl_matrix_get(&Zn.matrix, k, 0) == 1) {
                nest[k]--;
            }
        }

        // Sampling znk for k=1...K
        int nextUnsampled = bias;
        while (nextUnsampled < K - 1) {
            cout << "batch sample start with ";
            print_Zn(&Zn.matrix, K);
            nextUnsampled = parallel_sample_znk(N, n, K, nextUnsampled, D, nest, s2Rho, s2Y, C, R, p, &Zn.matrix,
                                                &Qnon_view.matrix, &Enon_view.matrix,
                                                Snon, Znon, Rho, s2y_p, Y, lambdanon);
            cout << "batch sample complete\n\n";
        }
        // sample last k
        sample_znk(N, n, K, K - 1, D, nest[K - 1], s2Rho, s2Y, C, R, p, &Zn.matrix, &Qnon_view.matrix,
                   &Enon_view.matrix,
                   Snon, Znon, Rho, s2y_p, Y, lambdanon);

        cout << "final Zn is ";
        print_Zn(&Zn.matrix, K);

        // based on new Zn, update nest
        for (int k = bias; k < K; k++) {
            if (gsl_matrix_get(&Zn.matrix, k, 0) == 1) {
                nest[k]++;
            }
        }

        // todo verify nest count

        end = chrono::steady_clock::now();
        LOG(OUTPUT_INFO, "Sample all K cost = %lld [ms]",
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

            // compute new full P
            gsl_matrix_set_identity(P);
            matrix_multiply(Z, Z, P, 1, 1 / s2B, CblasNoTrans, CblasTrans);

            // compute new Pnon
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

            Z_view = gsl_matrix_submatrix(Z, 0, 0, K, N);

            // compute new full Q
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

            // compute new Qnon
            Znon = gsl_matrix_calloc(K, N - 1);
            remove_col(K, N, n, Znon, &Z_view.matrix);
            compute_inverse_Q_directly(N - 1, K, Znon, beta, &Qnon_view.matrix);
            LOG(OUTPUT_DEBUG, "Removing a feature column ldet_Q=%f, ldet_Qnon = %f\n", ldet_Q[0],
                lndet_get(&Qnon_view.matrix, K * K, K * K, 0));

            // compute new full eta
            matrix_multiply(ZoZ, vecRho, eta, 1, 0, CblasNoTrans, CblasNoTrans);

            // compute new Etanon
            Enon_view = gsl_matrix_submatrix(etanon, 0, 0, K * K, 1);
            normal_update_eta(Znon, Rho, n, &Enon_view.matrix);

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
            // only need p[0] from the last k
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


        //update Qnon by adding Zn
        Qnon_view = gsl_matrix_submatrix(Qnon, 0, 0, K * K, K * K);
        Z_view = gsl_matrix_submatrix(Z, 0, 0, K, N);
        // compute full Q with new Z
        compute_inverse_Q_directly(N, K, &Z_view.matrix, beta, &Qnon_view.matrix);
        gsl_matrix_memcpy(Q, Qnon);

        Enon_view = gsl_matrix_submatrix(etanon, 0, 0, K * K, 1);

        // compute full eta
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

    gsl_matrix_free(s2y_p);
    gsl_matrix_free(ZoZ);

    return K;
}