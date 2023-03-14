//
// Created by su999 on 2022/7/5.
//

#include "ExtendFunction.h"

/**
 *  This file is the C++ version wrapper function
 *  The main.cpp to learn how to invoke this function
 *
 * */


using namespace std;

/**
Function to call inference routine for GLFM model from Python code
        Inputs:
            Xin: observation matrix ( numpy array [D*N] )
            Cin: string array of length D
            Zin: latent feature binary matrix (numpy array [K*N] )
            NETin: string determines the type of edges in the adjacency matrix
            Ain: observation matrix of the adjacency matrix ( numpy array [N*N] )
            Fin: vector of transform functions indicators
            F:transform functions indicator for a network with weighted edges
        *** (the following are optional parameters) ***
            bias: number of columns that should not be sampled in matrix Zin
            s2Y: variance for pseudo-observations Y
            s2u: auxiliary variance noise
            s2B: variance for feature values
            s2H: variance for the homophily matrix elements
            alpha: mass parameter for the IBP
            Nsim: number of iterations
            maxK: máximum number of latent features (for memory allocation)
            missing: value of missings (should be an integer or nan)

        Outputs:
            B_out: feature matrix: np.array of dimensions (D,Kest,maxR) where D is
                the number of dimensions, Kest is the number of inferred latent
                features, and maxR is the maximum number of categories
            Z_out: activation matrix: np.arry of dimensions (Kest,N) where Kest is
                the number of inferred latent features, and N = number of obs.
            theta_out: auxiliary variables for ordinal variables, ndarray of size
                (D,maxR) where D = nr. of dimensions, maxR = max nr. of
                categories
            H_out: homophily matrix
**/

void
infer(double *Xin, char *Cin, double *Zin, char NETin, double *Ain, double *Fin, int N, int D, int K, double F,
      int bias, double s2u, double s2B, double s2H, double alpha, int Nsim, int maxK, double missing) {
    gsl_matrix_view Xview, Zview, Aview;
    gsl_matrix *X;
    gsl_matrix *A;
    gsl_matrix *Zm;

    LOG(OUTPUT_DEBUG, "N=%d, D=%d, K=%d", N, D, K);


    // transpose input matrices in order to be able to call inner C function
    if (strlen(Cin) != D) {
        LOG(OUTPUT_NORMAL, "EXCEPTION! Size of C and X are not consistent!");
        return;
    }

    // gsl_matrix_view_array takes 1-D array
    Zview = gsl_matrix_view_array(Zin, K, N);
    // we need to allocate input matrix Z to [maxK*N] matrix
    Zm = &Zview.matrix;
    gsl_matrix *Z = gsl_matrix_calloc(maxK, N);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            gsl_matrix_set(Z, k, i, gsl_matrix_get(Zm, k, i));
        }
    }

    char *C = new char[D];
    for (int d = 0; d < D; d++) {
        // convert to lower case
        C[d] = (char) tolower(Cin[d]);
    }


    //  Determine the type of Adjacency matrix whether it is a binary matrix or matrix contains positive weights of each edge
    char Net[1];
    Net[0] = (char) tolower(NETin);

    //...............BODY CODE.......................
    Xview = gsl_matrix_view_array(Xin, D, N);
    X = &Xview.matrix;

    // set up the adjacency matrix
    Aview = gsl_matrix_view_array(Ain, N, N);
    A = &Aview.matrix;


    auto **B = (gsl_matrix **) malloc(D * sizeof(gsl_matrix *));
    auto **theta = (gsl_vector **) malloc(D * sizeof(gsl_vector *));

    // initialize the free parameter of truncated Guassian
    gsl_matrix *H = gsl_matrix_calloc(maxK, maxK);
    auto *w = (double *) malloc(D * sizeof(double));
    auto *mu = (double *) malloc(D * sizeof(double));
    auto *s2Y = (double *) malloc(D * sizeof(double));
    auto *R = new int32_t[D];
    LOG(OUTPUT_DEBUG, "In C++: transforming input data...");
    // always return 1
    int maxR = initialize_func(N, D, maxK, missing, X, C, B, theta, R, Fin, mu, w, s2Y);

    LOG(OUTPUT_DEBUG, "done");


    LOG(OUTPUT_DEBUG, "maxR = %d", maxR);



    //...............Inference Function.......................##
    LOG(OUTPUT_DEBUG, "\nEntering C++: Running Inference Routine...\n");
    double s2Rho;
    if (Net[0] == 'w') {
        s2Rho = 2;
    } else {
        s2Rho = 1;
    }


    int Kest = IBPsampler_func(missing, X, C, Net, Z, B, theta,
                               H, A, R, &Fin[0], F, &mu[0], &w[0],
                               maxR, bias, N, D, K, alpha, s2B, &s2Y[0], s2Rho, s2H, s2u, maxK, Nsim);
    LOG(OUTPUT_DEBUG, "\nBack to Python: OK\n");



    //...............Set Output Pointers.......................##
    auto **Z_out = (double **) malloc(Kest * sizeof(double *));
    for (int i = 0; i < Kest; i++) {
        Z_out[i] = (double *) malloc(N * sizeof(double));
    }

    auto **H_out = (double **) malloc(Kest * sizeof(double *));
    for (int i = 0; i < Kest; i++) {
        H_out[i] = (double *) malloc(Kest * sizeof(double));
    }

    auto ***B_out = (double ***) malloc(D * sizeof(double **));
    for (int i = 0; i < D; i++) {
        B_out[i] = (double **) malloc(Kest * sizeof(double *));
        for (int j = 0; j < Kest; j++) {
            B_out[i][j] = (double *) malloc(maxR * sizeof(double));
        }
    }

    auto **theta_out = (double **) malloc(D * sizeof(double *));
    for (int i = 0; i < D; i++) {
        theta_out[i] = (double *) malloc(maxR * sizeof(double));
    }


    LOG(OUTPUT_DEBUG, "Kest=%d, N=%d\n", Kest, N);


    for (int i = 0; i < N; i++) {
        for (int k = 0; k < Kest; k++) {
            Z_out[k][i] = gsl_matrix_get(Z, k, i);
        }
    }


    for (int i = 0; i < Kest; i++) {
        for (int k = 0; k < Kest; k++) {
            H_out[k][i] = gsl_matrix_get(H, k, i);
        }
    }


    LOG(OUTPUT_DEBUG, "Z_out loaded");


    gsl_matrix_view Bd_view;
    gsl_matrix *BT;
    int idx_tmp;
    LOG(OUTPUT_DEBUG, "B_out[D,Kest,maxR] where D=%d, Kest=%d, maxR=%d", D, Kest, maxR);

    for (int d = 0; d < D; d++) {
        if (C[d] == 'o') {
            idx_tmp = 1;
        } else {
            idx_tmp = R[d];
        }
        Bd_view = gsl_matrix_submatrix(B[d], 0, 0, Kest, idx_tmp);
        BT = gsl_matrix_alloc(idx_tmp, Kest);
        gsl_matrix_transpose_memcpy(BT, &Bd_view.matrix);

        for (int k = 0; k < Kest; k++) {
            for (int i = 0; i < idx_tmp; i++) {
                B_out[d][k][i] = gsl_matrix_get(BT, i, k);
            }
        }
        gsl_matrix_free(BT);
    }


    LOG(OUTPUT_DEBUG, "B_out loaded");


    for (int d = 0; d < D; d++) {
        for (int i = 0; i < maxR; i++) {
            if (C[d] == 'o' && i < (R[d] - 1)) {
                theta_out[d][i] = gsl_vector_get(theta[d], i);
            }
        }
    }


    LOG(OUTPUT_DEBUG, "theta_out loaded");


    //..... Free memory.....
    for (int d = 0; d < D; d++) {
        gsl_matrix_free(B[d]);
        if (C[d] == 'o') {
            gsl_vector_free(theta[d]);
        }
    }
    gsl_matrix_free(Z);

    delete[] C;
    delete[] R;
}