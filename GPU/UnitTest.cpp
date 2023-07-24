//
// Created by su999 on 2023/7/22.
//

#include "UnitTest.h"

using namespace std;

void testFullEtaComputation() {
    int K = 2;
    int N = 3;
    gsl_matrix_wrapper Z(K, N, new int[]{0, 1, 0, 1, 0, 1});
    gsl_matrix_wrapper Rho(N, N, 0);
    gsl_matrix_wrapper Eta(K * K, 1);

    compute_full_eta(Z.matrix, Rho.matrix, Eta.matrix);

    showMatrix(Eta.matrix);

    // correct res is 4 8 8 16
}

void testRankOneEtaUpdate() {
    int K = 2;
    int N = 3;
    int n = 1;
    gsl_matrix_wrapper Z(K, N, new int[]{0, 1, 0, 1, 0, 1});
    gsl_matrix_wrapper Rho(N, N, 0);
    gsl_matrix_wrapper Eta(K * K, 1);
    gsl_matrix_wrapper Etanon(K * K, 1);
    gsl_matrix_view zn = gsl_matrix_submatrix(Z.matrix, 0, n, K, 1);


    compute_full_eta(Z.matrix, Rho.matrix, Eta.matrix);

    rank_one_update_eta(K, N, n, Z.matrix, &zn.matrix, Rho.matrix, Eta.matrix, Etanon.matrix);

    showMatrix(Etanon.matrix);

    // correct answer 0 0 0 16
}

void testQComputation(){
    int N = 3;
    int K = 2;
    double beta = 0.5;
    gsl_matrix_wrapper Z(K, N, new int[]{0, 1, 0, 1, 0, 1});
    gsl_matrix_wrapper Q(K * K, K * K);

    compute_inverse_Q_directly(N, K, Z.matrix, beta, Q.matrix);

    showMatrix(Q.matrix);

    // correct res
//    0.666667, 0.000000, 0.000000, 0.000000,
//    0.000000, 0.400000, 0.000000, 0.000000,
//    0.000000, 0.000000, 0.400000, 0.000000,
//    0.000000, 0.000000, 0.000000, 0.222222,

    // if change Z to 0, 1, 0, 1, 1, 1
//    0.96585365853658536581	-0.25365853658536585361	-0.25365853658536585364	0.058536585365853658526
//    -0.25365853658536585364	0.45853658536585365852	0.058536585365853658521	-0.13658536585365853658
//    -0.25365853658536585364	0.058536585365853658523	0.45853658536585365851	-0.13658536585365853657
//    0.058536585365853658528	-0.13658536585365853658	-0.13658536585365853657	0.18536585365853658536
}





void showMatrix(gsl_matrix *m) {
    for (int i = 0; i < m->size1; i++) {
        for (int j = 0; j < m->size2; j++) {
            cout << to_string(gsl_matrix_get(m, i, j)) << ", ";
        }
        cout << endl;
    }
}





gsl_matrix_wrapper::gsl_matrix_wrapper(int size1, int size2, double *input) {
    matrix = gsl_matrix_calloc(size1, size2);
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            gsl_matrix_set(matrix, i, j, input[i * size2 + j]);
        }
    }
    delete[] input;
}

gsl_matrix_wrapper::gsl_matrix_wrapper(int size1, int size2, int *input) {
    matrix = gsl_matrix_calloc(size1, size2);
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            gsl_matrix_set(matrix, i, j, input[i * size2 + j]);
        }
    }
    delete[] input;
}

gsl_matrix_wrapper::gsl_matrix_wrapper(int size1, int size2, int start) {
    matrix = gsl_matrix_calloc(size1, size2);
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            gsl_matrix_set(matrix, i, j, start + i * size2 + j);
        }
    }
}

gsl_matrix_wrapper::gsl_matrix_wrapper(int size1, int size2) {
    matrix = gsl_matrix_calloc(size1, size2);
}

gsl_matrix_wrapper::~gsl_matrix_wrapper() {
    if (matrix != nullptr) {
        gsl_matrix_free(matrix);
    }
}