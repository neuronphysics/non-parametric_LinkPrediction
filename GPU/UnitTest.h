//
// Created by su999 on 2023/7/22.
//

#ifndef GLFM_CUDA_ACC_UNITTEST_H
#define GLFM_CUDA_ACC_UNITTEST_H

#include "Utils.h"

void showMatrix(gsl_matrix *m);

void testFullEtaComputation();

void testRankOneEtaUpdate();

void testQComputation();

class gsl_matrix_wrapper {
public:
    gsl_matrix *matrix;

public :
    gsl_matrix_wrapper(int size1, int size2, double * input);
    gsl_matrix_wrapper(int size1, int size2, int * input);
    gsl_matrix_wrapper(int size1, int size2, int start);
    gsl_matrix_wrapper(int size1, int size2);
    ~gsl_matrix_wrapper();
};

#endif //GLFM_CUDA_ACC_UNITTEST_H
