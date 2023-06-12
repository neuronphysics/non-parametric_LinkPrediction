//
// Created by su999 on 2022/7/5.
//

#ifndef GLFM_C_PART_EXTENDFUNCTION_H
#define GLFM_C_PART_EXTENDFUNCTION_H

#include "InferenceFunctionsNetwork.h"

/**
 *  This file is the C++ version wrapper function
 *  The main.cpp to learn how to invoke this function
 *
 * */


void infer(double *Xin, char *Cin, double *Zin, char NETin, double *Ain, double * Fin, int N, int D, int K, double F,
           int bias = 0, double s2u = 1.0, double s2B = 1.0, double s2H = 0.005, double alpha = 1.0, int Nsim = 100,
           int maxK = 50, double missing = -1);

#endif //GLFM_C_PART_EXTENDFUNCTION_H
