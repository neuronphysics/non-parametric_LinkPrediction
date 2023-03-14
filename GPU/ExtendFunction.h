//
// Created by su999 on 2022/7/5.
//

#ifndef GLFM_C_PART_EXTENDFUNCTION_H
#define GLFM_C_PART_EXTENDFUNCTION_H

#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
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
