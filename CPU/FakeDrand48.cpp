//
// Created by su999 on 2022/7/5.
//

#include "FakeDrand48.h"

/**
 *  This file is a simulation of Drand, only use this file when running under windows
 *
 * */

double drand48()
{
    rseed = (_a * rseed + _c) & 0xFFFFFFFFFFFFLL;
    unsigned int x = rseed >> 16;
    return 	((double)x / (double)_m);

}

void srand48(unsigned int i)
{
    rseed  = (((long long int)i) << 16) | rand();
}