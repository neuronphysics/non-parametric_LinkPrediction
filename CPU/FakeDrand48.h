#ifndef DRAND48_H
#define DRAND48_H

#include <cstdlib>

#define _m 0x100000000LL
#define _c 0xB16
#define _a 0x5DEECE66DLL

/**
 *  This file is a simulation of Drand, only use this file when running under windows
 *
 * */
static unsigned long long rseed = 1;

double drand48();

void srand48(unsigned int i);

#endif