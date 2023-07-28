#include <memory.h>
#include <math.h>

/*
 * Copyright (c) 2008, Christian Mendl
 * All rights reserved.
 *
 */
 // Modified by Fernando Rozenblit in 2017 from the ran1 and gasdev mex files by Christian Mendl
 // Extended by Sören Zapp in 2018 to include ranb and to improve performance

#include "rng_gasdev_ran1.h"
#include "Python.h"

// function declarations
double gasdev(Seed& seed);
double ran1(Seed& seed);
unsigned char ranb(Seed& seed);
PyObject* ranb_vec(Seed& seed, unsigned int num);
PyObject* ran1_vec(Seed& seed, unsigned int num);

// need some constants
const long IA = 16807;
const long IM = 2147483647;
const double AM = 1.0/IM;
const long IQ = 127773;
const long IR = 2836;
const long NDIV = 1+(IM-1)/NTAB;
const double EPS = 1.2e-7;
const double RNMX = 1.0-EPS;

double ran1(Seed& seed)
{
    int j;
    long k;
    // static long iy=0;
    // static long iv[NTAB];
    double temp;

    if (seed.idum <= 0 || !seed.iy) {
        if (-seed.idum < 1) seed.idum=1;
        else seed.idum = -seed.idum;
        for (j=NTAB+7;j>=0;j--) {
            k=seed.idum/IQ;
            seed.idum=IA*(seed.idum-k*IQ)-IR*k;
            if (seed.idum < 0) seed.idum += IM;
            if (j < NTAB) seed.iv[j] = seed.idum;
        }
        seed.iy=seed.iv[0];
    }
    k=(seed.idum)/IQ;
    seed.idum=IA*(seed.idum-k*IQ)-IR*k;
    if (seed.idum < 0) seed.idum += IM;
    j=seed.iy/NDIV;
    seed.iy=seed.iv[j];
    seed.iv[j] = seed.idum;
    if ((temp=AM*seed.iy) > RNMX) return RNMX;
    else return temp;
}


/*
 * Usually I'd use modulo, as suggested by the normal RNG algorithms, but this does not conform with how ran1 was used
 * previously.
 */
unsigned char ranb(Seed& seed)
{
    return ran1(seed) > 0.5;
}


double gasdev(Seed& seed)
{
    //static int seed.iset = 0;
    //static double seed.gset = 0;

    double fac,rsq,v1,v2;

    if (seed.idum < 0) seed.iset=0;
    if  (seed.iset == 0) {
        do {
            v1=2.0*ran1(seed)-1.0;
            v2=2.0*ran1(seed)-1.0;
            rsq=v1*v1+v2*v2;
        } while (rsq >= 1.0 || rsq == 0.0);
        fac=sqrt(-2.0*log(rsq)/rsq);
        seed.gset=v1*fac;
        seed.iset=1;
        return v2*fac;
    } else {
        seed.iset=0;
        return seed.gset;
    }
}


/*
 * Re-locating the for-loop into C results in almost a two-fold increase in performance.
 *
 * %timeit -n10 -r10 stim, seed = ranb(-10000, 45000000)
 *   910 ms ± 23.2 ms per loop (mean ± std. dev. of 10 runs, 10 loops each)
 *
 * %timeit -n10 -r10 stim, seed = ranb_vec(-10000, 45000000)
 *   555 ms ± 2.12 ms per loop (mean ± std. dev. of 10 runs, 10 loops each)
 */
PyObject* ranb_vec(Seed& seed, unsigned int num) {
    if (num < 2)
        return PyLong_FromLong(ranb(seed));

    PyObject* list = PyList_New(num);

    for (size_t i = 0; i < num; i++)
        PyList_SetItem(list, i, PyLong_FromLong(ranb(seed)));

    return list;
}


PyObject* ran1_vec(Seed& seed, unsigned int num) {
    if (num < 2)
        return PyFloat_FromDouble(ran1(seed));

    PyObject* list = PyList_New(num);

    for (size_t i = 0; i < num; i++)
        PyList_SetItem(list, i, PyFloat_FromDouble(ran1(seed)));

    return list;
}
