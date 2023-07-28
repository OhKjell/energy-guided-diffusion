#
# To compile:
#   python setup.py build_ext --inplace
#
#
# To use:
#
# import randpy_wn
#
# (random_number, seed)  = randpy_wn.gasdev(seed)
# (random_numbers, seed) = randpy_wn.gasdev(seed, n)
#
# (random_number, seed)  = randpy_wn.ran1(seed)
# (random_numbers, seed) = randpy_wn.ran1(seed, n)
#
# (random_number, seed)  = randpy_wn.ranb(seed)
# (random_numbers, seed) = randpy_wn.ranb(seed, n)
#
#
#
#  Note: seed can be either a negative integer or a dictionary returned by the randpy_wn functions
#
#  For Windows and Python 2.7, you need to install Visual Studio 2008 (or the Visual Studio C++ Tools for Python 2.7)
#
#  (Fernando Rozenblit, 2017)
#
#
#  Modified to add ranb and improve performance.
#  (Sören Zapp, 2018)
#
#  Fixed compile flags for newer setuptools.
#  (Sören Zapp, 2021)


# distutils: language=c++

cdef extern from "rng_gasdev_ran1.h":
    struct Seed:
        long idum
        long iy
        long iv[32]  # const long NTAB = 32
        int iset
        double gset

cdef extern from "rng_gasdev_ran1.cpp":
    list c_ran1_vec "ran1_vec" (Seed& seed, unsigned int num)
    list c_ranb_vec "ranb_vec" (Seed& seed, unsigned int num)
    double c_gasdev "gasdev" (Seed& seed)

cpdef make_seed(seed):
    cdef Seed c_seed

    if isinstance(seed, dict):
        c_seed = seed
    else:
        c_seed.idum = seed

    return c_seed


def ran1(seed, n=1):
    cdef Seed c_seed = make_seed(seed)
    return (c_ran1_vec(c_seed, n), c_seed)


def ranb(seed, n=1):
    cdef Seed c_seed = make_seed(seed)
    return (c_ranb_vec(c_seed, n), c_seed)


def gasdev(seed, n=1):
    cdef Seed c_seed = make_seed(seed)

    if n > 1:
        res = [c_gasdev(c_seed) for x in xrange(n)]
    else:
        res = c_gasdev(c_seed)

    return (res, c_seed)
