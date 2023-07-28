#ifndef H_RNG_GASDEV_RAN1
#define H_RNG_GASDEV_RAN1

/*
 * Copyright (c) 2008, Christian Mendl
 * All rights reserved.
 *
 */
 // Modified by Fernando Rozenblit in 2017 from the ran1 and gasdev mex files by Christian Mendl

const long NTAB = 32;

// seed also contains 'static' variables used in 'gasdev'
struct Seed
{
	long idum;
	long iy;
	long iv[NTAB];
	int iset;
	double gset;

	Seed()
	: idum(-1), iy(0), iset(0), gset(0) {
	}
};

#endif
