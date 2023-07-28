To compile:
  python setup.py build_ext --inplace

To use:

  import randpy

  rnd_number, seed  = randpy.gasdev(seed)
  rnd_numbers, seed = randpy.gasdev(seed, n)

  rnd_number, seed  = randpy.ran1(seed)
  rnd_numbers, seed = randpy.ran1(seed, n)

Note: seed can be either a negative integer or the seed returned by the gasdev/ran1 functions

For Windows and Python 2.7, you need to install Visual Studio 2008 (or the Visual Studio C++ Tools for Python 2.7)
