"""
randpy
======
::

  >>> import randpy_wn

  >>> random_number,  seed = randpy.gasdev(seed)
  >>> random_numbers, seed = randpy.gasdev(seed, n)

  >>> random_number,  seed = randpy.ran1(seed)
  >>> random_numbers, seed = randpy.ran1(seed, n)

  >>> random_number,  seed = randpy.ranb(seed)
  >>> random_numbers, seed = randpy.ranb(seed, n)


.. note:: ``seed`` can be either a negative integer or a dictionary
    returned by the randpy functions
"""
from .randpy import gasdev, ran1, ranb

__all__ = [
    "gasdev",
    "ran1",
    "ranb",
]

# Remove randpy_wn as reference
randpy = None
del randpy
