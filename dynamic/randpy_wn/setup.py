import os
from setuptools import setup
from Cython.Build import cythonize

# The __init__.py interferes with setuptools, so rename it after
# os.rename('__init__.py', '__init__.py.disabled')

setup(
    name="randpy1",
    ext_modules=cythonize("randpy.pyx", language_level=2),
    zip_safe=False,
)

# Rename it back
# os.rename('__init__.py.disabled', '__init__.py')
