"""
light  Copyright (C) 2020  privateos
This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
This is free software, and you are welcome to redistribute it
under certain conditions; type `show c' for details.
"""

import numpy
#import cupy
backend = numpy

def set_cupy_as_backend():
    global backend
    backend = cupy

def set_numpy_as_backend():
    global backend
    backend = numpy

def set_backend(new_backend):
    global backend
    backend = new_backend
