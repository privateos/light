"""
    light--a deep learning framework based on numpy/cupy for the purpose of education
    Copyright (C) 2020  privateos

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
