import ctypes
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
handle = ctypes.CDLL(dir_path + "/libneural_network.so")

handle.NNTest.argtypes = [ctypes.c_int]

def My_Function(num1, num2):
    return handle.NNTest(num1, num2)

My_Function(150, 120)
#