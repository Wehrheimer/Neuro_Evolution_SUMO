# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:34:43 2019

@author: peters
"""

import multiprocessing
import sys

def f(x):
    created = multiprocessing.Process()
    current = multiprocessing.current_process()
    print ('running:', current.name, current._identity)
    print ('created:', created.name, created._identity)
    sys.stdout
    return created._identity

if __name__ == "__main__":
    p = multiprocessing.Pool(6)
    b=p.map(f, range(6))
    print(b)