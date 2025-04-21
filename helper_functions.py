#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 00:44:13 2025

@author: lenti
"""
import numpy as np

def find_decimal_point(val):
    '''Funktion som finder den betydende cifre i given værdi, og outputter cifrens placering som integer'''
    if val>0:
        return int(-np.floor(np.log10(val)))
    else:
        return 1