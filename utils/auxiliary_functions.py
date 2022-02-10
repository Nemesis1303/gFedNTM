# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                        AUXILIARY FUNCTIONS                             ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import numpy as np

def get_type_from_string(str_dtype):
    # @ TODO: Make match-case when Pytorch works with Python 3.10
    print(str_dtype)
    if str_dtype == "float32":
        dytpe = np.float32
    elif str_dtype == "float64":
        dytpe = np.float64
    return dytpe
