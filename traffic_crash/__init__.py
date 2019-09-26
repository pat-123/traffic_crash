# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:38:47 2019

@author: PAT
"""


import os
import sys
from sklearn.datasets import load_iris

import pandas as pd
import seaborn  as sb
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

from pprint import pprint
import numpy as np


import geojson


# add to sys paths
#sys.path.append('C:\\Users\\PAT\\Documents\\edwisor\\projects\\bigmart_sales')
def rt_parent():
    parent_dir = 'TBD---------'
    return parent_dir