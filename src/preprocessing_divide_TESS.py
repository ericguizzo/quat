from __future__ import print_function
import configparser
import random
import utility_functions as uf
import preprocessing_utils as pre
import numpy as np
import os, sys

p_path = ""
t_path = ""

p = np.load(p_path, allow_pickle=True).item.()
t = np.load(t_path, allow_pickle=True).item.()

k = list(p.keys())
len_first_actor = len(p[k[0]])
bound = int(len_first_actor / 2)
new_p = {} 
new_t = {} 

new_dict[0] = 