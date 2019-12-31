# -*- coding: utf-8 -*-
'''
工具调用示例
'''
import kashgari
import os
from data_tools import clean_data

time_ner = kashgari.utils.load_model('models/time_ner.h5')

for root, dirs, files in os.walk('data/data_time'):
    for file in files:
        clean_data('data/data_time/' + file, 'data/data_all/' + file, time_ner) 
        
org_loc_per_ner = kashgari.utils.load_model('models/org_loc_per_ner.h5')

for root, dirs, files in os.walk('data/data_org_loc_per'):
    for file in files:
        clean_data('data/data_org_loc_per/' + file, 'data/data_all/' + file, org_loc_per_ner) 
