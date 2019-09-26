# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:34:22 2019

@author: PAT
"""

from pathlib import Path

data_dir = Path('C:/Users/PAT/Documents/edwisor/projects/traffic_crash/data/raw')
json_data_path = data_dir  / 'pedestrian-crashes-chapel-hill-region.json'
train_data_path = data_dir  / 'train.csv'
submitted_data_path = data_dir / '../submitted/submission.csv'
processed_data_path = data_dir / '../processed/processed.csv'
