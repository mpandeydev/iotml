# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 20:05:24 2018

@author: satya
"""

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

gpus = get_available_gpus()