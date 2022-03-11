# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#
"""
DESCRIPTION.

Authors: Chen Chao (chenchao214@outllook.com)
"""
import os


def get_data_dir():
    """
    Returns:
    """
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")
    return data_dir
