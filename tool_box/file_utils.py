# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#
"""
tool for file io

Authors: Chen Chao (chenchao23@baidu.com)
"""
import json
import os
import pickle as pkl


def write_bytes_to_disk(bytes_content, file_path):
    """
    bytes数据写进磁盘
    """
    with open(file_path, "wb") as fout:
        fout.write(bytes_content)
    print("file saved in {}".format(file_path))


def read_file_from_disk(file_path):
    """
    从磁盘中读文件，返回bytes
    """
    with open(file_path, "rb") as f:
        bytes_content = f.read()
    return bytes_content


def mkdir(dir_path):
    """
    make directory
    Args:
        dir_path: str. directory path
    Returns:
        None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_pkl(pkl_file: str):
    try:
        data = pkl.load(open(pkl_file, 'rb'), encoding='iso-8859-1')
    except ValueError as e:
        print(e)
        import pickle5
        with open(pkl_file, "rb") as f:
            data = pickle5.load(f)
    return data


def read_json(json_file):
    """
    Args:
        json_file:

    Returns:

    """
    with open(json_file) as f:
        data = json.load(f)
    return data


def write_json(data, json_file):
    """

    Args:
        data:
        json_file:

    Returns:

    """
    if os.path.exists(json_file):
        print("Warning: override {}".format(json_file))
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)
