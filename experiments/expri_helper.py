#!/usr/bin/env python
# coding: utf-8

import os
import json


def get_path(data_name, dat_ls_file='expri.json'):
    assert os.path.exists(dat_ls_file)
    with open(dat_ls_file) as f:
        dat_dict = json.load(f)
    return os.path.join(dat_dict['dat_root'], dat_dict[data_name.lower()])
    