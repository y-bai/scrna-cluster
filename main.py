#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2022/6/21 9:30
@License: (C) Copyright 2013-2017. 
@File: main.py
@Desc:

'''

import argparse
import logging


def main(in_args):
    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='scDualGN model')
    args.add_argument('-a', '--alpha',
                      default=None, type=bool,
                      help='')

    in_args = args.parse_args()
    main(in_args)
