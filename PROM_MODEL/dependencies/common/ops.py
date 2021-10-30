#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------
# File  : ops.py
# Author: liushichang
# Date  : 2021/3/24
# Desc  : 各类型算子
# Contact : liushichang@meituan.com
# ----------------------------------

def seqOp(x, y):
    """seqFunc"""
    x.append(y)
    return x


def combOp(x, y):
    """combFunc"""
    return x+y
