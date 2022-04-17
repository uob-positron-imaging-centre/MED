#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utilities.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 31.03.2022


import numpy as np


def str_summary(obj):
    sobj = str(obj.__class__.__name__)

    if isinstance(obj, np.ndarray):
        sobj = f"np.{sobj}[{obj.shape}, {obj.dtype}]"

    # TODO: special-case pd.DataFrame
    return sobj
