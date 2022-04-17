#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : example_hills.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.04.2022


import medeq
import numpy as np
from scipy.stats.qmc import Sobol, discrepancy


# Compare discrepancy of a standard Sobol sequence and a DVASampler
dims = 2
num_samples = 16
seed = 42

parameters = medeq.create_parameters(
    [f"D{i}" for i in range(dims)],
    minimums = [0] * dims,
    maximums = [1] * dims,
)


samples_sobol = Sobol(dims, seed = seed).random(num_samples)

med = medeq.MED(parameters, seed = seed)
samples_dva = med.sample(num_samples)

print(f"Sobol      discrepancy: {discrepancy(samples_sobol)}")
print(f"DVASampler discrepancy: {discrepancy(samples_dva)}")
