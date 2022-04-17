#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : example_hills.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.04.2022


import medeq
import numpy as np


parameters = medeq.create_parameters(
    ["A", "B"],
    minimums = [-2 * np.pi, -2 * np.pi],
    maximums = [2 * np.pi, 2 * np.pi],
)


def instrument(x):
    return np.sin(1.1 * x[0]) + np.cos(0.5 * x[1])


med = medeq.MED(parameters, seed = 123)

# Initial parameter sampling
med.sample(24)
med.evaluate(instrument)

# Expand parameter bounds by 200%
med.parameters.iloc[1, :2] *= 2

# Sample expanded parameter space
med.sample(24)
med.evaluate(instrument)
med.save()

print(med)

# Discover underlying equation
med.discover(
    unary_operators = ["cos"],
    maxsize = 20,
    populations = 32,
    ncyclesperiteration = 64,
    multithreading = True,
)
