#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : usage.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.04.2022


import medeq
import numpy as np


# Create DataFrame of MED free parameters and their bounds
parameters = medeq.create_parameters(
    ["velocity", "viscosity", "radius"],
    minimums = [-9, -9, -9],
    maximums = [10, 10, 10],
)


def instrument(x):
    '''Example unknown "experimental response" - a complex non-convex function.
    '''
    return x[0] * np.sin(0.5 * x[1]) + np.cos(1.1 * x[2])


# Create MED object, keeping track of free parameters, samples and results
med = medeq.MED(parameters)

# Initial parameter sampling
med.sample(24)
med.evaluate(instrument)

# New sampling, targeting most uncertain parameter regions
med.sample(16)
med.evaluate(instrument)

# Add previous / manually-evaluated responses
med.augment([[0, 0, 0]], [1])

# Save all results to disk - you can load them on another machine
med.save("med_results")

# Discover underlying equation; tell MED what operators it may use
med.discover(
    binary_operators = ["+", "-", "*", "/"],
    unary_operators = ["cos"],
)

# Plot interactive 2D slices of responses and uncertainties
med.plot_gp()
