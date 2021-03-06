#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : template_med_script.py
# License: GNU v3.0


'''Run a user-defined simulation script with a given set of free parameter
values, then save the `response` value to disk.

MED can take an arbitrary simulation script that defines its set of free
parameters between two `# MED PARAMETERS START / END` directives and
substitutes them with a MED-generated sample. After running the simulation, it
saves the `response` variable to disk.

This simulation setup is achieved via a form of metaprogramming: the user's
code is modified to change the `parameters` to what is sampled at each run,
then code is injected to save the `response` variable. This generated script is
called in a massively parallel environment with two command-line arguments:

    1. The path to this run's `parameters`, as generated by MED.
    2. A path to save the user-defined `error` variable to.

You can find them in the `med_seed<seed>/results` directory.
'''


import os
import sys
import pickle


###############################################################################
# MED INJECT USER CODE START ##################################################
# MED INJECT USER CODE END   ##################################################
###############################################################################


# Save the user-defined `error` and `extra` variables to disk.
with open(sys.argv[2], "wb") as f:
    pickle.dump(response, f)

if "extra" in locals() or "extra" in globals():
    path = os.path.split(sys.argv[2])
    path = os.path.join(path[0], path[1].replace("result", "extra"))
    with open(path, "wb") as f:
        pickle.dump(extra, f)
