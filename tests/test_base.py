#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_base.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 17.04.2022


import medeq
import pytest


def test_parameters():
    # Simple, correct test
    parameters = medeq.create_parameters(
        ["A", "B", "C"],
        [1, 2, 3],
        [2, 3, 4],
    )
    print(parameters)

    # Stress, correct test
    nparam = 100
    parameters = medeq.create_parameters(
        [f"P{i}" for i in range(nparam)],
        [0 for _ in range(nparam)],
        [9 for _ in range(nparam)],
    )
    print(parameters)

    # Invalid tests
    with pytest.raises(ValueError):
        # Number of values too high
        parameters = medeq.create_parameters(
            ["A", "B"],
            [1, 2, 3],
            [2, 3, 4],
        )

    with pytest.raises(ValueError):
        # Number of values too low
        parameters = medeq.create_parameters(
            ["A", "B"],
            [1],
            [2],
        )

    with pytest.raises(ValueError):
        # Minimums larger than maximums
        parameters = medeq.create_parameters(
            ["A", "B"],
            [1, 2],
            [1, 2],
        )
