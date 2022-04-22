#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_med.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 16.04.2022


import os
import textwrap

import numpy as np

import medeq
import pytest


def test_sampler_decorator():

    @medeq.sampler
    def usersampler(med, n):
        return np.random.random((n, len(med.parameters)))

    parameters = medeq.create_parameters(
        ["A", "B"],
        [0, 0],
        [9, 9],
    )

    med = medeq.MED(parameters, seed = 123)
    samples = usersampler.sample(med, 5)
    assert samples.ndim == 2
    assert samples.shape[0] == 5
    assert samples.shape[1] == 2
    assert np.all(samples >= 0)
    assert np.all(samples < 1)


def printer(msg, obj):
    over = "=" * 80
    line = "-" * 80
    print((
        f"{over}\n"
        f"{msg}\n"
        f"{line}\n"
        f"{obj}\n"
        f"{over}\n"
    ))


def test_local_single():
    '''Test MED running locally with a single response / GP.
    '''

    def instrument(x):
        return np.sin(1.1 * x[0]) + np.cos(0.5 * x[1])

    parameters = medeq.create_parameters(
        ["A", "B"],
        [0, 0],
        [9, 9],
    )

    # Test attributes were created
    med = medeq.MED(parameters, seed = 123)
    printer("med:", med)
    printer("med.queue", med.queue)
    printer("med.evaluated", med.evaluated)
    printer("med.responses", med.responses)
    printer("med.results:", med.results)

    meddir = "med_single_local"
    if not os.path.isdir(meddir):
        os.mkdir(meddir)

    # Save and load empty MED
    med.save(f"{meddir}/med_empty")
    medeq.MED.load(f"{meddir}/med_empty")

    # Save and load MED with queued samples, but no results
    med.sample(24)

    med.save(f"{meddir}/med_queued")
    medeq.MED.load(f"{meddir}/med_queued")

    # Save and load MED with evaluated responses
    med.evaluate(instrument)
    printer("med.results:", med.results)

    med.save(f"{meddir}/med_evaluated")
    medeq.MED.load(f"{meddir}/med_evaluated")

    # Save and load MED with augmented responses
    manual = [[1, 1], [2, 2], [3, 3]]
    med.augment(manual, [instrument(m) for m in manual])

    med.save(f"{meddir}/med_augmented")
    medeq.MED.load(f"{meddir}/med_augmented")

    # Test wrong number of response names
    with pytest.warns(RuntimeWarning):
        med = medeq.MED(parameters, ["resp1", "resp2"], seed = 123)
        med.sample(24)
        med.evaluate(instrument)

    with pytest.warns(RuntimeWarning):
        med = medeq.MED(parameters, [], seed = 123)
        med.sample(24)
        med.evaluate(instrument)


def test_local_multi():
    '''Test MED running locally with multiple responses / GPs.
    '''

    def instrument(x):
        return [np.sin(1.1 * x[0]) + np.cos(0.5 * x[1]), x[0] - x[1]]

    parameters = medeq.create_parameters(
        ["A", "B"],
        [0, 0],
        [9, 9],
    )

    # Test attributes were created
    med = medeq.MED(parameters, seed = 123)
    printer("med:", med)
    printer("med.queue", med.queue)
    printer("med.evaluated", med.evaluated)
    printer("med.responses", med.responses)
    printer("med.results:", med.results)

    meddir = "med_multi_local"
    if not os.path.isdir(meddir):
        os.mkdir(meddir)

    # Save and load empty MED
    med.save(f"{meddir}/med_empty")
    medeq.MED.load(f"{meddir}/med_empty")

    # Save and load MED with queued samples, but no results
    med.sample(24)

    med.save(f"{meddir}/med_queued")
    medeq.MED.load(f"{meddir}/med_queued")

    # Save and load MED with evaluated responses
    med.evaluate(instrument)
    printer("med.results:", med.results)

    med.save(f"{meddir}/med_evaluated")
    medeq.MED.load(f"{meddir}/med_evaluated")

    # Save and load MED with augmented responses
    manual = [[1, 1], [2, 2], [3, 3]]
    med.augment(manual, [instrument(m) for m in manual])

    med.save(f"{meddir}/med_augmented")
    medeq.MED.load(f"{meddir}/med_augmented")

    # Test wrong number of response names
    with pytest.warns(RuntimeWarning):
        med = medeq.MED(parameters, ["resp1"], seed = 123)
        med.sample(24)
        med.evaluate(instrument)

    with pytest.warns(RuntimeWarning):
        med = medeq.MED(parameters, [], seed = 123)
        med.sample(24)
        med.evaluate(instrument)



def test_parallel_single():
    '''Test MED running a user-script in parallel with a single response / GP.
    '''

    # Simple, idiomatic, correct test
    code = textwrap.dedent('''
        # MED PARAMETERS START
        import sys
        import medeq

        parameters = medeq.create_parameters(
            ["A", "B", "C"],
            [-5., -5., -5.],
            [10., 10., 10.],
        )
        # MED PARAMETERS END

        print("Example stdout message.")
        print("Example stderr message.", file=sys.stderr)

        values = parameters["value"]
        response = values["A"]**2 + values["B"]**2
    ''')

    file = "med_single_script1.py"
    with open(file, "w") as f:
        f.write(code)

    med = medeq.MED(file, seed = 123)
    print(med)

    meddir = "med_single_parallel"
    if not os.path.isdir(meddir):
        os.mkdir(meddir)

    # Will save into default directory hierarchy
    med.sample(8)
    med.evaluate()
    print(med)

    # Change saving directory
    med.save(f"{meddir}/med_evaluated")
    med.sample(8)
    med.evaluate()
    print(med)

    med2 = medeq.MED.load(f"{meddir}/med_evaluated")
    print(med2)

    # Simple, idiomatic, correct test with named responses given in dict
    code = textwrap.dedent('''
        # MED PARAMETERS START
        import sys
        import medeq

        parameters = medeq.create_parameters(
            ["A", "B", "C"],
            [-5., -5., -5.],
            [10., 10., 10.],
        )
        # MED PARAMETERS END

        print("Example stdout message.")
        print("Example stderr message.", file=sys.stderr)

        a, b, c = parameters["value"]
        response = dict(
            resp1 = a**2 + b**2,
        )
    ''')

    file = "med_multi_script2.py"
    with open(file, "w") as f:
        f.write(code)

    med3 = medeq.MED(file, seed = 123)
    print(med3)

    med3.save(f"{meddir}/med_named_responses")
    med3.sample(8)
    med3.evaluate()

    med3.save()
    print(med3)

    # Ensure named responses are correctly extracted, saved and loaded
    med3 = medeq.MED.load(f"{meddir}/med_named_responses")
    assert med3.response_names[0] == "resp1"


def test_parallel_multi():
    '''Test MED running a user-script in parallel with multiple responses/GPs.
    '''

    # Simple, idiomatic, correct test
    code = textwrap.dedent('''
        # MED PARAMETERS START
        import sys
        import medeq

        parameters = medeq.create_parameters(
            ["A", "B", "C"],
            [-5., -5., -5.],
            [10., 10., 10.],
        )
        # MED PARAMETERS END

        print("Example stdout message.")
        print("Example stderr message.", file=sys.stderr)

        a, b, c = parameters["value"]
        response = [a**2 + b**2, a + b + c**2]
    ''')

    file = "med_multi_script1.py"
    with open(file, "w") as f:
        f.write(code)

    med = medeq.MED(file, seed = 124)
    print(med)

    meddir = "med_multi_parallel"
    if not os.path.isdir(meddir):
        os.mkdir(meddir)

    # Will save into default directory hierarchy
    med.sample(8)
    med.evaluate()
    print(med)

    # Change saving directory
    med.save(f"{meddir}/med_evaluated")
    med.sample(8)
    med.evaluate()
    print(med)

    med2 = medeq.MED.load(f"{meddir}/med_evaluated")
    print(med2)

    # Simple, idiomatic, correct test with named responses given in dict
    code = textwrap.dedent('''
        # MED PARAMETERS START
        import sys
        import medeq

        parameters = medeq.create_parameters(
            ["A", "B", "C"],
            [-5., -5., -5.],
            [10., 10., 10.],
        )
        # MED PARAMETERS END

        print("Example stdout message.")
        print("Example stderr message.", file=sys.stderr)

        a, b, c = parameters["value"]
        response = dict(
            resp1 = a**2 + b**2,
            resp2 = a + b + c**2,
        )
    ''')

    file = "med_multi_script2.py"
    with open(file, "w") as f:
        f.write(code)

    med3 = medeq.MED(file, seed = 123)
    print(med3)

    med3.save(f"{meddir}/med_named_responses")
    med3.sample(8)
    med3.evaluate()

    med3.save()
    print(med3)

    # Ensure named responses are correctly extracted, saved and loaded
    med3 = medeq.MED.load(f"{meddir}/med_named_responses")
    assert med3.response_names[0] == "resp1"
    assert med3.response_names[1] == "resp2"
