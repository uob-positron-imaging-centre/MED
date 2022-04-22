#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : base.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 29.03.2022


import textwrap

import numpy as np
import pandas as pd




def create_parameters(
    variables = [],
    minimums = [],
    maximums = [],
    values = None,
    **kwargs,
):
    '''Create a ``pandas.DataFrame`` storing ``MED`` free parameters' names,
    bounds.

    This is simply a helper returning a ``pandas.DataFrame`` with the format
    required by e.g. ``pymed.Sampling``.

    Only the `variables`, `minimums` and `maximums` are necessary. If unset,
    the initial ``values`` are set to halfway between the lower and upper
    bounds; the initial standard deviation ``sigma`` is set to 40% of this
    range, so that the entire space is explored.

    Parameters
    ----------
    variables : list[str], default []
        A list of the free parameters' names.

    minimums : list[float], default []
        A list with the same length as ``variables`` storing the lower bound
        for each corresponding variable.

    maximums : list[float], default []
        A list with the same length as ``variables`` storing the lower bound
        for each corresponding variable.

    values : list[float], optional
        The optimisation starting values; not essential as ACCES samples the
        space randomly anyways. If unset, they are set to halfway between
        ``minimums`` and ``maximums``.

    sigma : list[float], optional
        The initial standard deviation in each variable, setting how far away
        from the initial ``values`` the parameters will be sampled; the
        sampling is Gaussian. If unset, they are set to 40% of the data range
        (i.e. ``maximums`` - ``minimums``) so that the entire space is
        initially explored. ACCES will adapt and minimise this uncertainty.

    **kwargs : other keyword arguments
        Other columns to include in the returned parameters DataFrame, given
        as other lists with the same length as ``variables``.

    Returns
    -------
    pandas.DataFrame
        A table storing the intial ``value``, ``min``, ``max`` and ``sigma``
        (columns) for each free parameter (rows).

    Examples
    --------
    Create a DataFrame storing two free parameters, specifying the minimum and
    maximum bounds; notice that the starting guess and uncertainty are set
    automatically.

    >>> import coexist
    >>> parameters = coexist.create_parameters(
    >>>     variables = ["cor", "separation"],
    >>>     minimums = [-3, -7],
    >>>     maximums = [+5, +3],
    >>> )
    >>> parameters
                value  min  max  sigma
    cor           1.0 -3.0  5.0    3.2
    separation   -2.0 -7.0  3.0    4.0
    '''

    if not len(variables) == len(minimums) == len(maximums) or \
            (values is not None and len(values) != len(variables)):
        raise ValueError(textwrap.fill((
            "The input iterables `variables`, `minimums`, `maximums` and "
            "`values` (if defined) must all have the same lengths."
        )))

    minimums = np.array(minimums, dtype = float)
    maximums = np.array(maximums, dtype = float)

    if values is None:
        values = (maximums + minimums) / 2
    else:
        values = np.array(values, dtype = float)

    if not np.all(minimums < maximums):
        raise ValueError(textwrap.fill((
            "The input `minimums` must contain values strictly smaller than "
            "those in `maximums`."
        )))

    parameters = pd.DataFrame(
        data = {
            "value": values,
            "min": minimums,
            "max": maximums,
            **kwargs,
        },
        index = variables,
    )

    return parameters
