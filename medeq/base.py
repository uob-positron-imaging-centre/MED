#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : base.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 29.03.2022


import  shutil
import  textwrap
import  pathlib
import  subprocess

import  numpy           as      np
import  pandas          as      pd

from    .__version__    import  __version__




def install(julia_project = None, verbose = True):
    try:
        import pysr
        pysr.install(julia_project, not(verbose))
    except ImportError:
        print((
            "PySR / Julia not found - please follow installation instructions "
            "from the docs."
        ))
        raise

    # Create separate Julia environment for MED, as using GLMakie through
    # PyJulia segfaults
    julia = get_julia_executable()
    julia_project, is_shared = get_julia_project(julia_project)

    if is_shared:
        io = "stderr" if verbose else "devnull"
        activate_project = (
            "using Pkg\n"
            f'Pkg.activate("{pysr.sr._escape_filename(julia_project)}", '
            f'shared = Bool({int(is_shared)}), io={io})\n'
        )

        subprocess.run([
            julia,
            "-e",
            f'''
            {activate_project}
            Pkg.add("GLMakie", io={io})
            Pkg.add("Colors", io={io})

            Pkg.instantiate(io={io})
            Pkg.precompile(io={io})
            '''
        ])




def get_julia_executable():

    candidates = ["julia", "julia.exe", "julia.cmd"]
    for c in candidates:
        executable = shutil.which(c)
        if executable is not None:
            return executable

    raise FileNotFoundError




def get_julia_project(julia_project):
    if julia_project is None:
        is_shared = True
        julia_project = f"medeq-{__version__}"
    else:
        is_shared = False
        julia_project = pathlib.Path(julia_project)
    return julia_project, is_shared





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
    required by e.g. ``medeq.Sampling``.

    Only the `variables`, `minimums` and `maximums` are necessary. If unset,
    the initial ``values`` are set to halfway between the lower and upper.

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
