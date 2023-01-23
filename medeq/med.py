#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : med.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 29.03.2022


import  os
import  re
import  sys
import  contextlib
import  pickle
import  inspect
import  textwrap
import  warnings
import  subprocess
import  time
from    datetime            import  datetime

import  numpy               as      np
import  pandas              as      pd

import  cma
from    fvgp.gp             import  GP
from    scipy.stats.qmc     import  discrepancy, LatinHypercube

import  toml
from    tqdm                import  tqdm

from    coexist             import  schedulers
from    coexist.code_trees  import  code_contains_variable
from    coexist.code_trees  import  code_substitute_variable

from    coexist.plots       import  format_fig
from    coexist.utilities   import  autorepr, SignalHandlerKI

import  plotly.express      as      px
import  plotly.graph_objs   as      go
from    plotly.subplots     import  make_subplots

import  medeq
from    .__version__        import  __version__
from    .utilities          import  str_summary


# Optional imports
try:
    from pysr import PySRRegressor
except ImportError:
    class PySRNotInstalled:
        pass
    PySRRegressor = PySRNotInstalled




signal_handler = SignalHandlerKI()




def validate_parameters(parameters):
    '''Validate the free parameters provided or extracted from a user script (a
    ``pandas.DataFrame``).
    '''
    if not isinstance(parameters, pd.DataFrame):
        raise ValueError(textwrap.fill((
            "The `parameters` given is not a `pandas.DataFrame` instance (or "
            f"subclass thereof). Received `{type(parameters)}`."
        )))

    columns_needed = ["min", "max", "value"]
    if not all(c in parameters.columns for c in columns_needed):
        raise ValueError(textwrap.fill((
            "The `parameters` DataFrame provided must have at least three "
            "columns defined: ['min', 'max', 'value']. Found these: "
            f"`{parameters.columns}`. You can use the "
            "`medeq.create_parameters` function as a helper."
        )))

    if not np.all(parameters["min"] < parameters["max"]):
        raise ValueError(textwrap.fill((
            "The `parameters` DataFrame must have all values in the column "
            "'min' strictly smaller than in the column 'max'."
        )))


def upscale(x, lo, hi):
    '''Scale the input values `x` from [0, 1) up to [lo, hi).
    '''
    return lo + (hi - lo) * x


def downscale(x, lo, hi):
    '''Scale the input values `x` from [lo, hi) down to [0, 1).
    '''
    return (x - lo) / (hi - lo)


def sampler(f):
    '''Decorator making a user-defined function a MED sampler.

    A MED sampler is simply an object defining the method ``.sample(n, med)``,
    where ``med`` is a complete MED instance (for e.g. using historical
    sampling information) and ``n`` is the number of samples between [0, 1) to
    return.

    This decorator simply attaches a method ``.sample`` to the given function
    that forwards the function call to the function itself.

    Examples
    --------

    .. code-block:: python

        import medeq
        import numpy as np

        @medeq.sampler
        def user_sampler(n, med = None):
            num_parameters = len(med.parameters)
            return np.random.random((n, num_parameters))
    '''
    f.sample = f.__call__
    return f




class DVASampler:
    '''Parameter sampler that targets the most uncertain regions while
    maximising the area covered.
    '''

    def __init__(self, d, seed = None):
        self.d = int(d)
        self.seed = seed


    def sample(self, n, med):
        # Save the MED instance as an attribute to be accessed in `.cost`
        if med is None:
            class MEDNotGiven:
                gp = None
            med = MEDNotGiven()
        elif not isinstance(med, MED):
            raise TypeError(textwrap.fill((
                "The input `med` argument may be either `None` or an instance "
                f"of `medeq.MED`. Received `{type(med)}`."
            )))

        # Draw starting guesses from well-spread out Latin Hypercube
        sampler0 = LatinHypercube(self.d, seed = self.seed)
        x0 = sampler0.random(n).flatten()
        bounds = [np.zeros(n * self.d), np.ones(n * self.d)]

        es = cma.CMAEvolutionStrategy(x0, 0.4, cma.CMAOptions(
            bounds = bounds,
            seed = self.seed,
            # maxfevals = 100_000,
            verbose = -9,
        ))
        # es.logger = cma.CMADataLogger(os.path.join(folder, "cache", ""))

        try:
            with tqdm() as pbar:
                iterations = 0
                nit = 100
                while not es.stop():
                    es.optimize(self.cost, iterations = nit, args = (med,))
                    iterations += nit

                    pbar.update(nit)
                    pbar.set_description((
                        "DVASampler | "
                        f"Fitness: {es.result.fbest:4.4e} | "
                        f"Convergence: {es.sigma:4.4e} : "
                    ))

        except KeyboardInterrupt:
            line = "*" * 80 + "\n"
            print((
                f"{line}Caught Ctrl-C; stopping optimisation and collecting "
                f"current best results.\n{line}"
            ), file = sys.stderr)

        # es.optimize(self.cost)
        return es.result.xbest.reshape(-1, self.d)


    def cost(self, x, med):
        x = x.reshape(-1, self.d)

        if med.gp is None:
            uncertainty = [1]
        else:
            # Evaluate mean uncertainty for each GP
            uncertainty = np.ones((len(med.gp)))
            for i in range(len(med.gp)):
                uncertainty[i] = med.gp[i].posterior_covariance(
                    x, variance_only = True
                )["v(x)"].mean()

            # Take previous samples into consideration for discrepancy
            prev = downscale(
                med.evaluated,
                med.parameters["min"].to_numpy(),
                med.parameters["max"].to_numpy(),
            )
            x = np.vstack((prev, x))

            # Remove samples outside given parameters range
            x = x[(x < 1).all(axis = 1) & (x >= 0).all(axis = 1)]

        return discrepancy(x) / np.prod(uncertainty)


    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(d={self.d}, seed={self.seed})"




class RandomSampler:

    def __init__(self, d, seed = None):
        self.d = int(d)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)


    def sample(self, n, med = None):
        return self.rng.random((n, self.d))


    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(d={self.d}, seed={self.seed})"




class LatticeSampler:

    def __init__(self, d, seed = None):
        self.d = int(d)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)


    def sample(self, n, med = None):
        # Generate regular lattice with the closest number of points to `n`
        nd = int(np.ceil(n ** (1 / self.d)))
        lattice = np.meshgrid(*([np.linspace(0, 0.999999, nd)] * self.d))
        lattice = np.vstack([lat.flatten() for lat in lattice]).T

        # Remove some random points to get down to exactly `n` samples
        remove = self.rng.integers(0, len(lattice), size = len(lattice) - n)
        mask = np.full(len(lattice), True)
        mask[remove] = False

        return lattice[mask]


    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(d={self.d}, seed={self.seed})"




@autorepr
class MEDPaths:
    '''Structure handling IO and storing all paths relevant for MED.

    Attributes
    ----------
    directory : str
        Path to the MED directory, e.g. ``med_seed123``.
    '''

    def __init__(self, directory):
        # Root directory / prefix
        self.directory = directory

        # All samples requested and responses found / given, then queued ones
        self.results = os.path.join(self.directory, "results.csv")
        self.queue = os.path.join(self.directory, "queue.csv")
        self.setup = os.path.join(self.directory, "setup.toml")
        self.sampler = os.path.join(self.directory, "sampler.pickle")

        self.script = os.path.join(self.directory, "med_script.py")
        self.results_dir = os.path.join(self.directory, "results")
        self.outputs = os.path.join(self.directory, "outputs")


    def update_paths(self, prefix):
        '''Translate all paths saved in this class relative to a new `prefix`
        (which will replace the `directory` attribute).

        Please ensure that the `prefix` directory contains MED-created files.
        '''

        self.directory = prefix
        for attr in ["results", "queue", "setup", "sampler", "script",
                     "results_dir", "outputs"]:
            prev = getattr(self, attr)

            if prev is not None:
                new_path = os.path.join(prefix, os.path.split(prev)[1])
                setattr(self, attr, new_path)




class MED:
    '''Autonomously explore system responses and discover underlying physical
    laws / correlations.

    Exploring systems responses can be done in one of two ways:

    1. Locally / manually: running experiments / simulations, then feeding
       results back to MED.
    2. Massively parallel: for complex simulations that can be launched in
       Python, MED can automatically change simulation parameters and run them
       in parallel on OS processes (locally) or SLURM jobs (distributed
       clusters).

    A typical local workflow is:

    1. Define free parameters to explore as a ``pd.DataFrame`` - you can use
       the ``medeq.create_parameters`` function for this.

    >>> import medeq
    >>> parameters = medeq.create_parameters(
    >>>     ["A", "B"],
    >>>     minimums = [-5., -5.],
    >>>     maximums = [10., 10.],
    >>> )
    >>> print(parameters)
       value  min   max
    A    2.5 -5.0  10.0
    B    2.5 -5.0  10.0

    2. Create a ``medeq.MED`` object and generate samples (i.e. parameter
       combinations) to evaluate - the default sampler covers the parameter
       space as efficiently as possible, taking previous results into account;
       use the ``MED.sample(n)`` method to get ``n`` samples to try.

    >>> med = medeq.MED(parameters)
    >>> print(med)
    MED(seed=42)
    ---------------------------------------
    parameters =
         value  min   max
      A    2.5 -5.0  10.0
      B    2.5 -5.0  10.0
    response_names =
      None
    ---------------------------------------
    sampler =   DVASampler(d=2, seed=42)
    samples =   np.ndarray[(0, 2), float64]
    responses = NoneType
    epochs =    list[0, tuple[int, int]]

    >>> med.sample(5)
    array([[-3.33602115, -0.45639296],
           [ 5.55496225,  5.554965  ],
           [ 2.72771903, -3.48852585],
           [-0.45639308,  8.33602069],
           [ 8.48852568,  2.27228172]])

    3. For a local / offline workflow, these samples can be evaluated in one of
       two ways:

       - Evaluate samples manually, offline - i.e. run experiments,
         simulations, etc. and feed them back to MED.
       - Let MED evaluate a simple Python function / model.

    >>> # Evaluate samples manually - run experiments, simulations, etc.
    >>> to_evaluate = med.queue
    >>> responses = [1, 2, 3, 4, 5]
    >>> med.evaluate(responses)
    >>>
    >>> # Or evaluate simple Python function / model
    >>> def instrument(sample):
    >>>     return sample[0] + sample[1]
    >>>
    >>> med.evaluate(instrument)
    >>> med.results
              A         B  variance   response
    0 -3.336021 -0.456393  0.037924  -3.792414
    1  5.554962  5.554965  0.111099  11.109927
    2  2.727719 -3.488526  0.007608  -0.760807
    3 -0.456393  8.336021  0.078796   7.879628
    4  8.488526  2.272282  0.107608  10.760807

    For a massively parallel workflow, e.g. using a complex simulation, all
    you need is a standalone Python script that:

    - Defines its free parameters between two "# MED PARAMETERS START / END"
      directives.
    - Runs the simulation in _any_ way - define simulation inline, launch it
      on a supercomputer and collect results, etc.
    - Defines a variable "response" for the simulated output of interest -
      either as a single number or a list of numbers (multi-response).

    Here is a simple example of a MED script:

    ::

        # In file `simulation_script.py`

        # MED PARAMETERS START
        import medeq

        parameters = medeq.create_parameters(
            ["A", "B", "C"],
            [-5., -5., -5.],
            [10., 10., 10.],
        )
        # MED PARAMETERS END

        # Run simulation in any way, locally, on a supercomputer and collect
        # results - then define the variable `response` (float or list[float])
        values = parameters["value"]
        response = values["A"]**2 + values["B"]**2

    If you have previous, separate experimental data, you can ``MED.augment``
    the dataset of responses:

    >>> # Augment dataset of responses with historical data
    >>> samples = [
    >>>     [1, 1],
    >>>     [2, 2],
    >>>     [1, 2],
    >>> ]
    >>>
    >>> responses = [1, 2, 3]
    >>> med.augment(samples, responses)

    To discover the underlying equation, you need to install Julia (a
    beautiful, high-performance programming language) on your system and the
    PySR library:

    1. Install Julia manually (see `https://julialang.org/downloads/`_).
    2. ``pip install pysr``
    3. ``python -c 'import pysr; pysr.install()'``

    And now discover underlying equations!

    >>> med.discover(binary_operators = ["+", "*"])
    Hall of Fame:
    -----------------------------------------
    Complexity  Loss       Score     Equation
    1           2.412e+01  5.296e-01  B
    3           0.000e+00  1.151e+01  (A + B)

    Attributes
    ----------
    parameters : pd.DataFrame
        A

    response_names : list[str] or None
        A

    sampler : object
        Any Python object defining a method ``.sample(n, med)`` returning ``n``
        samples to evaluate.

    scheduler : coexist.schedulers.Scheduler subclass or None
        An object implementing the ``coexist.schedulers.Scheduler`` interface,
        defining a method for scheduling function evaluations in a massively
        parallel context. Only relevant if ``parameters`` is given as a user
        script.

    samples : (M, P) np.ndarray
        A

    responses : (N, K) np.ndarray or None
        A

    response_names : list[str] or None
        A

    epochs : list[tuple[int, int]]
        A

    seed : int
        A

    verbose : int
        A

    queue : np.ndarray
        [Generated]

    evaluated : (N, P) np.ndarray
        [Generated]

    results : pd.DataFrame
        [Generated]

    variances : (N, K) np.ndarray
        [Generated]

    gp : list[fvgp.gp.GP] or None
        [Internal]

    sr : pysr.PySRRegressor or None
        [Internal]

    paths : medeq.med.MEDPaths or None
        [Internal]
    '''

    def __init__(
        self,
        parameters,
        response_names = None,
        sampler = DVASampler,
        scheduler = schedulers.LocalScheduler(),
        seed = None,
        verbose = 3,
    ):
        # Type-checking
        if isinstance(parameters, pd.DataFrame):
            # DataFrame given directly; experiments will be run offline
            validate_parameters(parameters)
            self.parameters = parameters
            self.scheduler = None
            self.script = None
        else:
            # User-script was given; extract parameters and set scheduler
            self._generate_script(parameters, scheduler)

        if response_names is not None:
            if isinstance(response_names, str):
                response_names = [response_names]
            else:
                response_names = [str(resp) for resp in response_names]

        self.response_names = response_names

        if seed is None:
            # Generate random 3-digit seed
            self.seed = np.random.randint(100, 1000)
        else:
            self.seed = int(seed)

        if inspect.isclass(sampler):
            # Construct given sampler class with the number of dimensions
            self.sampler = sampler(len(self.parameters), seed = self.seed)
        else:
            # The given sampler is already constructed
            self.sampler = sampler

        if not hasattr(sampler, "sample"):
            raise TypeError(textwrap.fill((
                "The input `sampler` must define a method `.sample(n)` to "
                "draw `n` samples. If you use a custom object or function "
                "(and not a class) you can use the `@sampler` decorator to "
                "automatically add it; check the `medeq` docs for more."
            )))

        self.verbose = int(verbose)

        # Setting inner attributes
        self.samples = np.empty((0, len(self.parameters)))
        self.responses = None
        self.epochs = []

        self.gp = None
        self.sr = None

        self.paths = None


    @property
    def queue(self):
        # Samples that were not evaluated yet
        ran = 0 if self.responses is None else len(self.responses)
        return self.samples[ran:]


    @property
    def evaluated(self):
        # Samples that already have computed responses
        ran = 0 if self.responses is None else len(self.responses)
        return self.samples[:ran]


    @property
    def variances(self):
        if self.responses is None:
            # If no responses were evaluated and no response names were given,
            # assume a single response
            if self.response_names is None:
                nresp = 1
            else:
                nresp = len(self.response_names)
        else:
            nresp = self.responses.shape[1]

        if self.gp is None:
            return np.empty((0, nresp))

        return np.vstack([gp.variances for gp in self.gp]).T


    @property
    def results(self):
        # Extract current responses if any were found
        if self.responses is None:
            # If no responses were evaluated and no response names were given,
            # assume a single response
            if self.response_names is None:
                nresp = 1
            else:
                nresp = len(self.response_names)

            responses = np.empty((0, nresp))
        else:
            responses = self.responses
            nresp = responses.shape[1]

        # Extract variances
        variances = self.variances

        # Extract column names and ensure correct number thereof
        if nresp == 1:
            variance_names = ["variance"]
        else:
            variance_names = [f"variance{i}" for i in range(nresp)]

        if self.response_names is None:
            if nresp == 1:
                response_names = ["response"]
            else:
                response_names = [f"response{i}" for i in range(nresp)]
        else:
            response_names = self.response_names

        # Results DataFrame column names
        columns = self.parameters.index.to_list()
        columns += variance_names + response_names

        # Results DataFrame data
        data = np.c_[self.evaluated, variances, responses]

        return pd.DataFrame(data, columns = columns)


    def save(self, directory = None):
        # Instantiate MEDPaths object handling directory hierarchy
        if directory is None:
            # Include the random seed used in the `med_seed<seed>` dirpath
            directory = f"med_seed{self.seed}"
        else:
            directory = str(directory)

        self.paths = MEDPaths(directory)

        # Create directories
        if not os.path.isdir(self.paths.directory):
            os.mkdir(self.paths.directory)

        # Save MED setup
        if self.gp is None:
            hyperparameters = None
        else:
            hyperparameters = [g.hyperparameters.tolist() for g in self.gp]

        setup = {
            "seed": self.seed,
            "verbose": self.verbose,
            "epochs": self.epochs,
            "parameter_names": self.parameters.index.to_list(),
            "response_names": self.response_names,
            "hyperparameters": hyperparameters,
            "parameters": self.parameters.to_dict(),
        }

        now = datetime.now().strftime("%H:%M:%S on %d/%m/%Y")
        with open(self.paths.setup, "w") as f:
            f.write(f"# Generated by MED-{__version__} at {now}\n")
            toml.dump(setup, f)

        # Save sampler as a binary pickle file to completely reconstruct the
        # arbitrary user-provided object
        with open(self.paths.sampler, "wb") as f:
            pickle.dump(self.sampler, f)

        # Results and outputs directory are only relevant if we have a MED
        # script rather than simple parameters and "offline" evaluation
        if self.script is not None:
            with open(self.paths.script, "w") as f:
                f.writelines(self.script)

            if not os.path.isdir(self.paths.results_dir):
                os.mkdir(self.paths.results_dir)

            if not os.path.isdir(self.paths.outputs):
                os.mkdir(self.paths.outputs)

        # Save information about the run
        readmefile = os.path.join(self.paths.directory, "readme.rst")
        with open(readmefile, "w", encoding = "utf-8") as f:
            f.write(textwrap.dedent(f'''
                MED System Response Exploration Directory
                -----------------------------------------

                This directory was generated by MED-{__version__} at {now}.
                MED setup:

                ::

            '''))

            f.write(textwrap.indent(str(self), '    '))

        # Save evaluated & queued samples and responses found
        self.results.to_csv(self.paths.results, index = None)
        pd.DataFrame(self.queue, columns = self.parameters.index).to_csv(
            self.paths.queue,
            index = None,
        )


    @staticmethod
    def load(dirpath):
        '''Load MED instance from a directory (eg "med_seed123").
        '''
        med = MED.__new__(MED)
        paths = MEDPaths(dirpath)

        with open(paths.setup) as f:
            setup = toml.load(f)

        with open(paths.sampler, "rb") as f:
            med.sampler = pickle.load(f)

        med.seed = setup["seed"]
        med.verbose = setup["verbose"]
        med.parameters = pd.DataFrame.from_dict(setup["parameters"])
        med.response_names = setup.get("response_names", None)
        med.epochs = [tuple(e) for e in setup["epochs"]]

        # Schedulers are platform-specific; when loading MED instances in
        # platform-agnostic manners, schedulers must be left out
        med.scheduler = None

        # Set derived attributes
        med.samples = np.empty((0, len(med.parameters)))
        med.responses = None

        med.gp = None
        med.sr = None

        # Load evaluated and queued samples and responses
        if os.path.isfile(paths.results):
            data = pd.read_csv(paths.results).to_numpy(dtype = float)
            samples = data[:, :len(med.parameters)]

            med.samples = samples
            if len(med.samples):
                nresp = (data.shape[1] - len(med.parameters)) // 2
                med.responses = data[:, -1] if nresp == 1 else data[:, -nresp:]
                med._train()

        if os.path.isfile(paths.queue):
            queue = pd.read_csv(paths.queue).to_numpy()
            if len(queue):
                med.samples = np.vstack((med.samples, queue))

        med.paths = paths
        return med


    def sample(self, n):
        new = self.sampler.sample(n, self)
        new = np.asarray(new, dtype = float)

        # If it's a simple (row) vector, transform it into a 2D column
        if new.ndim == 1:
            new = new[:, np.newaxis]

        # Ensure correct shape
        if new.ndim != 2 or new.shape[0] != n or \
                new.shape[1] != len(self.parameters):

            if hasattr(self.sampler, "__qualname__"):
                sampler_name = self.sampler.__qualname__
            else:
                sampler_name = self.sampler.__class__.__name__

            raise ValueError(textwrap.fill((
                f"The samples returned by the `{sampler_name}` sampler must "
                "be a 2D matrix with shape (N, D), where N is the number of "
                "samples requested and D is the number of dimensions. "
                f"Received an array with shape `{new.shape}`."
            )))

        # Ensure correct values
        if np.any(new < 0) or np.any(new > 1):
            if hasattr(self.sampler, "__qualname__"):
                sampler_name = self.sampler.__qualname__
            else:
                sampler_name = self.sampler.__class__.__name__

            num_outside = np.sum(new < 0) + np.sum(new > 1)

            raise ValueError(textwrap.fill((
                f"The samples returned by the `{sampler_name}` sampler must "
                f"have values between [0, 1). Received {num_outside} values "
                f"outside the range.\n< 0:{new[new < 0]}\n> 1:{new[new > 1]}."
            )))

        # Scale samples from [0, 1) up to real parameter bounds
        new = upscale(
            new,
            self.parameters["min"].to_numpy(),
            self.parameters["max"].to_numpy(),
        )

        self.samples = np.vstack((self.samples, new))
        return new


    def subset(self, select):
        '''Select a subset of the current samples, returning a new ``MED``
        object that can e.g. discover equations for only the selected subset.
        '''

        newmed = MED(
            self.parameters,
            response_names = self.response_names,
            sampler = self.sampler,
            seed = self.seed,
            verbose = self.verbose,
        )

        # Keep track of generated paths, e.g. MED script
        newmed.paths = self.paths

        newsamples = self.samples[select]
        newresponses = self.responses[select]
        newmed.augment(newsamples, newresponses)

        return newmed


    def evaluate(self, f = None):
        '''Evaluate the current samples online or offline.

        There are 3 possible workflows:

        1. The user evaluated the `MED.queue` values separately (e.g. ran
           experiments) - then simply supply a NumPy array of responses.

        >>> med.evaluate([1, 2, 3])

        2. A simple Python function is supplied that will be evaluated for each
           sample; the function must accept a single NumPy vector.

        ::

            def instrument(params):
                # `params` is a NumPy array of parameter combinations to try
                return params[0] + params[1]

            med.evaluate(instrument)

        3. If a separate Python script was provided when the class was created,
           nothing else is needed; this function will launch jobs and collect
           responses from the user script.
        '''

        # Check samples have been generated
        if len(self.queue) == 0:
            raise ValueError(textwrap.fill((
                "No samples were generated for evaluation. Run the "
                "`MED.sample(n)` method to generate samples first."
            )))

        if f is None:
            # If evaluating a user script, generate directory hierarchy
            if self.paths is None or not os.path.isdir(self.paths.directory):
                self.save()
            responses = self._evaluate_script()
        elif callable(f):
            responses = self._evaluate_function(f)
        else:
            responses = np.asarray(f, dtype = float)

        # Ensure 2D shape
        if responses.ndim == 1:
            responses = responses[:, np.newaxis]

        # Type-check responses found
        if len(responses) != len(self.queue):
            raise ValueError(textwrap.fill((
                "Incorrect number of responses given; expected "
                f"{len(self.queue)} responses for each sample generated. "
                f"Received {len(responses)} responses."
            )))

        # Set / extend the inner responses attribute
        if self.responses is None:
            start = 0
            self.responses = responses
        else:
            start = len(self.responses)
            self.responses = np.concatenate((self.responses, responses))

        # Keep track of sampling epochs
        end = len(self.responses)
        self.epochs.append((start, end))

        self._check_response_names()
        self._train()

        # If a user script was evaluated, immediately save results, as they are
        # assumed to be long simulations running offline / asynchronously
        if f is None:
            self.save(self.paths.directory)


    def _evaluate_script(self):
        # Evaluate the current samples in `queue` using the user-script.

        # Aliases
        param_names = self.parameters.index
        start_index = 0 if self.responses is None else len(self.responses)
        queue = self.queue

        # For every sample to try, start a separate OS process that runs the
        # `med_seed<seed>/med_script.py` script, which computes and saves
        # responses
        processes = []

        # Schedule processes using given scheduler
        if not hasattr(self, "scheduler_cmd"):
            self.scheduler_cmd = self.scheduler.generate()

        # These are this epoch's paths to save the simulation outputs to; they
        # will be given to `self.paths.script` as command-line arguments
        parameters_paths = [
            os.path.join(
                self.paths.results_dir,
                f"parameters.{start_index + i}.pickle",
            ) for i in range(len(queue))
        ]

        response_paths = [
            os.path.join(
                self.paths.results_dir,
                f"response.{start_index + i}.pickle",
            ) for i in range(len(queue))
        ]

        # Catch the KeyboardInterrupt (Ctrl-C) signal to shut down the spawned
        # processes before aborting.
        try:
            signal_handler.set()

            # Spawn a separate process for every sample to try / sim to run
            for i, sample in enumerate(queue):
                # Create new set of parameters and save them to disk
                parameters = self.parameters.copy()

                for j, sval in enumerate(sample):
                    parameters.at[param_names[j], "value"] = sval

                with open(parameters_paths[i], "wb") as f:
                    pickle.dump(parameters, f)

                processes.append(
                    subprocess.Popen(
                        self.scheduler_cmd + [
                            self.paths.script,
                            parameters_paths[i],
                            response_paths[i],
                        ],
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,
                        universal_newlines = True,      # outputs in text mode
                    )
                )

            # Gather results
            responses, stdout_rec, stderr_rec, crashed = \
                self._gather_results(processes, response_paths)

        except KeyboardInterrupt:
            for proc in processes:
                proc.kill()

            raise

        finally:
            signal_handler.unset()

        self._print_status_eval(stdout_rec, stderr_rec, crashed)

        # Find number of error values returned for one successful simulation
        nresp = 1
        for i, resp in enumerate(responses):
            if resp is not None:
                if nresp == 1:
                    nresp = len(resp)
                    continue

                if len(resp) != nresp:
                    raise ValueError(textwrap.fill((
                        f"The simulation at index {start_index + i} returned "
                        f"{len(resp)} responses, while previous simulations "
                        f"had {nresp} responses."
                    )))

        # Substitute results that are None (i.e. crashed) with rows of NaNs
        for i in range(len(responses)):
            if responses[i] is None:
                responses[i] = np.full(nresp, np.nan)

        return np.array(responses)


    def _evaluate_function(self, f):
        samples = self.queue
        responses = [f(s) for s in samples]
        return np.array(responses, dtype = float)


    def augment(self, samples, responses):
        '''Augment set of samples with manually-evaluated responses.
        '''

        samples = np.asarray(samples, dtype = float)
        responses = np.asarray(responses, dtype = float)

        # If samples is a simple (row) vector, transform it into a 2D column
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]

        if responses.ndim == 1:
            responses = responses[:, np.newaxis]

        # Type-check samples and responses shapes
        if samples.ndim != 2 or samples.shape[1] != len(self.parameters):
            raise ValueError(textwrap.fill((
                "The input `samples` must have the same number of columns "
                "as the number of dimensions, i.e. number of parameters (= "
                f"{len(self.parameters)}). Received array with shape "
                f"`{samples.shape}`."
            )))

        if len(responses) != len(samples):
            raise ValueError(textwrap.fill((
                "The input `responses` must be a vector / matrix with the "
                "same number of values / rows as `samples`. Received "
                f"{len(responses)} responses for {len(samples)} samples."
            )))

        # Append new samples and responses to current ones. Also move the
        # previously unevaluated samples to the end
        nev = len(self.responses) if self.responses is not None else 0

        self.samples = np.vstack((
            self.samples[:nev],
            samples,
            self.samples[nev:]
        ))

        if self.responses is None:
            self.responses = responses
        else:
            self.responses = np.concatenate((self.responses, responses))

        self._check_response_names()
        self._train()


    def discover(
        self,
        binary_operators = ["+", "-", "*", "/"],
        unary_operators = [],

        maxsize = 50,
        maxdepth = None,

        niterations = 100,
        populations = 32,

        parsimony = 0.0032,
        constraints = None,
        nested_constraints = None,

        denoise = False,
        select_k_features = None,

        turbo = True,
        equation_file = "equations.csv",

        progress = False,
        **kwargs,
    ):
        # Check we have data to discover equations for
        if not len(self.responses):
            raise ValueError(textwrap.fill((
                "No responses were saved in MED. Before discovering equations "
                "please sample the parameter space and collect some values, "
                "e.g. with `MED.sample` and `MED.evaluate`."
            )))

        # Flush stdout, as the Julia backend will overwrite it
        print(flush = True)

        # TODO: use equation_file to save equations found
        self.sr = PySRRegressor(
            binary_operators = binary_operators,
            unary_operators = unary_operators,

            maxsize = maxsize,
            maxdepth = maxdepth,

            niterations = niterations,
            populations = populations,

            parsimony = parsimony,
            constraints = constraints,
            nested_constraints = nested_constraints,

            denoise = denoise,
            select_k_features = select_k_features,

            equation_file = equation_file,
            temp_equation_file = True,

            progress = progress,
            **kwargs,
        )

        names = [c.replace(" ", "_") for c in self.parameters.index]
        self.sr.fit(self.evaluated, self.responses, variable_names = names)
        return self.sr


    def _generate_script(self, script_path, scheduler):
        # Given a path to a user-defined simulation script, extract the free
        # parameters and generate the MED script.

        # Type-check and generate scheduler commands
        if not isinstance(scheduler, schedulers.Scheduler):
            raise TypeError(textwrap.fill((
                "The input `scheduler` must be a subclass of `coexist."
                f"schedulers.Scheduler`. Received {type(scheduler)}."
            )))

        self.scheduler = scheduler

        # Extract parameters and generate ACCES script
        with open(script_path, "r") as f:
            user_code = f.readlines()

        # Find the two parameter definition directives
        params_start_line = None
        params_end_line = None

        regex_prefix = r"#+\s*MED\s+PARAMETERS"
        params_start_finder = re.compile(regex_prefix + r"\s+START")
        params_end_finder = re.compile(regex_prefix + r"\s+END")

        for i, line in enumerate(user_code):
            if params_start_finder.match(line):
                params_start_line = i

            if params_end_finder.match(line):
                params_end_line = i

        if params_start_line is None or params_end_line is None:
            raise NameError(textwrap.fill((
                f"The user script found in file `{script_path}` did not "
                "contain the blocks `# MED PARAMETERS START` and "
                "`# MED PARAMETERS END`. Please define your simulation "
                "free parameters between these two comments / directives."
            )))

        # Execute the code between the two directives to get the initial
        # `parameters`. `exec` saves all the code's variables in the
        # `parameters_exec` dictionary
        user_params_code = "".join(
            user_code[params_start_line:params_end_line]
        )
        user_params_exec = dict()
        exec(user_params_code, user_params_exec)

        if "parameters" not in user_params_exec:
            raise NameError(textwrap.fill((
                "The code between the user script's directives "
                "`# MED PARAMETERS START` and "
                "`# MED PARAMETERS END` does not define a variable "
                "named exactly `parameters`."
            )))

        validate_parameters(user_params_exec["parameters"])
        self.parameters = user_params_exec["parameters"]

        if not code_contains_variable(user_code, "response"):
            raise NameError(textwrap.fill((
                f"The user script found in file `{script_path}` does not "
                "define the required variable `response`."
            )))

        # Substitute the `parameters` creation in the user code with loading
        # them from a MED-defined location
        parameters_code = [
            "\n# Unpickle `parameters` from this script's first " +
            "command-line argument and set\n",
            '# `med_id` to a unique simulation ID\n',
            code_substitute_variable(
                user_code[params_start_line:params_end_line],
                "parameters",
                ('with open(sys.argv[1], "rb") as f:\n'
                 '    parameters = pickle.load(f)\n')
            )
        ]

        # Also define a unique ACCESS ID for each simulation
        parameters_code += (
            'med_id = int(sys.argv[1].split(".")[-2])\n'
        )

        # Read in the `async_access_template.py` code template and find the
        # code injection directives
        template_code_path = os.path.join(
            os.path.split(medeq.__file__)[0],
            "template_med_script.py"
        )

        with open(template_code_path, "r") as f:
            template_code = f.readlines()

        for i, line in enumerate(template_code):
            if line.startswith("# MED INJECT USER CODE START"):
                inject_start_line = i

            if line.startswith("# MED INJECT USER CODE END"):
                inject_end_line = i

        generated_code = "".join((
            template_code[:inject_start_line + 1] +
            user_code[:params_start_line + 1] +
            parameters_code +
            user_code[params_end_line:] +
            template_code[inject_end_line:]
        ))

        self.script = generated_code


    def _gather_results(self, processes, response_paths):
        # Load parallel-executed results from user script
        if not hasattr(self, "_stdout"):
            self._stdout = None

        if not hasattr(self, "_stderr"):
            self._stderr = None

        responses = []
        stdout_rec = []
        stderr_rec = []
        crashed = []

        # Occasionally check if jobs finished
        wait = 0.1          # Time between checking results
        waited = 0.         # Total time waited
        logged = 0          # Number of times logged remaining simulations
        tlog = 30 * 60      # Time until logging remaining simulations again

        while wait != 0:
            done = sum((p.poll() is not None for p in processes))

            if done == len(processes):
                wait = 0
                for i, proc in enumerate(processes):
                    proc_index = int(proc.args[-1].split(".")[-2])
                    stdout, stderr = proc.communicate()

                    # If a *new* output message was recorded in stdout, log it
                    if len(stdout) and stdout != self._stdout:
                        stdout_rec.append(proc_index)
                        self._stdout = stdout

                        stdout_log = os.path.join(
                            self.paths.outputs,
                            f"stdout.{proc_index}.log",
                        )
                        with open(stdout_log, "w") as f:
                            f.write(self._stdout)

                    # If a *new* error message was recorded in stderr, log it
                    if len(stderr) and stderr != self._stderr:
                        stderr_rec.append(proc_index)
                        self._stderr = stderr

                        stderr_log = os.path.join(
                            self.paths.outputs,
                            f"stderr.{proc_index}.log",
                        )
                        with open(stderr_log, "w") as f:
                            f.write(self._stderr)

                    # Load result if the file exists, otherwise set it to NaN
                    if os.path.isfile(response_paths[i]):
                        with open(response_paths[i], "rb") as f:
                            response = pickle.load(f)

                            # If it's a dictionary[resp_name -> resp_val]
                            if hasattr(response, "items") and \
                                    callable(response.items):

                                resp = []
                                resp_names = []
                                for k, v in response.items():
                                    resp_names.append(k)
                                    resp.append(v)

                                resp = np.array(resp, dtype = float)
                                self.response_names = resp_names

                            # If it's a list-like of responses
                            elif hasattr(response, "__iter__"):
                                resp = np.array(response, dtype = float)

                            # If it's a single response
                            else:
                                resp = np.array([response], dtype = float)

                            responses.append(resp)
                    else:
                        responses.append(None)
                        crashed.append(proc_index)

            # Every `remaining` seconds print remaining jobs
            if (
                self.verbose >= 4 and
                wait != 0 and
                waited > (logged + 1) * tlog
            ):
                logged += 1
                tlog *= 1.5

                remaining = " ".join([
                    p.args[-1].split(".")[-2]
                    for p in processes if p.poll() is None
                ])

                minutes = int(waited / 60)
                if minutes > 60:
                    timer = f"{minutes // 60} h {minutes % 60} min"
                else:
                    timer = f"{minutes} min"

                print((
                    f"  * Remaining jobs after {timer}:\n" +
                    textwrap.indent(textwrap.fill(remaining), "  * ")
                ), flush = True)

            # Wait for increasing numbers of seconds until checking for results
            # again - at most 1 minute
            time.sleep(wait)
            waited += wait
            wait = min(wait * 1.5, 60)

        return responses, stdout_rec, stderr_rec, crashed


    def _print_status_eval(self, stdout_rec, stderr_rec, crashed):
        # Print logged stdout and stderr messages and crashed simulations
        # after evaluating an epoch.
        line = "-" * 80
        if len(stderr_rec):
            stderr_rec_str = textwrap.fill(" ".join(
                str(s) for s in stderr_rec
            ))
            print(
                line + "\n" +
                "New stderr messages were recorded while running jobs:\n" +
                textwrap.indent(stderr_rec_str, "  ") + "\n" +
                f"The error messages were logged in:\n  {self.paths.outputs}",
                flush = True,
            )

        if len(stdout_rec):
            stdout_rec_str = textwrap.fill(" ".join(
                str(s) for s in stdout_rec
            ))
            print(
                line + "\n" +
                "New stdout messages were recorded while running jobs:\n" +
                textwrap.indent(stdout_rec_str, "  ") + "\n" +
                f"The output messages were logged in:\n  {self.paths.outputs}",
                flush = True,
            )

        if len(crashed):
            crashed_str = textwrap.fill(" ".join(
                str(c) for c in crashed
            ))

            print(
                line + "\n" +
                "No results were found for these jobs:\n" +
                textwrap.indent(crashed_str, "  ") + "\n" +
                "They crashed or terminated early; for details, check the "
                f"output logs in:\n  {self.paths.outputs}\n"
                "The error values for these simulations were set to NaN.",
                flush = True,
            )

        if len(stderr_rec) or len(stdout_rec) or len(crashed):
            print(line + "\n")


    def _train(self):
        # Train fvgp.GP Gaussian Processes surrogates for each response
        if self.responses.ndim == 1:
            nresp = 1
            responses = self.responses[:, np.newaxis]
        else:
            nresp = self.responses.shape[1]
            responses = self.responses

        # First time training, create new GP (single response) / GPs (multi)
        if self.gp is None:
            self.gp = [None] * nresp

            for i in range(nresp):
                self.gp[i] = GP(
                    len(self.parameters),
                    points = self.evaluated,
                    values = responses[:, i],
                    init_hyperparameters = np.ones(1 + len(self.parameters)),
                    variances = np.abs(0.01 * responses[:, i]),
                    use_inv = True,
                )

        # New training data, update GPs
        else:
            for i in range(len(self.gp)):
                self.gp[i].update_gp_data(
                    self.evaluated,
                    responses[:, i],
                    np.abs(0.01 * responses[:, i]),
                )

        # Train GPs' hyperparameters using our internal CMA-ES method
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            for gp in self.gp:
                gp.train(None, method = self._train_gp_method)


    def _train_gp_method(self, gp):
        nparams = 1 + len(self.parameters)
        x0 = np.ones(nparams)
        bounds = [[0.] * nparams, [None] * nparams]

        es = cma.CMAEvolutionStrategy(x0, 1, cma.CMAOptions(
            bounds = bounds,
            seed = self.seed,
            verbose = -9,
        ))
        # es.logger = cma.CMADataLogger(os.path.join(folder, "cache", ""))

        es.optimize(gp.log_likelihood)
        return es.result.xbest


    def _check_response_names(self):
        nresp = 1 if self.responses.ndim == 1 else self.responses.shape[1]

        if self.response_names is None:
            # Generate response names for each response found
            if nresp == 1:
                self.response_names = ["response"]
            else:
                self.response_names = [f"response{i}" for i in range(nresp)]

        else:
            # Check the correct number of response names is saved and append /
            # remove the necessary number of responses
            nnames = len(self.response_names)
            if nresp > nnames:
                msg = "\n" + textwrap.fill((
                    f"The number of responses found (={nresp}) is larger "
                    f"than the number of response names given (={nnames}). "
                    f"Appending {nresp - nnames} additional response names."
                ))
                warnings.warn(msg, RuntimeWarning, stacklevel = 3)

                self.response_names += [
                    f"response{i}"
                    for i in range(nnames, nresp)
                ]
            elif nresp < nnames:
                msg = "\n" + textwrap.fill((
                    f"The number of responses found (={nresp}) is smaller "
                    f"than the number of response names given (={nnames}). "
                    f"Removing last {nnames - nresp} response names."
                ))
                warnings.warn(msg, RuntimeWarning, stacklevel = 3)

                del self.response_names[nresp:]


    def plot_gp(
        self,
        response = 0,
        resolution = (32, 32),
        verbose = True,
    ):
        '''Plot interactive 2D slices of the `response` and uncertainty.
        '''
        # Compute index of response to plot
        try:
            rid = int(response)
        except ValueError:
            rid = self.response_names.index(response)

        # Type-check responses
        resolution = tuple([int(r) for r in resolution])
        if len(resolution) != 2:
            raise ValueError(textwrap.fill((
                "The input `resolution` must be a tuple with two integers, "
                "e.g. `resolution = (32, 32)`."
            )))

        # Save results
        if self.paths is None:
            self.save()
        else:
            self.save(self.paths.directory)

        medpath = self.paths.directory

        # Path to Julia plotting code
        script = os.path.join(
            os.path.split(medeq.__file__)[0],
            "plot_gp.jl",
        )

        # Run subprocess OS command using the Julia executable
        julia = medeq.base.get_julia_executable()

        subprocess.run([
            julia,
            script,
            medpath,
            str(rid + 1),
            str(resolution[0]),
            str(resolution[1]),
        ])


    def plot_response(
        self,
        f = None,
        colors = px.colors.qualitative.Set1[1:],
    ):
        # Plot samples and uncertainty (columns) for each response (rows)
        nrows = len(self.gp)
        ncols = 2
        subplot_titles = ["MED Sampling", "GP Uncertainty"]

        if f is not None:
            ncols = 3
            subplot_titles += ["Real Response"]

        nx = 100
        ny = 100

        mins = self.parameters["min"]
        maxs = self.parameters["max"]

        x = np.linspace(mins[0], maxs[0], nx)
        y = np.linspace(mins[1], maxs[1], ny)

        xx, yy = np.meshgrid(x, y)

        # Create Figure
        fig = make_subplots(
            rows = nrows,
            cols = ncols,
            subplot_titles = subplot_titles,
        )

        for igp, gp in enumerate(self.gp):
            prediction = gp.posterior_mean(
                np.c_[xx.flatten(), yy.flatten()]
            )["f(x)"].reshape(nx, ny)

            uncertainty = gp.posterior_covariance(
                np.c_[xx.flatten(), yy.flatten()],
                variance_only = True,
            )["v(x)"].reshape(nx, ny)

            if f is not None:
                real = f(
                    np.vstack((xx.flatten(), yy.flatten()))
                ).reshape(nx, ny)

            fig.add_trace(
                go.Heatmap(
                    x = x,
                    y = y,
                    z = prediction,
                    colorscale = "magma",
                    showscale = False,
                ),
                row = 1 + igp,
                col = 1,
            )

            fig.add_trace(
                go.Heatmap(
                    x = x,
                    y = y,
                    z = uncertainty,
                    colorscale = "magma",
                    colorbar_title = "Variance",
                ),
                row = 1 + igp,
                col = 2,
            )

            if f is not None:
                fig.add_trace(
                    go.Heatmap(
                        x = x,
                        y = y,
                        z = real,
                        colorscale = "magma",
                        showscale = False,
                    ),
                    row = 1 + igp,
                    col = 3,
                )

            for i in range(2 if f is None else 3):
                separate = np.full(len(self.evaluated), True)

                for j, ep in enumerate(self.epochs):
                    separate[ep[0]:ep[1]] = False

                    ic = j
                    while ic >= len(colors):
                        ic -= len(colors)

                    fig.add_trace(
                        go.Scatter(
                            x = self.evaluated[ep[0]:ep[1], 0],
                            y = self.evaluated[ep[0]:ep[1], 1],
                            mode = "markers",
                            marker_color = colors[ic],
                            marker_symbol = "x",
                            hovertext = f"Epoch {j}",
                            showlegend = False,
                        ),
                        row = igp + 1,
                        col = i + 1,
                    )

                separate = self.evaluated[separate]
                if len(separate):
                    fig.add_trace(
                        go.Scatter(
                            x = separate[:, 0],
                            y = separate[:, 1],
                            mode = "markers",
                            marker_color = "white",
                            marker_symbol = "x",
                            hovertext = "Manual",
                            showlegend = False,
                        ),
                        row = igp + 1,
                        col = i + 1,
                    )

        format_fig(fig)
        return fig


    def __repr__(self):
        parameters = textwrap.indent(str(self.parameters), '  ')

        if self.response_names is None:
            response_names = "  None"
        else:
            response_names = textwrap.indent(
                textwrap.fill(", ".join(self.response_names)),
                '  ',
            )

        samples = str_summary(self.samples)
        responses = str_summary(self.responses)
        epochs = f"list[{len(self.epochs)}, tuple[int, int]]"

        if self.scheduler is not None:
            scheduler = f"scheduler = {self.scheduler.__class__.__name__}\n"
        else:
            scheduler = ""

        docstr = (
            f"MED(seed={self.seed})\n"
            "--\n"
            f"parameters = \n{parameters}\n"
            f"response_names = \n{response_names}\n"
            "--\n"
            f"sampler =   {self.sampler}\n"
            f"{scheduler}"
            f"samples =   {samples}\n"
            f"responses = {responses}\n"
            f"epochs =    {epochs}\n"
        ).split("\n")

        maxline = max(len(d) for d in docstr)
        for i in range(len(docstr)):
            if len(docstr[i]) > 1 and docstr[i][1] == "-":
                docstr[i] = "-" * maxline

        return "\n".join(docstr)
