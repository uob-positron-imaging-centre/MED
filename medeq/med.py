#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : med.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 29.03.2022


import  os
import  sys
import  contextlib
import  pickle
import  inspect
import  textwrap
import  warnings
from    datetime            import  datetime

import  numpy               as      np
import  pandas              as      pd

import  cma
from    fvgp.gp             import  GP
from    scipy.stats.qmc     import  discrepancy, LatinHypercube

import  toml
from    tqdm                import  tqdm
from    coexist.plots       import  format_fig
from    coexist.utilities   import  autorepr

import  plotly.express      as      px
import  plotly.graph_objs   as      go
from    plotly.subplots     import  make_subplots

from    .__version__        import  __version__
from    .utilities          import  str_summary


# Optional imports
try:
    import pysr
    from pysr import PySRRegressor
    pysr.silence_julia_warning()
except ImportError:
    class PySRNotInstalled:
        pass
    PySRRegressor = PySRNotInstalled




def validate_parameters(parameters):
    '''Validate the free parameters provided or extracted from a user script (a
    ``pandas.DataFrame``).
    '''
    if not isinstance(parameters, pd.DataFrame):
        raise ValueError(textwrap.fill((
            "The `parameters` given is not a `pandas.DataFrame` instance (or "
            f"subclass thereof). Received `{type(parameters)}`."
        )))

    columns_needed = ["min", "max"]
    if not all(c in parameters.columns for c in columns_needed):
        raise ValueError(textwrap.fill((
            "The `parameters` DataFrame provided must have at least two "
            "columns defined: ['min', 'max']. Found these: "
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

    A MED sampler is simply an object defining the method ``.sample(med, n)``,
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
        def user_sampler(med, n):
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


    def sample(self, med, n):
        # Save the MED instance as an attribute to be accessed in `.cost`
        if med is None:
            class MEDNotGiven:
                gp = None
            self.med = MEDNotGiven()
        elif isinstance(med, MED):
            self.med = med
        else:
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
                    es.optimize(self.cost, iterations = nit)
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


    def cost(self, x):
        x = x.reshape(-1, self.d)

        if self.med.gp is None:
            uncertainty = 1
        else:
            uncertainty = self.med.gp.posterior_covariance(
                x, variance_only = True
            )["v(x)"].mean()

            # Take previous samples into consideration for discrepancy
            prev = downscale(
                self.med.evaluated,
                self.med.parameters["min"].to_numpy(),
                self.med.parameters["max"].to_numpy(),
            )
            x = np.vstack((prev, x))

        return discrepancy(x) / uncertainty


    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(d={self.d}, seed={self.seed})"




class RandomSampler:

    def __init__(self, d, seed = None):
        self.d = int(d)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)


    def sample(self, med, n):
        return self.rng.random((n, self.d))


    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(d={self.d}, seed={self.seed})"




class LatticeSampler:

    def __init__(self, d, seed = None):
        self.d = int(d)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)


    def sample(self, med, n):
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
        self.outputs = os.path.join(self.directory, "outputs")


    def update_paths(self, prefix):
        '''Translate all paths saved in this class relative to a new `prefix`
        (which will replace the `directory` attribute).

        Please ensure that the `prefix` directory contains MED-created files.
        '''

        self.directory = prefix
        for attr in ["results", "queue", "setup", "sampler", "script",
                     "outputs"]:
            prev = getattr(self, attr)

            if prev is not None:
                new_path = os.path.join(prefix, os.path.split(prev)[1])
                setattr(self, attr, new_path)




class MED:
    '''Autonomously explore system responses and discover underlying physical
    laws / correlations.

    Attributes
    ----------
    parameters : pd.DataFrame
        A

    response_names : list[str] or None
        A

    sampler : object
        Any Python object defining a method ``.sample(med, n)`` returning ``n``
        samples to evaluate.

    samples : np.ndarray
        A

    responses : np.ndarray or None
        A

    epochs : list[tuple[int, int]]
        A

    seed : int
        A

    verbose : int
        A

    queue : np.ndarray
        [Generated]

    evaluated : np.ndarray
        [Generated]

    results : pd.DataFrame
        [Generated]

    gp : fvgp.gp.GP or None
        [Internal]

    sr : pysr.PySRRegressor or None
        [Internal]

    paths : medeq.med.MEDPaths or None
        [Internal]

    '''

    def __init__(
        self,
        parameters,             # TODO: Or script
        response_names = None,
        sampler = DVASampler,
        seed = None,
        verbose = 3,
    ):
        # TODO: only generate script if needed

        # Type-checking
        validate_parameters(parameters)
        self.parameters = parameters

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
            if responses.ndim == 1:
                responses = responses[:, np.newaxis]
            nresp = responses.shape[1]

        # Extract variances
        if self.gp is None:
            variances = np.empty((0, nresp))
        else:
            # TODO: handle multiple GPs / variances
            variances = self.gp.variances

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
            hyperparameters = self.gp.hyperparameters

        setup = {
            "seed": self.seed,
            "verbose": self.verbose,
            "epochs": self.epochs,
            "response_names": self.response_names,
            "hyperparameters": hyperparameters.tolist(),
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
        if False:
            # TODO: Save script too
            if not os.path.isdir(self.paths.results):
                os.mkdir(self.directory)

            if not os.path.isdir(self.outputs):
                os.mkdir(self.directory)

        # Save information about the run
        readmefile = os.path.join(self.paths.directory, "readme.rst")
        with open(readmefile, "w", encoding = "utf-8") as f:
            f.write(textwrap.dedent(f'''
                MED System Response Exploration Directory
                -----------------------------------------

                This directory was generated by MED at {now}.
            '''))

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
        new = self.sampler.sample(self, n)
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
        2. A simple Python function is supplied that will be evaluated for each
           sample; the function must accept a single NumPy vector.
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
            responses = self._evaluate_script()
        elif callable(f):
            responses = self._evaluate_function(f)
        else:
            responses = np.asarray(f, dtype = float)

        # Type-check responses found
        if responses.ndim != 1 or len(responses) != len(self.queue):
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

        # Type-check samples and responses shapes
        if samples.ndim != 2 or samples.shape[1] != len(self.parameters):
            raise ValueError(textwrap.fill((
                "The input `samples` must have the same number of columns "
                "as the number of dimensions, i.e. number of parameters (= "
                f"{len(self.parameters)}). Received array with shape "
                f"`{samples.shape}`."
            )))

        if responses.ndim != 1 or len(responses) != len(samples):
            raise ValueError(textwrap.fill((
                "The input `responses` must be a 1D vector with the same "
                "number of values as the number of rows in `samples`. "
                f"Received {len(responses)} responses for {len(samples)} "
                "samples."
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
        parsimony = 1e-4,
        constraints = None,
        model_selection = "best",
        niterations = 100,
        ncyclesperiteration = 32,
        populations = 32,
        use_symbolic_utils = True,
        multithreading = True,
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
            parsimony = parsimony,
            constraints = constraints,
            model_selection = model_selection,
            niterations = niterations,
            ncyclesperiteration = ncyclesperiteration,
            populations = populations,
            use_symbolic_utils = use_symbolic_utils,
            multithreading = multithreading,
            equation_file = "equations",
            temp_equation_file = "equations_temp",
            progress = progress,
            **kwargs,
        )

        names = [c.replace(" ", "_") for c in self.parameters.index]
        self.sr.fit(self.evaluated, self.responses, variable_names = names)
        return self.sr


    def _train(self):
        if self.gp is None:
            self.gp = GP(
                len(self.parameters),
                points = self.evaluated,
                values = self.responses,
                init_hyperparameters = np.ones(1 + len(self.parameters)),
                variances = np.abs(0.01 * self.responses),
                use_inv = True,
            )
        else:
            self.gp.update_gp_data(
                self.evaluated,
                self.responses,
                np.abs(0.01 * self.responses),
            )

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.gp.train(None, method = self._train_gp_method)


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


    def plot_response(
        self,
        f = None,
        colors = px.colors.qualitative.Set1[1:],
    ):

        nrows = 1
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

        prediction = self.gp.posterior_mean(
            np.c_[xx.flatten(), yy.flatten()]
        )["f(x)"].reshape(nx, ny)

        uncertainty = self.gp.posterior_covariance(
            np.c_[xx.flatten(), yy.flatten()],
            variance_only = True,
        )["v(x)"].reshape(nx, ny)

        if f is not None:
            real = f(np.vstack((xx.flatten(), yy.flatten()))).reshape(nx, ny)

        # Create Figure
        fig = make_subplots(
            rows = nrows,
            cols = ncols,
            subplot_titles = subplot_titles,
        )

        fig.add_trace(
            go.Heatmap(
                x = x,
                y = y,
                z = prediction,
                colorscale = "magma",
                showscale = False,
            ),
            row = 1,
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
            row = 1,
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
                row = 1,
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
                    row = 1,
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
                    row = 1,
                    col = i + 1,
                )

        format_fig(fig)
        return fig


    def __repr__(self):
        parameters = textwrap.indent(str(self.parameters), '  ')
        response_names = textwrap.indent(
            textwrap.fill(", ".join(self.response_names)),
            '  ',
        )

        samples = str_summary(self.samples)
        responses = str_summary(self.responses)
        epochs = f"list[{len(self.epochs)}, tuple[int, int]]"

        docstr = (
            f"MED(seed={self.seed})\n"
            "--\n"
            f"parameters = \n{parameters}\n"
            f"response_names = \n{response_names}\n"
            "--\n"
            f"sampler =   {self.sampler}\n"
            f"samples =   {samples}\n"
            f"responses = {responses}\n"
            f"epochs =    {epochs}\n"
        ).split("\n")

        maxline = max(len(d) for d in docstr)
        for i in range(len(docstr)):
            docstr[i] += (maxline - len(docstr[i]) + 1) * " "
            if docstr[i][1] == "-":
                docstr[i] = "-" * maxline

        return "\n".join(docstr)
