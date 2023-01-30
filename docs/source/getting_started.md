# Getting Started

These instructions will help you get started with MED; the only quirk of the library is that we
need to install Julia beforehand - this is a modern, powerful programming language that resembles
MATLAB, but is compiled to high-performance machine code (kinda like C/C++/Rust) - it's really lovely
and I highly recommend it, but we don't need to write any Julia in MED. The *symbolic regression*
engine that we use to derive equations automatically is written in Julia - which is why it's powerful
enough to derive and fit thousands of equations per second.

You can find a nice Julia configuration tutorial for [Visual Studio Code here](https://www.julia-vscode.org/docs/dev/gettingstarted/).


## Installation

Before the ``medeq`` library is published to PyPI, you can install it directly from this GitHub repository: 

```
$> pip install git+https://github.com/uob-positron-imaging-centre/MED
```

Alternatively, you can download all the code and run `pip install .` inside its
directory:

```
$> git clone https://github.com/uob-positron-imaging-centre/MED
$> cd MED
$MED> pip install .
```

If you would like to modify the source code and see your changes without reinstalling the package, use the `-e` flag for a *development installation*:

```
$MED> pip install -e .
```

###  Julia

To discover underlying equations and see interactive plots of system responses,
uncertainties and model outputs, you need to install Julia (a
beautiful, high-performance programming language) on your system and the
PySR library:

1. Install Julia manually (see [Julia downloads](https://julialang.org/downloads/), version >=1.8 is recommended).
2. `import medeq; medeq.install()`
