***************
Getting Started
***************
These instructions will help you get started with KonigCell; this package
interfaces with optimised C subroutines that must be compiled for every OS.


Prerequisites
-------------
If you install this package from PyPI (``pip install konigcell``) or conda-forge
(``conda install konigcell``), it should download pre-compiled wheels - so no
prior configuration should be needed.

To *build* this package on your specific machine, you will need a C compiler -
the low-level C code does not use any tomfoolery, so any compiler since the
2000s should do.

This package supports Python 3.6 and above (though it might work with even older
versions).


Installation
------------
Before the package is published to PyPI, you can install it directly from this GitHub
repository: 

::

    pip install git+https://github.com/anicusan/KonigCell

Alternatively, you can download all the code and run `pip install .` inside its
directory:

::

    git clone https://github.com/anicusan/KonigCell
    cd KonigCell
    pip install .

If you would like to modify the source code and see your changes without reinstalling
the package, use the `-e` flag for a *development installation*:

::

    pip install -e .


Optional Dependencies
---------------------
The ``konigcell`` library can offer some extra functionality if optional dependencies
are found:

- **plotly**: for the `*_trace` plotting utilities for `Pixels` and `Voxels`.
- **matplotlib** for the `plot_*` plotting utilities for `Pixels` and `Voxels`.
- **pyvista** for the `volumetric` and `vtk` utilities for `Voxels`.
