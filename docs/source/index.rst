..
   File   : index.rst
   License: GNU v3.0
   Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
   Date   : 28.06.2020


=================================
The MedEQ Library's Documentation
=================================

Autonomously driving equation discovery, from the micro to the macro, from
laptops to supercomputers.

It builds on the fantastic ``PySR`` (https://github.com/MilesCranmer/PySR) and
``fvGP`` (https://github.com/lbl-camera/fvGP) libraries to create a user-facing
package offering:

- Discovery of symbolic **closed-form equations** that model multiple responses.
- Efficient **parameter sampling** for planning experimental / simulational campaigns.
- System multi-response uncertainty quantification - and specifically **targeting high-variance parameter regions**.
- **Automatic parallelisation** of complex user simulation scripts on OS Processes and distributed supercomputers.
- Interactive plotting of responses, uncertainties, discovered model outputs.
- Language-agnostic saving of results found.

MedEQ was developed to discover physical laws and correlations in chemical engineering, but it is
data-agnostic - and works with both simulated and experimental results in any domain.


Tutorials and Documentation
===========================
At the top of this page, see the "Getting Started" tab for installation help; the
"Tutorials" section has some explained high-level examples of the library. Finally,
all exported functions are documented in the "Manual".


Contributing
============
You are more than welcome to contribute to this library in the form of library
improvements, documentation or helpful examples; please submit them either as:

- GitHub issues.
- Pull requests.
- Email me at <a.l.nicusan@bham.ac.uk>.


Citing
======
If you use this library in your research, you are kindly asked to cite:

    <Paper after publication>


This library would not have been possible without the excellent `PySR` library
(https://github.com/MilesCranmer/PySR) which forms the core of the equation
discovery subroutine; if you use MedEQ in your work, please also cite:

    Cranmer M, Sanchez Gonzalez A, Battaglia P, Xu R, Cranmer K, Spergel D, Ho S. Discovering symbolic models from deep learning with inductive biases. Advances in Neural Information Processing Systems. 2020;33:17429-42.


Licensing
=========
MedEQ is licensed under the GPL v3.0 license.



Indices and tables
==================

.. toctree::
   :caption: Documentation
   :maxdepth: 2

   getting_started
   tutorials/index
   manual/index


Pages

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
