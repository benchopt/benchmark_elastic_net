Elastic Net Benchmark
=====================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to elastic net regression:


$$ \\min_w \\frac{1}{2n} \\Vert y - Xw \\Vert^2 + \\lambda \\ (\\rho \\Vert w \\Vert_1 + \\frac{1 - \\rho}{2} \\Vert w \\Vert^2)$$

where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features
, $\\rho \\in (0, 1]$ is the ``l1_ratio`` and


$$y \\in \\mathbb{R}^n , \\ X = [x_1^\\top, \\dots, x_n^\\top]^\\top \\in \\mathbb{R}^{n \\times p}$$

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_elastic_net
   $ benchopt run benchmark_elastic_net

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_elastic_net -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_elastic_net/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_elastic_net/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
