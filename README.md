# modsoft

This repository provides a reference implementation of the soft clustering algorithm for graphs described
in the following research paper:

**Modularity-based Sparse Soft Graph Clustering**, Alexandre Hollocou, Thomas Bonald, Marc Lelarge.
**AISTATS 2019**

It contains a Python version and a faster Cython version of the algorithm.

## Build the Cython version

In order to build the modsoft package:
- Go to the modsoft directory
- Do: `python setup.py build_ext --inplace`
- You can then use "import modsoft" if you copy the modsoft directory to your project.

## Example

Check out the Jupyter notebook example.ipynb.
