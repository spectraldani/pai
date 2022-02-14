# Parallel Active Inference - (PAI)

This repositiory contains the supplementary code for the paper "Parallel MCMC Without Embarrassing Failures" presented at AISTATS 2022.

Currently, the repostory contain the necessary GP model, mean function, and acquistion function code in the `lib` directory. The `run_toy_4modes.ipynb` notebook contains a simple demonstration of the method on the dataset from the paper's section 4.1.

## How to install the dependencies

Using conda, the environment.yml file can be used to install all the appropriate libraries. Just run `conda env create -f environment.yml
` and a new `ebmcgp` can be activated.

## Known issues when running on Windows

For Windows users, PyStan might no automatically choose the correct C++ compiler, here's how to fix this:
1. Find the directory of `distutils` by running:
    ```python
    import distutils
    print(distutils.__file__)
    # Outputs something like: C:\Users\xxx\miniconda\envs\ebmcgp\lib\distutils\__init__.py
   ```
2. Edit or create the file `distutils.cfg` at the path found above. The new content of this file should be:
   ```
   [build]
   compiler=mingw32
   ```
3. Done!
