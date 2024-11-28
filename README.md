# A dual-pairing summation-by-parts finite difference framework for nonlinear conservation laws

This repository contains the code to generate the figures and tables presented
in https://arxiv.org/abs/2411.06629

## Running the code

### Clone Repository

First clone the repository and switch to the project directory.

```bash
git clone https://github.com/Dougal-s/paper-2024-DP-SBP-nonlinear.git
cd paper-2024-DP-SBP-nonlinear
```

### Install Dependencies

The code requires [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl)
to be installed on the global environment. To do this run the command

```bash
julia -e 'using Pkg; Pkg.add("DrWatson")'
```

To install the project dependencies, run the command

```bash
julia --project='.' -e 'using Pkg; Pkg.instantiate()'
```

### Execute Scripts

Each file in `scripts` corresponds to an individual test case and can be
executed by running

```
julia scripts/the_script_name
```

Generated figures will be placed in the `plots` directory and data/tables with
be placed in the `data` directory.

As the scripts cycle through a variety of parameter combinations, it is
recommended to run them with the `--threads auto` flag set to multithreaded
execution.


### System Image Compilation

Due to the number of scripts, it may be desirable to reduce the time to first
execution by generating a custom system image using
[PackageCompiler.jl](https://github.com/JuliaLang/PackageCompiler.jl). This can
be done by executing the script `compile/compile.sh`. *Note that on my
machine, this took ~5min and used all 16gb of available memory.* Once the
system image has been generated, scripts can be run using:

```
julia -Jpath/to/sysimage.so path/to/script.jl
```

## Authors
 - Kenneth Duru (University of Texas at El Paso, US; Australian National University, Canberra, Australia)
 - Dougal Stewart (University of Melbourne, Australia)
 - Nathan Lee (University of New South Wales, Australia)
