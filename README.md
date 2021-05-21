# PyOptiX

Python bindings for OptiX 7 - this branch also contains an experimental
implementation of an OptiX kernel written in Python, compiled with
[Numba](https://numba.pydata.org).


## Installation

### OptiX SDK

Install the [OptiX 7.3.0
SDK](https://developer.nvidia.com/optix/downloads/7.3.0/linux64).


### Conda environment

Create an environment containing pre-requisites:

```
conda create -n pyoptix python numpy conda-forge::cupy pybind11 pillow cmake numba
```

Activate the environment:

```
conda activate pyoptix
```

### PyOptiX installation

Build and install PyOptiX into the environment with:

```
export PYOPTIX_CMAKE_ARGS="-DOptiX_INSTALL_DIR=<optix install dir>"
pip3 install --global-option build --global-option --debug .
```

`<optix install dir>` should be the OptiX 7.3.0 install location - for example,
`/home/gmarkall/numbadev/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64`.


## Running the example

The example can be run from the root of the repository with:

```
python examples/hello.py
```

If the example runs successfully, a square will be rendered:

![Example output](example_output.png)


## Explanation

The Python implementation of the OptiX kernel and Numba extensions consists of
three parts, all in [examples/hello.py](examples/hello.py):

- Generic OptiX extensions for Numba - these implement things like
  `GetSbtDataPointer`, etc., and are a sort of equivalent of the implementations
  in the headers in the OptiX SDK.
- The user's code, which I tried to write exactly as I'd expect a PyOptiX Python
  user to write it - it contains declarations of the data structures as in
  hello.h, and the kernel as in hello.cu - you can, in this example modify the
  Python `__raygen__hello` function and see the changes reflected in the output
  image.
- Code that should be generated from the user's code - these tell Numba how to
  support the data structures that the user declared, and how to create them
  from the `SbtDataPointer`, etc. I've handwritten these for this example, to
  understand what a code generator should generate, and because it would have
  taken too long and been too risky to write something to generate this off the
  bat. The correspondence between the user's code and the "hand-written
  generated" code is mechanical - there is aclear path to write a generator for
  these based on the example code.
