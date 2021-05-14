# PyOptiX

Python bindings for OptiX 7.

## Installation

### OptiX SDK

Install the OptiX 7.2.0 SDK from https://developer.nvidia.com/optix/downloads/7.2.0/linux64



### Conda environment

Create an environment containing pre-requisites:

```
conda create -n pyoptix python numpy conda-forge::cupy pybind11 pillow cmake
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

`<optix install dir>` should be the OptiX 7.2.0 install location - for example,
`/home/gmarkall/numbadev/NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64`.


## Running the example

The example can be run from the root of the repository with:

```
python examples/hello.py
```

It may be necessary to modify the include path used by nvrtc to find the OptiX
headers at their actualy installed location. For example:

```diff
diff --git a/examples/hello.py b/examples/hello.py
index 16a153e..317cc37 100755
--- a/examples/hello.py
+++ b/examples/hello.py
@@ -73,7 +73,7 @@ def compile_cuda( cuda_file ):
         #'-IC:\\ProgramData\\NVIDIA Corporation\OptiX SDK 7.2.0\include',
         #'-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.1\include'
         '-I/usr/local/cuda/include',
-        '-I/home/kmorley/Code/support/NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64/include/'
+        '-I/home/gmarkall/numbadev/NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64/include/'
         ] )
     return ptx
 
```

If the example runs successfully, a green square will be rendered.
