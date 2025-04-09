# GPU-troubleshoot

This page is contains advices on installation, testing and troubleshooting of GPU dependencies required for GPU-version configuration of `DeBCR`.

## Contents

- [Check GPU installation](#check-gpu-installation)
- [Tested configurations](#tested-configurations)
- [Source CUDA](#source-cuda)
- [Install cuDNN](#install-cudnn)

## Check GPU installation

To check that **TensorFlow** library, needed for our model usage, recognizes available GPU device(s):
1. Activate corresponding `debcr` environment, if you use a Python package manager, for example by
```bash
micromamba activate debcr
```
2. Check that your GPUs are visible to the **TensorFlow** by
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

For a single GPU device, you should see similar output as below:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If your GPU devices list is empty, please check that:
* a GPU device is available and connected to your machine
* **CUDA Driver** is installed, which is compatible to your GPU device
* **CUDA Tollkit** is installed, which is compatible to your GPU device and to the **TensorFlow** versions we support (see our [tested configurations](#tested-configurations))
* **CUDA Tollkit** is sourced, especially if installed in non-standard location (see how to [source CUDA](#source-cuda))
* **cuDNN** library is installed/configured (see how to [install cuDNN](#install-cudnn))
* activated correct package environment for `debcr` (i.e. with TensorFlow-GPU installed), to check which tensorflow version is installed you can type
```bash
pip list | grep tensorflow
```

> **Note**
> <br/> A proper CUDA and cuDNN installation and configuration might be tricky, especially if you work on an HPC cluster. Thus, try to contact your local system administrator first, before trying to install it yourself. 

## Tested configurations

We developed/tested `DeBCR` using the following configuration:
- [Python-3.9](https://www.python.org/downloads/release/python-390/)
- TensorFlow-2.11 as a deep learning backend
- [CUDA-11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive)
- cuDNN-v8.4.0 for CUDA-11.x from [cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive)
- OS Linux (Ubuntu 20.04)

If you are exploring other combinations of Python / CUDA / cuDNN / TensorFlow versions, you may find useful the [TensorFlow compatibility table](https://www.tensorflow.org/install/source#tested_build_configurations).

## Source CUDA

To source existing **CUDA** installation you need to set `CUDA_HOME` to the corresponding location by
```bash
export CUDA_HOME=/path/to/your/cuda-11.7
```
and add the respective paths to the `PATH` and `LD_LIBRARY_PATH` environment variables:
```bash
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```

To check if the correct **CUDA** version is sourced, you may look at `nvcc` compiler version by
```bash
nvcc --version
```

To make this changes permament, you can add them in your `~/.bashrc` file.

## Install cuDNN

To install **cuDNN** library,
1. Obtain from [cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive) the necessary **cuDNN** tar archive (see [tested configurations](#tested-configurations)), e.g. for cuDNN-8.4.0
```bash
wget https://developer.nvidia.com/compute/cudnn/secure/8.4.0/local_installers/11.6/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz
```
3. Unpack the obtained **cuDNN** archive:
```bash
tar -xvf cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz
```
4. Copy **cuDNN** files into the **CUDA Toolkit** location:
```bash
cp -P cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/include/cudnn.h /path/to/cuda-11.7/include
cp -P cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/lib/libcudnn* /path/to/your/cuda-11.7/lib64/
chmod a+r /path/to/cuda-11.7/lib64/libcudnn*
```
Unless the default CUDA path was changed during its installation, you may find it at `/usr/local/cuda-xx.x`. In the default path case you may need to add `sudo` before each command for the access permissions.

<details>
<summary><b> Alternative cuDNN installation if you cannot install cuDNN as described above. (not preferred) </b></summary>
</br>
If you have no access permissions to copy cuDNN files to the existing CUDA directory and cannot ask local system administrator to help you doing that, there is a rather cumbersome and inconvenient, but sometimes working option to install cuDNN. 

Install the **CUDA Toolkit** and **cuDNN** via your Python package envoronment manager by
```bash
micromamba install -c conda-forge cudatoolkit=11.7 cudnn=8.4
``` 
and [check GPU installation](#check-gpu-installation) again.

If the GPU list is still empty, it might be, for example, bacause **cuDNN** from conda-forge is missing libraries or **TensorFlow** cannot find its location. You can further try to re-install **cuDNN** via pip by
```bash
pip install nvidia-pyindex
pip install nvidia-cudnn-cu115
```
and make the following export for the current bash session (or add it permanently to your `~/.bashrc`) to point **cuDNN** location where **TensorFlow** can find it:
```bash
export LD_LIBRARY_PATH=/path/to/micromamba/envs/debcr/lib/python3.9/site-packages/nvidia/cudnn/lib/:${LD_LIBRARY_PATH}
```
with the actual location/name of your package manager (`micromamba` in this example) and name of your `DeBCR` environment (`debcr` in this example), where you also just installed **cuDNN** using commands above.

Finally, try again to [check your GPU installation](#check-gpu-installation).

</details>
</br>

For example of **CUDA-11.7** and **cuDNN-8.5.0** installation/configuration on OS Linux (Ubuntu 20.04), you may also find useful [this code snippet on Github Gist](https://gist.github.com/verazuo/19f381e4e2e546a9edcf66fc103d24a4), which can be adapted to other libraries/OS versions.