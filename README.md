# DeBCR

**DeBCR** is a Python-based framework for light microscopy data enhancement, including denoising and deconvolution.

As an enhancement core, **DeBCR** implements a multi-scale sparsity-efficient deep learning model [m-rBCR](https://doi.org/10.1007/978-3-031-73226-3_22).

As a framework, **DeBCR** provides user interfaces such as:
- [`debcr`](https://github.com/DeBCR/DeBCR) - a Python-based API library for scripting, e.g. using [Jupyter Notebook/Lab](https://jupyter.org/)
- [`napari-debcr`](https://github.com/DeBCR/napari-debcr/) - an add-on GUI plugin for [Napari viewer](https://github.com/napari/napari)

### License
This is an open-source project and is licensed under [MIT license](https://github.com/DeBCR/DeBCR/blob/main/LICENSE).

### Contact
For any questions or bug-reports on `debcr` please use dedicated [GitHub Issue Tracker](https://github.com/DeBCR/DeBCR/issues).

## Installation

There are two hardware-based installation options for `debcr`:
- `debcr[tf-gpu]` - for a GPU-based trainig and prediction (**recommended**);
- `debcr[tf-cpu]` - for a CPU-only execution (note: training on CPUs might be quite slow!).

### GPU prerequisites

For a GPU version you need:
- a GPU device with at least 12Gb of VRAM;
- a compatible CUDA Toolkit (recommemded: [CUDA-11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive));
- a compatible cuDNN library (recommemded: v8.4.0 for CUDA-11.x from [cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive)).

For more info on GPU dependencies please check our [GPU-advice page](https://github.com/DeBCR/DeBCR/blob/main/docs/GPU-advice.md). 

### Create a package environment (optional)

For a clean isolated installation, we advice using one of Python package environment managers, for example:
- `micromamba`/`mamba` (see [mamba.readthedocs.io](https://mamba.readthedocs.io/))
- `conda-forge` (see [conda-forge.org](https://conda-forge.org/))

Create an environment for `debcr` using
```bash
micromamba env create -n debcr python=3.9 -y
```
and activate it for further installation or usage by
```bash
micromamba activate debcr
```

### Install DeBCR

Install one of the `DeBCR` versions:
- GPU (**recommended**; backend: TensorFlow-GPU-v2.11):
  ```bash
  pip install 'debcr[tf-gpu]'
  ```
- CPU (*limited*; backend: TensorFlow-CPU-v2.11)
  ```bash
  pip install 'debcr[tf-cpu]'
  ```

### Test GPU visibility

For a GPU version installation, it is recommended to check if your GPU device is recognised by **TensorFlow** using
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

which for a single GPU device should produce a similar output as below:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If your GPU device list is empty, please check our [GPU-advice page](https://github.com/DeBCR/DeBCR/blob/main/docs/GPU-advice.md). 

### Install Jupyter

To use `debcr` as a Python library (API) interactively, please also install [Jupyter Notebook/Lab](https://jupyter.org/install), for example
```bash
pip install jupyterlab
```

## Usage

To learn using `debcr` as a python library (API) interactively, follow our notebook tutorials:

| Notebook tutorial                                                     | Purpose | Hardware | Inputs |
| :---------------------------------------------------------------- | :------ | :------- | :------- | 
| [debcr_predict.ipynb](https://github.com/DeBCR/DeBCR/blob/main/notebooks/debcr_predict.ipynb)  | enhanced prediction | CPU/GPU | pre-processed input data (NPZ/NPY), </br> trained DeBCR model. |
| [debcr_train.ipynb](https://github.com/DeBCR/DeBCR/blob/main/notebooks/debcr_train.ipynb)    | model training | GPU | training/validation data (NPZ/NPY). |
| [debcr_preproc.ipynb](https://github.com/DeBCR/DeBCR/blob/main/notebooks/debcr_preproc.ipynb)      | raw data pre-processing | CPU | raw data (TIF/TIFF, JPG/JPEG, PNG). |

To use these notebooks,
1. activate `debcr` environment, if was inactive, by
```bash
micromamba activate debcr
```
2. start Jupyter session at the notebooks location (download them from the [DeBCR GitHub](https://github.com/DeBCR/DeBCR))
```bash
jupyter-lab
```

### Example data and trained model weights

Based on several previously published datasets (from [CARE](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.FDFZOF), [DeepBacs](https://doi.org/10.5281/zenodo.12626121), and [TA-GAN](https://doi.org/10.5281/zenodo.7908913)), we prepared four example datasets and trained `m-rBCR` model weights to both evaluate our model and serve as the example data/weights for notebook tutorials.

The datasets are distributed as NumPy (.npz) arrays in three essential sets (train, validation and test), available along with the trained model weights on Zenodo: [10.5281/zenodo.12626121](https://zenodo.org/doi/10.5281/zenodo.12626121).

## About model

The core **DeBCR** enhancement model **m-rBCR** approximates imaging process inversion with deep convolutional neural network (DCNN), based on compact BCR-representation ([Beylkin G. et al., *Comm. Pure Appl. Math*, 1991](https://onlinelibrary.wiley.com/doi/10.1002/cpa.3160440202)) for convolutions and its DCNN implementation as proposed in BCR-Net ([Fan Y. et al., *J. Comput. Phys.*, 2019](https://www.sciencedirect.com/science/article/pii/S0021999119300762)):

![DeBCR network structure](https://github.com/DeBCR/DeBCR/raw/main/docs/images/DeBCR_structure.jpg?raw=true)

In contrast to the traditional single-stage residual BCR learning process, the core DeBCR model integrates feature maps from multiple resolution levels:
![DeBCR multi-resolution](https://github.com/DeBCR/DeBCR/raw/main/docs/images/DeBCR_multires.jpg?raw=true)

The example of the **DeBCR** performance on the low/high exposure confocal data of *Tribolium castaneum* sample from the **CARE** work ([Weigert et al., *Nat. Methods*, 2018](https://www.nature.com/articles/s41592-018-0216-7)) is shown below:
![DeBCR LM](https://github.com/DeBCR/DeBCR/raw/main/docs/images/DeBCR_LM.jpg?raw=true)

For more details on the multi-stage residual BCR (m-rBCR) architechture implemented within DeBCR framework see:

Li, R., Kudryashev, M., Yakimovich, A. Solving the Inverse Problem of Microscopy Deconvolution with a Residual Beylkin-Coifman-Rokhlin Neural Network. *ECCV 2024*, *Lecture Notes in Computer Science*, vol 15133. Springer, Cham. https://doi.org/10.1007/978-3-031-73226-3_22

<!--
For more details on implementaion and benchmarks please see our recent preprint:
Li R., Yushkevich A., Chu X., Kudryashev M., Yakimovich A. Denoising, Deblurring, and optical Deconvolution for cryo-ET and light microscopy with a physics-informed deep neural network DeBCR. *bioRxiv*, 2024.
-->
