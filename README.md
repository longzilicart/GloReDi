# GloReDi

This repository contains the implementation of "Learning to Distill Global Representation for Sparse-View CT" in ICCV2023. It includes two main components:

**(1) simulator protocol for Sparse-View CT**: A simulator protocol for sparse-view CT during the training process in CUDA. Designed as an easy-to-use wrapper, it is compatible with various networks and datasets.

**(2) Fourier network and distillation framework**: An effective image-domain-only method for sparse-view CT reconstruction.



## Updates
- [x] clean code and init commit



## Data Preparation
The DeepLesion dataset is available at [DeepLesion](https://nihcc.app.box.com/v/ChestXray-NIHCC), and the AAPM-Myo dataset can be downloaded from [CT Clinical Innovation Center](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/).


## Requirements
To set up the environment, please refer to requirements.txt. Notably, we utilize [torch radon](https://github.com/faebstn96/torch-radon) for efficient projection and forward-back projection. Follow these steps for installation, ensuring compatibility with the specified versions of CUDA and Torch:
```shell
git clone https://github.com/matteo-ronchetti/torch-radon.git
cd torch-radon
python setup.py install
```


## Training and Testing
Please refer to ```RUN.sh``` in the src directory for examples of training and testing


## Citation
If you find our work and code helpful, please kindly cite our paper:
```
@InProceedings{GloReDi_2022,
    author    = {Li, Zilong and Ma, Chenglong and Chen, Jie and Zhang, Junping and Shan, Hongming},
    title     = {Learning to Distill Global Representation for Sparse-View CT},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {21196-21207}
}
```

Also, welcome to visit our other works [FreeSeed](https://github.com/Masaaki-75/freeseed). 
```
@inproceedings{ma2023freeseed,
  title={FreeSeed: Frequency-band-aware and Self-guided Network for Sparse-view CT Reconstruction},  
  author={Ma, Chenglong and Li, Zilong and Zhang, Yi and Zhang, Junping and Shan, Hongming}, 
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023},
  year={2023}
}
```



## Contact

If you have any question, please feel free to concat me at longzilipro@gmail.com


