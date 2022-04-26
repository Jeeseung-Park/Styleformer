## [CVPR 2022] Styleformer - Official PyTorch implementation

![Teaser image](./docs/style_mixing.png)

**Styleformer: Transformer based Generative Adversarial Networks with Style Vector**<br>
Jeeseung Park, Younggeun Kim<br>
https://arxiv.org/abs/2106.07023

Abstract: *We propose Styleformer, a generator that synthesizes image using style vectors based on the Transformer structure. In this paper, we effectively apply the modified Transformer structure (e.g., Increased multi-head attention and Pre-layer normalization) and attention style injection which is style modulation and demodulation method for self-attention operation. The new generator components have strengths in CNN's shortcomings, handling long-range dependency and understanding global structure of objects. We propose two methods to generate high-resolution images using Styleformer.
First, we apply Linformer in the field of visual synthesis (Styleformer-L), enabling Styleformer to generate higher resolution images and result in improvements in terms of computation cost and performance. This is the first case using Linformer to image generation. Second, we combine Styleformer and StyleGAN2 (Styleformer-C) to generate high-resolution compositional scene efficiently, which Styleformer captures long-range-dependencies between components.
With these adaptations, Styleformer achieves comparable performances to state-of-the-art in both single and multi-object datasets. Furthermore, groundbreaking results from style mixing and attention map visualization demonstrate the advantages and efficiency of our model.*


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleformer-transformer-based-generative/image-generation-on-celeba-64x64)](https://paperswithcode.com/sota/image-generation-on-celeba-64x64?p=styleformer-transformer-based-generative)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleformer-transformer-based-generative/image-generation-on-stl-10)](https://paperswithcode.com/sota/image-generation-on-stl-10?p=styleformer-transformer-based-generative)

<div align="left">
  <img src="docs/overall_architecture.png" style="float:left" width="250px">
  <img src="docs/cityscape_paper.png" style="float:right" width="250px"> 
</div>


## Requirements

* We have done all testing and development using 4 Titan RTX GPUs with 24GB.
* 64-bit Python 3.7 and PyTorch 1.7.1. 
* Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`.  We use the Anaconda3 2020.11 distribution which installs most of these by default.

## Pretrained pickle

[CIFAR-10](https://drive.google.com/file/d/1z7sNrq_iGXgt3Tzl3NxRTEXHKdw_AzSZ/view?usp=sharing)
Styleformer-Large with FID 2.82 IS 9.94

[STL-10](https://drive.google.com/file/d/1fpWR9sOQA5KApeGlP7hWTi8S6bpcn5Bt/view?usp=sharing)
Styleformer-Medium with FID 15.17 IS 11.01 

[CelebA](https://drive.google.com/file/d/1nyYxhRKE-kNMFRO5Ijx8N_1KOSX5jh_V/view?usp=sharing)
Styleformer-Linformer with FID 3.66

[LSUN-Church](https://drive.google.com/file/d/1X3yPt__srOuK8pRr0z4GKvtyjnEKYQOU/view?usp=sharing)
Styleformer-Linformer with FID 7.99

## Generating images

Pre-trained networks are stored as `*.pkl` files that can be referenced using local filenames

```.bash
# Generate images using pretrained_weight 
python generate.py --outdir=out --seeds=100-105 \
    --network=path_to_pkl_file
```

Outputs from the above commands are placed under `out/*.png`, controlled by `--outdir`. Downloaded network pickles are cached under `$HOME/.cache/dnnlib`, which can be overridden by setting the `DNNLIB_CACHE_DIR` environment variable. The default PyTorch extension build directory is `$HOME/.cache/torch_extensions`, which can be overridden by setting `TORCH_EXTENSIONS_DIR`.


## Preparing datasets


**CIFAR-10**: Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/cifar-10-python.tar.gz --dest=~/datasets/cifar10.zip
```

**STL-10**: Download the stl-10 dataset 5k training, 100k unlabeled images from [STL-10 dataset page](https://cs.stanford.edu/~acoates/stl10/) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/cifar-10-python.tar.gz --dest=~/datasets/stl10.zip \
    ---width=48 --height=48
```

**CelebA**: Download the CelebA dataset Aligned&Cropped Images from [CelebA dataset page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/cifar-10-python.tar.gz --dest=~/datasets/stl10.zip \
    ---width=64 --height=64
```


**LSUN Church**: Download the desired categories(church) from the [LSUN project page](https://www.yf.io/p/lsun/) and convert to ZIP archive:

```.bash

python dataset_tool.py --source=~/downloads/lsun/raw/church_lmdb --dest=~/datasets/lsunchurch.zip \
    --width=128 --height=128
```



## Training new networks

In its most basic form, training new networks boils down to:

```.bash
python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1 --batch=32 --cfg=cifar --g_dict=256,64,16 \
    --num_layers=1,2,2 --depth=32
```

* `--g_dict=` it means 'Hidden size' in paper, and it must be match with image resolution.
* `--num_layers=` it means 'Layers' in paper, and it must be match with image resolution.
* `--depth=32` it means minimum required depth is 32, described in Section 2 at paper.
* `--linformer=1` apply informer to Styleformer.

Please refer to [`python train.py --help`](./docs/train-help.txt) for the full list. 
To train STL-10 dataset with same setting at paper, please fix the starting resolution 8x8 to 12x12 at training/networks_Generator.py. 



## Quality metrics

Quality metrics can be computed after the training:

```.bash
# Pre-trained network pickle: specify dataset explicitly, print result to stdout.
python calc_metrics.py --metrics=fid50k_full --data=~/datasets/lsunchurch.zip \
    --network=path_to_pretrained_lsunchurch_pkl_file
    
python calc_metrics.py --metrics=is50k --data=~/datasets/lsunchurch.zip \
    --network=path_to_pretrained_lsunchurch_pkl_file    
```

## Citation
If you found our work useful, please don't forget to cite
```
@misc{park2021styleformer,
      title={Styleformer: Transformer based Generative Adversarial Networks with Style Vector}, 
      author={Jeeseung Park and Younggeun Kim},
      year={2021},
      eprint={2106.07023},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



The code is heavily based on the [stylegan2-ada-pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch)

