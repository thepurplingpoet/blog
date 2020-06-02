---
toc: true
layout: post
description: Installing the Fastai library (developmental version 2) on Windows 10
categories: [markdown]
title: Installing the Fastai library on Windows 10
---

# Installing the Fastai library on Windows 10

Here are the steps I took to install fastai v2 on my windows machine. This version of fastai is still in development as of 3rd of June,
2020 so I would recommend using v1 instead for production. (But then, would you use windows for production ? ;) )

### Use the instructions at https://dev.fast.ai/ to install an editable version
```
git clone https://github.com/fastai/fastai2
cd fastai2
conda env create -f environment.yml
source activate fastai2
pip install -e ".[dev]"

git clone https://github.com/fastai/fastcore
cd fastcore
pip install -e ".[dev]"
```
The above commands should install both fastai and fastcore in editable mode. So, if you update one, remember to update the other as well.


### Here's what I tried after installing:
```
from fastai2.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224), num_workers=0)

learn = cnn_learner(dls, resnet34, metrics=error_rate)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
learn.fine_tune(1)
```

I have an RTX 2060 in my machine and training took 256 seconds compared to Google Colaboratory which took 89 seconds.

### Things to Note

- If you see a ``` CUDA Runtime Error ```, set the  ``` num_workers = 0 ``` in the dataloader function.
- For ``` OSError: Image file is truncated ``` , use this piece of code 
``` 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

I would try to add more points here once I explore further. 
