# MSA-FM

Official repository of **MSA-FM** (Journal of Electronic Imaging).

Code for our paper **MSA-FM** on visible–infrared person re-identification (VI-ReID).

## Requirements

Python 3.8 is recommended. Install the dependencies with:

```bash
pip install -r requirements.txt
```

Training a ResNet-50 model on SYSU-MM01 needs roughly 22 GB of GPU memory
(batch size 128).

## Data preparation

### SYSU-MM01

1. Download SYSU-MM01 and unzip it into a folder named `SYSU-MM01`
   (containing `cam1` … `cam6`).
2. Download the evaluation split files `rand_perm_cam.mat` and
   `rand_perm_cam_test.mat` from the official
   [SYSU-MM01 evaluation tools](https://github.com/wuancong/SYSU-MM01/tree/master/evaluation/data_split)
   and put **both** of them into `SYSU-MM01/exp/`.
   The evaluation code loads `rand_perm_cam_test.mat` for the 1-shot protocol
   (`rand_perm_cam.mat` is only used by the commented-out variant in `engine/__init__.py`).

### RegDB

Download RegDB into a folder named `RegDB` (images plus the `idx/` train/test split lists).

### Dataset paths

Paths are configured in `configs/default/dataset.py` and are relative to the working
directory by default:

```python
dataset_cfg.sysu.data_root  = "SYSU-MM01"
dataset_cfg.regdb.data_root = "RegDB"
```

Either change these entries or run the scripts from the directory that contains the
dataset folders.

## Training

```bash
python train.py --cfg configs/SYSU.yml
python train.py --cfg configs/RegDB.yml
```

Checkpoints, if written, go to `checkpoints/<dataset>/`; logs go to `logs/`.

## Testing

`test.py` loads a trained checkpoint and runs evaluation only:

```bash
python test.py --cfg configs/SYSU.yml --checkpoint checkpoints/sysu/model.pth
python test.py --cfg configs/RegDB.yml --checkpoint checkpoints/regdb/model.pth
```

Without `--checkpoint` the script exits with a `FileNotFoundError`.

This code is developed upon the official SAAI implementation (ICCV 2023).
