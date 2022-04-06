# DIDA-Net

Pytorch implementation for DIDA-Net (["Zhongying Deng, Kaiyang Zhou, Da Li, Junjun He, Yi-Zhe Song, Tao Xiang. Dynamic Instance Domain Adaptation. arXiv:2203.05028"](https://arxiv.org/abs/2203.05028))

## Installation

Please install the [Dassl codebase](https://github.com/KaiyangZhou/Dassl.pytorch#get-started), and then copy the files in this repository to Dassl. 
When asked to overwrite, please say yes (some `__init__.py` files may be overwritten, these files is modified to include the backbone used for DIDA-Net. 
Other files may also different from the latest Dassl codebase, e.g. `dassl/optim/optimizer.py` is updated to set different learning rate for new layers, and 
`dassl/data/transforms/transforms.py` is from an old version of Dassl codebase which does not use center crop for testing image).
Then `pytorch 1.7.1 + cuda 10.1, python 3.7` should be installed.

## Training

Create a folder like `output/didanet_pacs` (under the Dassl root path) where checkpoint and log can be saved.

Then run the bash script as
```bash
bash train_didanet.sh /path/to/your/dataset
```

The experiments on PACS will start running. In the script `train_didanet.sh`, `$DATA` denotes the directory where datasets are located. For example, there should be a folder named `pacs` under such directory, i.e., `/path/to/your/dataset/pacs`. 
For the experiments on Digit-Five and DomainNet, modify the `--source-domains`, `--target-domains` and related config files, e.g., `--dataset-config-file configs/datasets/da/digit5.yaml` and `--config-file configs/trainers/da/didanet/digit5_dida_net.yaml`.

The detailed training settings are in the folder named `configs`, such as datasets (see `configs/datasets/da`), lr, optimizer etc. (see `configs/trainers/da/didanet`)

The most important files are under the folder of `dassl`: 
* Implementation of DIDA-Net can be found in `dassl/modeling/backbone/resnet_dida.py` (for PACS and DomainNet where ResNet is used as backbone) and`dassl/modeling/backbone/cnn_digit5_m3sda_dida.py` (for Digit-Five);

The trained DIDA-Net model on Sketch domain of PACS can be found [here](https://drive.google.com/drive/folders/1mLIkm-CburEhI27tT8CPkFK991qXaj6r?usp=sharing). This model gives 86.00% on the Sketch domain.

## Test

Similar to `train_didanet.sh`, testing can be done like this:

```
DATA=/root_path/to/your/dataset
CUDA_VISIBLE_DEVICES=0 python tools/train.py --root $DATA --trainer FixMatch \
 --source-domains cartoon art_painting photo --target-domains sketch \
 --dataset-config-file configs/datasets/da/pacs.yaml --config-file configs/trainers/da/didanet/pacs_dida_net.yaml \
 --output-dir output/didanet_pacs/sketch \
 --eval-only \
 --model-dir output/didanet_pacs/sketch \
 --load-epoch 20 \
 MODEL.BACKBONE.NAME resnet18_dida
```

## Citation

If you find this code useful, please consider citing the following paper:
```
@article{deng2022dynamic,
  title={Dynamic Instance Domain Adaptation},
  author={Deng, Zhongying and Zhou, Kaiyang and Li, Da and He, Junjun and Song, Yi-Zhe and Xiang, Tao},
  journal={arXiv preprint arXiv:2203.05028},
  year={2022}
}
```