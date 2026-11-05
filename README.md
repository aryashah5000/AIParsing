# AIParsing
This respository includes a PyTorch implementation of the TIP2022 paper AIParsing:Anchor-Free Instance-Level Human Parsing. 

## Requirements:<br>
python 3.7<br>
PyTorch 1.7.1<br>
cuda 10.1 <br>

The detail environment can be find in AIParsing_env.yaml.

## Compiling<br>

Apex install:

git clone https://github.com/NVIDIA/apex <br>
cd apex <br>
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ <br>
or python setup.py build develop <br>

cd models<br>
sh make.sh<br>

cd ..<br>
LDSHARED="g++ -shared -static-libstdc++" python setup.py build develop<br>


## Dataset and pretrained model<br>
Plesae download [CIHP](https://drive.google.com/open?id=1OLBd23ufm6CU8CZmLEYMdF-x2b8mRgxV) dataset<br>

[Well-trained models on the CIHP and LV-MHP datasets](https://tjueducn-my.sharepoint.com/:f:/g/personal/zhangsanyi_tju_edu_cn/EhLaPw8f2nBAsosG1FaOZ4MBkmcQkTy61SrvbOZ7jR9xHA?e=wbbmgq) (MM:aiparsing)<br>


## Evaluation<br>
bash test_CIHP_R50_75epoch.sh<br>

## Training<br>
bash train_CIHP_R50_75epoch.sh<br>

The launchers require the pinned Python 3.7/PyTorch 1.7.1 environment. The
current shell must not be the default `base` environment:

```bash
conda env create -f AIParsing_env.yaml
conda activate segformerhuman
```

If the environment already exists, use `conda activate segformerhuman` before
running the commands above. If creation stopped while building `mmcv-full` or
`pycocotools`, those packages were the failure: AIParsing does not use mmcv,
and the environment file now installs pycocotools from conda instead of
compiling an obsolete pip source archive. Repair the existing environment with:

```bash
conda install -n segformerhuman -c conda-forge pycocotools=2.0.2
conda activate segformerhuman
```

The launchers now stop with an actionable message instead of cascading shell
errors when the environment is wrong.

## Reproducing CIHP and grouped mIoU

This repository contains the AIParsing implementation; it does not contain
UniParser or InclusiveHuman-10k annotations. The provided CIHP archive has the
standard 20 CIHP labels and therefore cannot produce disability-related
mIoU. It can still be used to reproduce the CIHP baseline:

```bash
bash tools/setup_CIHP.sh ./CIHP-20261030T004640Z-1-001/CIHP
bash train_CIHP_R50_75epoch.sh
bash test_CIHP_R50_75epoch.sh
```

For InclusiveHuman-10k, point `--ground-truth` and `--annotation-file` at that
dataset and pass its exact taxonomy names. The evaluator reports the overall,
disability-related, and common-class mIoU from the same pixel confusion matrix:

```bash
python tools/evaluate_grouped_miou.py \
  --predictions checkpoints/CIHP_R50_75epoch/test/parsing_predict \
  --ground-truth datasets/CIHP/val_seg \
  --annotation-file datasets/CIHP/annotations/CIHP_val.json \
  --common-classes Face \
  --disability-classes Prosthetic-arm
```

The final example intentionally fails on CIHP because `Prosthetic-arm` is not
a CIHP class; replace both groups with labels from InclusiveHuman-10k.

### AMD ROCm

The legacy environment is CUDA-only. For an AMD GPU, install the host ROCm
driver/runtime first, then create the modern environment:

```bash
bash tools/setup_rocm.sh
conda activate aiparsing-rocm
bash make.sh
LDSHARED="g++ -shared -static-libstdc++" python setup.py build_ext --inplace
```

The ROCm environment removes the old Apex dependency and uses device-aware
PyTorch tensor creation. The custom `.cu` extensions still require a ROCm
build check: current PyTorch no longer ships the legacy THC headers used by
AIParsing's deformable convolution, deformable pooling, and focal-loss
kernels. Until those kernels are migrated to current ATen/HIP APIs, the
ROCm environment is a porting baseline, not a claim of full training
compatibility.

On Ubuntu, the HIP build also needs the system hipBLASLt headers:

```bash
sudo apt update
sudo apt install libhipblaslt-dev ninja-build
```

## Acknowledgment  
This project is created based on the [Parsing R-CNN](https://github.com/soeaver/Parsing-R-CNN), [CenterMask](https://github.com/youngwanLEE/CenterMask)

If this code is helpful for your research, please cite the following paper:

<p>
@article{AIParsing2022,<br>
&emsp;  title={AIParsing: Anchor-Free Instance-Level Human Parsing},<br>
&emsp;   author={Sanyi Zhang, Xiaochun Cao, Guo-jun Qi, Zhanjie Song, Jie Zhou},<br>
&emsp;       journal={IEEE Transactions on Image Processing (TIP)},<br>
&emsp;       year={2022},<br>
&emsp;       volume={31},<br>
&emsp;       pages={5599-5612}<br> 
}
  </p>
