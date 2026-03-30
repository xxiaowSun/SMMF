# SMMF
This repository is the official PyTorch implementation of "Scalable Medical Multimodal Fusion via Symmetric Consistency Modeling".

## Installation
Code developed and tested in Python 3.9.0 using PyTorch 2.0.0. Please refer to their official websites for installation and setup.

Some major requirements are given below:

```python
numpy~=1.26.2
scikit-learn~=1.2.2
scipy~=1.10.1
torch~=2.0.0
torch-geometric~=2.0.4
nilearn~=0.10.1
```

Alternatively, you can choose to run the following code to install the required environment:
```shell
pip install -r requirements.txt
```

## Data

### ADHD-200
The pre-processed ADHD-200 data upload address is as follows:

#### Google Drive

Link：https://drive.google.com/drive/folders/19HoajzuBFIV0dVGLtWv_jx2c0qg9srX_?usp=sharing 


#### Baidu Cloud Drive

Link：https://pan.baidu.com/s/16sqz0fZvuSHHypMkLtikbA 

Password：qj12

### ABIDE
To fetch ABIDE public datasets.
```shell
python fetch_abide.py
```

## Training and Testing

Classification Task ( Default dataset is ADHD-200 )
```shell
python train_attention.py --train 1
```

