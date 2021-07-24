# Multi-Branch Neural Architecture Search for Lightweight Image Super-resolution

Basic implementation of MBNAS from Multi-Branch Neural Architecture Search for Lightweight Image Super-resolution.

- Uses Tensorflow to define and train the child network / Controller network.
- `Controller` manages the training and evaluation of the Controller RNN
- `ChildNetwork` handles the training and evaluation of the Child network

# Usage
At first, you should download the training dataset(DIV2K) and test datasets(Set5, Set14, B100, Urban100).

For full training details, please see `train.py`.
The metrics and results can be generated with `evaluate.py`

You can search promising networks by MBNAS

```shell
$ python ./src/DIV2K/train.py --output_dir='./outputs/x2_search' --num_epochs=500 --controller_training=True
```
Train MBNASNet searched by MBNAS with, 
```shell
$ python ./src/DIV2K/train.py --output_dir='./outputs/x2_scratch' --fine_tune=False --child_fixed_arc_Low='0 0 1 2' --child_fixed_arc_Mid='0 0 1 2' --child_fixed_arc_High='2 0 1 2' --controller_training=False
```

After training MBNASNet from scratch, finetune MBNASNet with, 
```shell
$ python ./src/DIV2K/train.py --output_dir='./outputs/x2_finetune' --checkpoint='./outputs/x2_scratch/model.ckpt-1000000' --fine_tune=True --child_fixed_arc_Low='0 0 1 2' --child_fixed_arc_Mid='0 0 1 2' --child_fixed_arc_High='2 0 1 2' --controller_training=False
```

Finally, evaluate MBNASNet with,
```shell
$ python ./src/DIV2K/evaluate.py 
```

You can search and train for x3 image super-resolution task by adding argument --child_upsample_size=3

# Result
We construct MBNASNet with 4 MSBs, and each MSB has 4 layers. 

The sequence of MBNASNet for x2 scale image super-resolution is 
```shell
child_fixed_arc_Low='0 0 1 2'
child_fixed_arc_Mid='0 0 1 2'
child_fixed_arc_High='2 0 1 2'
```
For x2 scale super-resolution task, we evaluated the performance(PSNR and SSIM) of our MBNASNet on four datasets(Set5, Set14, B100, Urban100).

<img src="https://github.com/Junem360/MBNAS/blob/main/images/quantitative_result.png" height=70% width=70%>
<img src="https://github.com/Junem360/MBNAS/blob/main/images/qualitative_result.png" height=70% width=70% >


# Requirements
- Tensorflow-gpu >= 1.13
- scipy >= 1.5.0
- numpy >= 1.18
- OpenCV >= 3.4.0

# Acknowledgements
We referred the codes of ENAS([melodyguan/enas](https://github.com/melodyguan/enas)).
