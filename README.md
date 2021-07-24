# Neural Architecture Search for Image Super-Resolution Using Densely Connected Search Space: DeCoNAS

Basic implementation of MBNAS from Multi-Branch Neural Architecture Search for Lightweight Image Super-resolution.

- Uses Tensorflow to define and train the child network / Controller network.
- `Controller` manages the training and evaluation of the Controller RNN
- `ChildNetwork` handles the training and evaluation of the Child network

# Usage
At first, you should download the training dataset(DIV2K) and test datasets(Set5, Set14, B100, Urban100).

For full training details, please see `train.py`.
The metrics and results can be generated with `evaluate.py`

You can search promising networks by MBNAS

$ python ./src/DIV2K/train.py --output_dir='./outputs/x2_search' --num_epochs=500 --controller_training=True
```
Train DeCoNASNet searched by DeCoNAS with, 
```shell
$ python ./src/DIV2K/train.py --output_dir='./outputs/x2_scratch' --fine_tune=False --child_fixed_arc='1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1 1 0 0 0 0 1' --controller_training=False
```

After training DeCoNANet from scratch, finetune DeCoNASNet with, 
```shell
$ python ./src/DIV2K/train.py --output_dir='./outputs/x2_finetune' --checkpoint='./outputs/x2/model.ckpt-931000' --fine_tune=True --child_fixed_arc_Low='0 0 1 2' --child_fixed_arc_Mid='0 0 1 2' --child_fixed_arc_High='2 0 1 2' --controller_training=False
```

Finally, evaluate DeCoNASNet with,
```shell
$ python ./src/DIV2K/evaluate.py 
```

# Result
We construct MBNASNet with 4 MSBs, and each MSB has 4 layers. 

The sequence of MBNASNet for x2 scale image super-resolution is 
```shell
child_fixed_arc_Low='0 0 1 2'
child_fixed_arc_Mid='0 0 1 2'
child_fixed_arc_High='2 0 1 2'
```
For x2 scale super-resolution task, we evaluated the performance(PSNR and SSIM) of our MBNASNet on four datasets(Set5, Set14, B100, Urban100).


<img src="https://github.com/Junem360/MBNAS/blob/master/images/MBNAS_result.png" height=100% width=100%>


# Requirements
- Tensorflow-gpu >= 1.13
- scipy >= 1.5.0
- numpy >= 1.18
- OpenCV >= 3.4.0

# Acknowledgements
We referred the codes of ENAS([melodyguan/enas](https://github.com/melodyguan/enas)).
