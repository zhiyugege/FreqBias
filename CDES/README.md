## Bias Hypothesis Ι : Spectral Density

<div align="center">
    <img src="Image/CDES.png" alt="image-20220922143512377" style="zoom:100%;" />
    <p> Convolutional Density Enhancement Strategy (CDES) </p>
</div>

**Hypothesize**: Spectral density can serve as an explanation for frequency bias in image classification tasks. We propose a framework called **Convolutional Density Enhancement Strategy (CDES)** to modify the spectral density of natural images.

## Requirements

1. [Python](https://www.python.org/)
2. [Pytorch](https://pytorch.org/)
3. [Torchvision](https://pytorch.org/vision/stable/index.html)
4. [Robustness](https://github.com/MadryLab/robustness)

## Generate toy dataset of strong&weak class related

**Train toy kernels**

Since simply adding noise is difficult to modify the spectral density as we expect, we first propose to perform convolution operations on original images with a set of trainable convolution filters.

```
cd SpectralDensity
python train_toy_models.py --number 16 --kernel_size 7 --cuda_index 0 --weight_decay 1e-4 --momentum 0.9 --epochs 20 --lr 0.01 --step_lr 5 --step_lr_gamma 0.1 --exp_name SCR --batch_size 100 --workers 4
```
* **--number** (int): number of convolutional kernels, default 16.
* **--kernel_size** (int): size of convolutional kernels, default 7.
* **--exp_name** (str): The type of generate dataset. default [SCR, WCR].

**Generate toy datasets**

$\mathcal{L} = \sum\limits_{k=0}^{H/2-1} || AI_{k}(X^{Conv})-AI_{k}(X^{Target}) ||_{2}^{2} + || X^{(Conv)} -X ||_{2}^{2}$

```
cd SpectralDensity
python generate_toy_dataset.py --exp_name SCR --cuda_index 0
```
* **--cuda_index** (int): your gpu ids, default 0.
* **--exp_name** (str): The type of generate dataset, default [SCR, WCR].

**Visualization of SCR-dataset(take class 0 as an example)**
<div align="center">
    <img src="Image/SCR/0.png" alt="image-20220922143512377" style="zoom:100%;" />
</div>

**Visualization of WCR-dataset**
<div align="center">
    <img src="Image/WCR/all.png" alt="image-20220922143512377" style="zoom:100%;" />
</div>


## Feature Discrimination Observations

<div align="center">
    <img src="Image/Result.png" alt="image-20220922143512377" style="zoom:100%;" />
</div>

## Learning Priority Observations

<div align="center">
    <img src="Image/Result2.png" alt="image-20220922143512377" style="zoom:100%;" />
</div>


For any questions, welcome to contact {**zylin**, **yifeigao**}@bjtu.edu.cn

## References

The code refers to https://github.com/MadryLab/robustness and https://github.com/cc-hpc-itwm/UpConv

## Citation

If you find this repo useful for your research, please consider citing the paper
```
@inproceedings{ijcai2022p101,
  title     = {Investigating and Explaining the Frequency Bias in Image Classification},
  author    = {Lin, Zhiyu and Gao, Yifei and Sang, Jitao},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {717--723},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/101},
  url       = {https://doi.org/10.24963/ijcai.2022/101},
}
```
