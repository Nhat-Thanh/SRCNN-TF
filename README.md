# [TensorFlow] Super-Resolution CNN

Implementation of SRCNN model in **Image Super-Resolution using Deep Convolutional Network** paper with Tensorflow 2x. We used Adam with optimize tuned hyperparameters instead of SGD + Momentum. W implement 3 models in the paper, SRCNN-915, SRCNN-935, SRCNN-955.


## Train
You run this command to begin the training:
```
python train.py  --steps=1000000                   \
                 --architecture="915"              \
                 --batch_size=128                  \
                 --save-best-only=1                \
                 --save-every=1000                 \
                 --ckpt-dir="checkpoint/SRCNN915"  
```
- **architecture** accepts 3 values: 915, 935, 955. They are orders of kernel size.
- **save-best-only**: if it is **0**, model weights will be saved every **save-every** steps.


**NOTE**: if you want to retrain a model, you can delete all files in **checkpoint** directory. Your checkpoint will be saved when above command finishs and can be used for next times, so you can train this model on Colab without taking care of GPU limit.

We trained 3 models in 1000000 steps, you can get them here:
- [SRCNN-915.h5](checkpoint/SRCNN915/SRCNN-915.h5)
- [SRCNN-935.h5](checkpoint/SRCNN935/SRCNN-935.h5)
- [SRCNN-955.h5](checkpoint/SRCNN955/SRCNN-955.h5)


## Demo 
After Training, you can test the model with this command, the result is the **sr.png**.
```
python demo.py --image-path="dataset/test2.png"                \
               --architecture="915"                            \
               --ckpt-path="checkpoint/SRCNN915/SRCNN-915.h5"  \
               --scale=2
```

We evaluated the model with Set5 and Set14 dataset by PSNR:

<div align="center">

| Methods               | Set5 x2 | Set5 x3 | Set5 x4 | Set14 x2 | Set14 x3 | Set14 x4 |
|:---------------------:|:-------:|:-------:|:-------:|:--------:|:--------:|:--------:|
| SRCNN-915             | 35.8345 |	34.3566 | 31.9265 |	32.7506  | 31.3271  | 29.5111  |
| SRCNN-935 			| 36.3159 |	34.4074 | 31.9210 |	33.0301  | 31.3659  | 29.5404  |
| SRCNN-955 			| 36.0525 | 34.3292 | 32.9078 |	32.9502  | 31.2873  | 29.5225  |

</div>

<div align="center">
  <img src="./README/example.png" width="1000">  
  <p><strong>Bicubic x2 (left), SRCNN-935 x2 (right).</strong></p>
</div>
Source: game ZingSpeed Mobile

## References
- Image Super-Resolution Using Deep Convolutional Networks: https://arxiv.org/pdf/1501.00092.pdf
- SRCNN Matlab code: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
- T91: http://vllab.ucmerced.edu/wlai24/LapSRN/
- Set5, Set14: https://github.com/jbhuang0604/SelfExSR#comparison-with-the-state-of-the-art