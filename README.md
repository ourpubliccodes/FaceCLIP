# FaceCLIP 
Pytorch implementation for Facial Expression Genaration from Text with CLIP. The goal is to generate realistic facial expression images from pure text, and also allow the user to sematiclly edit basic facial attributes using natural language descriptions to present complex attribute combinations, in one framework. 

### Overview
<img src="archi.png" width="900px" height="248px"/>

**[Facial Expression Genaration from Text with CLIP].**  
Wenwen Fu, [Wenjuan Gong](https://www.wenjuangong.com/), Chenyang Yu, Wei Wang, Jordi Gonzalez.

### Data

1. Download the original data for [AffectNet](http://mohammadmahoor.com/affectnet/) and [RAF-DB](http://www.whdeng.cn/raf/model1.html), and save both into `data/`
2. Preprocess the facial images from [AffectNet] dataset and save the images to `data/affectnet/images/`
3. Preprocess the facial images from [RAF-DB] dataset and save the images to `data/affectnet/images/`
4. Based on
     ```
    python crop_alignface/align_images.py
    ```

### Training
All code was developed and tested on Linux with Python 3.7 (Anaconda) and torch '1.10.1+cu111'.

#### [DAMSM](https://github.com/taoxugit/AttnGAN) model includes text encoder and image encoder
- Pre-train DAMSM model for bird dataset:
```
python pretrain_DAMSM.py --cfg cfg/DAMSM/.yml --gpu 0
```
- Pre-train DAMSM model for coco dataset: 
```
python pretrain_DAMSM.py --cfg cfg/DAMSM/coco.yml --gpu 1
```
#### ControlGAN model 
- Train ControlGAN model for bird dataset:
```
python main.py --cfg cfg/train_bird.yml --gpu 2
```
- Train ControlGAN model for coco dataset: 
```
python main.py --cfg cfg/train_coco.yml --gpu 3
```

`*.yml` files include configuration for training and testing.


#### Pretrained DAMSM Model
- [DAMSM for bird](https://drive.google.com/file/d/1dbdCgaYr3z80OVvISTbScSy5eOSqJVxv/view?usp=sharing). Download and save it to `DAMSMencoders/`
- [DAMSM for coco](https://drive.google.com/file/d/1k8FsZFQrrye4Ght1IVeuphFMhgFwOxTx/view?usp=sharing). Download and save it to `DAMSMencoders/`
#### Pretrained ControlGAN Model
- [ControlGAN for bird](https://drive.google.com/file/d/1g1Kx5-hUXfJOGlw2YK3oVa5C9IoQpnA_/view?usp=sharing). Download and save it to `models/`

### Testing
- Test ControlGAN model for bird dataset:
```
python main.py --cfg cfg/eval_bird.yml --gpu 4
```
- Test ControlGAN model for coco dataset: 
```
python main.py --cfg cfg/eval_coco.yml --gpu 5
```
### Evaluation

- To generate images for all captions in the testing dataset, change B_VALIDATION to `True` in the eval_*.yml. 
- Inception Score for bird dataset: [StackGAN-inception-model](https://github.com/hanzhanggit/StackGAN-inception-model).
- Inception Score for coco dataset: [improved-gan/inception_score](https://github.com/openai/improved-gan/tree/master/inception_score).

### Code Structure
- code/main.py: the entry point for training and testing.
- code/trainer.py: creates the main networks, harnesses and reports the progress of training.
- code/model.py: defines the architecture of ControlGAN.
- code/attention.py: defines the spatial and channel-wise attentions.
- code/VGGFeatureLoss.py: defines the architecture of the VGG-16.
- code/datasets.py: defines the class for loading images and captions.
- code/pretrain_DAMSM.py: creates the text and image encoders, harnesses and reports the progress of training. 
- code/miscc/losses.py: defines and computes the losses.
- code/miscc/config.py: creates the option list.
- code/miscc/utils.py: additional functions.

### Citation

If you find this useful for your research, please use the following.

```
@article{li2019control,
  title={Controllable text-to-image generation},
  author={Li, Bowen and Qi, Xiaojuan and Lukasiewicz, Thomas and H.~S.~Torr, Philip},
  journal={arXiv preprint arXiv:1909.07083},
  year={2019}
}
```

### Acknowledgements
This code borrows heavily from [AttnGAN](https://github.com/taoxugit/AttnGAN) repository. Many thanks.
