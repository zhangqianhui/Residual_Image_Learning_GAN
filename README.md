# Residual_Image_Learning_GAN
Tensorflow Implement of Paper [Learning Residual Images for Face Attribute Manipulation](https://arxiv.org/abs/1612.05363), which has been accepted in CVPR 2017. We need implement this paper and compare our results with that, because Auther does't public their code. I think this paper is a good paper and they give a perfect idea for facial visual manipulating with images residual learning. This difference with original paper is that this implements use Instance_norm instead of batch_normal. You can adjust important weights for getting more perfect results.

![image](imgs/paper_caption.PNG)


## Prerequisites

+ [Tensorflow](http://tensorflow.org/)


## Datasets
We use the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets. The code will crop and resize images to 128x128.

~~~
---------------------------------------------
The training data folder should look like : 
<train_data_root>
                |--image1
                |--image2
                |--image3
                |--image4...
---------------------------------------------
~~~

## Running

    $ python main.py --IMAGE_PATH /home/?/data/celebA/
 
## Experiments

The man face:

![](imgs/m.png)

The residual face:

![](imgs/m_r.png)

Man-to-Woman Face:

![](imgs/m_wm.png)

--------------------

The woman face:

![](imgs/wm.png)

The residual face:

![](imgs/wm_r.png)

Woman-to-Man Face:

![](imgs/wm_m.png)



## Acknowledgement
+ [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
