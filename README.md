# U-Net with TensorFlow 2.1
This project is my own custom implementation of U-Net for image segmentation using TensorFlow v2. Feel free to play around with the code and let me know of any problems.
The file names of the project are self explanatory and the codes have comments for more details.
I have also provided some examples of running the model in images directory.

## The Training Loss vs. Test Loss
The code generates this graph in the end. However, I have run this code and I provide my own generated images:
![Training Loss vs. Test Loss across epoches](https://github.com/k-timy/U-Net-TF/blob/master/images/training%20and%20validation%20loss.png)

## Some Segmented Samples:
Here are some of the predicted samples and their true segmetation masks. Notice that how close some generated masks are to the true masks and how different are others. I posted a few of both correctly and incorrectly segmented samples:

![Sample 1](https://github.com/k-timy/U-Net-TF/blob/master/images/s1.png)
![Sample 2](https://github.com/k-timy/U-Net-TF/blob/master/images/s2.png)
![Sample 3](https://github.com/k-timy/U-Net-TF/blob/master/images/s3.png)
![Sample 4](https://github.com/k-timy/U-Net-TF/blob/master/images/s4.png)
![Sample 5](https://github.com/k-timy/U-Net-TF/blob/master/images/s5.png)
![Sample 6](https://github.com/k-timy/U-Net-TF/blob/master/images/s6.png)

## Discussion
As a matter of fact, there are more effient and performant state of the art methods for image segmentation. Also, I have not applied data augmentation methods that were implemented in the U-Net paper. However, I decided to make this code public because it was one of my first steps in getting my hands on the image segmentation and I had the intention to do this a while ago. Though I did not have enough time or I kept procastinating it :D.
During the nation-wide quarantine due to the COVID-19 outbreak in the USA, I found some free time to publish some piece of my code on my github account.

## References
I implemented this code based on the original paper and the [example](https://www.tensorflow.org/tutorials/images/segmentation) available on the website of TensorFlow.
Click on [this link](https://arxiv.org/abs/1505.04597) to view the original paper that proposed U-Net architecture.
