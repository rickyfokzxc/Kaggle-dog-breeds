# Kaggle-dog-breeds
### ```Kaggle dog breed identification challenge```

https://www.kaggle.com/c/dog-breed-identification

### Pretrained models in Pytorch
We used the pretrained model downloadable at the link below. At the time of writing this can be done by the command

```pip install pretrainedmodels```

https://github.com/Cadene/pretrained-models.pytorch

## Main Results

|Ensemble | Test accuracy|
| ------------- |:----------------|
|[resnext101_64x4d, inceptionv4, inceptionresnetv2]*2 + resnext101_64x4d | 94.2 |
|[resnext101_64x4d, inceptionresnetv2]*3 + resnext101_64x4d | 94.1|
|[resnext101_64x4d, inceptionv4, inceptionresnetv2]| 94.6 |

![Training curves](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/errors.png)

### Examples of correct classifications

Bull Mastiff             |  French bulldog | Kerry blue terrier | Walker Hound
:-------------------------:|:-------------------------:|:------:|:---:|
![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/correct/bull_mastiff.jpg)  |  ![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/correct/french_bulldog.jpg) |![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/correct/kerry_blue_terrier.jpg) | ![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/correct/walker_hound.jpg) |

## Examples of wrong classifications
We show the incorrectly classified dog breeds side by side with pictures of the correct breeds from Google search. These images are picked randomly. In most, if not all, cases the predicted breed is closer to the image found by Google. The algorithm is doing a better job than the person labeling the data.

Image ID             | Prediction | Label | 
:---------------------:|:-----:|:------:|
5353afdcaf0ad24196c1fdf1276ca644| African Hunting dog | Dhole |
![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/incorrect/african_hunting_dog_(dhole).jpg)| ![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/truth/African_hunting_dog.jpg)|![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/truth/dhole.jpg)|
6487f0f5886fa4e46ef422d8069acb8d | Boston Bull | Pug |
![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/incorrect/boston_bull_(pug).jpg) | ![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/truth/Boston%20bull.jpg) | ![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/truth/pug.jpg)|
97338b5e572b8154cf9aa1e7fc507c92 | Bernese Mountain Dog | Greater Swiss Mountain Dog |
![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/incorrect/bernese_mountain_dog_(greater_swiss_mountain_dog).jpg) |![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/truth/Bernese%20mountain%20dog.jpg)|![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/truth/greater%20swiss%20mountaindogs.jpg)|
83bcff6b55ee179a7c123fa6103c377a | Stratfordshire Bull Terrier | American Stratfordshire Terrier |
![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/incorrect/staffordshire_bullterrier_(american_staffordshire_terrier).jpg)|![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/truth/Stratfordshire%20bull%20terrier.jpg)|![](https://github.com/rickyfokzxc/Kaggle-dog-breeds/blob/master/truth/AmericanStaffordshireTerrierMountbrier.jpg)|



### Training strategy:
#### First stage
1) Load a pretrained model. Freezing all layers except the last linear layer.
2) Train a few epochs with cosine annealing on the learning rate, restarting every epoch.
3) Adam optimizer is used, initial learning rate is tuned to give the highest validation accuracy in this stage.

#### Second stage
1) Unfreeze earlier weights with learning rate 10 times smaller than the last linear layer. 
2) Learning rate cosine annealing now restarts slower and slower each time.

#### Parallelization
Training batches are split into mini-batches and sent to 4 GPUs. 


## Factors that improve predictive accuracy
1) Cosine annealing of the learning rate with restarts at every epoch. The network can reach almost the highest predictive accuracy in a few epochs. Allowing the learning rate to restart helps the model escape from bad local minima. This implementation of the cosine annealing rate is a proxy for the cyclical learning rate that has been shown to perform much better than conventional learning rate schedules, see https://arxiv.org/abs/1506.01186.
2) Training the pretrained model in evaluation mode (model.eval()) results in a much higher accuracy. In training mode, training accuracy for inception hovers around 80% (training and validation). In evaluation mode, both accuracies get above 90%.
3) Unfreezing the models and train with a small learning rate after fine-tuning helps resnext101_64x4d and inceptionresnetv2.

## Tuning the initial learning rate
The batch size is fixed to be 64. The validation accuracy for each model in the first training stage (all but the last layer frozen) is shown below. Each model is trained for 10 epochs. The chosen initial learning rate is 0.0003.

| Learning Rate |   inceptionv4   | inceptionresnetv2 | resnext101_64x4d|
| ------------- |:----------------|:------------------|:----------------|
| 0.01          |0.9286           |0.9257             | 0.9091          |
| 0.001         |0.9257           |0.9365             | 0.9247          |  
| 0.0001 |0.9335| 0.9335| 0.9218|


#### Model Ensembles
An average of model ensembles are known to outperform a single model. The code here is an incremental implementation of model averaging. The models are trained one by one over 4 GPUs using the same training and validation sets. For instance, a resnet, say, is trained on 4 GPUs using data parallelism before the next model is trained. After training each model, the classification probabilities for the validation set is sent to the CPU to save GPU memory. Only one copy of the probabilties is saved at any time. The probabilties are simply updated by summing with those from a new model. The CPU only stores one copy of the probability at all times and this allows the code to calculate an arbitrary large ensemble. Updating the class probabilities by summing rather than saving one set for each model does not lose information. Because the ensemble average prediction is obtained by taking the maximum of the class probabilities for each class / image.

The validation probability sent to the CPU is the one corresponding to the highest validation accuracy during training. This can be seen as using an early stopping strategy to prevent overfitting for each model.

