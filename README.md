# Kaggle-dog-breeds
### ```Kaggle dog breed identification challenge```

https://www.kaggle.com/c/dog-breed-identification

### Pretrained models in Pytorch

https://github.com/Cadene/pretrained-models.pytorch

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
1) Cosine annealing of the learning rate. The network can reach almost the highest predictive accuracy in a few epochs. Allowing the learning rate to restart helps the model escape from bad local minima.
2) Training the pretrained model in evaluation mode (model.eval()) results in a much higher accuracy. In traning mode, training accuracy for inception hovers around 80% (training and validation). In evalution mode, both accuracies get above 90%.
3) Unfreezing the models and train with a small learning rate after fine-tuning helps resnext101_64x4d and inceptionresnetv2.

## Tuning the initial learning rate
The batch size is fixed to be 64. The validation accuracy for each model in the first tuning stage (all but the last layer frozen) is shown below. Each model is trained for 10 epochs. 

| Learning Rate |   inceptionv4   | inceptionresnetv2 | resnext101_64x4d|
| ------------- |:----------------|:------------------|:----------------|
| 0.01          |0.9286           |0.9257             | 0.9091          |
| 0.001         |0.9257           |0.9365             | 0.9247          |  
| 0.0001 |0.9335| 0.9335| 0.9218|


#### Model Ensembles
An average of model ensembles are known to outperform a single model. The code here is an incremental implementation of model averaging. The models are trained one by one over 4 GPUs using the same training and validation sets. For instance, a resnet, say, is trained on 4 GPUs using data parallelism before the next model is trained. After training each model, the classification probabilities for the validation set is sent to the CPU to save GPU memory. Only one copy of the probabilties is saved at any time. The probabilties are simply updated by summing with those from a new model. The CPU only stores one copy of the probability at all times and this allows the code to calculate an arbitrary large ensemble. Updating the class probabilities by summing rather than saving one set for each model does not lose information. Because the ensemble average prediction is obtained by taking the maximum of the class probabilities for each class / image.

The validation probability sent to the CPU is the one corresponding to the highest validation accuracy during training. This can be seen as using an early stopping strategy to prevent overfitting for each model.
