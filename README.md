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

#### Model Ensemble
At the end of each epoch, the logits are saved to disk whenever the validation accuracy reaches a record high. The logits of different models are combined by averaging and normalizing. The prediction is given by the combined logits.

## Factors that improve predictive accuracy
1) Input image size before cropping. The larger the better.
2) Cosine annealing of the learning rate. The network can reach almost the highest predictive accuracy in a few epochs. Allowing the learning rate to restart helps the model escape from bad local minima.
3) Training the pretrained model in evaluation mode (model.eval()) results in a much higher accuracy. In traning mode, training accuracy for inception hovers around 80% (training and validation). In evalution mode, both accuracies get above 90%.

## Tuning the initial learning rate
The batch size is fixed to be 64. The validation accuracy for each model in the first tuning stage (all but the last layer frozen) is shown below. Each model is trained for 10 epochs.

| Learning Rate | resnext101_64x4d | inceptionv4 | inceptionresnetv2 |
| ------------- |:----------------|:-----------|:-----------------|
| 0.01          |0.9286            |0.9257       | 0.9091|
| 0.001         |0.9257            |0.9365       | 0.9247|  

