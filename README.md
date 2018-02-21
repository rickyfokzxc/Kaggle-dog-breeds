# Kaggle-dog-breeds
```
### Kaggle dog breed identification challenge
```
https://www.kaggle.com/c/dog-breed-identification

### Pretrained models in Pytorch

https://github.com/Cadene/pretrained-models.pytorch

### Training strategy:
#### First stage
1) Load a pretrained model. Freezing all layers except the last linear layer.
2) Train a few epochs with cosine annealing on the learning rate, restarting every epoch.
3) Adam optimizer is used, initial learning rate is tuned to give the highest validation accuracy in this stage.

#### Second stage
1) Unfreeze earlier weights with learning rate 1000 times smaller than the last linear layer. 
