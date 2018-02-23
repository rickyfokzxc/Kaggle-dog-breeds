import numpy as np
import pandas as pd
import torch
import copy
import pretrainedmodels
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.label import LabelEncoder
from torch import nn,optim
from torch.optim import lr_scheduler

from utils import *

BATCH_SIZE = 64 
EPOCHS = 1 
EPOCHS_2 = 1 
LEARNING_RATE = 0.001
CUDA = torch.cuda.is_available()

### Loading data ####
data_train_csv = pd.read_csv('data/labels.csv')
filenames = data_train_csv.id.values
le = LabelEncoder()
labels = le.fit_transform(data_train_csv.breed)
num_classes = len(le.classes_)

### Splitting data set into training and validation data ####
filenames_train , filenames_val ,labels_train, labels_val =\
    train_test_split(filenames,labels,test_size=0.1,stratify=labels)

##Load from pretrained model ####


#model_names = ['resnext101_64x4d','inceptionv4','inceptionresnetv2']*2
model_names = ['resnet18','resnet18']

model_sum_best_probs = 0
best_ensemble_val_acc = 0
best_model =0 
#looping over all models
for model_idx in range(len(model_names)):
    model_name = model_names[model_idx]

    ####Model settings for loading data ####
    model_classes = pretrainedmodels.pretrained_settings[model_name]['imagenet']['num_classes']
    model_ft = pretrainedmodels.__dict__[model_name](num_classes=model_classes, pretrained='imagenet')
    mu = model_ft.mean
    sigma = model_ft.std 
    image_size = model_ft.input_size[1]
    
    ### set up data loaders for pytorch, shuffling training set each epoch ####
    dog_train = get_train_dataset(filenames_train,labels_train,BATCH_SIZE, mu, sigma, image_size,1)
    dog_val = get_val_dataset(filenames_val,labels_val,BATCH_SIZE, mu, sigma, image_size)
 
    
    print(model_name)
    
    #Freezing weights
    for param in model_ft.parameters():
        param.requires_grad = False
    
    #Replacing the last linear layer, unfrozen weights by default
    num_ftrs = model_ft.last_linear.in_features
    model_ft.last_linear  = nn.Linear(num_ftrs, num_classes)
    
    #Send model to GPU#
    if CUDA:
        model_ft = model_ft.cuda()
    
    #Cross-entropy loss function
    criterion_train = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss(size_average=False)
    
    ## Optimizer ##
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()),lr=LEARNING_RATE)
    
    ## mini-batch data parallelism using 4 GPUs
    print('Stage 1:')
    net = torch.nn.DataParallel(model_ft, device_ids=[0,1,2,3])
    state = {'val_acc':[],'best_val_acc':0}
    state,best_logits = train(EPOCHS,dog_train,dog_val,model_name,optimizer,net,criterion_train,criterion_val,state,CUDA)

    ########### STAGE 1 Finishes, Stage 2 below ####
    #### Setting up differential learning rates ####
    
    #Optimizer, have to filter out the frozen weight
    ignored_params = list(map(id, model_ft.last_linear.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model_ft.parameters())
    for param in model_ft.parameters():
        param.requires_grad = True 
    optimizer2 = torch.optim.SGD([
                {'params': base_params},
                {'params': model_ft.last_linear.parameters(), 'lr': LEARNING_RATE/10}
            ], lr=LEARNING_RATE/100,momentum=0.9)
    
    print('Stage 2:')
    
    #Replicating model onto GPUs again
    net = torch.nn.DataParallel(model_ft, device_ids=[0,1,2,3])
    
    #Stage 2 training
    state,best_logits_2 = train(EPOCHS_2,dog_train,dog_val,model_name,optimizer2,net,criterion_train,criterion_val,state,CUDA)
    if len(best_logits_2) != 0:
        best_logits = best_logits_2

    print('best validation accuracy: {:.4f}'.format(state['best_val_acc']))
    best_logits = best_logits.T - np.max(best_logits,axis=1)
    best_probs = np.exp(best_logits)
    best_probs /= np.sum(best_probs,0)
    best_probs = best_probs.T


    model_sum_best_probs += best_probs

    predictions = np.argmax(model_sum_best_probs,axis=1)
    correct = np.sum(predictions==labels_val)
    accuracy = correct / len(dog_val.dataset)
    print('Current Ensemble predictive accuracy:i ' + str(accuracy))

    if best_ensemble_val_acc < accuracy:
        best_ensemble_val_acc = accuracy
        best_model = model_idx+1
    print('Record best ensemble predictive accuracy: ' + str(best_ensemble_val_acc))
    print('Record best model ensemble: ' + str(model_names[:best_model]) +'\n')
    #print(best_ensemble_val_acc)

predictions = np.argmax(model_sum_best_probs,axis=1)
correct = np.sum(predictions==labels_val)
accuracy = correct / len(dog_val.dataset)
correct_filenames = filenames_val[predictions==labels_val]
correct_labels = le.inverse_transform(labels_val[predictions==labels_val])

print('Correct classifications:')
print('file name, dog breed')
print(np.vstack((correct_filenames[:10], correct_labels[:10])).T)

incorrect_filenames = filenames_val[predictions!=labels_val]
predicted_labels = le.inverse_transform(predictions[predictions!=labels_val][:10])
true_labels = le.inverse_transform(labels_val[predictions!=labels_val][:10])
print('incorrect filenames, predictions, true labels')

dummy = np.vstack((incorrect_filenames[:10], predicted_labels[:10]))
print(np.vstack((dummy,true_labels)))
