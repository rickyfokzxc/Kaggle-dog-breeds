from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import torch
import numpy as np
from torch.optim import lr_scheduler

class DogDataset(Dataset):

    def __init__(self, filenames,labels,root_dir,transform=None):
        assert len(filenames)==len(labels)
        self.filenames = filenames
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        label = self.labels[item]
        img_name = os.path.join(self.root_dir,self.filenames[item]+'.jpg')

        with Image.open(img_name) as f:
            img = f.convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.labels is None:
            return img,self.filenames[item]
        else:
            return img,self.labels[item]


def get_train_dataset(filenames,labels,batch_size,m,s,image_size,shuffle,rootdir='data/train'):
    composed = transforms.Compose([
                                   transforms.Resize(450),
                                   transforms.RandomResizedCrop(image_size, scale=(0.75, 1)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=m,std=s)
                                   ])

    dog_trainset = DogDataset(filenames, labels, transform=composed,root_dir=rootdir)
    dog_train = DataLoader(dog_trainset, batch_size, shuffle,pin_memory=True,num_workers=4)
    return dog_train

def get_val_dataset(filenames,labels,batch_size,m,s,image_size,rootdir='data/train'):
    composed = transforms.Compose([
                                   transforms.Resize(450),
                                   transforms.RandomResizedCrop(image_size, scale=(0.75, 1)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=m,std=s)
                                   ])
    dog_valset = DogDataset(filenames, labels, transform=composed,root_dir=rootdir)
    dog_val = DataLoader(dog_valset, batch_size, False,pin_memory=True,num_workers=4)
    return dog_val
'''
def get_test_dataset(filenames,labels,batch_size,m,s,rootdir='data/train'):
    composed = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                   transforms.Normalize(mean=m,std=s)
                                   ])
    dog_testset = DogDataset(filenames,labels,transform=composed,root_dir=rootdir)
    dog_test = DataLoader(dog_testset, batch_size, False,pin_memory=True,num_workers=4)
    return dog_test
'''

def train_epoch(net,scheduler, data_iter,criterion,optimizer,use_cuda,mode='eval'):
    
    if mode == 'eval':
        net.eval() #using BN in eval mode during training
    else:
        net.train()
    correct = 0
    for batch_idx,(x,y) in enumerate(data_iter):
        if use_cuda:
            x,y = x.cuda(),y.cuda()
        x = Variable(x)
        y = Variable(y)
    
        #zeroing gradients
        optimizer.zero_grad()

        logits = net(x)#128x120
        loss = criterion(logits,y)
        loss.backward()
        optimizer.step()
        scheduler.step() #scheduler step inside batch loop to set period = 1 epoch 

        cur_correct =  torch.sum(logits.data.max(dim=-1)[1]==y.data)
        cur_accuracy = cur_correct/len(x)
        correct += cur_correct

    accuracy = correct / len(data_iter.dataset)
    return accuracy

def val_epoch(net,data_iter,criterion,use_cuda,model_name='',best_val=0,num_classes=120):
    test_loss = 0
    correct = 0
    start_idx =0
    net.eval()
    #logits_cpu = np.array([]).reshape(0,num_classes)
    for batch_idx,(x,y) in enumerate(data_iter):
        if use_cuda:
            x,y = x.cuda(),y.cuda()
        x = Variable(x)
        y = Variable(y)
        logits = net(x)
        loss = criterion(logits,y)
        test_loss += loss.data[0]
        correct += torch.sum((logits.data.max(dim=-1)[1])==y.data)
        if start_idx == 0:
            logits_cpu = logits.data.cpu()
            start_idx = 1
        else:
            logits_cpu = torch.cat((logits_cpu,logits.data.cpu()),0)
            
    test_loss /= len(data_iter.dataset)
    accuracy = correct / len(data_iter.dataset)
    logits_cpu = logits_cpu.numpy()
    return accuracy,logits_cpu

def train(EPOCHS,dog_train,dog_val,model_name,optimizer,net,criterion_train,criterion_val,state,CUDA):
    correct = 0 
    start_idx = 0
    mult_len = 1 #scale factor of scheduler period
    best_logits = []
    for epoch in range(1,EPOCHS+1):
        if epoch == mult_len:
            Tmax = mult_len*len(dog_train)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, Tmax, eta_min=1e-9)
            mult_len *= 2 #period multiplies by 2 every period
        train_acc = train_epoch(net,scheduler,dog_train,criterion_train,optimizer,CUDA)
        val_acc,logits_cpu = val_epoch(net,dog_val,criterion_val,CUDA)
        print('(Training accuracy,  Validation Accuracy) = ({:.4f},{:.4f})'.format(train_acc, val_acc))
        state['val_acc'].append(val_acc)
        if val_acc > state['best_val_acc']:
            state['best_val_acc'] = val_acc
            best_logits = logits_cpu 
    '''
    logits_cpu = logits_cpu.T - np.max(logits_cpu,axis=1) 
    probs = np.exp(logits_cpu)
    probs /= np.sum(probs,0)
    probs = probs.T
    '''
    #np.savetxt('probs_128'+model_name+'.txt',probs,fmt='%.6e',delimiter=', ')
    
    return state, best_logits 
