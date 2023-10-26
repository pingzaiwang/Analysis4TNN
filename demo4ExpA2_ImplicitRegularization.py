import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '8'

import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import numpy as np
from models import TNNdct3Layers
import matplotlib.pyplot as plt

from functions import f_seconds2sentence
from layers import ratio_spec2frob,t_spectral_norm

    
def fgsm_attack(model, loss, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    model.zero_grad()
    #print(f'outputs.len={len(outputs)},lavels.len={len(labels)}')
    cost = loss(outputs.view(1,-1), labels.view(1,-1)).backward()

    attack_images = images + epsilon*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images
    

def f_erank_of_weights(model):
    v_erank = torch.zeros(3)
    for i in range(3):
        W = getattr(model, f'layer{i+1}').weight
        v_erank[i] = ratio_spec2frob(W)
    return v_erank

def f_fnorm_of_weights(model):
    v_fnorm = torch.zeros(3)
    for i in range(3):
        W = getattr(model, f'layer{i+1}').weight
        v_fnorm[i] = torch.norm(W)
    return v_fnorm     

def f_spec_norms_of_weights(model):
    dic_spec_norms = {}
    for i in range(3):
        W = getattr(model, f'layer{i+1}').weight
        dic_spec_norms[i] = t_spectral_norm(W)
    return dic_spec_norms 

def main():
    # Hyperparameters
    batch_size = 80
    learning_rate = 1e-4
    num_epochs = 30000
    epsilon = 16/255

    num_channel = 28
    dim_input = 28
    dim_latent = 256

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    mnist_train = MNIST(root='./data', train=True, transform=ToTensor(), download=True)

    # Filter dataset for digits 3 and 7
    idx = np.where((mnist_train.targets == 3) | (mnist_train.targets == 7))[0]
    mnist_train.targets = mnist_train.targets[idx]
    mnist_train.data = mnist_train.data[idx]
    mnist_train.targets[ mnist_train.targets == 3] = 0
    mnist_train.targets[ mnist_train.targets == 7] = 1

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    # Model
    model = TNNdct3Layers(num_channel,dim_input,dim_latent,num_classes=2).to(device)

    # Loss and optimizer
    loss = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Adversarial training
    time_a = time.time()
    v_loss_adv_epochs = torch.zeros(num_epochs)
    m_erank_epochs = torch.zeros(num_epochs,3)
    m_fnorm_epochs = torch.zeros(num_epochs,3)
    #dict_spec_norms_epochs = {} 
    for epoch in range(num_epochs):
        total_loss_adv = 0
        total_num_example = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.float().to(device), labels.float().to(device)
            attack_images = fgsm_attack(model, loss, images, labels, epsilon)

            outputs = model(attack_images).squeeze()
            cost = loss(outputs, labels)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            total_loss_adv += cost.item()
            total_num_example += labels.shape[0]

            if (i+1) % 20 == 0:
                time_z= time.time() - time_a
                time_r = (num_epochs*len(train_loader)/(epoch*len(train_loader)+i+1) - 1)*time_z
                print("Used time:"+f_seconds2sentence(time_z)+"   Remaining time:"+f_seconds2sentence(time_r))
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {cost.item():.4f}")


        v_loss_adv_epochs[epoch] = total_loss_adv/total_num_example
        m_erank_epochs[epoch,:] = f_erank_of_weights(model).squeeze()
        m_fnorm_epochs[epoch,:] = f_fnorm_of_weights(model).squeeze()
        #dict_spec_norms_epochs[epoch] = f_spec_norms_of_weights(model)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'v_loss_adv_epochs': v_loss_adv_epochs,
            'm_erank_epochs': m_erank_epochs,
            'm_fnorm_epochs': m_fnorm_epochs,
            #'dict_spec_norms_epochs': dict_spec_norms_epochs
        }
        checkpoint_no_model = {
            'epoch': epoch,
            'v_loss_adv_epochs': v_loss_adv_epochs,
            'm_erank_epochs': m_erank_epochs,
            'm_fnorm_epochs': m_fnorm_epochs,
            #'dict_spec_norms_epochs': dict_spec_norms_epochs
        }        
        # plot the information each epoch
        if epoch > 0:
            x = range(epoch)
            # fnorm
            plt.clf()
            plt.plot(x, m_fnorm_epochs.detach().numpy()[x,0], label='Layer 1')
            plt.plot(x, m_fnorm_epochs.detach().numpy()[x,1], label='Layer 2')
            plt.plot(x, m_fnorm_epochs.detach().numpy()[x,2], label='Layer 3')
            plt.xlabel('Epoch')
            plt.ylabel('F-Norm of weights')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'op-3L-{dim_latent}-fnorms.png')
            plt.clf()
            # erank
            plt.plot(x, m_erank_epochs.detach().numpy()[x,0], label='Layer 1')
            plt.plot(x, m_erank_epochs.detach().numpy()[x,1], label='Layer 2')
            plt.plot(x, m_erank_epochs.detach().numpy()[x,2], label='Layer 3')
            plt.xlabel('Epoch')
            plt.ylabel('Square-root of effective rank')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'op-3L-{dim_latent}-erank.png')
            plt.clf()
            # loss
            plt.plot(x, v_loss_adv_epochs.detach().numpy()[x])
            plt.xlabel('Epoch')
            plt.ylabel('Adversarial training loss')
            #plt.legend()
            plt.grid(True)
            plt.savefig(f'op-3L-{dim_latent}-advloss.png')
            # do not save model
            torch.save(checkpoint_no_model, f'checkpoints/op-3L-{dim_latent}no-model.pth')
        # Save the information after each epoch
        if (epoch+1)%100 == 0:
            torch.save(checkpoint, f'checkpoints/over-para-3Layer-epoch{epoch+1}-latentdim{dim_latent}.pth')

if __name__ == "__main__":
    main()
