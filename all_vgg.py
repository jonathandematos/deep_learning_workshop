import cv2
import numpy as np
from torchvision.models import vgg11 as VGG
from PIL import Image
import torch.nn as nn
from torchvision import datasets, transforms

net_model = VGG(weights="DEFAULT")

print(net_model)

image = cv2.imread('uepg_sm.jpg')

if image is None:
    print('Could not read image')
 
# Decomposicao da VGG
layers = list(net_model.children())[0][:8]
model_conv1 = nn.Sequential(*layers)  

layers = list(net_model.children())[0][:15]
model_conv2 = nn.Sequential(*layers)   

layers = list(net_model.children())[0][:22]
model_conv3 = nn.Sequential(*layers) 

layers = list(net_model.children())[0][:29]
model_conv4 = nn.Sequential(*layers)

transform=transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
	])
    
im_tensor = transform(image)

# Primeiro conjunto de camadas convolucionais
activation = model_conv1(im_tensor)

activations = list()

for i in range(0,activation.shape[0]):
    c = activation[i].detach().numpy()
    c = (c - c.min()) / (c.max() - c.min())*255
    activations.append(Image.fromarray( c.astype('uint8'), 'L'))

j = 0
for i in activations:
    i.save("1/activation_{:03d}.png".format(j))
    j+=1

# Segundo conjunto de camadas convolucionais
activation = model_conv2(im_tensor)

activations = list()

for i in range(0,activation.shape[0]):
    c = activation[i].detach().numpy()
    c = (c - c.min()) / (c.max() - c.min())*255
    activations.append(Image.fromarray( c.astype('uint8'), 'L'))

j = 0
for i in activations:
    i.save("2/activation_{:03d}.png".format(j))
    j+=1

# Terceiro conjunto de camadas convolucionais
activation = model_conv3(im_tensor)

activations = list()

for i in range(0,activation.shape[0]):
    c = activation[i].detach().numpy()
    c = (c - c.min()) / (c.max() - c.min())*255
    activations.append(Image.fromarray( c.astype('uint8'), 'L'))

j = 0
for i in activations:
    i.save("3/activation_{:03d}.png".format(j))
    j+=1

# Quarto conjunto de camadas convolucionais
activation = model_conv4(im_tensor)

activations = list()
flatten = list()

for i in range(0,activation.shape[0]):
    c = activation[i].detach().numpy()
    c = (c - c.min()) / (c.max() - c.min())*255
    d = c.flatten().reshape(1,-1)
    print(d.shape)
    activations.append(Image.fromarray( c.astype('uint8'), 'L'))
    flatten.append(Image.fromarray( d.astype('uint8'), 'L'))

j = 0
for i in activations:
    i.save("4/activation_{:03d}.png".format(j))
    j+=1

for i in flatten:
    i.save("4/flatten_{:03d}.png".format(j))
    j+=1
