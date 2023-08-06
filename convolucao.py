import cv2
import numpy as np
from torchvision.models import vgg11 as VGG
from PIL import Image
import torch.nn as nn
from torchvision import datasets, transforms

# Carregando VGG pretreinada
net_model = VGG(weights="DEFAULT")

# Formato e valor dos filtros
print(net_model.features[0].weight.shape)
print(net_model.features[0].weight[0][0])

# Carregando imagem
image = cv2.imread('uepg_sm.jpg')

if image is None:
    print('Could not read image')

# Convolucao da imagem com um dos filtros usando OpenCV
weights = net_model.features[0].weight[0][0].detach().numpy()
image_filtered = cv2.filter2D(src=image, ddepth=-1, kernel=weights)

cv2.imwrite('filtered_vgg_0_0.jpg', image_filtered)
