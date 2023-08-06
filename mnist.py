'''
Fonte: https://github.com/pytorch/examples/blob/main/mnist/main.py
'''

#import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#from cv2 import *
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# Uma rede muito simples
# Duas camadas convolucionais
# Seguidas cada uma de uma ReLU
# Um MaxPooling para reduzir a imagem
# Flatten para converter a imagem em um vetor unidimensional
# Duas camadas Fully Connected
# Seguidas cada uma de um dropout
# Softmax no final para gerar algo semelhante a probabilidade
# de a imagem estar em cada uma das 10 classes
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

device = torch.device("cpu")

'''
im = Image.open(r"numero.png")
im = im.convert("L")
aaa = np.array(im)

tt = transforms.ToTensor()
nm = transforms.Normalize((0.1307,), (0.3081,))
im = nm(tt(aaa))
#im.show()

im = torch.unsqueeze(im, dim=0)
'''

# Modelo sera executado na CPU e nao na GPU
model = Net().to("cpu")

# Um objeto do pytorch para transformar arrays em Tensores normalizados
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# Carregamento da base MNIST a partir do site do pytorch
# Esta funcao encapsula o download e salva a base em um diretorio
# neste caso o diretorio e ./data
dataset1 = datasets.MNIST('./data', train=True, download=True,
                   transform=transform)
dataset2 = datasets.MNIST('./data', train=False,
                   transform=transform)
            
# Construcao do dataloader, este eh um iterador que cria batches automaticos e aleatorios
# para o treinamento, eh uma classe do proprio pytorch para facilitar o treinamento
train_loader = torch.utils.data.DataLoader(dataset1, num_workers=1, shuffle=1, batch_size=512)
test_loader = torch.utils.data.DataLoader(dataset2, num_workers=1, shuffle=1, batch_size=512)

# O otimizador eh o algoritmo que calculara quanto os pesos devem ser alterados a cada
# iteracao baseados nas derivadas parciais
optimizer = optim.Adadelta(model.parameters(), lr=1.0)

# O scheduler altera a taxa de aprendizado da rede a medida que ela eh treinada
# normalmente a taxa de aprendizado pode ser alta no inicio e ir diminuindo
# a medida que a precisao do modelo aumenta, fazendo ajustes mais finos nos pesos
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Verificacao da rede sem treino, assim conseguimos ver que ela melhora com o treinamento
model.eval() # Modo inferencia, sem calculo de derivadas parciais
test_loss = 0
correct = 0
with torch.no_grad(): # Derivadas parciais desabilitadas
    for data, target in test_loader:
        data, target = data.to(), target.to(device)
        output = model(data)
                # Calculo da funcao de perda (loss function) que diz o quanto
                # o modelo esta longe de um bom resultado, ou seja, o quanto ele
                # esta errado 
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('Teste loss: {:.4f}, Teste precisao: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# Loop para treinar por N epocas, ou seja, quantas vezes o modelo vera o 
# conjunto de treino inteiro
epochs = 2

total_loss = 0

for epoch in range(epochs):
        
    # Ativando gradientes, modo treino    
    model.train()
    
    print("Epoca {:d} de {:d}, loss: {:.6f}".format(epoch, epochs, total_loss))
    total_loss = 0
    pbar = tqdm(total=len(train_loader.dataset))

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Zeramos os resultados das derivadas parciais anteriores para
        # evitar que o treinamento tenha que armazenar muitas informacoes e
        # estoure a memoria
        optimizer.zero_grad()
                
        # As imagens do batch sao fornecidas para o modelo (rede)
        output = model(data)
                
        # A funcao de perda calcula o quanto a rede fez a previsao errada
        # das classes, seria algo proximo da precisao do modelo
        loss = F.nll_loss(output, target)
                
        # Calculo do gradiente gerado na funcao de perda para cada peso e cada imagem
        loss.backward()
                
        # Ajuste dos pesos de acordo com o gradiente gerado na funcao de perda
        # para este batch (subconjunto de imagens de treino)
        # Esta funcao, apesar de parecer simples e pequena, faz todo o trabalho de
        # treinamento. Este eh o coracao do treinamento
        optimizer.step()
                
        pbar.update(512)

        total_loss += loss.item()
        
        # Ajuste (reducao) da taxa de aprendizagem
    scheduler.step()
    pbar.close()

print("Epoca {:d} de {:d}, loss: {:.6f}".format(epoch, epochs, total_loss))

# Verificacao da rede sem treino, assim conseguimos ver que ela melhora com o treinamento
model.eval() # Modo inferencia, sem calculo de derivadas parciais
test_loss = 0
correct = 0
with torch.no_grad(): # Derivadas parciais desabilitadas
    for data, target in test_loader:
        data, target = data.to(), target.to(device)
        output = model(data)
                # Calculo da funcao de perda (loss function) que diz o quanto
                # o modelo esta longe de um bom resultado, ou seja, o quanto ele
                # esta errado 
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('Final: Teste loss: {:.4f}, Teste precisao: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    
exit(0)
