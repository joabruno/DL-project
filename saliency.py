from typing import Counter
import torch
from torchvision import datasets, models, transforms, utils
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image
dirname = os.path.dirname(__file__)
weighted_data_loader_model = os.path.join(dirname, 'Model/newmodel_weightedloader')
normal_data_loader_model = os.path.join(dirname, 'Model/newmodel_dataloader')
data = os.path.join(dirname, 'banana')
data = data.replace("\\", "/")
weighted_data_loader_model = weighted_data_loader_model.replace("\\", "/")
normal_data_loader_model = normal_data_loader_model.replace("\\", "/")

data_dir = data



data_transform = {'test':transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                  ]) }

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in ['test']}

data_loader = {x:torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=1, num_workers=0) for x in ['test']}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8*56*56, 56)
        self.fc2 = nn.Linear(56, 3)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = F.max_pool2d(self.relu(self.conv1(x)), 2)
        x = F.max_pool2d(self.relu(self.conv2(x)), 2)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x)) #softmax
        x = self.fc2(x)
        return x


dev = torch.device("cuda")
net = Net()
net.to(dev)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
cross_el = nn.CrossEntropyLoss()


net.load_state_dict(torch.load(normal_data_loader_model))
net.eval()



counter = 0
for data in data_loader['test']:
    if  counter == 10:
        break
    x, y = data
    
    x = x.cuda()

    x.requires_grad_()
    
    output = net(x)

    # Catch the output
    output_idx = output.argmax()
    output_max = output[0, output_idx]

    # Do backpropagation to get the derivative of the output based on the image
    output_max.backward()


    saliency, _ = torch.max(x.grad.data.abs(), dim=1) 
    saliency = saliency.reshape(224, 224)

    # Reshape the image
    x = x.reshape(-1, 224, 224)

    # Visualize the image and the saliency map
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle('The Image and Its Saliency Map')
    plt.show()
    counter += 1