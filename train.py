import numpy as np
import torch
from torchvision import datasets, models, transforms, utils
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
dirname = os.path.dirname(__file__)
weighted_data_loader_model = os.path.join(dirname, 'Model/newmodel_weightedloader')
normal_data_loader_model = os.path.join(dirname, 'Model/newmodel_dataloader')
data = os.path.join(dirname, 'banana')
data = data.replace("\\", "/")
weighted_data_loader_model = weighted_data_loader_model.replace("\\", "/")
normal_data_loader_model = normal_data_loader_model.replace("\\", "/")

data_dir = data



begin_time = datetime.datetime.now()


data_transform = {'train':transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),

                  'test':transforms.Compose([
                      transforms.Resize((224, 224)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ]),
                  'val':transforms.Compose([
                      transforms.Resize((224, 224)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ]) }



image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in ['train', 'test', 'val']}

#Oversampling attempt
targets = image_datasets['train'].classes
classes, class_count = np.unique(image_datasets['train'].targets, return_counts=True)
weight = 1. / torch.tensor(class_count, dtype=torch.float) 
class_weight = weight[classes]
samples_weight = [weight[x] for x in image_datasets['train'].targets]
sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

targets_val = image_datasets['val'].classes
classes, class_count = np.unique(image_datasets['val'].targets, return_counts=True)
weight = 1. / torch.tensor(class_count, dtype=torch.float) 
class_weight = weight[classes]
samples_weight_val = [weight[x] for x in image_datasets['val'].targets]

sampler_val = torch.utils.data.sampler.WeightedRandomSampler(samples_weight_val, len(samples_weight_val), replacement=True)

#print(list(sampler))
data_loader = {x:torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=124, num_workers=0) for x in ['train', 'test', 'val']}

weighted_loader = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=124, num_workers=0, sampler=sampler) for x in ['train', 'test', 'val']}
weighted_loader_val = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=124, num_workers=0, sampler=sampler_val) for x in ['val']}

class_names = image_datasets['train'].classes

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


inputs, classes = next(iter(weighted_loader['train']))

out = utils.make_grid(inputs)


#imshow(out, title=[class_names[x] for x in classes])


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
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
dev = torch.device("cuda")
net = Net()
net.to(dev)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
cross_el = nn.CrossEntropyLoss()

EPOCHS = 29


for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    net.train()
    for data in data_loader['train']:
        
        x, y = data
        ys, y_count = np.unique(y, return_counts=True)
        print(y_count)
        x,y = x.cuda(), y.cuda()
        net.zero_grad()
        output = net(x)
        loss_train = cross_el(output, y)
        #writer.add_scalar("Loss/train", loss_train, epoch)
        loss_train.backward()
        optimizer.step()

        # print statistics
        running_loss = 0.0
        running_loss += loss_train.item()
        print("Epoch: ", epoch + 1, " batchsize: 124", "-----loss: ", running_loss)
        running_loss = 0.0
    for val_data in data_loader['val']:
        x, y = val_data
        x,y = x.cuda(), y.cuda()
        
        val_output = net(x)
        loss = cross_el(val_output, y)
        
        #writer.add_scalar("Loss/val", loss, epoch)
    writer.add_scalars(f'loss train vs val', {
            'train loss': loss_train,
            'val loss': loss,
        }, epoch)
writer.flush()

correct = 0
total = 0

torch.save(net.state_dict(), weighted_data_loader_model)
print("Time: ", datetime.datetime.now() - begin_time)