import torch
from torchvision import datasets, models, transforms, utils
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import f1_score

dirname = os.path.dirname(__file__)
weighted_data_loader_model = os.path.join(dirname, 'Model/newmodel_weightedloader')
normal_data_loader_model = os.path.join(dirname, 'Model/newmodel_dataloader')
data = os.path.join(dirname, 'banana')
data = data.replace("\\", "/")
weighted_data_loader_model = weighted_data_loader_model.replace("\\", "/")
normal_data_loader_model = normal_data_loader_model.replace("\\", "/")

data_dir = data

data_transform = {'train':transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),

                  'test':transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                  ]) }

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in ['train', 'test']}

#Oversampling attempt
targets = image_datasets['test'].classes
classes, class_count = np.unique(image_datasets['test'].targets, return_counts=True)
weight = 1. / torch.tensor(class_count, dtype=torch.float) 
class_weight = weight[classes]
samples_weight = [weight[x] for x in image_datasets['test'].targets]

sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
data_loader = {x:torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=1, num_workers=0) for x in ['train', 'test']}

weighted_loader = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=1, num_workers=0, sampler=sampler) for x in ['train', 'test']}


class_names = image_datasets['train'].classes


inputs, classes = next(iter(data_loader['train']))

out = utils.make_grid(inputs)

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


net.load_state_dict(torch.load(normal_data_loader_model))
net.eval()

correct = 0
total = 0
acc_avg = 0
y_pred = []
y_actual = []
with torch.no_grad():
    for data in data_loader['test']:
        x, y = data
        y_actual += y
        x,y = x.cuda(), y.cuda()
        
        #x.requires_grad_()
        output = net(x)

        # output_idx = output.argmax()
        # output_max = output[0, output_idx]

        # output_max.backward()

        # saliency, _ = torch.max(x.grad.data.abs(), dim=1) 
        # saliency = saliency.reshape(224, 224)
        for idx, i in enumerate(output):
            y_pred.append(torch.argmax(i).to("cpu"))
            if torch.argmax(i) == y[idx]:
                correct +=1
                # x = x.reshape(-1, 224, 224)

                # # Visualize the image and the saliency map
                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(x.cpu().detach().numpy().transpose(1, 2, 0))
                # ax[0].axis('off')
                # ax[1].imshow(saliency.cpu(), cmap='hot')
                # ax[1].axis('off')
                # plt.tight_layout()
                # fig.suptitle('Guessed: '+ str(torch.argmax(i)) + " but is actually: " + str(y[idx]))
                # plt.show()
            else:
                #wrong prediction
                
                None
                
            total +=1

print(f'accuracy: {round(correct/total, 3)}')
print("f1 score ", f1_score(y_actual, y_pred, average='macro'))
