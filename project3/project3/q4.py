import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lr')
args = parser.parse_args()
learning_rate = float(args.lr)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

batch_size = 32
print("Loading data set.")
train_dataset = datasets.CIFAR10('./data',
                               train=True,
                               download=True,
                               transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.ToTensor()
                               ]))

test_dataset = datasets.CIFAR10('./data',
                               train=False,
                               download=True,
                               transform=transforms.Compose([
                               transforms.Grayscale(),
                               transforms.ToTensor()
                               ]))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size = batch_size,
                                            shuffle=True)

def partition_training_set(partition_idx, dataset):
    length = len(dataset)
    split_list = list(range(length))
    split_idx = int(np.floor(partition_idx * length))
    new_partition, rest_of_data = split_list[split_idx:], split_list[:split_idx]
    new_partition_subset = torch.utils.data.Subset(dataset, new_partition)
    rest_of_data_subset = torch.utils.data.Subset(dataset, rest_of_data)

    return new_partition_subset, rest_of_data_subset


validation_set, training_set = partition_training_set(float(4/5), train_dataset)

validation_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                                batch_size = batch_size,
                                                shuffle=True)

training_data = []

partition_idx =[float(3/4), float(2/3), float(1/2)]

for i in partition_idx:
    part, rest  = partition_training_set(i, training_set)
    training_data.append(part)
training_data.append(rest)


train_loader = []
for x in training_data:
   part_data_loader = torch.utils.data.DataLoader(dataset=x,
                                        batch_size = batch_size,
                                        shuffle=True)
   train_loader.append(part_data_loader)

for subset_loader in train_loader:
    for (X_train, y_train) in subset_loader:
        print('X_train:', X_train.size(), 'type:', X_train.type())
        print('y_train:', y_train.size(), 'type:', y_train.type())
        break

#here I added the 2 layers
# the one from all the inputs then they network through
# the 2 layers of hidden nodes
# the dropout ratio is .2 for both layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32, 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)

#here this creates the network of nodes
# as the fc1 connects to the fc2
# the activation function that I used here is the same one that I used in
#problm 2 because it would create a better side by side comparison.
    def forward(self, x):
        x = x.view(-1, 32*32)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)

        return F.log_softmax(self.fc3(x), dim=1)

model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)

def train(loss_train, accuracy_train, loader, epoch, log_interval=200):
    # Set model to training mode
    model.train()
    total_loss = 0
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()
        # Pass data through the network
        output = model(data)
        # Calculate loss
        loss = criterion(output, target)
        total_loss += loss.data.item()
        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.data.item()))

    total_loss /= len(loader)
    loss_train.append(total_loss)

def validate(loss_vector, accuracy_vector, loader):
     model.eval()
     val_loss, correct = 0, 0
     for data, target in loader:
         data = data.to(device)
         target = target.to(device)
         output = model(data)
         val_loss += criterion(output, target).data.item()
         pred = output.data.max(1)[1] # get the index of the max log-probability
         correct += pred.eq(target.data).cpu().sum()

     val_loss /= len(loader)
     loss_vector.append(val_loss)

     accuracy = 100. * correct.to(torch.float32) / len(loader.dataset)
     accuracy_vector.append(accuracy)

     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
         val_loss, correct, len(loader.dataset), accuracy))

epochs = 5

lossv, accv = [], []
loss_train, accuracy_train = [], []
for x in train_loader:
    for epoch in range(1, epochs + 1):
        train(loss_train, accuracy_train, x, epoch )
        validate(lossv, accv, validation_loader)




plt.plot(np.arange(1,(epochs*4)+1), lossv, 'b-', label="validation loss")
plt.title('Loss vs # of Epochs (Relu)')
plt.plot(np.arange(1,(epochs*4)+1), loss_train, 'r-', label="training loss")
plt.legend(loc="upper left")
plt.xlabel("Number of Epochs")
plt.ylabel("Error")
plt.subplots_adjust(hspace=0.5)

loss_train, accuracy_train = [], []
print("Testing data results")
validate(loss_train, accuracy_train, test_loader)


try:
    plt.show()

except:
    print("Cannot show graph")

plt.savefig('q4_1.png')
