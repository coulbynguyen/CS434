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
parser.add_argument('d')
parser.add_argument('m')
parser.add_argument('wd')
parser.add_argument
args = parser.parse_args()
learning_rate = .1

# arg_d = float(args.d)
# arg_m = float(args.m)
# arg_wd = float(args.wd)

test_results = []
# mymomentums = [0.0, 0.25, 0.5, 0.75, 1.0]
arg_m = 0.0
# arg_d = 0.0
arg_wd = 0.0
mydropouts = [0.0, 0.25, 0.5, 0.75, 1.0]
# myweight_decays = [0.0, 0.25, 0.5, 0.75, 1.0]

#this for loop iterates through my drop out values and grabs the corresponding testing accuracy
#for this i choose to keep the arg_m and arg_wd to an arbitrary value of 0 as to see what would the
#dropouts do without any interference
for arg_d in mydropouts:
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

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(32*32, 100)
            self.fc1_drop = nn.Dropout(arg_d)
            # self.fc2 = nn.Linear(50, 50)
            # self.fc2_drop = nn.Dropout(0.2)
            self.fc3 = nn.Linear(100, 10)

        def forward(self, x):
            x = x.view(-1, 32*32)
            x = F.relu(self.fc1(x))
            x = self.fc1_drop(x)
            return F.log_softmax(self.fc3(x), dim=1)

    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=arg_m, weight_decay=arg_wd)
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
         test_results.append(accuracy)

    epochs = 5

    lossv, accv = [], []
    loss_train, accuracy_train = [], []
    for x in train_loader:
        for epoch in range(1, epochs + 1):
            train(loss_train, accuracy_train, x, epoch )
            # validate(lossv, accv, validation_loader)
    validate(loss_train, accuracy_train, test_loader)

plt.title('Test Accuracy vs different values of D')
plt.plot(mydropouts, test_results, 'r-', label="Dropout")
plt.legend(loc="upper left")
plt.xlabel("Size of Dropout")
plt.ylabel("Test Accuracy")

try:
    plt.show()

except:
    print("Cannot show graph")

plt.savefig('q3_d.png')
