from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from Qnet import Quantumnet
import os
filtered_classes = ['cat','dog']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
graytransform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(28,antialias=False),
     transforms.Grayscale(num_output_channels=1),
     transforms.Normalize((0.485), (0.225))])
trainset = datasets.CIFAR10(root='./data/CIFAR', train=True,
                            download=True,transform=graytransform)

testset = datasets.CIFAR10(root='./data/CIFAR', train=False,
                           download=True,transform=graytransform)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

filtered_labels=[classes.index(cl) for cl in filtered_classes]
sub_indices={'train': [], 'val': []}
image_datasets_full={'train': trainset, 'val': testset}
for phase in ['train', 'val']:
    for idx, label in enumerate(image_datasets_full[phase].targets):
        if label in filtered_labels:
            sub_indices[phase].append(idx)
image_datasets = {x: torch.utils.data.Subset(image_datasets_full[x], sub_indices[x])
                  for x in ['train', 'val']}
torch.manual_seed(0)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=64, shuffle=True, num_workers=0) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


def labels_to_filtered(labels):
    """Maps CIFAR labels (0,1,2,3,4,5,6,7,8,9) to the index of filtered_labels"""
    return [filtered_labels.index(label) for label in labels]
class Net(nn.Module):
    def __init__(self,ftr_out):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, ftr_out)
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


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss=0
    running_corrects=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), torch.tensor(labels_to_filtered(target)).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        preds = output.argmax(dim=1,keepdim=False)
        running_corrects += torch.sum(preds == target.data)
        #print(preds.shape, target.data.shape)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    #print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
   # print(f'Best val Acc: {best_acc:4f}')
    print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), torch.tensor(labels_to_filtered(target)).to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='c2qex')
    parser.add_argument('--Quantum', default=0,
                        help='For training classical or quantum')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')

    args = parser.parse_args()

    torch.manual_seed(0)
    print(bool(int(args.Quantum)))




    if bool(int(args.Quantum)):
        print('.....train quantum.....')
        model = Net(len(filtered_classes))
        model.fc2=Quantumnet(128,len(filtered_classes),device)
    else:
        print('.....train classical .......')
        model = Net(len(filtered_classes))
    model=model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.7)
    for epoch in range(1, args.epochs + 1):
        train(model, device, dataloaders['train'], optimizer, epoch)

        scheduler.step()
    test(model, device, dataloaders['val'])

    if args.Quantum:
        torch.save(model.state_dict(), "cifar_cqnn.pt")
    else:
        torch.save(model.state_dict(), "cifar_cnn.pt")



if __name__ == '__main__':
    main()
