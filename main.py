import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_dataset
from models import Net


def test_data(net, classes, testloader, PATH):
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    #net.load_state_dict(torch.load(PATH))
    outputs = net(images)

    #Testing the model on test data
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        i = 0
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = (100 * correct) / total
    print('Accuracy of the network on the 10000 test images: %d' % (accuracy))
    return accuracy



def check_class_performance(net, classes, testloader):

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))



def main(args):
    trainloader, testloader = load_dataset(args.dataset)
    print(len(trainloader), len(testloader))

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for epoch in range(args.epochlen):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        test_accuracy = test_data(net, classes, testloader, args.path)


        #torch.save(net.state_dict(), args.path)

    check_class_performance(net, classes, testloader)
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="")
    parser.add_argument("--path", type=str, default="./cifar_model.pth", help="")
    parser.add_argument("--epochlen", type=int, default=2, help="")
    args = parser.parse_args()
    main(args)
