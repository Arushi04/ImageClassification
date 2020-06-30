import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import matplotlib.pyplot as plt
from dataset import load_dataset
from models import Net


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


def test_data(net, classes, testloader, criterion):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    outputs = net(images)

    #Testing the model on test data
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        test_loss = 0.0
        for i, data in enumerate(testloader, 0):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += loss.item()

    test_accuracy = (100 * correct) / total
    return test_loss, test_accuracy


def main(args):
    trainloader, testloader, classes = load_dataset(args.dataset)
    print(len(trainloader), len(testloader))

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter(log_dir=os.path.join(args.outdir, "tb/"), purge_step=0)

    # Plot confusion matrix
    title = ("Confusion matrix", None)
    X_test = []
    Y_test = []
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true = labels.numpy()
        y_pred = predicted.numpy()

    cm = multilabel_confusion_matrix(y_true, y_pred)
    print(cm)
    print(classification_report(y_true, y_pred))

    correct = 0
    total = 0
    iter = 0
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
            if iter % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, iter, running_loss / 2000))
                writer.add_scalar('Training Loss', running_loss, iter)
                running_loss = 0.0
            iter += 1

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = (100 * correct) / total
        writer.add_scalar('Train Accuracy', train_accuracy, epoch+1)

        test_loss, test_accuracy = test_data(net, classes, testloader, criterion)
        print(f"Accuracy of the network on the 10000 test images for epoch {epoch+1} is {test_accuracy}")
        writer.add_scalar('Testing Loss', test_loss, epoch+1)
        writer.add_scalar('Test Accuracy', test_accuracy, epoch+1)
    check_class_performance(net, classes, testloader)


    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="")
    parser.add_argument("--outdir", type=str, default="./output/", help="")
    parser.add_argument("--epochlen", type=int, default=1, help="")
    args = parser.parse_args()
    main(args)
