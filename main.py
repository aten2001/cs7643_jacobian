from __future__ import division
import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from jacobian import JacobianReg

from models.lenet import *
from models.resnet import *
from models.vgg import *

from logger import get_logger
from option import Options

args = Options().parse()



def eval(device, model, loader, criterion, lambda_JR):
    '''
    Evaluate a model on a dataset for Jacobian regularization
    Arguments:
        device (torch.device): specifies cpu or gpu training
        model (nn.Module): the neural network to evaluate
        loader (DataLoader): a loader for the dataset to eval
        criterion (nn.Module): the supervised loss function
        lambda_JR (float): the Jacobian regularization weight
    Returns:
        correct (int): the number correct
        total (int): the total number of examples
        loss_super (float): the supervised loss
        loss_JR (float): the Jacobian regularization loss
        loss (float): the total combined loss
    '''

    correct = 0
    total = 0 
    loss_super_avg = 0 
    loss_JR_avg = 0 
    loss_avg = 0

    # for eval, let's compute the jacobian exactly
    # so n, the number of projections, is set to -1.
    reg_full = JacobianReg(n=-1)
    for data, targets in loader:
        data = data.to(device)
        
        data.requires_grad = True # this is essential!
        targets = targets.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
        loss_super = criterion(output, targets) # supervised loss
        loss_JR = reg_full(data, output) # Jacobian regularization
        loss = loss_super + args.lambda_JR*loss_JR # full loss
        loss_super_avg += loss_super.data*targets.size(0)
        loss_JR_avg += loss_JR.data*targets.size(0)
        loss_avg += loss.data*targets.size(0)

    loss_super_avg /= total
    loss_JR_avg /= total
    loss_avg /= total
    return correct, total, loss_super, loss_JR, loss


def train_model(model, device, logger):
    
    if args.dataset == 'mnist':
        # load MNIST trainset and testset
        mnist_mean = (0.1307,)
        mnist_std = (0.3081,)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mnist_mean, mnist_std)]
        )
        trainset = datasets.MNIST(root='./data', train=True, 
            download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True
        )
        testset = datasets.MNIST(root='./data', train=False, 
            download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=True
        )
        if args.defense != None:
            if args.defense == "rand_size":
                new_W = np.random.randint(24, 28)
                new_H = np.random.randint(24, 28)
                padding_left = np.random.randint(0, 28 - new_W + 1)
                padding_right = 28 - new_W - padding_left
                padding_top = np.random.randint(0, 28 - new_H + 1) # left, top, right and bottom
                padding_bottom = 28 - new_H - padding_top
                logger.info('Width: %d, height: %d, padding_left: %d, padding_right: %d, padding_top: %d, padding_bottom: %d' %
	                        (new_W, new_H, padding_left, padding_right, padding_top, padding_bottom)
	                )
                transform_defense = transforms.Compose(
                    [transforms.Resize((new_H, new_W)), 
                     transforms.Pad((padding_left, padding_top, padding_right, padding_bottom)), 
                     transforms.ToTensor(), transforms.Normalize(mnist_mean, mnist_std)]
                )
            else:
                print("Invalid defense name.")
                raise
            testset_defense = datasets.MNIST(root='./data', train=False, 
                download=True, transform=transform_defense
            )
            testloader_defense = torch.utils.data.DataLoader(
                testset_defense, batch_size=args.batch_size, shuffle=True
            )    
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    
    
    # initialize the loss and regularization
    criterion = nn.CrossEntropyLoss()
    reg = JacobianReg(n=args.n_proj) # if n_proj = 1, the argument is unnecessary

    # initialize the optimizer
    # including additional regularization, L^2 weight decay
    optimizer = optim.SGD(model.parameters(), 
        lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    # eval on testset before any training
    print('eval on testset before any training...')

    correct_i, total, loss_super_i, loss_JR_i, loss_i = eval(
        device, model, testloader, criterion, args.lambda_JR
    )

    # train
    
    print('training started...')
    
    for epoch in range(args.epochs):
        logger.info("=========================================")
        logger.info("Epoch {0}".format(epoch+1))
        logger.info("=========================================")
        running_loss_super = 0.0
        running_loss_JR = 0.0
        for idx, (data, target) in enumerate(trainloader):        

            data, target = data.to(device), target.to(device)
            data.requires_grad = True # this is essential!

            optimizer.zero_grad()

            output = model(data) # forward pass

            loss_super = criterion(output, target) # supervised loss
            loss_JR = reg(data, output)   # Jacobian regularization
            loss = loss_super + args.lambda_JR*loss_JR # full loss

            loss.backward() # computes gradients

            optimizer.step()

            # print running statistics
            running_loss_super += loss_super.item()
            running_loss_JR += loss_JR.item()
            if idx % 100 == 99:    # print every 100 mini-batches
                
                # Logging
                logger.info('[%d, %5d] supervised loss: %.3f, Jacobian loss: %.3f' %
                        (
                            epoch + 1, 
                            idx + 1, 
                            running_loss_super / 100,  
                            running_loss_JR / 100, 
                        )
                )

                

                
                running_loss_super = 0.0
                running_loss_JR = 0.0


        if args.val == 1:
	        # evaluate test accuracy & loss
	        correct_f, total, loss_super_f, loss_JR_f, loss_f = eval(
		        device, model, testloader, criterion, args.lambda_JR
		    )


	        logger.info('test results: accuracy: %.3f, supervised loss: %.3f, Jacobian loss: %.3f, total loss: %.3f' %
	                        (
	                            correct_f/total, 
	                            loss_super_f, 
	                            loss_JR_f,  
	                            loss_f, 
	                        )
	                )



    # eval on testset after training
    print('eval on testset after training...')

    correct_f, total, loss_super_f, loss_JR_f, loss_f = eval(
        device, model, testloader, criterion, args.lambda_JR
    )

    # print results
    logger.info("=========================================")
    logger.info("Test set results")
    logger.info("=========================================")
    logger.info('Test set results on %s with lambda_JR=%.3f.' % (args.dataset, args.lambda_JR))
    logger.info('Before training:')
    logger.info('accuracy: %d/%d=%.3f' % (correct_i, total, correct_i/total))
    logger.info('supervised loss: %.3f' % loss_super_i)
    logger.info('Jacobian loss: %.3f' % loss_JR_i)
    logger.info('total loss: %.3f' % loss_i)

    logger.info('After %d epochs of training:' % args.epochs)
    logger.info('accuracy: %d/%d=%.3f' % (correct_f, total, correct_f/total))
    logger.info('supervised loss: %.3f' % loss_super_f)
    logger.info('Jacobian loss: %.3f' % loss_JR_f)
    logger.info('total loss: %.3f' % loss_f)
    
    
    # eval using defense
    if args.defense != None:
        print('eval on testset with defense...')

        correct_d, total, loss_super_d, loss_JR_d, loss_d = eval(
            device, model, testloader_defense, criterion, args.lambda_JR
        )

        # print results
        logger.info("After training with defense: " + args.defense)
        logger.info('accuracy: %d/%d=%.3f' % (correct_d, total, correct_d/total))
        logger.info('supervised loss: %.3f' % loss_super_d)
        logger.info('Jacobian loss: %.3f' % loss_JR_d)
        logger.info('total loss: %.3f' % loss_d)
        

    return model
    
def main():
    '''
    Train MNIST with Jacobian regularization.
    '''
    
    logger = get_logger(name=args.name_log)
    
#     seed = 1
#     batch_size = 64
#     epochs = 5

#     lambda_JR = .1

#     # number of projections, default is n_proj=1
#     # should be greater than 0 and less than sqrt(# of classes)
#     # can also set n_proj=-1 to compute the full jacobian
#     # which is computationally inefficient
#     n_proj = 1 

    # setup devices
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")


    # initialize the model
    if args.model == 'lenet_dropout':
        model = LeNet_dropout()
    elif args.model == 'lenet_standard':
        model = LeNet_standard()
    elif args.model == 'vgg11':
        model = VGG('VGG11')
    elif args.model == 'resnet18':
        model = ResNet18()        
        
    model.to(device)
    
    
    print('model loaded...')
    
    model = train_model(model, device, logger)



if __name__ == '__main__':
    main()
