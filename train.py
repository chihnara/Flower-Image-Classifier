from torchvision import transforms, datasets, models
import torch
from collections import OrderedDict
from torch import nn, optim
import argparse
import json


def custom_args():
    parser = argparse.ArgumentParser (description = "Image Classifier Parameters")
    parser.add_argument ('data_dir', help = '(Mandatory) Provide data directory.', type = str)
    parser.add_argument ('--save_dir', help = '(Optional) Provide directory to save checkpoints.', type = str)
    parser.add_argument ('--arch', help = '(Optional) Provide choice of architecture from vgg16(default) and alexnet.', type = str)
    parser.add_argument ('--lrn', help = '(Optional) Provide learning rate. Default value = 0.003', type = float)
    parser.add_argument ('--hidden_units', help = '(Optional) Hidden units in Classifier. Default value = 4096', type = int)
    parser.add_argument ('--epochs', help = '(Optional) Number of epochs. Default = 5', type = int)
    parser.add_argument ('--gpu', help = "(Optional) To use GPU", action="store_true")
    
    args = parser.parse_args()
    return args


def load_model(arch, hidden_units):
    if not hidden_units:
        hidden_units = 4096
    if arch == "alexnet":
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(9216, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
    else:
        arch = "vgg16"
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        
    return model, arch


def validation(model, loader, criterion, device):
    loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
#         calculate validation loss
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        loss += batch_loss.item()

#         calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return loss, accuracy


def main():
    args = custom_args()
    
    if not args.data_dir:
        print("ERROR: data_dir can not be empty.")
        return
    
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    model, arch = load_model(args.arch, args.hidden_units)
    
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
        
    model.to(device)
    
    if not args.lrn:
        args.lrn = 0.001
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lrn)
    
    if not args.epochs:
        args.epochs = 5
        
    steps = 0
    running_loss = 0
    print_every = 40

    for epoch in range(args.epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)

                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
     
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'classifier': model.classifier,
                 'class_to_idx': model.class_to_idx,
                 'state_dict': model.state_dict(),
                 'arch': arch}
    
    if args.save_dir:
        torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
    else:
        torch.save (checkpoint, 'checkpoint.pth')
    
if __name__=="__main__": 
    main() 
    