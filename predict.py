import PIL
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
import torch
import argparse
import json

def custom_args():
    parser = argparse.ArgumentParser (description = "Image Predictor Parameters")
    parser.add_argument ('image_dir', help = '(Mandatory) Provide path to the image.', type = str)
    parser.add_argument ('load_dir', help = '(Mandatory) Provide path to the checkpoint.', type = str)
    parser.add_argument ('--top_k', help = '(Optional) Enter K for top K most likely classes. Default value = 1', type = int)
    parser.add_argument ('--category_names', help = '(Optional) Mapping of categories to real names. JSON file name to be provided.', type = str)
    parser.add_argument ('--gpu', help = "(Optional) To use GPU", action="store_true")
    
    args = parser.parse_args()
    return args


def load_checkpoint(load_dir):
    checkpoint = torch.load(load_dir)
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        model = models.vgg16(pretrained = True)

    for param in model.parameters(): 
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = PIL.Image.open(image)
    width, height = image.size

    if width < height: 
        new_size=[256, 256**600]
    else: 
        new_size=[256**600, 256]
        
    image.thumbnail(size=new_size)
    center = width/4, height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    image = image.crop((left, top, right, bottom))

    np_image = np.array(image)/255 

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image-mean)/std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image


def predict(image_path, model, top_k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval();

    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to(device)
    log_ps = model.forward(torch_image)
    ps = torch.exp(log_ps)
    top_value, top_class = ps.topk(top_k)
 
    top_value = np.array(top_value.detach())[0] 
    top_class = np.array(top_class.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]
    
    return top_value, top_class


def main():
    args = custom_args()
    
    if not args.load_dir:
        print("ERROR: load_dir can not be empty.")
        return
    
    if not args.image_dir:
        print("ERROR: image_dir can not be empty.")
        return
    
    model = load_checkpoint(args.load_dir)
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
            pass
    
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    
    if not args.top_k:
        args.top_k = 1

    ps, classes = predict(args.image_dir, model, args.top_k, device)
    
    class_names = [cat_to_name [item] for item in classes]
    
    for l in range (args.top_k):
        print("Number: {}/{}.. ".format(l+1, args.top_k),
              "Class name: {}.. ".format(class_names[l]),
              "Probability: {:.3f}% ".format(ps[l]*100),
             )

if __name__=="__main__": 
    main() 