import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from PIL import Image
import numpy as np

def predict_func(img_name):
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=2)

    model.load_state_dict(torch.load("static/models/cat-dog-model.pth", map_location=torch.device('cpu')))

    val_transforms = transforms.Compose([
                                     transforms.CenterCrop((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = cv2.imread(img_name)   #Read the image
    img = Image.fromarray(image)      #Convert the image to an array
    img = val_transforms(img)     #Apply the transformations
    img = img.view(1,3,224,224)       #Add batch size
    img = Variable(img)
    #Wrap the tensor to a variable

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()

    output = model(img)
    _, predicted = torch.max(output, 1)
    if predicted==0:
        p = 'Cat'
    else:
        p = 'Dog'
    return  p
