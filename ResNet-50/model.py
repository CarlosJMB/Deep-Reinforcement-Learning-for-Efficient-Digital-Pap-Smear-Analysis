import torchvision.models as models
import torchvision
import torch.nn as nn

def build_model(weights=True, fine_tune=True, num_classes=1):
    if weights:
        print('[INFO]: Loading pre-trained weights')
    elif not weights:
        print('[INFO]: Not loading pre-trained weights')
    model = models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
         
    # change the final classification head, it is trainable
    model.fc = nn.Linear(512, num_classes)
    return model
