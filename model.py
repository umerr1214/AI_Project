import torch.nn as nn
import torchvision.models as models


# Loads a **ResNet18 model pre-trained on ImageNet**, a huge dataset 
# with 1,000 classes (e.g., cats, dogs, airplanes).
# This gives you a model that has already **learned useful features**, 
# like detecting edges, textures, and object parts — so we don’t need 
# to start from scratch.

def get_resnet18_model(num_classes, freeze_backbone=True):
    # Load pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)

    # freeze all layers except final layer
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False # tells PyTorch not to update the weights because the early layers are general feature detectors and we need those

    # Replace the final fully connected layer with custom classifier
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),   #reducing features to 256
        nn.ReLU(),  #adds non-linearity
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model


# import torch.nn.functional as F
# class ImprovedCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(ImprovedCNN, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(128 * 16 * 16, 256)  # adjust if input image size changes
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

