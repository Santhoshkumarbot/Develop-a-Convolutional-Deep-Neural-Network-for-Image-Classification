# Develop a Convolutional Deep Neural Network for Image Classification
# NAME : SANTHOSHKUMAR.P
# REG.NO : 212224040295
## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
The objective of this project is to develop a Convolutional Neural Network (CNN) model for image classification. The model classifies fashion images into different categories automatically. It learns important features from input images and predicts the correct class. The performance of the model is evaluated using accuracy, confusion matrix, and classification report.

The dataset used is the Fashion MNIST dataset provided by Zalando Research. It contains 70,000 grayscale images of size 28×28 pixels. The dataset has 60,000 training images and 10,000 testing images. It consists of 10 clothing categories such as T-shirt, Dress, Coat, Shoe, and Bag. The dataset is used to train and test the CNN model for image classification.
## Neural Network Model
<img width="1147" height="800" alt="image" src="https://github.com/user-attachments/assets/d8cad49c-d220-4dc4-bf6f-780b779fbbb5" />


## DESIGN STEPS
### STEP 1: 


Import the required libraries and configure the execution device (CPU/GPU) for building the CNN model.

### STEP 2: 

Load the Fashion MNIST dataset, apply image preprocessing, and create training and testing data loaders.

### STEP 3: 

Design and initialize the Convolutional Neural Network (CNN) architecture with convolution, pooling, and fully connected layers.

### STEP 4: 

Define the loss function and optimizer, then train the CNN model using the training dataset.

### STEP 5: 

Evaluate the trained model using the testing dataset and calculate accuracy, confusion matrix, and classification report.

### STEP 6: 
Test the model with a new image and display the actual and predicted class labels.



## PROGRAM
### Name: ARUL KUMARAN S A

### Register Number: 212224040030

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

transform=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5,),(0.5))  
])

train_set=torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,
transform=transform)
test_set=torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,
transform=transform)
im,lbl=train_set[0]
print(im.shape)
print(len(train_set))
print(len(test_set))

trl=DataLoader(train_set,batch_size=32,shuffle=True)
tstl=DataLoader(test_set,batch_size=32,shuffle=False)

class CNNclassifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.c2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.c3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.l1=nn.Linear(128*3*3,64)
        self.l2=nn.Linear(64,32)
        self.l3=nn.Linear(32,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.c1(x)))
        x=self.pool(torch.relu(self.c2(x)))
        x=self.pool(torch.relu(self.c3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        x=self.l3(x)
        return x
from torchsummary import summary

model=CNNclassifier1()
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
if torch.cuda.is_available():
    print(torch.cuda.is_available())
    device=torch.device('cuda')
    model.to(device)


summary(model,input_size=(1,28,28))
epochs=3
running_loss=0.0

for i in range(epochs):
    model.train()
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b)
        loss.backward()
        op.step()
        running_loss+=loss.item()
    print(f"Loss:{i}",running_loss/len(trl))

t=0
c=0
act=[]
pre=[]
model.eval()
with torch.no_grad():
    for img,labels in tstl:
        output=model(img)
        _,predicted=torch.max(output,1)
        t=t+labels.size(0)
        c+=(predicted==labels).sum().item()
        pre.extend(predicted.cpu().numpy())
        act.extend(labels.cpu().numpy())
accuracy=c/t*100
print("Accuracy Score:",accuracy)
conf_matrix=confusion_matrix(act,pre)
class_report=classification_report(act,pre,target_names=test_set.classes)
print("Classification Report:",class_report)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=test_set.classes,yticklabels=test_set.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

with torch.no_grad():
    img1,label=test_set[0]
    output=model(img1.unsqueeze(0))
    _,pred=torch.max(output,1)
    classes=test_set.classes
    img1 = img1 * 0.5 + 0.5
    plt.imshow(img1.squeeze(),cmap="gray")
    plt.title("Predicted Image")
    plt.axis('off')
    plt.show()
    print(f"Actual:{classes[label]}\n")
    print(f"Predicted:{classes[pred.item()]}\n")
original_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=None
)

image, label = original_dataset[0]

plt.imshow(image, cmap='gray')
plt.title(original_dataset.classes[label])
plt.axis('off')
plt.show()

```

### OUTPUT

## Training Loss per Epoch
<img width="669" height="538" alt="image" src="https://github.com/user-attachments/assets/29c50c86-6124-4ccb-ad79-47d382fffc64" />




<img width="314" height="86" alt="image" src="https://github.com/user-attachments/assets/672e5294-7e16-46bb-ab32-7b598c0cac71" />


## Confusion Matrix


<img width="758" height="654" alt="image" src="https://github.com/user-attachments/assets/b38d37f0-17af-4a41-adac-ab558bcfd3c3" />


## Classification Report

<img width="756" height="417" alt="image" src="https://github.com/user-attachments/assets/1294b454-1624-408d-b5ba-d10562f137ec" />


### New Sample Data Prediction
<img width="521" height="657" alt="image" src="https://github.com/user-attachments/assets/b4b5f127-4603-4cbe-bbe8-8df50b1e9a82" />


## RESULT
The CNN model was successfully trained and tested on the Fashion MNIST dataset, achieving accurate image classification with performance evaluation metrics.
