#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Python Version: 3.7.4
#PyTorch Version: 1.7.1


# In[ ]:


import numpy as np 
import pandas as pd
import os
##Please copy and paste the path of the directory where the folder 'stat946winter2021' is present in the line below
BASE_PATH='/home/bharat/Desktop/STAT_946_Data_Challenge_1/stat946winter2021/'
train_dataset=pd.read_csv(os.path.join(BASE_PATH,'train_labels.csv'))
test_dataset=pd.read_csv(os.path.join(BASE_PATH,'test_labels.csv'))


# In[ ]:


# Check the head of the train_dataset dataframe
train_dataset.head(3)


# In[ ]:


# Check the head of the test_dataset dataframe
test_dataset.head(3)


# In[ ]:


#Plot Images to take a look at the X-Ray images
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(32, 32)) #Size of Figure
columns = 3 #Columns in fig
rows = 5 #Rows in Fig
#Total images = row* columns
for i in range(1,rows*columns+1):
    IMG_PATH=BASE_PATH+'train/'
    img=Image.open(os.path.join(IMG_PATH,train_dataset.iloc[i][0]))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()


# In[ ]:


#Import various libraries for CNN based image classification

from torchsummary import summary
import PIL
import sys
import torch
from time import time
import torchvision
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import torchvision.transforms as transforms

# I have used Efficientnet3 pre-trained model for the classification task
from efficientnet_pytorch import EfficientNet


# #Data Loader

# In[ ]:


#Define the Dataset class to pair images and labels

class Dataset(data.Dataset):
    def __init__(self,csv_path,images_path,transform=None):
        self.train_set=pd.read_csv(csv_path) #Read The CSV and create the dataframe
        self.train_path=images_path #Images Path
        self.transform=transform # Augmentation Transforms
    def __len__(self):
        return len(self.train_set)
    
    def __getitem__(self,idx):
        file_name=self.train_set.iloc[idx][0] 
        label=self.train_set.iloc[idx][1]
        img=Image.open(os.path.join(self.train_path,file_name)) #Loading Image
        if self.transform is not None:
            img=self.transform(img)
        return img,label


# #Defining Transforms and Parameters for Training

# In[ ]:


# Set the learning rate

learning_rate=1e-4


# In[ ]:


# Prepare the dataset consisting of the training images and the corressponding labels

training_set_untransformed=Dataset(os.path.join(BASE_PATH,'train_labels.csv'),os.path.join(BASE_PATH,'train/'))
print(type(training_set_untransformed))


# In[ ]:


# Define a transform operation that applies transformations to an image

transform_train = transforms.Compose([transforms.Resize((224,224)),transforms.RandomApply([
        torchvision.transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip()],0.7),
		transforms.ToTensor()])


# In[ ]:


#Create transformed images from the training dataset such that the minority class is upsampled and the 
#resulting classes are equal in number in the new set

new_created_images=[]
for j in range (len(training_set_untransformed)):
    if training_set_untransformed[j][1]==1:
        for k in range(8):
            transformed_image = transform_train(training_set_untransformed[j][0])
            new_created_images.append((transformed_image,1))
    else:
        transformed_image = transform_train(training_set_untransformed[j][0])
        new_created_images.append((transformed_image,0))

print(len(new_created_images))                                                                                   


# In[ ]:


# Split the new set into a training and validation dataset in the 80:20 ratio

train_size = int(0.8 * len(new_created_images))
validation_size = len(new_created_images) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(new_created_images, [train_size,validation_size])


# In[ ]:


# Create batches of size 32 each from the training dataset

training_generator = data.DataLoader(train_dataset,shuffle=True,batch_size=32,pin_memory=True) 


# In[ ]:


# Enable GPU computation

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)


# #Importing the model

# In[ ]:


# Instantiate Efficientnet3 

model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)


# In[ ]:


# Load the model to device

model.to(device)


# In[ ]:


# Display the summary of the model

print(summary(model, input_size=(3, 224, 224)))


# In[ ]:


# Create a folder in the stat946winter2021 directory to save Weights

PATH_SAVE='./Weights/'
if(not os.path.exists(PATH_SAVE)):
    os.mkdir(PATH_SAVE)


# In[ ]:


# Make crossentropyloss as the criterion, set a learning rate decay and use Adam or weight update

criterion = nn.CrossEntropyLoss()
lr_decay=0.99
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[ ]:


#Create a class list

eye = torch.eye(2).to(device)
classes=[0,1]


# In[ ]:


# Create lists to record accuracy and loss and set the number of epochs ( I got 11 as the optimal number of epochs)

history_accuracy=[]
history_loss=[]
epochs = 11


# In[ ]:


# Train the model

for epoch in range(epochs):  
    running_loss = 0.0
    correct=0
    total=0
    class_correct = list(0. for _ in classes)
    class_total = list(0. for _ in classes)
    
    for i, data in enumerate(training_generator, 0):
        inputs, labels = data
        t0 = time()
        inputs, labels = inputs.to(device), labels.to(device)
        labels = eye[labels]
        optimizer.zero_grad()
        #torch.cuda.empty_cache()
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        c = (predicted == labels.data).squeeze()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = float(correct) / float(total)
        
        history_accuracy.append(accuracy)
        history_loss.append(loss)
        
        loss.backward()
        optimizer.step()
        
        for j in range(labels.size(0)):
            label = labels[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1
        
        running_loss += loss.item()
        
        print( "Epoch : ",epoch+1," Batch : ", i+1," Loss :  ",running_loss/(i+1)," Accuracy : ",accuracy,"Time ",round(time()-t0, 2),"s" )
    for k in range(len(classes)):
        if(class_total[k]!=0):
            print('Accuracy of %5s : %2d %%' % (classes[k], 100 * class_correct[k] / class_total[k]))
        
    print('[%d epoch] Accuracy of the network on the Training images: %d %%' % (epoch+1, 100 * correct / total))
    
    if epoch%10==0 or epoch==0:
        torch.save(model.state_dict(), os.path.join(PATH_SAVE,str(epoch+1)+'_'+str(accuracy)+'.pth'))
        
torch.save(model.state_dict(), os.path.join(PATH_SAVE,'Last_epoch'+str(accuracy)+'.pth'))


# In[ ]:


# Plot accuracy and loss histories

plt.plot(history_accuracy)
plt.plot(history_loss)


# In[ ]:


# Model evaluation 

model.eval()


# In[ ]:


# Apply the same transformations on the test set as the training set so that the data distribution remains the same 

test_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.RandomApply([
        torchvision.transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip()],0.7),
                                      transforms.ToTensor(),
                                     ])


# In[ ]:


# Define a function to predict an image

def predict_image(image):
    image_tensor = test_transforms(image)
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


# In[ ]:


# Predict accuracy of the test_set 

correct_counter=0
for i in range(len(validation_dataset)):
    image_tensor = validation_dataset[i][0].unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    if index == validation_dataset[i][1]:
        correct_counter+=1
print("Accuracy=",correct_counter/len(validation_dataset))


# In[ ]:


# Create a data frame from the sample submission file

submission=pd.read_csv(BASE_PATH+'sample_submission.csv')


# In[ ]:


# Check if the dataframe was created properly

submission.head(3)


# In[ ]:


# Create a new data frame with file name and label as column headers

submission_csv=pd.DataFrame(columns=['File','Label'])


# In[ ]:


# Predict test set images and add the predictions to the submission_csv dataframe

IMG_TEST_PATH=os.path.join(BASE_PATH,'test/')
for i in range(len(submission)):
    img=Image.open(IMG_TEST_PATH+submission.iloc[i][0])
    prediction=predict_image(img)
    submission_csv=submission_csv.append({'File': submission.iloc[i][0],'Label': prediction},ignore_index=True)
    if(i%10==0 or i==len(submission)-1):
        print('[',32*'=','>] ',round((i+1)*100/len(submission),2),' % Complete')


# In[ ]:


# Write the submision_csv dataframe to a csv file

submission_csv.to_csv('submission_epoch11_2.csv',index=False)

