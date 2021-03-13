
# coding: utf-8

# # CelebA DCGAN

# ## Setup

# In[3]:


import random
import torch
import pandas as pd
import numpy as np
from torchvision import transforms, datasets
import os
import zipfile
import gdown
from torch.utils.data import Dataset
from natsort import natsorted
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import torchvision.transforms as T
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries


# In[4]:


manual_seed = 999
random.seed(manual_seed)
torch.manual_seed(manual_seed)


# In[5]:


# Number of gpus available
ngpu = 1
device = torch.device('cuda:0' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')

device


# In[6]:


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


# ## Load Data

# In[9]:


data_root = 'data/celeba'
img_folder = f'{data_root}/img_align_celeba'


# In[11]:


# Root directory for the dataset
data_root = 'data/celeba'
# Path to folder with individual images
img_folder = f'{data_root}/img_align_celeba'
# URL for the CelebA dataset
url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'
# Path to download the dataset to
download_path = f'{data_root}/img_align_celeba.zip'
# Create required directories 
if not os.path.exists(data_root):
  os.makedirs(data_root)
  os.makedirs(img_folder)

# Download the dataset from google drive
gdown.download(url, download_path, quiet=False)

# Unzip the downloaded file 
with zipfile.ZipFile(download_path, 'r') as ziphandler:
  ziphandler.extractall(img_folder)


# In[11]:


class inception(nn.Module):
    def __init__(self, inDepth,oneByOne ,threeReduce, threeByThree, fiveReduce,fiveByFive, poolProj):
        super(inception, self).__init__()
        
        self.inception1 = nn.Sequential(
            nn.Conv2d(inDepth, oneByOne, kernel_size=1,padding=0),
            nn.BatchNorm2d(oneByOne),
            nn.ReLU(True))
        
        self.inception2a = nn.Sequential(
            nn.Conv2d(inDepth, threeReduce, kernel_size=1, padding=0),
            nn.BatchNorm2d(threeReduce),
            nn.ReLU(True))
        
        self.inception2b = nn.Sequential(
            nn.Conv2d(threeReduce, threeByThree, kernel_size=3, padding=1),
            nn.BatchNorm2d(threeByThree),
            nn.ReLU(True))
            
        self.inception3a = nn.Sequential(
            nn.Conv2d(inDepth, fiveReduce, kernel_size=1, padding=0),
            nn.BatchNorm2d(fiveReduce),
            nn.ReLU(True))

        self.inception3b = nn.Sequential(
            nn.Conv2d(fiveReduce, fiveByFive, kernel_size=5, padding=2),
            nn.BatchNorm2d(fiveByFive),
            nn.ReLU(True))

        self.inception4a = nn.MaxPool2d(3, stride=1, padding=1)


        self.inception4b = nn.Sequential(
            nn.Conv2d(inDepth, poolProj, kernel_size=1, padding=0),
            nn.BatchNorm2d(poolProj),
            nn.ReLU(True))    

    def forward(self, x):
      out1=self.inception1(x)
      out2=self.inception2a(x)
      out2=self.inception2b(out2)
      out3=self.inception3a(x)
      out3=self.inception3b(out3)
      out4=self.inception4a(x)
      out4=self.inception4b(out4)
      return torch.cat((out1,out2,out3,out4),1)


# In[12]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer0 = nn.Conv2d (3, 3, kernel_size=3, padding=1, stride=2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 31, kernel_size=5),
            nn.BatchNorm2d(31),
            nn.ReLU(True))
        
        self.layer2 = inception(31,10,10,12,2,4,8)
        self.poolingLayer0=nn.MaxPool2d(2, stride=2)
        self.layer3 = inception(34,14,14,16,4,12,10)
        self.layer4 = inception(52,14,12,20,2,10,10)
        self.poolingLayer1=nn.MaxPool2d(2, stride=2)
        self.layer5 = inception(54,20,16,24,4,16,16)
        self.layer6 = inception(76,22,18,26,8,16,16)
        self.layer7 = inception(80,20,16,24,6,18,18)
        self.poolingLayer2=nn.AvgPool2d(4, stride=2)
        self.layer11 = nn.Linear(320, 2)
        self.dropout = nn.Dropout(p=0.25)
        self.logsoftmax = nn.LogSoftmax()
        
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.poolingLayer0(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.poolingLayer1(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.dropout(out)
        out = self.poolingLayer2(out)
        out = out.view(out.size(0), -1)
        out= self.layer11(out)
        return self.logsoftmax(out)
net = CNN()
net = to_gpu(net)


# In[13]:


model = torch.load('data\\celeba\\model_attractive.pkl', map_location=torch.device('cpu'))
model.eval()
print(model)


# In[14]:


class CelebADatasetAttr(Dataset):
  def __init__(self, root_dir, transform=None, labels=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform 
    self.image_names = natsorted(image_names)
    self.labels = labels

  def __len__(self): 
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image 
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    import PIL.Image
    img = PIL.Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)
    label = self.labels[idx]
    if label == 0:
      img = torch.rand(3, 64, 64)

    return img, label


# In[15]:


# Spatial size of training images, images are resized to this size.
image_size = 64
# Transformations to be applied to each individual image sample
transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
])

transform_lime=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor()
])
#load labels
labels = pd.read_csv('data\\celeba\\list_attr_celeba.csv')
labels.loc[labels['Goatee'] == -1, 'Goatee'] = 0
labels = labels["Goatee"].tolist()

# Load the dataset from file and apply transformations

celeba_dataset_attr_lime = CelebADatasetAttr(f'{img_folder}/img_align_celeba', transform_lime, labels)


# In[16]:


# Batch size during training
batch_size = 1
# Number of workers for the dataloader
num_workers = 0 if device.type == 'cuda' else 2
num_workers = 0
# Whether to put fetched data tensors to pinned memory
pin_memory = True if device.type == 'cuda' else False


celeba_dataloader_atrr_lime = torch.utils.data.DataLoader(celeba_dataset_attr_lime,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                shuffle=True)


# LIME

# In[17]:


from lime import lime_image
from skimage.segmentation import mark_boundaries


# In[60]:


def predict_fn(img):
    img = img.reshape(3, 64, 64)
    img =  tensor_to_PIL(torch.from_numpy(img))
    img = lime_transf(img)
    img = np.array(img)
    img = img.reshape(1, 3, 64, 64)
    img = torch.from_numpy(img)
    img = to_gpu(img).float()
    return model(img).detach().cpu().numpy().astype('double')


# In[57]:


unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor, is_print=False):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    if is_print:
        plt.imshow(image)
        plt.axis('off')
        plt.figure(figsize=(3, 3))
        plt.show()
    return image


# In[58]:


image_size = 64
def get_lime_transform(): 
    transform = transforms.Compose([     transforms.Resize(image_size),    transforms.CenterCrop(image_size),    transforms.ToTensor(),    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    return transform
lime_transf = get_lime_transform()


# In[62]:


num_valid_images = 0
for i, data in enumerate(celeba_dataloader_atrr_lime, 0):
      # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    if labels[0]==0:
        continue
    num_valid_images += 1
    inputs_PIL = tensor_to_PIL(inputs[0], is_print=True)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(inputs_PIL), predict_fn, top_labels=2, hide_color=0,                                        num_samples=1000, batch_size=1)
    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=1, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    plt.imshow(img_boundry1)
    plt.axis('off')
    plt.figure(figsize=(3, 3))
    plt.show()

print('Finished LIME')

