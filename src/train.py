import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tnrange, tqdm_notebook
import src.readTrafficSigns as rTS
#from torchsummary import summary
from functools import reduce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('This Computation is running on {}'.format(device))

trainImages, trainLabels = rTS.readTrafficSigns('data/Final_Training')
print("Label: {}, Images: {}".format(len(trainLabels), len(trainImages))
plt.imshow(trainImages[42])
plt.show()
