import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cusLoader as cLd
from functools import reduce
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('This Computation is running on {}'.format(device))
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,50,5)
        self.conv1_bn = nn.BatchNorm2d(50)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(50,100,5)
        self.conv2_bn = nn.BatchNorm2d(100)
        self.conv3 = nn.Conv2d(100,150,5)
        self.conv3_bn = nn.BatchNorm2d(150)

        self.fc1 = nn.Linear(600,200)
        self.fc2 = nn.Linear(200,150)
        self.fc3 = nn.Linear(150,43)

    def forward(self,x):
        x = self.pool1(self.conv1_bn(self.conv1(x)))
        x = self.pool1(self.conv2_bn(self.conv2(x)))
        x = self.pool1(self.conv3_bn(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x

batch_size = 64

dataset_split = (0,0.7,0.2) # Splitting-Factors: Train (Placeholder: rest of all minus val and test), Validation, Test - Validation for future-feature

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = cLd.CustomImageDataset('data/Final_Training/Images',transform=transform)
size_all=len(dataset)
size_val = math.floor(size_all*dataset_split[1])
size_test = math.floor(size_all*dataset_split[2])
size_train = size_all - size_val - size_test
dat_train,dat_val,dat_test = torch.utils.data.random_split(dataset, [size_train, size_val, size_test])
trainloader = torch.utils.data.DataLoader(dat_train, batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(dat_test, batch_size=batch_size,shuffle=True)

classes = ('Tempo 20',     'Tempo 30',     'Tempo 50',     'Tempo 60',     'Tempo 70',
          'Tempo 80',     'Ende 80',      'Tempo 100',    'Tempo 120',    'Überhohlverbot',

          'Überhohlverbot für LKW',     'Vorfahrt an nächster Einmündung',      'Vorfahrtstraße',    'Vorfahrt gewähren',    'Halt. Vorfahrt gewähren',
          'Verbot für Fahrzeuge aller Art',     'Verbot für Fahrzeuge über 3,5t',      'Einfahrt verboten',    'Gefahrstelle',    'Kurve links',

          'Kurve rechts',     'Doppelkurve',      'unebene Fahrbahn',    'Schleudergefahr',    'verengte Fahrbahn',
          'Arbeitsstelle',     'Ampel',      'Fußgänger',    'Kinder',    'Radfahrer',

          'Schnee- oder Eisglätte',     'Wildwechsel',      'Ende sämtlicher streckenbezogener Geschwindigkeits-beschränkungen und Überholverbote',    'vorgeschriebene Fahrtrichtung rechts',    'vorgeschriebene Fahrtrichtung links',
          'vorgeschriebene Fahrtrichtung geradeaus',     'vorgeschriebene Fahrtrichtung geradeaus oder rechts',      'vorgeschriebene Fahrtrichtung geradeaus oder links',    'vorgeschriebene Vorbeifahrt rechts',    'vorgeschriebene Vorbeifahrt links',

          'Kreisverkehr',     'Ende Überhohlverbot',      'Ende Überhohlverbot für LKW',
)

sample_image = next(iter(trainloader))[0][0]

input_dim = reduce(lambda x, y: x*y, sample_image.shape)

net = Net().to(device) # net = Net().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.7)
epoch_loss = []

epochs = 300

for epoch in range(epochs):
    batch_loss = []
    out_ges = []
    lab_ges = []
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        # if epoch % 100 == 0:
        #     print("Output-Index: {}".format(torch.max(output,1)[1]))
        #     print("Target: {}".format(labels))
        #     print("Loss: {}".format(loss.item()))

        batch_loss.append(loss.item())
    epoch_loss.append(np.mean(batch_loss))
    print("Epoche {} mit Loss {} done".format(epoch+1,epoch_loss[-1]))

# plt.figure(figsize=(8,6))
# plt.plot(epoch_loss,c="#1ACC94")
# plt.title("LogLoss über Epochen")
# plt.xlabel("Epochen")
# plt.ylabel("LogLoss")
# plt.show()

gesamt = 0
falsch = 0
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    output = net(inputs)
    gesamt += len(torch.max(output,1)[1])
    falsch += torch.count_nonzero(torch.max(output,1)[1]-labels).item()
print("Im Trainingsdatensatz falsch erkannt: {} von {} ({}%)".format(falsch,gesamt,falsch*100/gesamt))

gesamt = 0
falsch = 0
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    output = net(inputs)
    gesamt += len(torch.max(output,1)[1])
    falsch += torch.count_nonzero(torch.max(output,1)[1]-labels).item()
print("Im Testdatensatz falsch erkannt: {} von {} ({}%)".format(falsch,gesamt,falsch*100/gesamt))

