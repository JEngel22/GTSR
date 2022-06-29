import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import csv

class CustomImageDataset(Dataset):
    def __init__(self, rootpath, transform=None, target_transform=None):
        data = []
        for c in range(0,43):
            prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
            gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            header = True
            for row in gtReader:
                if not header:
                    data.append([prefix + row[0],row[7]]) # 1st column is the filename, 8th the classId
                header = False
            gtFile.close()

        self.img_labels = pd.DataFrame(data)
        self.img_dir = rootpath
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path)
        image = image.resize((44,44),Image.ANTIALIAS)
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
