import os
import sys
import cv2
import torch
import torch.utils.data as data

class ListDataset(data.Dataset):
    '''Load RGB image/labels/boxes from a list file.
    List file format
      a.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
    '''
    def __init__(self, root, listFile, transform=None):

        self.root = root
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []
        
        if isinstance(listFile, list):
            tmpFile = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(listFile), tmpFile))
            listFile = tmpFile

        with open(listFile) as f:
            lines = f.readlines()
            self.numImgs = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            numBoxes = (len(splited) - 1) // 5
            box = []
            for i in range(numBoxes):
                xmin = float(splited[1+5*i])
                ymin = float(splited[2+5*i])
                xmax = float(splited[3+5*i])
                ymax = float(splited[4+5*i])
                width = float(xmax)-float(xmin)
                height = float(ymax)-float(ymin)
                c = splited[5+5*i]
                box.append([float(xmin),float(ymin),float(width),float(height),float(c)])
            self.boxes.append(torch.tensor(box,dtype=torch.float,requires_grad=False))

    def __getitem__(self, idx):

        # Load RGB image and boxes.
        fname = self.fnames[idx]
        image = cv2.cvtColor(cv2.imread(os.path.join(self.root, fname)), cv2.COLOR_BGR2RGB)
        boxes = self.boxes[idx].clone()
        
        if self.transform:
            image, data = self.transform(image, boxes)
        return image, data

    def __len__(self):
        return self.numImgs
