from PIL import Image
import numpy as np
import os.path as osp
import os
import sys
import torch

class PascalDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        if is_train:
            self.img_path = "Pascal/train_images/%s.jpg"
            self.seg_path = "Pascal/train_segmentations/%s.png"
            self.file_list = []
            with open("Pascal/train_id.txt", "r") as f:
                while True:
                    a = f.readline()
                    if a:
                         self.file_list.append(a.strip())
                    else:
                        break
        else:
            self.img_path = "Pascal/val_images/%s.jpg"
            self.seg_path = "Pascal/val_segmentations/%s.png"
            with open("Pascal/val_id.txt", "r") as f:
                while True:
                    a = f.readline()
                    if a:
                         self.file_list.append(a.strip())
                    else:
                        break
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, index):
        id = self.file_list[index]
        img = self.transform(np.asarray(Image.open(self.img_path % id).convert('RGB')).astype(np.uint8))
        seg = torch.from_numpy(np.asarray(Image.open(self.seg_path % id).convert('L'))).permute(2,0,1)
        print(np.max(seg), np.max(img), seg.shape, img.shape)
        return {'image': img, 'segmentation': seg}
        