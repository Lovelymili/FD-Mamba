import torchvision
import torch
import numpy as np
totensor = torchvision.transforms.ToTensor()

def transform_augment_cd(img, min_max=(0, 1)):
    img = totensor(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img

def transform_label_cd(label_img, min_max=None):
    label = np.array(label_img, dtype=np.int64)  
    return torch.from_numpy(label).long()


