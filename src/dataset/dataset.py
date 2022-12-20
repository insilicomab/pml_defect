import os
from PIL import Image
from torch.utils.data import Dataset


class DefectDataset(Dataset):
    def __init__(self, image_name_list, label_list, img_dir, transform=None, phase=None):
        self.image_name_list = image_name_list
        self.label_list = label_list
        self.img_dir = img_dir
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir, self.image_name_list[index])
        image = Image.open(image_path)
        image = self.transform(self.phase, image)
        label = self.label_list[index]
        
        return image, label