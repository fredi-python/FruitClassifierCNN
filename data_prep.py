# mapping

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


fruits_list = (os.listdir("data/train/train"))

mapping = {

}

mapping_value2class = {

}

for item in fruits_list:
    mapping[item] = fruits_list.index(item)
    mapping_value2class[fruits_list.index(item)] = item

# format:
# {0: 'Apple Braeburn', ...

# convert to tensor and resize to 100x100
transforms = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

class FruitsDataset(Dataset):
    def __init__(self, root_dir, transform=transforms):
        self.root_dir = root_dir
        self.transform = transform
        self.mapping = mapping
        self.images = []
        self.labels = []
        for label in os.listdir(root_dir):
            for image in os.listdir(os.path.join(root_dir, label)):
                self.images.append(os.path.join(root_dir, label, image))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.mapping[label]


