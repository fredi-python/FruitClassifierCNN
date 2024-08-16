import torch
from torch import nn
import torchvision
# import torchvision.transforms as transforms
from data_prep import *
import torch.nn.functional as F

class FruitClassifier(nn.Module):
    def __init__(self, num_fruits):
        super(FruitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 10, 5, 3, 3),
            nn.ReLU(),
            nn.Conv2d(10, 24, 5, 3, 3),
            nn.ReLU(),
            nn.Conv2d(24, 32, 5, 3, 3), #torch.Size([16, 32, 5, 5])
            nn.Flatten(1), # Exclude the batch size
            nn.Linear(800, 400), # mat1 and mat2 cannot be multiplied
            nn.ReLU(),
            nn.Linear(400, num_fruits)
        )


    def forward(self, x):
        #print("forward x before passing to layer:", x.shape)
        x = self.layer1(x)
        #print(x.shape)
        return x


#num_fruits = len(mapping)

#model = FruitClassifier(num_fruits)

#x = torch.rand((3, 100, 100))
#output = model(x)
#print(output)

#probabilities = F.softmax(output)
#print(probabilities)
#print("sum of probabilities:", sum(probabilities))

#chosen_fruit = torch.argmax(output)
#print(chosen_fruit)
#chosen_fruit = chosen_fruit.cpu().numpy()
#print(chosen_fruit)
#print(mapping_value2class[int(chosen_fruit)])

# 3 * 100 * 100
# = 30000 -> 800
# 1
