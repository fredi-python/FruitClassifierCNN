import torch
from torch import nn
import torchvision
#import torchvision.transforms as transforms
from data_prep import *
from torch.utils.data import dataloader
from modeling import FruitClassifier
from torchvision import utils
import os

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

batch_size = 512
num_epochs = 4

dataset = FruitsDataset("data/train/train")

dataloader = dataloader.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)

model = FruitClassifier(len(mapping))
model = model.to(device)

optimizer = torch.optim.AdamW((model.parameters()))
criterion = nn.CrossEntropyLoss()
# scheduler
loss_item = 0.0

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    running_loss = 0.0
    for step, batch in enumerate(dataloader, 1):  # start enumerating from 1
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        running_loss += loss.item()

        # Log the loss every 50 steps
        if step % 5 == 0:
            average_loss = running_loss / 5
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(dataloader)}], Training Loss: {average_loss:.4f}")
            running_loss = 0.0  # Reset the running loss after logging

    # Optionally, you can add a scheduler step here
    # scheduler.step()

print("Training completed.")

# Save the model
torch.save(model.state_dict(), "model.bin")


# Inference
torch.load("model.bin")
model.to("cpu")

# Inference
model.eval()
# Load a test image
test_image = Image.open("640x640.jpg")

# resize the image
#test_image = test_image.resize((100, 100))

# Preprocess the image
test_image = transforms(test_image)

# Save the test_image as jpg file
utils.save_image(test_image, "test_image.jpg")

print(test_image.shape)

# Add a batch dimension
test_image = test_image.unsqueeze(0)

# Get the model prediction
output = model(test_image)

# Get the predicted class
predicted_class = torch.argmax(output).item()

# Get the class name
predicted_class_name = mapping_value2class[predicted_class]

print(f"The predicted class is: {predicted_class_name}")
