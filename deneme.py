import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import xml.etree.ElementTree as xet
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision.models import inception_resnet_v2
import torch.nn as nn
import torch.optim as optim


# Read Data
df = pd.read_csv('labels.csv')
filename = df['filepath'][0]


def getFileName(filename):
    filename_img = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('./images', filename_img)
    return filepath_image


images_path = list(df['filepath'].apply(getFileName))

# Verify Labeled Data
file_path = images_path[0]
img = cv2.imread(file_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# Data Preprocessing
labels = df.iloc[:, 1:].values
data = []
output = []
for ind in range(len(images_path)):
    image = images_path[ind]
    img_arr = cv2.imread(image)
    h, w, d = img_arr.shape
    load_image = cv2.resize(img_arr, (224, 224))
    norm_load_img_arr = load_image / 255.0
    xmin, xmax, ymin, ymax = labels[ind]
    nxmin, nxmax = xmin / w, xmax / w
    nymin, nymax = ymin / w, ymax / w
    label_norm = (nxmin, nxmax, nymin, nymax)
    data.append(norm_load_img_arr)
    output.append(label_norm)

X = np.array(data, dtype=np.float32)
y = np.array(output, dtype=np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=0)


# Deep Learning Model Transfer Learning
class ObjectDetectionModel(nn.Module):
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        self.inceptionResNet = inception_resnet_v2(pretrained=True)
        for param in self.inceptionResNet.parameters():
            param.requires_grad = False
        self.headmodel = nn.Sequential(
            nn.Linear(1536, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.inceptionResNet(x)
        flattened = features.view(features.size(0), -1)
        output = self.headmodel(flattened)
        return output


model = ObjectDetectionModel()


# Model Training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100
batch_size = 10
X_train_tensor = torch.from_numpy(X_train.transpose(0, 3, 1, 2))
y_train_tensor = torch.from_numpy(y_train)
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save Model
torch.save(model.state_dict(), './models/object_detection.pt')


# Prediction Part
model = ObjectDetectionModel()
model.load_state_dict(torch.load('./models/object_detection.pt'))
model.eval()

path = './images/N1.jpeg'
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
outputs = model(image_tensor)
coords = outputs.detach().numpy()
coords = coords.squeeze()

# Denormalize the values
w, _, _ = image.shape
coords *= w

# Draw bounding box on the image
xmin, xmax, ymin, ymax = coords
cv2.rectangle(image, (int(xmin), int(ymin)),
              (int(xmax), int(ymax)), (0, 255, 0), 3)

plt.imshow(image)
plt.show()
