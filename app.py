import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Function to transform image
def transform_image(image):
    transform = Compose([Resize((224,224)), ToTensor()])
    return transform(image).unsqueeze(0)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load('waste_classifier.pth', map_location=device))

st.title('Waste Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is None:
    uploaded_file=st.camera_input('Take a picture')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image = transform_image(image).to(device)
    with torch.no_grad():
        model.eval()
        output = model(image)
        _, prediction = torch.max(output.data, 1)
        if prediction.item() == 0:
            st.write('The model predicts this image as category: Organic')
        else:
            st.write('The model predicts this image as category: Recycle')


