from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

from model import ImprovedCNN  # Make sure this class is defined in model.py

app = Flask(__name__)

# Load your model
model = ImprovedCNN(num_classes=10)
model.load_state_dict(torch.load('fruit_model.pth', map_location=torch.device('cpu')))
model.eval()

# Class labels (ensure order matches your training set)
classes = ['Apple', 'Banana', 'Blueberry', 'Guava', 'Mango', 'Orange', 'Peach', 'Pineapple', 'Strawberry', 'Watermelon']

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file).convert('RGB')
        img = transform(img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            label = classes[predicted.item()]
        return render_template('index.html', prediction=label)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
