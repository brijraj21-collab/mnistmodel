
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
import pickle
import io
from PIL import Image
from torchvision import transforms
from fastapi.middleware.cors import CORSMiddleware

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(64 * 12 * 12, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

app = Flask(__name__)
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)
# Load weights safely
device = torch.device("cpu")
model = SimpleCNN()
try:
    with open('model.pkl', 'rb') as f:
        model.load_state_dict(pickle.load(f))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file submitted'})
    
    file = request.files['file']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))
    tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        return jsonify({'prediction': int(predicted[0])})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
