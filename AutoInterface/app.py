from flask import Flask, request, render_template, jsonify
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
from io import BytesIO
import base64

app = Flask(__name__)

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define your autoencoder model (same as in your script)
class ColorAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 64, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.down5 = nn.Conv2d(512, 1024, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.up5 = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        d4 = self.relu(self.down4(d3))
        d5 = self.relu(self.down5(d4))
        u1 = self.relu(self.up1(d5))
        u2 = self.relu(self.up2(torch.cat((u1, d4), dim=1)))
        u3 = self.relu(self.up3(torch.cat((u2, d3), dim=1)))
        u4 = self.relu(self.up4(torch.cat((u3, d2), dim=1)))
        u5 = self.sigmoid(self.up5(torch.cat((u4, d1), dim=1)))
        return u5

# Load your model
model = ColorAutoEncoder().to(DEVICE)
model.load_state_dict(torch.load('F:\My_Projects\Sih\model.pth', map_location=DEVICE))
model.eval()

# Define the image preprocessing functions
def load_and_preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    color_img = transform(image)
    
    gray_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    gray_img = gray_transform(image)
    
    color_img = color_img.unsqueeze(0)  # Add batch dimension
    gray_img = gray_img.unsqueeze(0)
    
    return color_img, gray_img

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.permute(1, 2, 0)  # Change to HWC format
    tensor = tensor.detach().cpu().numpy()  # Convert to numpy array
    tensor = (tensor * 255).astype('uint8')  # Scale to [0, 255]
    return Image.fromarray(tensor)

def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file).convert("RGB")
        
        # Preprocess the image
        color_img, gray_img = load_and_preprocess_image(image)
        gray_img = gray_img.to(DEVICE)

        # Make prediction
        with torch.no_grad():
            prediction = model(gray_img)
        
        # Convert images to base64 for displaying on the frontend
        grayscale_base64 = pil_to_base64(image.convert("L"))
        predicted_base64 = pil_to_base64(tensor_to_image(prediction))
        color_base64 = pil_to_base64(tensor_to_image(color_img))

        return jsonify({
            'grayscale': grayscale_base64,
            'predicted': predicted_base64,
            'color': color_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)