from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import imageio.v3 as iio
import numpy as np

import gradio as gr


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# defining cnn -- maybe start with large kernel then decrease
class GalaxyNN(nn.Module):
    
   
    
    def __init__(self):
        super().__init__()
        self._example_input_array = torch.randn((1, 3, 77, 77))
        
        # Initialize the layers we need to build the network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=6), # first convolutional layer --> h = h - (5-1), w = w - (5-1)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # max pooling player --> h = h//2, w = w//2
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5), # second conv layer --> h = h - (5-1), w = w - (5-1)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(),
        )
        
        
        n_channels = self.conv4(self.conv3(self.conv2(self.conv1(self._example_input_array)))).view(-1).shape[0]

        self.fc1 = nn.Linear(n_channels, 120) # fully connected layer
        self.dropout1 = nn.Dropout(0.2) # adding dropout layers

        self.fc2 = nn.Linear(120, 80) # last fully connected layer should output the 37 label values
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(80, 60) 
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(60, 37) 
        self.dropout4 = nn.Dropout(0.2)
        

    def forward(self, x): # x is the image data (x from x, l = next(y))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = torch.flatten(x, 1) 

        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.dropout4(x)
        
        output = torch.sigmoid(x) # end with a sigmoid activation function
        return output
    
# Load  model
model = GalaxyNN()
MODEL_PATH = "lab3_data/galaxy_reduced_net.pth"

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


# class names 
class_names = [
    "Smooth and rounded, no features or disk",  # 0
    "Features or disk present",                 # 1
    "Star or artifact",                         # 2
    "Edge-on disk (disk viewed edge-on)",       # 3
    "Not edge-on (face-on or other)",           # 4
    "Bar feature present",                      # 5
    "No bar feature",                           # 6
    "Spiral arm pattern present",               # 7
    "No spiral arm pattern",                    # 8
    "No central bulge",                         # 9
    "Central bulge just noticeable",            # 10
    "Central bulge obvious",                    # 11
    "Central bulge dominant",                   # 12
    "Odd feature present",                      # 13
    "No odd feature",                           # 14
    "Completely round shape",                   # 15
    "In between round and cigar-shaped",        # 16
    "Cigar-shaped",                             # 17
    "Ring feature present",                     # 18
    "Lens or arc feature present",              # 19
    "Disturbed feature present",                # 20
    "Irregular feature present",                # 21
    "Other odd feature present",                # 22
    "Merger feature present",                   # 23
    "Dust lane present",                        # 24
    "Rounded bulge shape",                      # 25
    "Boxy bulge shape",                         # 26
    "No bulge (again)",                         # 27
    "Tightly wound spiral arms",                # 28
    "Medium wound spiral arms",                 # 29
    "Loosely wound spiral arms",                # 30
    "1 spiral arm",                             # 31
    "2 spiral arms",                            # 32
    "3 spiral arms",                            # 33
    "4 spiral arms",                            # 34
    "More than four spiral arms",               # 35
    "Cannot tell number of spiral arms"         # 36
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    # Read image as numpy array
    image_np = iio.imread(contents, extension='.jpg')  # or '.png', depending on your input
    # Resize if needed
    image_np = np.array(Image.fromarray(image_np).resize((77, 77)))
    # Reshape to (1, 3, 77, 77)
    if image_np.shape[-1] == 3:  # Ensure it's RGB
        image_np = image_np.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
    img_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)  # (1, 3, 77, 77)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        result = class_names[predicted.item()]
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

# Optional: serve static files (e.g., CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
