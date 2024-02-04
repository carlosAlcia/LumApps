import pathlib
import tkinter as tk
import tkinter.ttk as ttk
import pygubu
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import Image_Classifier as ImC
import Image_Transform as ImT
from tkinter import filedialog
from PIL import ImageGrab, Image
import io



PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "Main_screen_UI.ui"
PATH_NN = './cifar_net.pth'


# Define a Convolutional Neural Network :
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Main_screen:
    def __init__(self, master=None):
        # 1: Create a builder and setup resources path (if you have images)
        self.builder = builder = pygubu.Builder()
        builder.add_resource_path(PROJECT_PATH)

        # 2: Load a ui file
        builder.add_from_file(PROJECT_UI)

        # 3: Create the mainwindow
        self.mainwindow = builder.get_object('mainwindow', master)

        # Save the canvas object to show images.
        self.canvas = builder.get_object('canvas', master)
        self.image_container = self.canvas.create_image(0,0,anchor="nw", image=None)
        self.image_tk = None
        # Save the output label to print results to user.
        self.output_lb = builder.get_object('outputs_label', master)

        # Create an object to store the image
        self.picture = None

        # 4: Connect callbacks
        builder.connect_callbacks(self)

    def run(self):
        self.mainwindow.mainloop()

    # Callback of Load Image from file button.
    def load_image_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Files JPEG", "*.jpg;*.jpeg")])
        if file_path:
            self.picture = ImT.transform_image(file_path, from_path=True)
            image_scaled = ImT.resize_image(file_path, self.canvas.winfo_width(), self.canvas.winfo_height(), from_path=True)
            self.image_tk = ImT.get_image_tk(image_scaled)
            self.canvas.itemconfig(self.image_container, image=self.image_tk)
            self.canvas.update()


    def load_image_from_cb(self):
        capture = ImageGrab.grabclipboard()
        if capture is not None:
            # Convert the capture to bytes format bytes to open with Pillow
            imagen_bytes = io.BytesIO()
            capture.save(imagen_bytes, format="PNG")
            # Open the image with Pillow
            image = Image.open(imagen_bytes).convert("RGB")

            self.picture = ImT.transform_image(image)
            image_scaled = ImT.resize_image(image,  self.canvas.winfo_width(), self.canvas.winfo_height())
            self.image_tk = ImT.get_image_tk(image_scaled)
            self.canvas.itemconfig(self.image_container, image=self.image_tk)
            self.canvas.update()

        else:
            self.output_lb.config(text="No image in the clipboard")


    def predict_class(self):
        if (self.picture is not None):
            self.output_lb.config(text=ImC.predict(self.picture))
        else :
            self.output_lb.config(text="You must provide an image first")



if __name__ == '__main__':
    app = Main_screen()
    app.run()