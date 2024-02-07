# Code by Carlos Alvarez 2024.

import pathlib
import tkinter as tk
import tkinter.ttk as ttk
import pygubu
import Image_Classifier as ImC
import Image_Transform as ImT
from tkinter import filedialog
from PIL import ImageGrab, Image
import io



PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "Main_screen_UI.ui"


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

        # Save the progress bar to show graphically the confidence
        self.bar_confidence = builder.get_object('bar_confidence', master)

        # Definition of different styles to change confidence bar color
        self.style = ttk.Style()
        self.style.theme_use('clam') 
        self.style.configure("1.Horizontal.TProgressbar", troughcolor ='white', background='green') 
        self.style.configure("2.Horizontal.TProgressbar", troughcolor ='white', background='yellow')
        self.style.configure("3.Horizontal.TProgressbar", troughcolor ='white', background='red')

        # Create an object to store the image
        self.picture = None

        # 4: Connect callbacks
        builder.connect_callbacks(self)

    def run(self):
        self.mainwindow.mainloop()

    # Callback of Load Image from file button.
    def load_image_from_file(self):
        # Show window to select file
        file_path = filedialog.askopenfilename(filetypes=[("Files JPEG", "*.jpg;*.jpeg")])
        # Check if a valid file is selected.
        if file_path:
            # Transform picture and save in class variable for the prediction.
            self.picture = ImT.transform_image(file_path, from_path=True)

            # Resize the image to fit the canvas.
            image_scaled = ImT.resize_image(file_path, self.canvas.winfo_width(), self.canvas.winfo_height(), from_path=True)
            # Change the image to tk format.
            self.image_tk = ImT.get_image_tk(image_scaled)
            # Show the image in the canvas.
            self.canvas.itemconfig(self.image_container, image=self.image_tk)
            self.canvas.update()


    # Callback of Load image from clipboard button.
    def load_image_from_cb(self):
        # Get the capture from the clipboard.
        capture = ImageGrab.grabclipboard()
        #Check if there was any picture in the clipboard.
        if capture is not None:
            # Convert the capture to bytes format bytes to open with Pillow
            imagen_bytes = io.BytesIO()
            capture.save(imagen_bytes, format="PNG")
            # Open the image with Pillow
            image = Image.open(imagen_bytes).convert("RGB")

            # Transform picture and save in class variable for the prediction.
            self.picture = ImT.transform_image(image)

            # Resize the image to fit the canvas.
            image_scaled = ImT.resize_image(image,  self.canvas.winfo_width(), self.canvas.winfo_height())
            # Change the image to tk format.
            self.image_tk = ImT.get_image_tk(image_scaled)
            # Show the image in the canvas.
            self.canvas.itemconfig(self.image_container, image=self.image_tk)
            self.canvas.update()

        else:
            self.output_lb.config(text="No image in the clipboard")

    # Callback of the predict button.
    def predict_class(self):
        #First check if there is a picture in the class variable.
        if (self.picture is not None):
            # Call the predict function.
            text_result, confidence = ImC.predict(self.picture)

            #Set the result in the output label.
            self.output_lb.config(text=text_result)

            #Update the confidence bar with the value and color.
            self.bar_confidence.config(value=confidence)
            if (confidence >= 80) :
                self.bar_confidence.config(style="1.Horizontal.TProgressbar")
            if (confidence < 80 and confidence > 40):
                self.bar_confidence.config(style="2.Horizontal.TProgressbar")
            if (confidence <= 40) :
                self.bar_confidence.config(style="3.Horizontal.TProgressbar")

        else :
            self.output_lb.config(text="You must provide an image first")



if __name__ == '__main__':
    app = Main_screen()
    app.run()