import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from torchvision import transforms
import torch


 
def transform_image(path_image, from_path=False):
    """@brief : Function to transform an image to pytorch tensor normalized.
       @param : path_image [string]: Path of the image. 
       @returns : batch_images [Tensor]: Pytorch tensor with batch of 4 images."""

    #Load and resize the image
    image = resize_image(path_image, from_path=from_path)

    # Normalize the image
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image_transformed = transform(image)

    # Stack in 4 pictures batch to avoid size errors.
    batch_images = torch.stack([image_transformed] * 4)

    # Show the image to the user.
    imshow(image_transformed)

    return batch_images


def imshow(img):
    """@brief : Function to show an image.
       @param : img [Tensor]: Tensor normalized of the image. """
         
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def resize_image(image, width=32, height=32, from_path=False):
    if (from_path):
        # Open the image with PIL
        image = Image.open(image)
    # Resize to 32x32
    image = image.resize((width, height))
    return image

def get_image_tk(image):
    return ImageTk.PhotoImage(image)