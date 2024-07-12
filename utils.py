import numpy as np
from PIL import Image

def is_black_and_white(image_path):
    """
    Check if an image is black and white.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    bool: True if the image is black and white, False otherwise.
    """
    with Image.open(image_path) as img:
        # Convert the image to RGB if it's not
        img = img.convert('RGB')
        # Access pixel data
        pixels = list(img.getdata())
        for pixel in pixels:
            r, g, b = pixel
            if r != g != b:  # If any of the R, G, B values differ, it's not black and white
                return False
        return True