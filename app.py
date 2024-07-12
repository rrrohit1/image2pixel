# Approach 1: Direct Pixel Extraction

import numpy as np
from PIL import Image
from utils import is_black_and_white    

def direct_pixel_extraction(image_path):
    """
    Extracts pixel values from an image and returns a NumPy array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: A NumPy array containing the pixel values of the image. The shape of the array
        depends on whether the image is black and white or color. For black and white images, the shape
        is (height, width, 1) and for color images, the shape is (height, width, 3).

    """
    # Open the image using PIL
    img = Image.open(image_path)
    
    # Convert the image to RGB mode if it's not already
    img = img.convert('RGB')
    
    # Get the dimensions of the image
    width, height = img.size
    
    # store if the image is black and white in a variable
    isBnW = is_black_and_white(image_path)

    # Create a NumPy array to store the pixel values
    pixel_array = np.zeros((height, width, 1 if isBnW else 3), dtype=np.float32)
    
    # Iterate over each pixel in the image
    if isBnW:
         for y in range(height):
            for x in range(width):
                # Get the RGB values of the pixel
                r, g, b = img.getpixel((x, y))   
                
                # Store the R values in the NumPy array since the image is black and white
                # and all the RGB values are the same
                pixel_array[y, x] = [r/ 255.0]

    else:
        for y in range(height):
            for x in range(width):
                # Get the RGB values of the pixel
                r, g, b = img.getpixel((x, y))   
                
                # Store the RGB values in the NumPy array
                pixel_array[y, x] = [r/ 255.0, g/ 255.0, b/ 255.0]  # Normalize the values to the range [0, 1]
    
    return pixel_array

def main():
    # Example usage
    image_path = '/Users/rohitrawat/github-repos/image2pixel/test.png'
    result = direct_pixel_extraction(image_path)

    print(f"Image dimensions: {result.shape}")
    print("Sample pixel values:")
    print(result.flatten().reshape(-1, result.shape[0]))  # Print the first 5x5 pixels

if __name__ == "__main__":
    main()

