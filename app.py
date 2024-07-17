# Approach 1: Direct Pixel Extraction

import cv2
import argparse
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor
import time

def image_to_pixel_array_direct(image_path):
    """
    Convert an image file to a pixel array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing the pixel array and the image format.
            - The pixel array is a numpy array representing the image.
            - The image format is a string indicating the format of the image.
              Possible values are 'grayscale', 'color', or 'color with alpha'.

    Raises:
        ValueError: If the image file cannot be read or if the image format is unsupported.
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Unable to read the image. Please check the file path.")

    # Check if the image is grayscale or color
    if len(img.shape) == 2:
        # It's grayscale
        return img, "grayscale"
    elif len(img.shape) == 3:
        if img.shape[2] == 3:
            # It's a color image (BGR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "color"
        elif img.shape[2] == 4:
            # It's a color image with alpha channel (BGRA)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA), "color with alpha"
    
    raise ValueError("Unsupported image format")

# Approach 2: Pixel Extraction using Pretrained Model

def image_to_pixel_array_pretrained(image_path):
    """
    Convert an image file to a pixel array using a ViT preprocessor.

    Args:
        image_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing the pixel array and the image format.
            - The pixel array is a numpy array representing the image.
            - The image format is always 'color' for this method.

    Raises:
        ValueError: If the image file cannot be read.
    """
    # Load the image
    try:
        image = Image.open(image_path)
    except IOError:
        raise ValueError("Unable to read the image. Please check the file path.")

    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Initialize the ViT preprocessor
    preprocessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    # Preprocess the image
    inputs = preprocessor(images=image, return_tensors="np")
    
    # Get the pixel values
    pixel_array = inputs['pixel_values'][0]
    
    # The pixel_array will be in the shape (3, height, width)
    # Let's transpose it to (height, width, 3) for consistency with the original code
    pixel_array = np.transpose(pixel_array, (1, 2, 0))

    return pixel_array, "color"

def main():
    """
    Convert an image to its pixel constituents in a NumPy array using either direct or pretrained method.

    Usage:
        python app.py <image_path> [--method {direct,pretrained}]

    Arguments:
        image_path (str): Path to the input image file.
        --method (str): Method to use for pixel extraction. Either 'direct' or 'pretrained'. Default is 'direct'.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Convert an image to its pixel constituents in a NumPy array.")
    parser.add_argument("image_path", help="Path to the input image file")
    parser.add_argument("--method", choices=['direct', 'pretrained'], default='direct',
                        help="Method to use for pixel extraction. Either 'direct' or 'pretrained'. Default is 'direct'.")
    args = parser.parse_args()

    try:
        start_time = time.time()
        if args.method == 'direct':
            pixel_array, img_type = image_to_pixel_array_direct(args.image_path)        
        else:  # args.method == 'pretrained'
            pixel_array, img_type = image_to_pixel_array_pretrained(args.image_path)
        end_time = time.time()
        print(f"Time taken for {args.method} method: {round(end_time - start_time, 5)} seconds")

        print(f"Method used: {args.method}")
        print(f"Image type: {img_type}")
        print(f"Array shape: {pixel_array.shape}")
        print(f"Array data type: {pixel_array.dtype}")
        
        if img_type == "grayscale":
            print("\nFirst 5x5 pixel values:")
            print(pixel_array[:5, :5])
        else:
            print("\nFirst 5 pixel values (RGB):")
            print(pixel_array[:5, 0, :])

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()