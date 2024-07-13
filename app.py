# Approach 1: Direct Pixel Extraction

import cv2
import argparse

def image_to_pixel_array(image_path):
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

def main():
    """
    Convert an image to its pixel constituents in a NumPy array.

    Usage:
        python app.py <image_path>

    Arguments:
        image_path (str): Path to the input image file.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Convert an image to its pixel constituents in a NumPy array.")
    parser.add_argument("image_path", help="Path to the input image file")
    args = parser.parse_args()

    try:
        pixel_array, img_type = image_to_pixel_array(args.image_path)
        
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