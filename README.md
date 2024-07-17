# Image Pixel Extractor

This Python application extracts pixel values from images and normalizes them to the range [0, 1]. It supports both grayscale and RGB images. For grayscale images, it extracts a single intensity value per pixel, while for RGB images, it extracts and normalizes the three RGB values per pixel.

## Installation

Before running the Image Pixel Extractor, you need to set up a Python environment with all the necessary dependencies. This project uses Conda for environment management. Follow these steps to create and activate the Conda environment:

1. **Ensure Conda is Installed**: First, make sure you have Conda installed on your system. If not, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual).

2. **Create the Conda Environment**: Open a terminal and navigate to the project directory. Then, run the following command to create a Conda environment from the `environment.yaml` file:

    ```bash
    conda env create -f environment.yaml
    ```

3. **Activate the Environment**: Once the environment is ready, activate it using:

    ```bash
    conda activate image2pixel
    ```

    Make sure to replace `image2pixel` with the name of your Conda environment if is specified as anything else in the `environment.yaml` file.

## Usage

To use app.py, navigate to the directory containing the script in your terminal and run the following command:

```bash
python app.py <image_path> [--method METHOD]
```

- <image_path>: The path to the input image file you wish to process.
- --method: Optional. Specifies the method to use for pixel extraction. Can be either 'direct' or 'pretrained'. The default method is 'direct'.

### Examples

To process an image using the direct method:

```bash
python app.py /path/to/image.jpg 
```

To process an image using a pretrained model:

```bash
python app.py /path/to/image.jpg --method pretrained
```

## Features

- Extract and normalize pixel values from images.
- Support for both grayscale and RGB images.
- Outputs a NumPy array containing the normalized pixel values.

## Direct VS Pretrained approach

Below is a comparison of both methods based on various factors.

| Factor              | Direct Method                          | Pretrained Method                       |
|---------------------|----------------------------------------|-----------------------------------------|
| **Complexity**      | Simple, direct manipulation of pixels. | Uses advanced pretrained models.        |
| **Speed**           | Potentially faster for specific cases. | May introduce computational overhead.   |
| **Dependencies**    | Fewer, simpler dependencies.           | Requires deep learning frameworks and pretrained models. |
| **Robustness**      | May require manual tuning.             | More robust to variations in images.    |
| **Ease of Use**     | Straightforward for basic usage.       | Simplifies complex image processing tasks. |
| **Transparency**    | High, easier to debug.                 | Lower, due to the complexity of models. |
| **Use Case**        | Suitable for simple applications or when processing speed is critical. | Better for high accuracy and handling diverse image types. |

## Performance Metrics

The script measures the time taken to convert an image to a pixel array using both methods. It also prints the method used, the image type, the shape of the resulting pixel array, and its data type. For grayscale images, it displays the first 5x5 pixel values, and for RGB images, it shows the first 5 pixel values in the RGB channels.

## Error Handling

The script includes error handling to catch and report any issues that occur during the image conversion process.
