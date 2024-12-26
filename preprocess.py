from PIL import Image
import numpy as np
import cv2
from rembg import remove


for i in range(15):
    # Load the input image (image 1)
    input_path = './dataset/source/'+str(i+1)+".jpg"
    output_path = './dataset/input/'+str(i+1)+".jpg"

    # Open the image using PIL
    with open(input_path, 'rb') as inp_file:
        input_image = inp_file.read()

    # Remove the background
    result = remove(input_image)

    # Save the resulting image
    with open(output_path, 'wb') as out_file:
        out_file.write(result)

    print(f"Background removed and saved to {output_path}")
