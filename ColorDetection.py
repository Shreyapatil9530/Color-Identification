# Import necessary libraries
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

# Specify the path to your image
image_path = 'sample_image.jpg'  # Ensure this path is correct

# Read the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image at {image_path}. Please check the file path and ensure the file exists.")
else:
    print("The type of this input is {}".format(type(image)))
    print("Shape: {}".format(image.shape))

    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

    # Function to preprocess the image and fit KMeans
    def get_colors(image, k=5, image_processing_size=None):
        # Resize image if image_processing_size is set
        if image_processing_size is not None:
            image = cv2.resize(image, image_processing_size, interpolation=cv2.INTER_AREA)
        
        # Reshape the image to be a list of pixels
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(image)
        
        # Get the most common colors
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        return colors, labels

    # Function to convert RGB to hex
    def rgb_to_hex(rgb_color):
        hex_color = "#"
        for i in rgb_color:
            hex_color += ("{:02x}".format(int(i)))
        return hex_color

    # Function to plot the colors
    def plot_colors(colors, labels):
        # Create a histogram of label frequencies
        label_counts = Counter(labels)
        total_count = sum(label_counts.values())
        
        # Sort the labels based on the count
        label_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))
        
        # Create a figure to display the colors
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        start = 0
        for idx, count in label_counts.items():
            color = colors[idx]
            end = start + (count / total_count)
            
            ax.add_patch(plt.Rectangle((start, 0), end - start, 1, color=[color/255 for color in colors[idx]]))
            start = end
            
            ax.text((start + end) / 2, 0.5, rgb_to_hex(color), ha='center', va='center', color='white', fontsize=16)
        
        ax.set_axis_off()
        plt.show()

    # Get colors and labels
    colors, labels = get_colors(image_rgb, k=5)

    # Plot the colors
    plot_colors(colors, labels)
