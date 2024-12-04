import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import json

# Load the trained model
model = tf.keras.models.load_model('superclass_cnn_model.h5')

# Load class indices from JSON file
with open('class_indices.json', 'r') as json_file:
    class_indices = json.load(json_file)

# Reverse the class indices dictionary to get label names by index
class_labels = {v: k for k, v in class_indices.items()}

# Define function to preprocess image
def preprocess_image(image_path, target_size=(64, 64)):
    # Load the image and resize it
    img = load_img(image_path, target_size=target_size)
    # Convert the image to array
    img_array = img_to_array(img)
    # Expand dimensions to match model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image array
    img_array /= 255.0
    return img_array

# Define function to make a prediction and display top probabilities
def predict_image(image_path, class_labels):
    img_array = preprocess_image(image_path)
    # Get model predictions
    predictions = model.predict(img_array)[0]
    # Sort class probabilities in descending order
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_probs = predictions[sorted_indices]
    sorted_classes = [class_labels[i] for i in sorted_indices]
    
    # Get the top 5 classes and probabilities for Command Prompt
    top_classes = sorted_classes[:5]
    top_probs = sorted_probs[:5]

    # Print top 5 classes and probabilities to Command Prompt
    print("\nTop 5 Predictions:")
    for i, (cls, prob) in enumerate(zip(top_classes, top_probs), 1):
        print(f"{i}. {cls}: {prob:.2f}")

    # Get top 75 classes and probabilities for the diagram
    diagram_classes = sorted_classes[:75]
    diagram_probs = sorted_probs[:75]

    # Display top 75 class probabilities as a bar chart
    plt.figure(figsize=(12, 8))
    plt.barh(diagram_classes, diagram_probs, color='skyblue')
    plt.xlabel('Probability')
    plt.title('Top 75 Prediction Probabilities for Each Class')
    plt.gca().invert_yaxis()  # Invert y-axis to show highest probability at top
    plt.show()

# Open file dialog for user to select image
def select_image():
    # Initialize Tkinter root and hide main window
    root = Tk()
    root.withdraw()  # Hide root window
    # Open file dialog
    file_path = filedialog.askopenfilename(title="Select an image for prediction", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    root.destroy()  # Close Tkinter window after selection
    return file_path

# Select image and predict
image_path = select_image()
if image_path:
    print(f"Selected image path: {image_path}")
    predict_image(image_path, class_labels)
else:
    print("No image selected.")
