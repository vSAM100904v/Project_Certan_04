import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import json

# Load the trained model
model = tf.keras.models.load_model('superclass_cnn_model_with_not_butterfly.h5')

# Load class indices from JSON file
with open('class_indices.json', 'r') as json_file:
    class_indices = json.load(json_file)

# Reverse the class indices dictionary to get label names by index
class_labels = {v: k for k, v in class_indices.items()}

# Define function to preprocess image
def preprocess_image(image_path, target_size=(64, 64)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Define function to make a prediction
# Define function to make a prediction
def predict_image(image_path, class_labels):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)[0]
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_probs = predictions[sorted_indices]
    sorted_classes = [class_labels[i] for i in sorted_indices]
    
    # Display top-5 predictions in the terminal
    print("\nTop-5 Predictions:")
    for i in range(5):
        print(f"{i + 1}. {sorted_classes[i]}: {sorted_probs[i]:.2f}")
    
    top_class = sorted_classes[0]
    top_probability = sorted_probs[0]

    if top_class == "not_butterfly" and top_probability > 0.6:
        print(f"\nThe model predicts this image as 'not a butterfly' with a probability of {top_probability:.2f}")
    else:
        print(f"\nThe model predicts this image as '{top_class}' with a probability of {top_probability:.2f}")

    # Display bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_classes[:5], sorted_probs[:5], color='skyblue')  # Plot top-5 predictions
    plt.xlabel('Probability')
    plt.title('Top-5 Prediction Probabilities')
    plt.gca().invert_yaxis()
    plt.show()


# Select image using file dialog
def select_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image for prediction",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    root.destroy()
    return file_path

# Run prediction
image_path = select_image()
if image_path:
    print(f"Selected image path: {image_path}")
    predict_image(image_path, class_labels)
else:
    print("No image selected.")
