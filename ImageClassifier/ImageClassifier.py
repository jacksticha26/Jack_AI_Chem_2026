# Imports
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def load_and_preprocess_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at: {img_path}")

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array

def main():
    # Your image path
    img_path = "/Users/jacksticha/Desktop/guitar.jpg"

    # Updated confidence threshold
    CONFIDENCE_THRESHOLD = 0.50

    # Load model
    model = VGG16(weights='imagenet')

    # Preprocess image
    processed_image = load_and_preprocess_image(img_path)

    # Predict
    preds = model.predict(processed_image)

    # Decode top 5 predictions
    decoded = decode_predictions(preds, top=5)[0]

    print("\nFiltered Predictions (≥ 50% confidence):")

    found = False
    for (imagenet_id, label, confidence) in decoded:
        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"{label} ({confidence:.4f})")
            found = True

    if not found:
        print("No predictions met the 50% confidence threshold.")

if __name__ == "__main__":
    main()