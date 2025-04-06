import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create a Hands object for detecting hands in images
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)

# Directory where original images are stored
DATA_DIR = './data'  
# Directory where images with landmarks will be saved
LANDMARKS_DIR = './data_landmarks'  

# Create the directory for images with landmarks if it does not exist
if not os.path.exists(LANDMARKS_DIR):
    os.makedirs(LANDMARKS_DIR)

data = []  # List to store features (landmark coordinates)
labels = []  # List to store labels (class names)

# Iterate over each folder in the data directory
for class_folder in os.listdir(DATA_DIR):
    # Create a subfolder for the class (hand sign)
    landmark_folder = os.path.join(LANDMARKS_DIR, class_folder)
    # Create the directory if it doesn't exist
    if not os.path.exists(landmark_folder):
        os.makedirs(landmark_folder)  

    image_count = 0  # Counter for the number of processed images

    # Iterate over each image in the class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, class_folder)):
        img = cv2.imread(os.path.join(DATA_DIR, class_folder, img_path))  # Load the image
        # Check if the image was loaded successfully
        if img is None:  
            print(f"Error loading image: {os.path.join(DATA_DIR, class_folder, img_path)}")
            continue  

        # Convert the image from BGR to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # Process the image to detect hands
        results = hands.process(img_rgb)  
        # Check if any hands were detected
        if results.multi_hand_landmarks:  
            # Only process the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]  
            data_aux = []  # List to store features of the current hand

            # Extract the coordinates of the landmarks
            for landmark in hand_landmarks.landmark:  
                x = landmark.x  # Get the x coordinate
                y = landmark.y  # Get the y coordinate
                data_aux.append(x)  # Add the x coordinate to the feature list
                data_aux.append(y)  # Add the y coordinate to the feature list

            # Normalize the landmark coordinates based on image dimensions
            if len(data_aux) == 42:  
                data.append(data_aux)  # Add the features to the main data list
                labels.append(class_folder)  # Add the label to the labels list

                # Draw the landmarks on the image
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save the processed image in the corresponding subfolder
                image_count += 1  # Increment the image count
                landmark_image_path = os.path.join(landmark_folder, f"{class_folder}_{image_count}.jpg")  # Create the filename
                try:
                    cv2.imwrite(landmark_image_path, img)  # Save the image with landmarks
                except Exception as e:
                    print(f"Error saving image: {landmark_image_path}, {e}")

    print(f"Processed {image_count} images for class '{class_folder}'.")

# Check if any data has been collected before saving
if data and labels:
    # Open a file in write-binary mode
    with open('data.pickle', 'wb') as f:  
        # Save the data and labels as a dictionary
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Data successfully saved in 'data.pickle'.")  
else:
    print("No data found to save.")  

# Close all OpenCV windows
cv2.destroyAllWindows()  