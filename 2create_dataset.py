import cv2 
import mediapipe as mp 
import os  
import pickle  
import numpy as np  
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

# Number of frames in each gesture sequence
SEQUENCE_LENGTH = 5  
# Number of features per hand (21 landmarks * 3 coordinates)
FEATURES_PER_HAND = 63  
# Total features for 2 hands
TOTAL_FEATURES = FEATURES_PER_HAND * 2  

mp_hands = mp.solutions.hands  # Initialize MediaPipe hands module
mp_drawing = mp.solutions.drawing_utils  # Initialize drawing utilities for MediaPipe
hands = mp_hands.Hands(
    static_image_mode=True,  # Set to True for processing static images
    max_num_hands=2,  # Maximum number of hands to detect
    min_detection_confidence=0.5  # Minimum confidence for detection
)

# Directory to store captured images
DATA_DIR = './data'  
# Directory to store processed landmark images
LANDMARKS_DIR = './data_landmarks'  
# Create landmarks directory if it doesn't exist
os.makedirs(LANDMARKS_DIR, exist_ok=True)  

data = []  # List to store features for static data
labels = []  # List to store labels corresponding to the features
sequence_data = []  # List to store sequences of features

# Function to process hand landmarks and returns normalized features.
def process_landmarks(hand_landmarks):
    
    # Extract x, y coordinates from landmarks
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
     
    # Calculate the bounding box of the hand
    min_x, max_x = min(x_coords), max(x_coords)  # Find the smallest and largest x coordinates
    min_y, max_y = min(y_coords), max(y_coords)  # Find the smallest and largest y coordinates

    # Calculate the width and height of the hand
    hand_width = max(max_x - min_x, 0.001)  # Calculate the width of the hand; ensure it's at least 0.001 to avoid division by zero
    hand_height = max(max_y - min_y, 0.001)  # Calculate the height of the hand; ensure it's at least 0.001 to avoid division by zero
    
    features = []  # List to store normalized features

    # Loop through each landmark (point) on the hand
    for lm in hand_landmarks.landmark:  
        # Normalize the x-coordinate
        # lm.x is the original x-coordinate of the landmark
        # min_x + hand_width / 2 calculates the center of the hand's bounding box
        # We subtract this center from lm.x to shift the x-coordinate so that the center is at 0
        # Then we divide by hand_width to scale the value between -0.5 and 0.5
        norm_x = (lm.x - (min_x + hand_width / 2)) / hand_width
        
        # Normalize the y-coordinate
        # lm.y is the original y-coordinate of the landmark
        # min_y + hand_height / 2 calculates the center of the hand's bounding box in the y direction
        # We subtract this center from lm.y to shift the y-coordinate so that the center is at 0
        # Then we divide by hand_height to scale the value between -0.5 and 0.5
        norm_y = (lm.y - (min_y + hand_height / 2)) / hand_height
        
        # Append the normalized x and y coordinates, along with the original z value
        # lm.z is the depth coordinate of the landmark (how far it is from the camera)
        features.extend([norm_x, norm_y, lm.z])  

    # Return the list of features containing normalized x, y coordinates and z values
    return features  

# Loop through each folder in the data directory
for class_folder in os.listdir(DATA_DIR):  
    # Get the path for the current class
    class_dir = os.path.join(DATA_DIR, class_folder)  
    # Get the path for landmarks of the current class
    landmark_dir = os.path.join(LANDMARKS_DIR, class_folder)  
    # Create landmarks directory for the class if it doesn't exist
    os.makedirs(landmark_dir, exist_ok=True)  
    
    # Print the current class being processed
    print(f"Processing class: {class_folder}")  
    
    # Dictionary to hold sequences of images
    sequences = {}  
    # Loop through each image in the class directory
    for img_name in os.listdir(class_dir):  
        # Check if the image name indicates it is part of a sequence
        if '_seq' in img_name:  
            # Extract the sequence number from the filename
            seq_num = img_name.split('_seq')[1].split('_')[0]  
            # If the sequence number is not already in the dictionary
            if seq_num not in sequences: 
                # Create a new list for this sequence 
                sequences[seq_num] = []  
            # Add the image to the corresponding sequence list
            sequences[seq_num].append(img_name)   
    
    # Loop through each sequence
    for seq_num, img_list in sequences.items():  
        sequence_features = []  # List to store features for the current sequence
        img_list.sort()  # Sort the images in the sequence
        
        # Loop through each image in the sequence
        for img_name in img_list:  
            try:
                # Get the full path of the image
                img_path = os.path.join(class_dir, img_name)  
                # Read the image using OpenCV
                img = cv2.imread(img_path)  
                # Check if the image was loaded successfully
                if img is None:  
                    # Print a warning if the image could not be loaded
                    print(f"⚠️ Could not load image: {img_path}")  
                    # Skip to the next image
                    continue  
                
                # Convert the image from BGR to RGB format
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Process the image to detect hands  
                results = hands.process(img_rgb)  
                
                # Initialize an array for features
                frame_features = np.zeros(TOTAL_FEATURES, dtype=np.float32) 
                # Counter for detected hands 
                hands_detected = 0  
                
                # Check if any hands were detected
                if results.multi_hand_landmarks:  
                    # Count the number of detected hands
                    hands_detected = len(results.multi_hand_landmarks)  
                    
                    # Loop through each detected hand (we can detect up to 2 hands)
                    # results.multi_hand_landmarks contains the landmarks for all detected hands
                    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                        # Calculate the starting index in the frame_features array for the current hand
                        # Each hand has a fixed number of features (FEATURES_PER_HAND)
                        start_index = hand_index * FEATURES_PER_HAND  
                        
                        # Process the landmarks of the current hand to extract features
                        # The process_landmarks function returns the normalized features for the hand
                        # Store these features in the frame_features array at the calculated starting index
                        frame_features[start_index:start_index + FEATURES_PER_HAND] = process_landmarks(hand_landmarks)  
                    
                    # Draw landmarks on the image for visualization
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                
                # Save the image with landmarks
                cv2.imwrite(os.path.join(landmark_dir, f"landmarks_{img_name}"), img)  
                # Add the features of the current frame to the sequence
                sequence_features.append(frame_features)  
                
                # If at least one hand was detected
                if hands_detected > 0:  
                    # Append the features to the static data list
                    data.append(frame_features)  
                    # Append the corresponding label (gesture class)
                    labels.append(class_folder)  
                    
            # Catch any exceptions that occur during processing
            except Exception as e:  
                print(f"⚠️ Error processing {img_name}: {str(e)}") 
                # Skip to the next image 
                continue  

        # Check if the sequence has the required number of frames
        if len(sequence_features) == SEQUENCE_LENGTH:  
            # Append the sequence data to the list
            sequence_data.append({  
                'features': np.array(sequence_features, dtype=np.float32),  # Store the features as a NumPy array
                'label': class_folder  # Store the corresponding label
            })

# Check if there are any complete sequences to save
if sequence_data:  
    # Open a file to write the sequence data
    with open('sequence_data.pickle', 'wb') as f:  
        pickle.dump(sequence_data, f)  # Save the sequence data using pickle
    print(f"Data for {len(sequence_data)} sequences saved.")  # Print the number of sequences saved

# Check if there is any static data to save
if data:  
    # Open a file to write the static data
    with open('static_data.pickle', 'wb') as f:  
        pickle.dump({'data': np.array(data, dtype=np.float32), 'labels': labels}, f)  # Save the static data and labels
    print(f"Static data of {len(data)} samples saved.")  # Print the number of static samples saved

cv2.destroyAllWindows()  # Close all OpenCV windows that may be open