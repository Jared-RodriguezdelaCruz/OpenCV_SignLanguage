import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model
from collections import deque

# Sign Language Reference - https://youtu.be/BXD1wu6yEOQ?si=HZ0zVcQJAlV6Ezwq 

# Length of the sequence for dynamic prediction
SEQUENCE_LENGTH = 5 
# Confidence threshold for static predictions
STATIC_CONFIDENCE = 0.8  
# Confidence threshold for dynamic predictions
DYNAMIC_CONFIDENCE = 0.7  

# Load the static gesture recognition model
static_model = load_model('static_model.h5')  
# Load the dynamic gesture recognition model
sequence_model = load_model('sequence_model.h5')  

# Load label encoders for static and dynamic models
with open('static_label_encoder.pkl', 'rb') as f:
    static_le = pickle.load(f)
with open('sequence_label_encoder.pkl', 'rb') as f:
    sequence_le = pickle.load(f) 

# Access the hands module from MediaPipe
mp_hands = mp.solutions.hands  
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for real-time video processing
    max_num_hands=2,  # Allow detection of up to 2 hands
    min_detection_confidence=0.8  # Minimum confidence for hand detection
)

# Create a deque (double-ended queue) to store the most recent sets of features 
# extracted from video frames for dynamic prediction. It can hold a maximum of 
# SEQUENCE_LENGTH items (5 in this case). Older items will be removed automatically 
# when new ones are added.
sequence = deque(maxlen=SEQUENCE_LENGTH)  

# Initialize a variable to hold the current prediction message.
current_prediction = "Waiting for gesture..."

# Create another deque to store the last 10 predictions made by the model. 
# This helps in smoothing out the predictions over time by keeping track of 
# the most recent predictions, allowing the system to determine the most 
# frequently predicted gesture and reduce fluctuations in the output.
prediction_history = deque(maxlen=10)  

# Function to extract hand features from a video frame
def extract_features(frame):
    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    # Process the frame to detect hands
    results = hands.process(frame_rgb)  
    
    # Initialize an array for features (2 hands × 63 features)
    features = np.zeros(126, dtype=np.float32)  

    # List to store bounding boxes for each detected hand
    hand_boxes = []
    
    # Check if any hands are detected
    if results.multi_hand_landmarks:  
        # Loop through detected hands (up to 2)
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):  
            x_coords = [lm.x for lm in hand_landmarks.landmark]  # Get x coordinates of landmarks
            y_coords = [lm.y for lm in hand_landmarks.landmark]  # Get y coordinates of landmarks
            z_coords = [lm.z for lm in hand_landmarks.landmark]  # Get z coordinates of landmarks
            
            # Calculate the bounding box of the hand
            min_x, max_x = min(x_coords), max(x_coords)  # Find the smallest and largest x coordinates
            min_y, max_y = min(y_coords), max(y_coords)  # Find the smallest and largest y coordinates

            # Calculate the width and height of the hand
            hand_width = max(max_x - min_x, 0.001)  # Calculate the width of the hand; ensure it's at least 0.001 to avoid division by zero
            hand_height = max(max_y - min_y, 0.001)  # Calculate the height of the hand; ensure it's at least 0.001 to avoid division by zero
    
            # Calculate starting index for the current hand's features
            start_index = hand_index * 63  
            
            # Normalize the coordinates of each landmark
            # The following loop iterates over the indices and coordinates of the hand landmarks.
            # Each hand has 21 landmarks, and we are processing the x, y, and z coordinates for each.
            for lm_index, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
        
                # Normalize the x coordinate:
                # The normalization process adjusts the x coordinate to a range between -0.5 and 0.5.
                # 1. Subtracting the center of the hand (min_x + hand_width/2) from the x coordinate.
                # 2. Dividing the result by the hand's width to scale it relative to the hand's size.
                norm_x = (x - (min_x + hand_width/2)) / hand_width  # Normalize x coordinate
                
                # Normalize the y coordinate:
                # Similar to the x normalization, we adjust the y coordinate to a range between -0.5 and 0.5.
                # 1. Subtracting the center of the hand (min_y + hand_height/2) from the y coordinate.
                # 2. Dividing the result by the hand's height to scale it relative to the hand's size.
                norm_y = (y - (min_y + hand_height/2)) / hand_height  # Normalize y coordinate
                
                # Store the normalized coordinates in the features array:
                # The features array is structured to hold the normalized coordinates for each landmark.
                # Each hand has 21 landmarks, and each landmark has 3 coordinates (x, y, z).
                # The start_index variable indicates where to begin storing the features for the current hand.
                
                # Store the normalized x coordinate in the features array at the appropriate index.
                features[start_index + lm_index*3 + 0] = norm_x  
                
                # Store the normalized y coordinate in the features array at the appropriate index.
                features[start_index + lm_index*3 + 1] = norm_y  
                
                # Store the original z coordinate in the features array at the appropriate index.
                # The z coordinate is not normalized in this case; it is stored as is.
                features[start_index + lm_index*3 + 2] = z
            
            # Store bounding box for the current hand
            hand_boxes.append((int(min_x * frame.shape[1]), int(min_y * frame.shape[0]), 
                               int(max_x * frame.shape[1]), int(max_y * frame.shape[0])))
            
    # Return the extracted features
    return features, hand_boxes

# Open the default camera
cap = cv2.VideoCapture(0)  

while True:  
    # Read a frame from the camera
    success, frame = cap.read()  
    # Check if the frame was captured successfully
    if not success:  
        # Print an error message if not
        print("Camera error")  
        break

    # Extract hand features from the current frame
    features, hand_boxes = extract_features(frame)  

    # Check if features were successfully extracted
    if features is not None:  
        # Predict using the static model
        # The features are wrapped in a NumPy array to match the input shape expected by the model.
        # The model expects a batch of inputs, so we create a batch with a single sample by using np.array([features]).
        static_proba = static_model.predict(np.array([features]), verbose=0)[0]  
     
        # Get the index of the gesture with the highest probability
        static_pred_index = np.argmax(static_proba)  
        # Get the confidence score for the predicted gesture
        static_confidence = static_proba[static_pred_index]  

        # Add the extracted features to the sequence for dynamic prediction
        sequence.append(features)  

        # Check if the sequence has reached the required length
        if len(sequence) == SEQUENCE_LENGTH:  
            # Predict using the dynamic model
            # The sequence contains multiple sets of features (up to SEQUENCE_LENGTH) that represent the last few frames.
            # We wrap the sequence in a NumPy array to match the input shape expected by the model.
            # The model expects a batch of inputs, so we create a batch with a single sample by using np.array([sequence]).
            seq_proba = sequence_model.predict(np.array([sequence]), verbose=0)[0] 

            # Get the index of the gesture with the highest probability
            seq_pred_index = np.argmax(seq_proba)  
            # Get the confidence score for the predicted gesture
            seq_confidence = seq_proba[seq_pred_index]  

            # Compare dynamic and static predictions to determine the current prediction
            if seq_confidence > DYNAMIC_CONFIDENCE and seq_confidence > static_confidence:
                current_prediction = sequence_le.classes_[seq_pred_index]  # Update current prediction with dynamic prediction
                prediction_history.append(current_prediction)  # Store the current prediction in history
            elif static_confidence > STATIC_CONFIDENCE:
                current_prediction = static_le.classes_[static_pred_index]  # Update current prediction with static prediction
                prediction_history.append(current_prediction)  # Store the current prediction in history

        # Check if there are any predictions in history
        if prediction_history:  
            final_pred = max(set(prediction_history), key=prediction_history.count)  # Get the most frequent prediction
        else:
            final_pred = current_prediction  # If no history, use the current prediction


        # Draw rectangles and predictions for each detected hand
        for box in hand_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)  # Draw rectangle around hand
            cv2.putText(frame, f"Prediction: {final_pred}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
    # Show the frame with the prediction
    cv2.imshow('Sign Language Recognition', frame)  

    # Check if the 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  # Exit the loop

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows