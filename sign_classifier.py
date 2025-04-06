import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

# Load the model from a pickle file
model_dict = pickle.load(open('./model.p', 'rb'))  
# Extract the model from the loaded dictionary
model = model_dict['model']  

# Define the directory where the data is stored
DATA_DIR = './data'  
# Create a dictionary mapping indices to label names ({0: 'A', 1: 'B', 2: 'C'}) from the sorted list of class names in DATA_DIR
labels_dict = {i: name for i, name in enumerate(sorted(os.listdir(DATA_DIR)))}  
# Print the labels dictionary to the console
print(labels_dict)  

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)  

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands  # Access the hand tracking solution from MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Access drawing utilities for visualizing landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Access styles for drawing landmarks
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)  # Initialize the Hands model with specific parameters

while True:
    data_aux = []  # Initialize an empty list to store normalized landmark data
    x_coordinates = []  # Initialize an empty list to store x-coordinates of landmarks
    y_coordinates = []  # Initialize an empty list to store y-coordinates of landmarks

    success, frame = cap.read()  # Capture a frame from the video
    # Check if the frame was captured successfully
    if not success: 
        # Print an error message if the camera fails
        print("Camera error")  
        break  

    H, W, num_channels = frame.shape  # Get the height (H), width (W), and number of color channels (num_channels) of the captured frame    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame from BGR to RGB color space
    results = hands.process(frame_rgb)  # Process the RGB frame to detect hand landmarks

    # Check if any hand landmarks were detected
    if results.multi_hand_landmarks:  
        # Iterate over each detected hand
        for hand_landmarks in results.multi_hand_landmarks:  
            # Reset data for each hand
            data_aux = []  # Initialize an empty list to store normalized landmark data for the current hand
            x_coordinates = []  # Initialize an empty list to store x-coordinates of landmarks for the current hand
            y_coordinates = []  # Initialize an empty list to store y-coordinates of landmarks for the current hand

            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,  # Draw connections between hand landmarks
                mp_drawing_styles.get_default_hand_landmarks_style(),  # Get default style for landmarks
                mp_drawing_styles.get_default_hand_connections_style()  # Get default style for connections
            )

            # Iterate over each landmark in the detected hand
            for landmark in hand_landmarks.landmark:  
                x_coordinates.append(landmark.x)  # Append the x-coordinate of the landmark to the x_coordinates list
                y_coordinates.append(landmark.y)  # Append the y-coordinate of the landmark to the y_coordinates list

            # Normalize the landmark coordinates by subtracting the minimum x and y values
            for landmark in hand_landmarks.landmark:  
                data_aux.append(landmark.x - min(x_coordinates))  # Normalize x-coordinate
                data_aux.append(landmark.y - min(y_coordinates))  # Normalize y-coordinate

            # Calculate bounding box coordinates for the detected hand
            x1 = int(min(x_coordinates) * W) - 10  # Calculate the top-left x-coordinate of the bounding box
            y1 = int(min(y_coordinates) * H) - 10  # Calculate the top-left y-coordinate of the bounding box
            x2 = int(max(x_coordinates) * W) - 10  # Calculate the bottom-right x-coordinate of the bounding box
            y2 = int(max(y_coordinates) * H) - 10  # Calculate the bottom-right y-coordinate of the bounding box

            # Predict the sign based on the normalized landmark data and get probabilities
            probabilities = model.predict_proba([np.asarray(data_aux)])  
            predicted_index = np.argmax(probabilities)  # Get the index of the class with the highest probability
            predicted_character = labels_dict[predicted_index]  # Get the predicted character
            confidence_score = probabilities[0][predicted_index] * 100  # Calculate the confidence score as a percentage

            # Draw a black rectangle with a thickness of 4 pixels around the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  

            # Put the predicted character text and confidence score in green color
            cv2.putText(frame, f'{predicted_character} ({confidence_score:.2f}%)', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)  

    # Show the frame with the drawn landmarks and predictions in a window titled 'Sign Recognition'
    cv2.imshow('Sign Recognition', frame)  
    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows