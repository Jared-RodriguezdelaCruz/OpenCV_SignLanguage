import cv2 
import os  
import numpy as np  

# Directory where the captured data will be stored
DATA_DIR = 'data' 
# Check if the directory exists, if not, create the directory
if not os.path.exists(DATA_DIR):  
    os.makedirs(DATA_DIR)  

# Number of sequences to capture for each gesture
DATASET_SIZE = 200  
# Image size for better detection (width, height)
IMG_SIZE = (640, 480)
# Number of frames per dynamic sequence  
SEQUENCE_LENGTH = 5  
# Start capturing video from the default camera (0)
cap = cv2.VideoCapture(0)  

# Function to display text on the video frame
def show_text(frame, text, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX  
    scale = 0.8  
    thickness = 2  
    y = 40  

    # Split the text into multiple lines
    lines = text.split('\n')
    # Loop through each line of text  
    for line in lines:  
        # Get the size of the text
        size = cv2.getTextSize(line, font, scale, thickness)[0]
        # Center the text horizontally  
        x = (frame.shape[1] - size[0]) // 2  
        # Draw a filled rectangle behind the text
        cv2.rectangle(frame, (x - 10, y - size[1] - 10), (x + size[0] + 10, y + 10), (0, 0, 0), cv2.FILLED)
        # Put the text on the frame
        cv2.putText(frame, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)  
        # Move down for the next line of text
        y += 40  

# Function to capture sequences of frames for a specific gesture
def capture_sequences(class_name):
    # Create a directory for the specific gesture
    class_dir = os.path.join(DATA_DIR, class_name)  
    # Check if the directory exists, if not, create the directory
    if not os.path.exists(class_dir):  
        os.makedirs(class_dir)  

    # Tell user about the capture process
    print(f"\nCapturing {DATASET_SIZE} sequences for: {class_name}")  

    # List to store the current sequence of frames
    current_sequence = []  
    # Flag to indicate if capturing is active
    capture_active = False

    sequence_count = 0

    # Loop until the desired number of sequences is captured
    while sequence_count < DATASET_SIZE:
        # Read a frame from the camera
        success, frame = cap.read()  
        # Check if the frame was captured successfully
        if not success:  
            # Print an error message if not
            print("Camera error")  
            break
        
        # Display instructions if not currently capturing
        if not capture_active:
            show_text(frame, f'Press "Q" to start capturing sequences\nGesture: {class_name}\nSequences captured: {sequence_count}/{DATASET_SIZE}')
        # If capturing is active
        else:  
            show_text(frame, f'Capturing sequence {sequence_count+1}/{DATASET_SIZE}\nPerform the gesture now...\nFrames remaining: {SEQUENCE_LENGTH - len(current_sequence)}', (0, 0, 255))
        
        # Show the current frame in a window
        cv2.imshow('Gesture Capturer', frame)  

        # Wait for a key press for 1 millisecond
        key = cv2.waitKey(1)  
        # Check if 'Q' is pressed to start capturing
        if key & 0xFF == ord('q') and not capture_active:  
            # Set the capturing flag to True
            capture_active = True  
            # Reset the current sequence list
            current_sequence = []  
        
        # If capturing is active
        if capture_active:  
            # Resize the frame to the specified image size
            frame_resized = cv2.resize(frame, IMG_SIZE)
            # Add the resized frame to the current sequence
            current_sequence.append(frame_resized)  
            
            # Check if the current sequence is complete
            if len(current_sequence) >= SEQUENCE_LENGTH:  
                # Loop through each frame in the current sequence like 'gestureName_seqX_frameY.jpg'
                for i, frame_seq in enumerate(current_sequence):
                    # Create a filename for each frame in the sequence
                    filename = os.path.join(class_dir, f'{class_name}_seq{sequence_count}_frame{i}.jpg')  
                    # Save the frame as an image file
                    cv2.imwrite(filename, frame_seq)  
                
                print(f"Sequence {sequence_count + 1} saved")  # Tell user that the sequence has been saved
                current_sequence = []  # Reset the current sequence list
                capture_active = False  # Reset the capturing flag
                sequence_count += 1
                cv2.waitKey(500)  # Wait for half a second before capturing the next sequence

try:
    while True:  
        # Prompt the user for the gesture name
        class_name = input("\nGesture name ('exit' to finish): ").strip()  
        
        # Check if the user wants to exit
        if class_name.lower() ==  "exit": 
            # Exit the loop
            break  
            
        # Call the function to capture sequences for the entered gesture
        capture_sequences(class_name)  

# Catch any exceptions that occur during execution
except Exception as e:  
    print(f"\nError: {e}")

finally:
    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print("\nProgram completed successfully")
