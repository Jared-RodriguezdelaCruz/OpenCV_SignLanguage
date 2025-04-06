import os 
import cv2  

# Define the directory where images will be saved
DATA_DIR = 'data' 

if not os.path.exists(DATA_DIR):  
    os.makedirs(DATA_DIR) 

# Set the number of images to capture for each class
DATASET_SIZE = 100  
# Define the standard size for the images
IMG_SIZE = (224, 224) 
# Start capturing video from the default camera (0)
cap = cv2.VideoCapture(0)  

# Display centered text at the top with line breaks
def show_text(frame, texto):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8  
    thickness = 2 
    color = (0, 255, 0) 
    line_spacing = 40  
    y = 40 
    
    # Split the text into lines
    lines = texto.split('\n')
    
    # Loop through each line of text
    for line in lines:  
        # Get the size of the text
        size = cv2.getTextSize(line, font, scale, thickness)[0]  
        # Calculate the x position to center the text
        x = (frame.shape[1] - size[0]) // 2  
        # Draw a filled rectangle behind the text for better visibility
        cv2.rectangle(frame, (x - 10, y - size[1] - 10), (x + size[0] + 10, y + 10), (0, 0, 0), cv2.FILLED)
        # Draw the text on the frame
        cv2.putText(frame, line, (x, y), font, scale, color, thickness, cv2.LINE_AA) 
        # Move the y position down for the next line
        y += line_spacing 

try:
    while True:
        # Ask for the name of the class
        class_name = input("\nSign's name ('exit' to terminate): ").strip()
        
        if class_name.lower() == 'exit': 
            break
            
        # Create the path for the class directory
        class_dir = os.path.join(DATA_DIR, class_name)  
        # Create the directory if it doesn't exist
        if not os.path.exists(class_dir): 
            os.makedirs(class_dir)  

        # Inform the user about the capture process
        print(f"\nCapturing {DATASET_SIZE} images for: {class_name}")  

        # Find the next available image index
        existing_files = os.listdir(class_dir)
        existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.startswith(class_name)]
        next_index = max(existing_indices, default=-1) + 1  # Start from the next available index

        while True: 
            # Capture a frame from the camera
            success, frame = cap.read()  
            # Check if the frame was captured successfully
            if not success: 
                # Print an error message if the camera fails
                print("Error in camera")  
                break
            
            show_text(frame, f'Press "Q" to start capturing images for: {class_name}') 
            # Display the frame in a window
            cv2.imshow('Sign Capturer', frame)  
 
            # Wait until the user presses 'Q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit the loop if 'Q' is pressed

        # Countdown before capturing images
        countdown = 3
        while countdown > 0:
            # Capture a frame from the camera
            success, frame = cap.read()  
            if not success: 
                print("Error in camera")  
                break
            
            # Show the countdown message
            show_text(frame, f"Get ready! Capturing in {countdown}...") 
            # Display the countdown 
            cv2.imshow('Sign Capturer', frame)  
            
            # Decrease the countdown
            countdown -= 1
            
            # Wait for a short period to allow the countdown to be visible
            cv2.waitKey(1000)

        print("\nStart capturing!")
        # Loop to capture the specified number of images
        for i in range(DATASET_SIZE): 
            # Capture a frame from the camera
            success, frame = cap.read()  
            # Check if the frame was captured successfully
            if not success: 
                # Print an error message if the camera fails
                print("Error in camera") 
                break
            
            # Resize the image before saving it
            frame_resized = cv2.resize(frame, IMG_SIZE)  

            # Display the current image number and class name
            show_text(frame, f'Image {i + 1}/{DATASET_SIZE}\nSign: {class_name}') 
            # Show the frame in a window
            cv2.imshow('Sign Capturer', frame)  

            # Save the image in the folder (appending to existing images)
            filename = os.path.join(class_dir, f'{class_name}_{next_index}.jpg')  # Create the filename for the image
            cv2.imwrite(filename, frame_resized)  # Save the resized image to the specified path

            # Increment the index for the next image
            next_index += 1

            # Wait 75 ms between each capture to avoid blurry images
            cv2.waitKey(75)

        # Reset next_index for the next class capture
        next_index = 0  # Reset the index for the next class capture

# Catch any exceptions that occur
except Exception as e:  
    print(f"\nThere was an error: {e}")

finally: 
    cap.release()  # Release the camera resource
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print("\nProgram completed successfully")  # Inform the user that the program has finished