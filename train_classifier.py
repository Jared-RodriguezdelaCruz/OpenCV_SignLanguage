import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))  # Open the file in read-binary mode and load the data
    data = np.asarray(data_dict['data'])  # Convert the 'data' part of the dictionary to a NumPy array
    labels = np.asarray(data_dict['labels'])  # Convert the 'labels' part of the dictionary to a NumPy array
except FileNotFoundError:
    # If the file is not found, print an error message and exit the program
    print("Error: 'data.pickle' file not found. Please generate the data first.")
    exit()  # Exit the program if the file is not found


# Split the data into training and testing sets
# 80% of the data will be used for training and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


# Create a Random Forest Classifier model
# This line initializes a Random Forest Classifier. 
# A Random Forest is an ensemble learning method that uses multiple decision trees to improve classification accuracy.
#It combines predictions from several trees to make a final decision, which helps avoid fitting too closely to the training data
model = RandomForestClassifier()


# Train the model using the training data
# The fit method is called on the model to train it using the training data (x_train) and the corresponding labels (y_train).
# During this process, the model learns the patterns and relationships in the data to make predictions.
# The training process involves building multiple decision trees based on the training data.
model.fit(x_train, y_train)  


# Make predictions on the test data
# After the model is trained, we use it to make predictions on the test data (x_test).
# The predict method takes the test data as input and outputs the predicted labels (predictions).
# This allows us to see how well the model performs on unseen data.
predictions = model.predict(x_test)  


# Calculate the accuracy of the model
# The accuracy_score function compares the predicted labels (predictions) with the actual labels (y_test) from the test set.
# It calculates the proportion of correctly classified samples, giving us a measure of the model's performance.
# The score will be a value between 0 and 1, where 1 means perfect accuracy.
score = accuracy_score(predictions, y_test)  


# Print the accuracy as a percentage
print('{}% of samples were classified correctly!'.format(score * 100))  

f = open('model.p', 'wb')  # Open a file named 'model.p' in write-binary mode
pickle.dump({'model': model}, f)  # Save the trained model in the file as a dictionary
f.close()  # Close the file after saving
