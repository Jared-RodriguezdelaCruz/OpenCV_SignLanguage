import os
import pickle  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  
from tensorflow.keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU use (ignore GPU)

# Number of frames in each sequence
SEQUENCE_LENGTH = 5  
# 21 landmarks * 3 coordinates (x, y, z) for one hand
FEATURES_PER_HAND = 63  
# For two hands
TOTAL_FEATURES = FEATURES_PER_HAND * 2  

# Load and validate data with shape checks
def load_and_validate_data():
    try:
        # Load static data from a pickle file
        with open('static_data.pickle', 'rb') as f:
            static_data = pickle.load(f)
            static_data['data'] = np.array(static_data['data'], dtype=np.float32)  # Convert data to float32
            
        # Load sequence data from a pickle file
        with open('sequence_data.pickle', 'rb') as f:
            sequence_data = pickle.load(f)
            
            # List to store valid sequences
            valid_sequences = []  
            # Loop for each sequence in the sequence data
            for seq in sequence_data:
                # Check if the shape of features matches the expected shape
                if seq['features'].shape == (SEQUENCE_LENGTH, TOTAL_FEATURES):
                    # Add valid sequence to the list
                    valid_sequences.append(seq)  
            
            print(f"\nTotal sequences: {len(sequence_data)}")  # Print total number of sequences
            print(f"Valid sequences: {len(valid_sequences)}")  # Print number of valid sequences
            
            # Warn if any sequences were discarded
            if len(valid_sequences) < len(sequence_data):
                print(f"⚠️ {len(sequence_data) - len(valid_sequences)} invalid sequences were discarded")
            
            sequence_data = valid_sequences  # Update sequence data to only include valid sequences
            
        return static_data, sequence_data  # Return loaded and validated data
    
    # Print error message if loading fails
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")  
        return None, None  # Return None if there was an error

# Prepare static data with validation
def prepare_static_data(static_data):
    try:
        x = static_data['data']  # Get features from static data
        y = static_data['labels']  # Get labels from static data
        
        # Check dimensions of features
        if x.shape[1] != TOTAL_FEATURES:
            raise ValueError(f"Incorrect shape in static data. Expected: (n, {TOTAL_FEATURES}), Obtained: {x.shape}")
        
        # Create a label encoder
        le = LabelEncoder()  
        # Encode labels into integers
        y_labels_encoded = le.fit_transform(y)  
        # Convert integer labels to categorical format
        y_labels_categorical = to_categorical(y_labels_encoded)  
        
        # Split data into training and testing sets
        # The 'test_size=0.2' parameter indicates that 20% of the data will be used for testing, while 80% will be used for training.
        # The 'random_state=42' parameter ensures that the split is reproducible; using the same random state will yield the same split every time the code is run.
        # Return the split data and the label encoder
        return train_test_split(x, y_labels_categorical, test_size=0.2, random_state=42), le
    
    # Print error message if preparation fails
    except Exception as e:
        print(f"\n❌ Error preparing static data: {e}")  
        return None, None  # Return None if there was an error

# Prepare sequence data with validation
def prepare_sequence_data(sequence_data):
    try:
        # Stack features from each sequence into a 3D array
        x = np.stack([seq['features'] for seq in sequence_data])
        # Get labels for each sequence
        y = [seq['label'] for seq in sequence_data]  
        
        # Check dimensions of features
        if x.shape[1:] != (SEQUENCE_LENGTH, TOTAL_FEATURES):
            raise ValueError(f"Incorrect shape in sequences. Expected: (n, {SEQUENCE_LENGTH}, {TOTAL_FEATURES}), Obtained: {x.shape}")
        
        # Create a label encoder
        le = LabelEncoder()  
        # Encode labels into integers
        y_labels_encoded = le.fit_transform(y)  
        # Convert integer labels to categorical format, required format for Keras models
        y_labels_categorical = to_categorical(y_labels_encoded)  
        
        # Split data into training and testing sets
        # The 'test_size=0.2' parameter indicates that 20% of the data will be used for testing, while 80% will be used for training.
        # The 'random_state=42' parameter ensures that the split is reproducible; using the same random state will yield the same split every time the code is run.
        # Return the split data and the label encoder
        return train_test_split(x, y_labels_categorical, test_size=0.2, random_state=42), le
    
    # Print error message if preparation fails
    except Exception as e:
        print(f"\n❌ Error preparing sequences: {e}")  
        return None, None  # Return None if there was an error

# Build model for static data
def build_static_model(input_shape, num_classes):
    # Create a Sequential model, which allows us to build a neural network layer by layer.
    model = Sequential([
        # First dense layer with 256 units (neurons).
        # 'Dense' means that each neuron in this layer is connected to every neuron in the previous layer.
        # 'activation='relu'' means we use the ReLU activation function, which helps the model learn complex patterns.
        Dense(256, activation='relu', input_shape=input_shape),
        
        # Batch normalization layer normalizes the output of the previous layer.
        # This helps stabilize and speed up the training process.
        BatchNormalization(),
        
        # Dropout layer randomly sets a fraction (40% here) of the input units to 0 during training.
        # This helps prevent overfitting, which is when the model learns the training data too well and performs poorly on new data.
        Dropout(0.4),
        

        # Second dense layer with 128 units.
        # Again, we use the 'relu' activation function to help the model learn.
        Dense(128, activation='relu'),
        
        # Another batch normalization layer to normalize the output of the second dense layer.
        BatchNormalization(),
        
        # Another dropout layer, this time with a 30% dropout rate.
        Dropout(0.3),
        
        # Output layer with 'num_classes' units.
        # The 'softmax' activation function is used here, which converts the output into probabilities for each class.
        # This is useful for multi-class classification problems, where we want to predict one class out of many.
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model to prepare it for training.
    # The 'Adam' optimizer is used for updating the model weights during training.
    # The learning rate (0.001) controls how much to change the model in response to the estimated error each time the model weights are updated.
    # 'categorical_crossentropy' is the loss function used for multi-class classification problems.
    # 'metrics=['accuracy']' means we want to track the accuracy of the model during training and testing.
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Return the compiled model so it can be trained later.
    return model

# Build LSTM model for sequential data
def build_sequence_model(input_shape, num_classes):
    # Create a Sequential model, which allows us to build a neural network layer by layer.
    model = Sequential([
        # First LSTM layer with 128 units (neurons).
        # LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) that is well-suited for sequence prediction problems.
        # 'return_sequences=True' means that this layer will return the full sequence of outputs for each input sequence,
        # which is necessary for stacking another LSTM layer on top.
        LSTM(128, return_sequences=True, input_shape=input_shape),
        
        # Batch normalization layer normalizes the output of the previous layer.
        # This helps stabilize and speed up the training process by reducing internal covariate shift.
        BatchNormalization(),
        
        # Dropout layer randomly sets a fraction (40% here) of the input units to 0 during training.
        # This helps prevent overfitting, which occurs when the model learns the training data too well and performs poorly on new data.
        Dropout(0.4),
        

        # Second LSTM layer with 256 units.
        # This layer processes the output from the first LSTM layer.
        LSTM(256),  
        
        # Another batch normalization layer to normalize the output of the second LSTM layer.
        BatchNormalization(),
        
        # Another dropout layer, this time with a 50% dropout rate.
        Dropout(0.5),
        
        # Output layer with 'num_classes' units.
        # The 'softmax' activation function is used here, which converts the output into probabilities for each class.
        # This is useful for multi-class classification problems, where we want to predict one class out of many.
        Dense(num_classes, activation='softmax')  # Output layer with softmax activation for multi-class classification
    ])
    
    # Compile the model to prepare it for training.
    # The 'Adam' optimizer is used for updating the model weights during training.
    # The learning rate (0.0005) controls how much to change the model in response to the estimated error each time the model weights are updated.
    # 'categorical_crossentropy' is the loss function used for multi-class classification problems.
    # 'metrics=['accuracy']' means we want to track the accuracy of the model during training and testing.
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',  
        metrics=['accuracy']  
    )
    
    # Return the compiled model so it can be trained later.
    return model  # Return the compiled model

# Function for training the model
def train_model(model, x_train, y_train, x_test, y_test, model_name, epochs=50, patience=5):
    # Define callbacks for early stopping and model checkpointing.
    # Callbacks are functions that can be called at certain points during training to perform specific actions.
    callbacks = [
        # EarlyStopping callback will stop training if the validation loss does not improve for a specified number of epochs (patience).
        # 'restore_best_weights=True' means that the model will revert to the best weights observed during training when stopping.
        EarlyStopping(patience=patience, restore_best_weights=True),
        # ModelCheckpoint callback saves the model after every epoch if the validation loss improves.
        # The model is saved with a filename that includes the model name, allowing for easy identification.
        ModelCheckpoint(f'best_{model_name}_model.h5', save_best_only=True)
    ]
    
    # Train the model using the training data (X_train, y_train) and validate it using the test data (x_test, y_test).
    # The 'fit' method is used to train the model.
    history = model.fit(
        x_train, y_train,  # Training data and corresponding labels
        validation_data=(x_test, y_test),  # Data used for validation during training
        epochs=epochs,  # Number of epochs to train the model
        batch_size=32,  # Number of samples to process before updating the model weights
        callbacks=callbacks,  # Callbacks defined earlier for managing training
        verbose=1  # Print progress during training (1 means progress will be printed for each epoch)
    )
    
    # Save the trained model
    model.save(f'{model_name}_model.h5')  
    # Return the training history
    return history  

# Function to train and evaluate the model
def train_and_evaluate():
    # Load and validate data from the source.
    # This function is expected to return two datasets: static data and sequence data.
    static_data, sequence_data = load_and_validate_data()
    
    # Check if static data is available and contains data.
    if static_data and len(static_data['data']) > 0:
        # Indicate that we are starting to train the static model
        print("\n=== TRAINING STATIC MODEL ===")  
        
        # Prepare the static data for training and testing.
        # This function will return the training and testing sets (x_train, x_test, y_train, y_test)
        # and a label encoder for the static data.
        (x_train, x_test, y_train, y_test), static_le = prepare_static_data(static_data)
        
        # Check if the training data is not None before proceeding.
        if x_train is not None:
            # Build the static model using the specified input shape and number of classes.
            model = build_static_model((TOTAL_FEATURES,), y_train.shape[1])
            
            # Train the model using the training data and validate it with the test data.
            train_model(model, x_train, y_train, x_test, y_test, 'static', epochs=50) 
            
            # Save the label encoder for static data to a file for future use.
            with open('static_label_encoder.pkl', 'wb') as f:
                pickle.dump(static_le, f)
    
    # Check if sequence data is available and contains data.
    if sequence_data and len(sequence_data) > 0:
        # Indicate that we are starting to train the dynamic model
        print("\n=== TRAINING DYNAMIC MODEL ===")  
        
        # Prepare the sequence data for training and testing.
        # This function will return the training and testing sets (x_train, x_test, y_train, y_test)
        # and a label encoder for the sequence data.
        (x_train, x_test, y_train, y_test), seq_le = prepare_sequence_data(sequence_data)
        
        # Check if the training data is not None before proceeding.
        if x_train is not None:
            # Build the sequence model using the specified input shape and number of classes.
            model = build_sequence_model((SEQUENCE_LENGTH, TOTAL_FEATURES), y_train.shape[1])
            
            # Train the model using the training data and validate it with the test data.
            train_model(model, x_train, y_train, x_test, y_test, 'sequence', epochs=30, patience=8)
            
            # Save the label encoder for sequence data to a file for future use.
            with open('sequence_label_encoder.pkl', 'wb') as f:
                pickle.dump(seq_le, f)

# Main Program
print("\n" + "="*50)
print("  SIGN LANGUAGE TRAINING SYSTEM")
print("="*50 + "\n")

train_and_evaluate()  # Start the training and evaluation process

print("\nTraining completed. Models saved in:") # Indicate that training is complete
print("- static_model.h5")  # Indicate where the static model is saved
print("- sequence_model.h5")  # Indicate where the sequence model is saved
print("- *_label_encoder.pkl")  # Indicate where the label encoders are saved 