import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def handle_missing_data(df):
    """
    Handle missing data by imputing NaN values with the mean of each column.
    """
    # Only calculate the mean for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

def check_data_integrity(X, y):
    """
    Check if there are NaN or infinite values in the data.
    Replace them with 0 or a specified value.
    """
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print("Warning: Data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        print("Warning: Data contains infinite values.")

    # Replace NaN and infinite values with 0
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    return X, y

def prepare_multi_patient_data(file_paths, time_steps=5, max_length=None):
    """
    Prepare data from multiple patient CSV files for LSTM modeling.

    Parameters:
    -----------
    file_paths : list of str
        List of file paths for patient CSV files
    time_steps : int, optional (default=5)
        Number of time steps to use in each sequence
    max_length : int, optional
        Maximum sequence length to pad/truncate to

    Returns:
    --------
    X : numpy array
        Preprocessed input sequences
    y : numpy array
        Encoded labels
    """
    all_sequences = []
    all_labels = []

    # Preprocessing scalers to be applied consistently
    feature_scaler = StandardScaler()
    label_encoder = LabelEncoder()

    # Collect features from all patients
    all_features = []
    all_raw_labels = []

    # Read data from multiple files
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        # Handle missing data
        df = handle_missing_data(df)

        # Extract features and labels
        features = df[['time', 'steps_value', 'heart_rate_value', 'hrv']].values
        labels = df['Label'].values

        # Encode labels for each patient individually
        encoded_labels = label_encoder.fit_transform(labels)

        all_features.append(features)
        all_raw_labels.append(encoded_labels)

    # Combine all features for global scaling
    combined_features = np.vstack(all_features)
    feature_scaler.fit(combined_features)

    # Process each patient's data
    for features, labels in zip(all_features, all_raw_labels):
        # Scale features
        scaled_features = feature_scaler.transform(features)

        # Create sequences
        patient_sequences = []
        patient_labels = []

        for i in range(len(scaled_features) - time_steps + 1):
            # Create sequence of time_steps
            seq = scaled_features[i:i+time_steps]
            patient_sequences.append(seq)

            # Label is the last label in the sequence
            patient_labels.append(labels[i+time_steps-1])

        all_sequences.extend(patient_sequences)
        all_labels.extend(patient_labels)

    # Convert to numpy arrays
    X = np.array(all_sequences)
    y = to_categorical(all_labels)  # Convert labels to one-hot encoding

    # Optional: pad sequences to consistent length
    if max_length:
        X = pad_sequences(X, maxlen=max_length, padding='post', truncating='post')

    # Ensure data integrity
    X, y = check_data_integrity(X, y)

    return X, y, label_encoder

def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_multi_patient_lstm(file_paths):
    # Prepare data
    X, y, label_encoder = prepare_multi_patient_data(file_paths)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build model
    model = build_lstm_model(
        input_shape=(X.shape[1], X.shape[2]),
        num_classes=y.shape[1]
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    return model, label_encoder, X_test, y_test

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot the confusion matrix for the predictions
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

def calculate_class_accuracies(y_true, y_pred, labels):
    """
    Calculate accuracy for each individual class based on confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    accuracies = {}

    for i, label in enumerate(labels):
        # True positives (diagonal), False positives (column sums), and False negatives (row sums)
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        total = tp + fp + fn

        # Calculate accuracy for the class
        accuracy = tp / total if total != 0 else 0
        accuracies[label] = accuracy

    return accuracies

# Example of how to use with multiple patient files
if __name__ == "__main__":
    output_dir = "synthetic_patient_data_temporal"
    patient_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.csv')]

    model, label_encoder, X_test, y_test = train_multi_patient_lstm(patient_files)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Get predicted classes (labels)

    # Convert one-hot encoded labels back to integer labels for confusion matrix
    y_test_classes = np.argmax(y_test, axis=1)

    # Plot confusion matrix
    plot_confusion_matrix(y_test_classes, y_pred_classes, label_encoder.classes_)

    # Calculate and print accuracy for each class
    class_accuracies = calculate_class_accuracies(y_test_classes, y_pred_classes, label_encoder.classes_)
    for label, accuracy in class_accuracies.items():
        print(f"Accuracy for {label}: {accuracy:.4f}")

