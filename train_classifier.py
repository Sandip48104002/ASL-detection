import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the processed dataset
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Convert to NumPy arrays
data = np.array(data_dict['data'], dtype=np.float32)  # Ensure numeric format

# Encode labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data_dict['labels'])  # Convert ASL letters to numbers

# Validate shapes
print(f"Data shape: {data.shape}, Labels shape: {encoded_labels.shape}")

# Split dataset into train & test sets
x_train, x_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate Model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save Trained Model and Label Encoder
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)

print("Training complete. Model saved as 'model.p'.")
