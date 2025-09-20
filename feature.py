import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path to Dataset
DATA_DIR = r'C:\Users\HP\Downloads\archive\asl_dataset'

data = []
labels = []

# Mapping class labels (ASL characters) to numeric values
label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(DATA_DIR)))}

for dir_ in os.listdir(DATA_DIR):
    label = label_map[dir_]  # Convert label to numeric

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img_path_full = os.path.join(DATA_DIR, dir_, img_path)
        
        # Read Image
        img = cv2.imread(img_path_full)
        if img is None:
            print(f"Skipping {img_path_full}, failed to load.")
            continue  # Skip if image is not found

        # Convert Image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract Hand Landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize Features
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalize X
                    data_aux.append(y - min(y_))  # Normalize Y

                data.append(data_aux)
                labels.append(label)  # Use numeric label

# Ensure all feature vectors have the same length
max_length = max(len(d) for d in data)
data_padded = [d + [0] * (max_length - len(d)) for d in data]  # Padding

# Save Data as Pickle File
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data_padded, 'labels': labels}, f)

print("Feature extraction complete. Saved dataset as 'data.pickle'.")
