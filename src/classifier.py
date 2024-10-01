import json
import math

def calculate_distance(landmarks1, landmarks2):
    distance = 0
    for lm1, lm2 in zip(landmarks1, landmarks2):
        distance += math.sqrt((lm1['x'] - lm2['x'])**2 + (lm1['y'] - lm2['y'])**2 + (lm1['z'] - lm2['z'])**2)
    return distance

def load_training_data():
    training_data = []
    with open("pruebas/data/hand_positions.json", "r") as file:
        for line in file:
            training_data.append(json.loads(line))
    return training_data

def classify_hand_gesture(landmarks, training_data):
    min_distance = float('inf')
    best_match = None
    current_landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks.landmark]

    for data in training_data:
        distance = calculate_distance(current_landmarks, data["landmarks"])
        if distance < min_distance:
            min_distance = distance
            best_match = data["letter"]
    
    return best_match
