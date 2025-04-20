import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Text-to-Speech engine
engine = pyttsx3.init()

# Label mapping
labels_dict = {0: 'I need Water', 1: 'I need meds', 2: 'Call nurse'}

# Function to speak the detected text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Draw a rounded rectangle
def draw_rounded_rect(img, top_left, bottom_right, color, thickness=1, radius=10):
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Draw rectangles for rounded corners
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)  # Middle
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)  # Middle

    # Draw corner arcs
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)  # Top-left
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)  # Top-right
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)  # Bottom-left
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)  # Bottom-right

# Main loop
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # Draw the enhanced Speak button
            speak_button = (50, 50, 250, 150)  # x1, y1, x2, y2
            draw_rounded_rect(frame, (speak_button[0], speak_button[1]), (speak_button[2], speak_button[3]), (255, 0, 0), -1)

            # Add a border for better visibility
            draw_rounded_rect(frame, (speak_button[0], speak_button[1]), (speak_button[2], speak_button[3]), (0, 0, 0), 2)

            # Put text on the button
            cv2.putText(frame, 'Speak', (speak_button[0] + 50, speak_button[1] + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Check if the Speak button is clicked
            if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' key to speak
                speak(predicted_character)

    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
