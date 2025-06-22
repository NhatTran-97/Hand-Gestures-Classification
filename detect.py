import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import cv2
import mediapipe as mp
from nn_model import NeuralNetwork
import utils
import torch
import numpy as np


class HandLandmarksDetector():
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(False,max_num_hands=1,min_detection_confidence=0.5)

    def detectHand(self,frame):
        hands = []
        frame = cv2.flip(frame, 1)
        annotated_image = frame.copy()
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                for landmark in hand_landmarks.landmark:
                    x,y,z = landmark.x,landmark.y,landmark.z
                    hand.extend([x,y,z])
            hands.append(hand)
        return hands,annotated_image
    
    def detectRightHand(self, frame):
        hands = []
        frame = cv2.flip(frame, 1)
        annotated_image = frame.copy()
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks is not None and results.multi_handedness is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Right' or 'Left'

                if label == 'Right':
                    hand = []
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                    for landmark in hand_landmarks.landmark:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        hand.extend([x, y, z])
                    hands.append(hand)

        return hands, annotated_image


class Gestures_Recognition(object):
    def __init__(self,model_path):
        self.height = 720
        self.width = 1280
        self.detector = HandLandmarksDetector()
        self.signs = utils.label_dict_from_config_file("generate_data/hand_gesture.yaml")
        self.classifier = NeuralNetwork()
        self.classifier.load_state_dict(torch.load(model_path))

        self.classifier.eval()
    def run(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, self.width)
        cam.set(4, self.height)

        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break

            hands, img = self.detector.detectRightHand(frame)

            if len(hands) != 0:
                with torch.no_grad():
                    hand_landmark = torch.from_numpy(
                        np.array(hands[0], dtype=np.float32).flatten()
                    ).unsqueeze(0)

                    class_number = self.classifier.predict(hand_landmark).item()

                    if class_number != -1:
                        status_text = self.signs[class_number]
                        # ✅ Hiển thị kết quả lên ảnh
                        cv2.putText(img, f"Prediction: {status_text}", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    else:
                        cv2.putText(img, "No valid gesture", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # Hiển thị kết quả ảnh
            cv2.imshow("Hand Gesture Recognition", img)

            # Nhấn ESC hoặc 'q' để thoát
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    gr = Gestures_Recognition("./models/models_22-06 19:42_NeuralNetwork_best.pth")
    gr.run()
