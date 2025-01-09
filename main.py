import cv2
import mediapipe as mp
import pyautogui

# Initialize camera and Mediapipe
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()


def process_landmarks(landmarks, frame_width, frame_height):
    """
    Process hand landmarks to calculate screen coordinates for index finger and thumb.
    """
    screen_coords = {}
    for id, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        screen_coords[id] = (screen_width / frame_width * x, screen_height / frame_height * y)
    return screen_coords


while True:
    try:
        # Capture frame and preprocess
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                landmarks = hand.landmark
                coords = process_landmarks(landmarks, frame_width, frame_height)

                # Screen coordinates for index finger and thumb
                if 8 in coords and 4 in coords:
                    index_x, index_y = coords[8]
                    thumb_x, thumb_y = coords[4]

                    # Visualize index and thumb positions
                    cv2.circle(frame,
                               (int(index_x * frame_width / screen_width), int(index_y * frame_height / screen_height)),
                               10, (0, 255, 255), -1)
                    cv2.circle(frame,
                               (int(thumb_x * frame_width / screen_width), int(thumb_y * frame_height / screen_height)),
                               10, (255, 0, 255), -1)

                    # Mouse actions
                    distance = abs(index_y - thumb_y)
                    if distance < 20:  # Clicking
                        pyautogui.click()
                        pyautogui.sleep(0.5)
                    else:  # Moving
                        pyautogui.moveTo(index_x, index_y)

        # Show video feed
        cv2.imshow('Virtual Mouse', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
