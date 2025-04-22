import cv2
import mediapipe as mp
import pyautogui
import math
import time
import subprocess
import threading
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper


# booleans to set scrolling modes
scroll_up = False
scroll_down = False
clicking = False
ssing = False
recording = threading.Event()


# simple math distance formula to detect finger proximity
def distance(coords1, coords2):
    return math.sqrt((coords1[0]-coords2[0])**2 + (coords1[1]-coords2[1])**2)

# record audio until "trigger" is triggered
def record_audio_on_event(trigger, filename="output.wav"):
    recorded = []

    def callback(indata, frames, time_info, status):
        recorded.append(indata.copy())

    with sd.InputStream(samplerate=44100, channels=1, callback=callback):
        while trigger.is_set():  # Keep recording while the flag is True
            sd.sleep(100)

    audio = np.concatenate(recorded, axis=0)
    audio = np.int16(audio * 32767)
    wav.write(filename, 44100, audio)
    print(f"[Audio saved as {filename}]")
 

# initialize the media pipe hands (up to 2 hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, 
    )
mp_drawing = mp.solutions.drawing_utils


# Start capturing video from the webcam, try cam 1 if cam 0 does not work
# try:
#     cap = cv2.VideoCapture(1)
#     print("Camera 1 Activated")
# except:
#     cap = cv2.VideoCapture(0)
#     print("Camera 0 Activated")

# for camera_index in range(3):  # Try indices 0, 1, and 2
#         print(f"Trying to open camera with index {camera_index}...")
#         cap = cv2.VideoCapture(camera_index)
        
#         if not cap.isOpened():
#             print(f"Failed to open camera with index {camera_index}.")
#             continue
            
#         # Check if we can actually read from the camera
#         success, test_frame = cap.read()
#         if not success:
#             print(f"Camera opened but failed to read frame from index {camera_index}.")
#             cap.release()
#             continue
            
#         print(f"Successfully connected to camera with index {camera_index}")

cap = cv2.VideoCapture(1)

# Load the Whisper model (you can use "base", "small", "medium", "large")
model = whisper.load_model("small.en")

# # Set camera window properties to full screen
# cv2.namedWindow("Hand Tracking", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Create window
cv2.namedWindow("Hand Track Cam")


# Main loop, ran every frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Flip window(the frame array) by horizontal axis only
    frame = cv2.flip(frame, 1)

    # Find camera window dimensions
    h, w, c = frame.shape

    color = (255, 0, 0)

    # print(frame.shape)
    # print(pyautogui.size())
    # w, h = pyautogui.size()

    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Calculate the palm center by averaging the coordinates of landmarks:
            # 0: wrist, 5: index finger MCP, 9: middle finger MCP,
            # 13: ring finger MCP, 17: pinky finger MCP.
            indices = [0, 5, 9, 13, 17]

            # Retrieve landmark positions for thumb tip, index tip, and middle tip.
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Convert normalized coordinates(0-1) to screen's pixel coordinates
            thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_coords = (int(index_tip.x * w), int(index_tip.y * h))
            middle_coords = (int(middle_tip.x * w), int(middle_tip.y * h))
            ring_coords = (int(ring_tip.x * w), int(ring_tip.y * h))
            pinky_coords = (int(pinky_tip.x * w), int(pinky_tip.y * h))

            palm_x = sum([hand_landmarks.landmark[i].x for i in indices]) / len(indices)
            palm_y = sum([hand_landmarks.landmark[i].y for i in indices]) / len(indices)
            palm_coords = (int(palm_x * w), int(palm_y * h))
            

            # Optionally, draw all hand landmarks 
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            # DETECTIONS -------------------------------------------------------------------------

            # Check for scroll
            if distance(index_coords, middle_coords) < 38:
                scroll_up = True
            elif distance(middle_coords, ring_coords) < 38:
                scroll_down = True
            
            # Check for click
            if distance(middle_coords, thumb_coords) < 38:
                clicking = True

            # Check for ss
            if distance(ring_coords, palm_coords) < 25:
                ssing = True

            # Check for recording
            if distance(thumb_coords, palm_coords) < 60 and not recording.is_set():
                recording.set()
                threading.Thread(target=record_audio_on_event, args=(recording, ), daemon=True).start()
                print("Recording started...")
            elif not distance(thumb_coords, palm_coords) < 60 and recording.is_set():
                recording.clear()
                print("Recording complete.")

                # Transcribe the audio file
                result = model.transcribe("output.wav")

                # Print the transcribed text
                print("Transcribed Text:")
                print(result["text"])

            # EXECUTIONS --------------------------------------------------------------------------

            # Move cursor to index finger
            pyautogui.moveTo(index_coords[0], index_coords[1], _pause=False)

            # Scroll
            if scroll_up:
                pyautogui.scroll(-3, _pause=False)
                color = (0, 0, 255)
            elif scroll_down:
                pyautogui.scroll(3, _pause=False)
                color = (0, 0, 255)

            # Click
            if clicking:
                pyautogui.click(_pause=False)
                color = (0, 255, 0)

            # SS
            if ssing:
                pyautogui.screenshot(f"Hand Track Cam - {time.asctime()}")
                subprocess.run(["screencapture", f"/Users/kaito/Desktop/Hand Track Cam - {time.asctime()}.png"])
                color = (0, 255, 255)

            # Recording
            if recording.is_set():
                color = (255, 0, 255)
            

            # DRAWINGS ----------------------------------------------------------------------------

            # Draw node at palm
            cv2.circle(frame, palm_coords, 15, (0, 255, 0), -1)

            # Draw node at each fingertip
            cv2.circle(frame, thumb_coords, 10, (0, 255, 0), -1)
            cv2.circle(frame, index_coords, 10, (0, 255, 0), -1)
            cv2.circle(frame, middle_coords, 10, (0, 255, 0), -1)
            cv2.circle(frame, ring_coords, 10, (0, 255, 0), -1)
            cv2.circle(frame, pinky_coords, 10, (0, 255, 0), -1)

            # Draw connecting lines
            cv2.line(frame, thumb_coords, index_coords, color, 2)
            cv2.line(frame, index_coords, middle_coords, color, 2)
            cv2.line(frame, middle_coords, ring_coords, color, 2)
            cv2.line(frame, ring_coords, pinky_coords, color, 2)
        

        # RESET ----------------------------------------------------------------------------
        scroll_up = False
        scroll_down = False
        clicking = False
        ssing = False

        color = (255, 0, 0)



    # # Display the resulting frame.
    # cv2.imshow("Hand Track Cam", frame)

    # resized_frame = cv2.resize(frame, (2560, 1600))
    # resized_frame = cv2.resize(frame, (1600, 1000))
    cv2.imshow("Hand Track Cam", frame)

    # Press 'q' key to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
