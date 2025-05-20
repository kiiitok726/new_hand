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


# Webcam detection border


# Screen dimensions
fscreen_x, fscreen_y = pyautogui.size()

# Sensitivity ratio (this determines how small of a portion of your webcam should convert to a fullscreen coordinate)
border = 0.75
border_top_left = (1-border)/2
border_bottom_right = (1-border)/2 + border


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

    print("Recording started...", flush=True)

    with sd.InputStream(samplerate=44100, channels=1, callback=callback):
        while trigger.is_set():  # Keep recording while the flag is True
            sd.sleep(100)
    
    print("Recording complete.", flush=True)

    audio = np.concatenate(recorded, axis=0)
    audio = np.int16(audio * 32767)
    wav.write(filename, 44100, audio)
    print(f"[Audio saved as {filename}]", flush=True)

    # Transcribe the audio file
    result = model.transcribe("output.wav")

    # Print the transcribed text
    print("Transcribed Text:", flush=True)
    print(result["text"], flush=True)

    # Type the transcribed text
    pyautogui.typewrite(result["text"], interval=0.02)
 

# initialize the media pipe hands (up to 2 hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, 
    )
mp_drawing = mp.solutions.drawing_utils


## Initialize webcam, try cam 1 if cam 0 does not work
# try:
#     cap = cv2.VideoCapture(1)
#     print("Camera 1 Activated")
# except:
#     cap = cv2.VideoCapture(0)
#     print("Camera 0 Activated")


# # Initialize webcam, try multiple indices
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

## Initialize specific webcam
cap = cv2.VideoCapture(0)

# Create webcam window
cv2.namedWindow("Hand Track Cam")

# Load the Whisper model (you can use "base", "small", "medium", "large")
model = whisper.load_model("tiny.en")

# Load OpenCV's pre-classified models
face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


# MAIN LOOP (ran every frame) -------------------------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip window(the frame array) by horizontal axis only
    frame = cv2.flip(frame, 1)

    # Find webcam window dimensions
    h, w, c = frame.shape

    # draw the blue “active area” box
    x1 = int(border_top_left    * w)
    y1 = int(border_top_left    * h)
    x2 = int(border_bottom_right * w)
    y2 = int(border_bottom_right * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Default blue color for lines
    color = (255, 0, 0)

    # Grayscaling for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using preclassified model
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (f_x, f_y, f_w, f_h) in faces:
        cv2.rectangle(frame, (f_x,f_y), (f_x+f_w, f_y+f_h), (0, 255, 255), 2)
        face_roi_gray = gray[f_y:f_y+f_h, f_x:f_x+f_w]
        face_roi_color = frame[f_y:f_y+f_h, f_x:f_x+f_w]

        # Detect any smiles among detected faces
        smiles = smile_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.9,
            minNeighbors=30,
            minSize=(40, 40)
        )

        # If smile detected, take photo
        if len(smiles) > 0:
            cv2.putText(frame, "Smiling", (f_x, f_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            
            cv2.imwrite(f"/Users/kaito/Desktop/HTC_output/HTC photo - {time.asctime()}.png", frame)

    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process any present hands
    result = hands.process(rgb_frame)

    # Run gesture detection, analysis, drawings if any hands are detected
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
            thumb_coords = (int(np.interp(thumb_tip.x, [border_top_left, border_bottom_right], [0, fscreen_x])), int(np.interp(thumb_tip.y, [border_top_left, border_bottom_right], [0, fscreen_y])))
            index_coords = (int(np.interp(index_tip.x, [border_top_left, border_bottom_right], [0, fscreen_x])), int(np.interp(index_tip.y, [border_top_left, border_bottom_right], [0, fscreen_y])))
            middle_coords = (int(np.interp(middle_tip.x, [border_top_left, border_bottom_right], [0, fscreen_x])), int(np.interp(middle_tip.y, [border_top_left, border_bottom_right], [0, fscreen_y])))
            ring_coords = (int(np.interp(ring_tip.x, [border_top_left, border_bottom_right], [0, fscreen_x])), int(np.interp(ring_tip.y, [border_top_left, border_bottom_right], [0, fscreen_y])))
            pinky_coords = (int(np.interp(pinky_tip.x, [border_top_left, border_bottom_right], [0, fscreen_x])), int(np.interp(pinky_tip.y, [border_top_left, border_bottom_right], [0, fscreen_y])))
            # print(f"({thumb_tip.x}, {thumb_tip.y}) - > {thumb_coords}")
            # print(f"({index_tip.x}, {index_tip.y}) - > {index_coords}")
            # print(f"({middle_tip.x}, {middle_tip.y}) - > {middle_coords}")
            # print(f"({ring_tip.x}, {ring_tip.y}) - > {ring_coords}")
            # print(f"({pinky_tip.x}, {pinky_tip.y}) - > {pinky_coords}")
            # print()

            palm_x = sum([hand_landmarks.landmark[i].x for i in indices]) / len(indices)
            palm_y = sum([hand_landmarks.landmark[i].y for i in indices]) / len(indices)
            palm_coords = (int(np.interp(palm_x, [border_top_left, border_bottom_right], [0, fscreen_x])), int(np.interp(palm_y, [border_top_left, border_bottom_right], [0, fscreen_y])))


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

            if not distance(thumb_coords, palm_coords) < 60 and recording.is_set():
                recording.clear()


            # EXECUTIONS --------------------------------------------------------------------------

            # Move cursor to index finger
            pyautogui.moveTo(index_coords[0], index_coords[1], _pause=False)

            # Convert index_coords from float to int for OpenCV drawing
            index_coords_int = (int(index_coords[0] * w / fscreen_x), int(index_coords[1] * h / fscreen_y))
            coord_text = f"{int(index_coords[0])}, {int(index_coords[1])}"
            original_text = f"{index_tip.x}, {index_tip.y}\n{int(index_coords[0])}, {int(index_coords[1])}"

            # cv2.putText(frame, coord_text, (index_coords_int[0] + 20, index_coords_int[1] - 20),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, original_text, (index_coords_int[0] + 20, index_coords_int[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
                subprocess.run(["screencapture", f"/Users/kaito/Desktop/HTC_output/HTC ss - {time.asctime()}.png"])
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
