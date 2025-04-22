import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import sys

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Define the finger tip indices in MediaPipe hand model
    # 4 = Thumb tip, 8 = Index finger tip, 12 = Middle finger tip
    FINGER_TIP_INDICES = [4, 8, 12]
    INDEX_FINGER_TIP = 8
    INDEX_FINGER_PIP = 6  # Proximal interphalangeal joint (middle joint of index finger)

    # Define node properties
    NODE_RADIUS = 10
    NODE_COLOR = (0, 255, 0)  # Green
    ACTIVE_NODE_COLOR = (0, 0, 255)  # Red when scrolling
    LINE_COLOR = (255, 0, 0)  # Blue
    LINE_THICKNESS = 2

    # Scrolling parameters
    scroll_active = False
    prev_y = None
    Y_THRESHOLD = 8  # Minimum y movement to trigger scroll
    SCROLL_SPEED_MULTIPLIER = 0.1  # Adjust this to control scrolling sensitivity
    scroll_cooldown = 0
    COOLDOWN_FRAMES = 5  # Frames to wait before registering another scroll action

    # Try different camera indices if the default one fails
    for camera_index in range(3):  # Try indices 0, 1, and 2
        print(f"Trying to open camera with index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Failed to open camera with index {camera_index}.")
            continue
            
        # Check if we can actually read from the camera
        success, test_frame = cap.read()
        if not success:
            print(f"Camera opened but failed to read frame from index {camera_index}.")
            cap.release()
            continue
            
        print(f"Successfully connected to camera with index {camera_index}")
        
        # If we reach here, we have a working camera
        break
    else:
        # This runs if the for loop completes without a break
        print("Failed to find a working camera. Please check your camera connection and permissions.")
        return

    print("Gesture Controls:")
    print("- Move index finger up to scroll DOWN")
    print("- Move index finger down to scroll UP")
    print("- Quick flick up/down for faster scrolling")
    print("Press 'q' to quit the program.")
    
    try:
        while True:
            success, image = cap.read()
            if not success:
                print("Failed to capture frame. Retrying...")
                time.sleep(0.5)
                continue

            # Flip the image horizontally for a mirror effect
            image = cv2.flip(image, 1)
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = hands.process(image_rgb)
            
            # Update cooldown counter
            if scroll_cooldown > 0:
                scroll_cooldown -= 1
            
            # Get image dimensions - FIXED: Get dimensions here, before any references
            img_height, img_width, _ = image.shape
            
            # Status text variables
            status_text = "Status: Waiting for hand..."
            status_color = (255, 255, 255)
            
            # Draw hand landmarks and handle scrolling
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get coordinates of finger tips
                    finger_tip_coords = []
                    
                    for tip_idx in FINGER_TIP_INDICES:
                        landmark = hand_landmarks.landmark[tip_idx]
                        # Convert normalized coordinates to pixel coordinates
                        x, y = int(landmark.x * img_width), int(landmark.y * img_height)
                        finger_tip_coords.append((x, y))
                    
                    # Get index finger coordinates for scrolling
                    index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
                    index_tip_y = index_tip.y
                    
                    # Get index finger PIP joint to determine if finger is extended
                    index_pip = hand_landmarks.landmark[INDEX_FINGER_PIP]
                    
                    # RELAXED: Now we only require the finger to be visible, not necessarily extended
                    finger_visible = True  # Assume finger is visible if we have the landmarks
                    
                    # Handle scrolling logic with RELAXED requirements
                    if finger_visible and scroll_cooldown == 0:
                        if prev_y is not None:
                            # Calculate vertical movement
                            y_delta = (index_tip_y - prev_y) * 1000  # Scale up for better sensitivity
                            
                            # Only scroll if movement exceeds threshold
                            if abs(y_delta) > Y_THRESHOLD:
                                # INVERTED: Positive y_delta (finger moved down) now scrolls down
                                # Negative y_delta (finger moved up) now scrolls up
                                scroll_amount = int(y_delta * SCROLL_SPEED_MULTIPLIER)
                                
                                # Apply non-linear scaling for faster movements
                                if abs(scroll_amount) > 20:
                                    scroll_amount *= 1.5
                                
                                # INVERTED: Removed the negative sign to invert the scrolling direction
                                pyautogui.scroll(scroll_amount)
                                
                                scroll_active = True
                                # INVERTED: Updated status text to match new behavior
                                status_text = f"Scrolling: {'up' if scroll_amount < 0 else 'down'}"
                                status_color = (0, 255, 0)
                                
                                # Set cooldown to prevent too many scroll events
                                scroll_cooldown = COOLDOWN_FRAMES
                            else:
                                scroll_active = False
                                status_text = "Status: Index finger detected (move to scroll)"
                                status_color = (255, 165, 0)  # Orange
                        
                        prev_y = index_tip_y
                    else:
                        if not finger_visible:
                            prev_y = None
                            status_text = "Status: Show index finger to scroll"
                            status_color = (255, 255, 0)  # Yellow
                    
                    # Draw nodes at finger tips with different color for active scrolling
                    for i, (x, y) in enumerate(finger_tip_coords):
                        # Make index finger red when actively scrolling
                        node_color = ACTIVE_NODE_COLOR if (i == 1 and scroll_active) else NODE_COLOR
                        cv2.circle(image, (x, y), NODE_RADIUS, node_color, cv2.FILLED)
                    
                    # Connect thumb to index finger
                    if len(finger_tip_coords) >= 2:
                        cv2.line(image, finger_tip_coords[0], finger_tip_coords[1], 
                                LINE_COLOR, LINE_THICKNESS)
                    
                    # Connect index finger to middle finger
                    if len(finger_tip_coords) >= 3:
                        cv2.line(image, finger_tip_coords[1], finger_tip_coords[2], 
                                LINE_COLOR, LINE_THICKNESS)
                    
                    # Add labels
                    labels = ["Thumb", "Index", "Middle"]
                    for i, (x, y) in enumerate(finger_tip_coords):
                        cv2.putText(image, labels[i], (x-15, y-15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add status text
            cv2.putText(image, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(image, "Press 'q' to quit", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add instruction text
            cv2.putText(image, "Move UP = Scroll DOWN", (10, img_height - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, "Move DOWN = Scroll UP", (10, img_height - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the resulting image
            cv2.imshow('Hand Gesture Scroll Controller', image)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace
    
    finally:
        # Always release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Program terminated.")

if __name__ == "__main__":
    main()