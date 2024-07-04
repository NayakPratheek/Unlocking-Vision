import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Configure the Streamlit interface
st.set_page_config(layout="wide")

# Layout: two columns for displaying video feed and AI response
col1, col2 = st.columns([2, 1])

# Column 1: Video feed and Run checkbox
with col1:
    # run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

# Column 2: AI Response
with col2:
    output_text_area = st.title("Answer")
    output_text_area = st.subheader("")

# Configure Gemini AI with the provided API key
genai.configure(api_key="API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1080)  # Set the width of the video capture
cap.set(4, 720)  # Set the height of the video capture

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


def getHandInfo(img):
    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=False, flipType=True)
    # Check if any hands are detected
    if hands:
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        fingers = detector.fingersUp(hand)  # Count the number of fingers up for the first hand
        return fingers, lmList
    else:
        return None


def draw(canvas, info, prev_pos):
    fingers, lmList = info
    current_position = None

    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_position = lmList[8][0:2]  # Index finger tip position (x, y)
        if prev_pos is not None:
            # Draw a line from the previous position to the current position
            cv2.line(canvas, prev_pos, current_position, (255, 255, 0), 5)
        prev_pos = current_position  # Update the previous position to the current one
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up
        canvas = np.zeros_like(canvas)  # Reset the canvas to a blank image

    return current_position, canvas


def sendToAi(model, canvas, fingers):
    if fingers == [1, 1, 0, 0, 1]:  # Specific gesture (thumb, index, and pinky up)
        pil_image = Image.fromarray(
            cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))  # Convert canvas to RGB and then to PIL Image
        response = model.generate_content(["Solve", pil_image])  # Send the canvas image to Gemini AI
        return response.text  # Return the AI response text


prev_pos = None
canvas = None
output_text = None

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip horizontally for natural interaction

    if canvas is None:
        # Initialize canvas with the same size as img
        canvas = np.zeros_like(img)  # This creates a black canvas of the same dimensions as the frame

    info = getHandInfo(img)

    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(canvas, info, prev_pos)  # Update the canvas and previous position
        output_text = sendToAi(model, canvas, fingers)  # Send to AI based on specific gesture

    # Combine the original image with the canvas for display
    combined_img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(combined_img, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
