import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image

# API key and api model
genai.configure(api_key="AIzaSyDIvahoFu6sUfFJc7I4z7VLD9-GSUECZ3M")
model = genai.GenerativeModel('gemini-1.5-flash')


# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Hand detection
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        fingers = detector.fingersUp(hand)                # Count the number of fingers up for the first hand
        return fingers, lmList
    else:
        return None

# Drawing function
def draw(canvas, info, prev_pos):
    fingers, lmList = info
    current_position = None

    if fingers == [0, 1, 0, 0, 0]:  #finger position (only when index finger is up)
        current_position = lmList[8][0:2]  # Index finger tip positioning
        if prev_pos is not None:
            cv2.line(canvas, prev_pos, current_position, (255, 255, 0), 10)  # Drawing line
            prev_pos = current_position

    return current_position


# Sending request to Gemini AI
def sendToAi(model,canvas, fingers):
    if fingers == [1,1,0,0,1]:          #fingers position(swag gesture)
        pil_image = Image.fromarray(canvas)                              #taking the image from the canvas and converting to text
        response = model.generate_content(["Solve",pil_image])           #sending the converted text to ai
        print(response.text)


prev_pos = None
canvas = None

# webcam captures
while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flipping the img window

    if canvas is None:
        canvas = np.zeros_like(img)  # Initialize canvas with same size as img and setting its background to black

    info = getHandInfo(img)

    if info:
        fingers, lmList = info
        # print(fingers)
        prev_pos = draw(canvas, info, prev_pos)
        sendToAi(model,canvas,fingers)

    # Combine the original image with the canvas for display
    combined_img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    # Display
    cv2.imshow("Image", combined_img)
    cv2.imshow("Canvas", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' to exit the loop
        break

# To destroy all windows
cap.release()
cv2.destroyAllWindows()
