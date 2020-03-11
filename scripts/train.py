import cv2
import numpy as np

# Load image classifier for faces
face_classifier = cv2.CascadeClassifier('classification/haarcascade_frontalface_default.xml')

# Open a camera connection
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Unable to open camera stream on /dev/video0")
    exit(1)

# Ask the user for a name for this session
name = input("Name this session:\n> ")
# name = "TEST_NAME"

# Read the last session ID
sessionID: int = 0
with open("dataset/labels.pairs", "r") as fp:
    sessionID = int(len(fp.read().split("\n")))+1
    fp.close()

# Write the new session/label pair
with open("dataset/labels.pairs", "a") as fp:
    fp.writelines(f"{sessionID}:{name}\n")
    fp.close()

# Set frames left to read
remaining_frames = 200
print(f"Begining to read 200 frames of video")

while cam.isOpened() and remaining_frames >= 0:

    # Handle kill command
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # Read a frame from the camera
    ret, frame = cam.read()

    # Skip null frames
    if not ret:
        continue

    # Convert image to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find bounds of all faces in frame
    faces_bounds: list = face_classifier.detectMultiScale(frame, 1.3, 5)

    # Skip if no faces found
    if len(faces_bounds) == 0:
        continue

    # Sort faces by size
    faces_bounds = sorted(faces_bounds, key=lambda x: (x[2]*x[3]) * -1)

    # Select the largest face
    (x, y, w, h) = faces_bounds[0]

    # Get a cropped frame of the face
    cropped_face = frame[y:y+h, x:x+w]

    # Display the face
    cv2.imshow("Found Face", cropped_face)
    
    # Write the face to the dataset
    cv2.imwrite(f"tmp/Session.{sessionID}.{remaining_frames}.jpg", cropped_face)
    remaining_frames -= 1

# Close all started windows
print("Finished recording")
cv2.destroyAllWindows()

