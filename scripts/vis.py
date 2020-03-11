import cv2
import numpy as np


# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.createLBPHFaceRecognizer()


# Load the trained mode
recognizer.load('dataset/model.yml')

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(
    'classification/haarcascade_frontalface_default.xml')

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Open a camera connection
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Unable to open camera stream on /dev/video0")
    exit(1)

# Load faces list
people = {}
with open("dataset/labels.pairs", "r") as fp:
    for person in fp.read().split("\n"):

        person_dat = person.split(":")
        if len(person_dat) >= 2:
            people[int(person_dat[0])] = person_dat[1]

while True:

    # Handle kill command
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # Read a frame from the camera
    ret, im = cam.read()

    # Skip null frames
    if not ret:
        continue

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.4, 3)

    # For each face in faces
    for(x, y, w, h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 4)

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # If we know this ID, set it's name
        name = "Unknown"
        if Id in people.keys():
            name = (people[Id] if confidence > 50 else "Unknown") + \
                f" [{Id}][{round(confidence)}]"
            # print(confidence)

        # Put text describe who is in the picture
        cv2.putText(im, name, (x, y-40), font, 0.8, (0, 255, 0), 2)

    # Display the video frame with the bounded rectangle
    cv2.imshow('Faces viewer', im)

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
