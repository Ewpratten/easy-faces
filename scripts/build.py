import cv2, os
import numpy as np
from PIL import Image

# Create Local Binary Patterns Histograms for face recognition
recognizer = cv2.face.createLBPHFaceRecognizer()
detector = cv2.CascadeClassifier('classification/haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):

    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # Initialize empty face sample
    faceSamples=[]
    
    # Initialize empty id
    ids = []

    # Loop all the file path
    for imagePath in imagePaths:
        
        # Skip any gitkeep files
        if "gitkeep" in imagePath:
            continue

        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')

        # PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')

        # Get the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Get the face from the training images
        faces = detector.detectMultiScale(img_numpy)

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:

            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Add the ID to IDs
            ids.append(id)

    # Pass the face array and IDs array
    return faceSamples,ids

# Read data
faces,ids = getImagesAndLabels('tmp')

# Train & save model
recognizer.train(faces, np.array(ids))
recognizer.save('dataset/model.yml')