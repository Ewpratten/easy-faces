import cv2
import numpy as np

# Load image classifier for faces
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ask the user for a name for this session
name = input("Name this session:\n> ")

# Read the last session ID
sessionID: int = 0
with open("dataset/labels.pairs", "r") as fp:
    sessionID = int(len(fp.read.split("\n")))
    fp.close()
    
# Write the new session/label pair
with open("dataset/labels.pairs", "a") as fp:
    fp.writelines(f"{sessionID}:{name}")
    fp.close()
