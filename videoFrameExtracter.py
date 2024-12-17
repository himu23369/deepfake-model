import cv2
import os

def extract_frames(videoPath, outputPath = "Frames", frameRate = 1):
    videoCap = cv2.VideoCapture(videoPath)
    fps = videoCap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frameRate)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    
    count = 0
    frameNumber = 0

    while True:
        ret, frame = videoCap.read()
        if not ret:
            break

        if frameNumber % interval == 0:
            frameFilename = os.path.join(outputPath, f"frame{count}.jpg")
            cv2.imwrite(frameFilename,frame)
            count += 1
        frameNumber += 1
    
    videoCap.release()
