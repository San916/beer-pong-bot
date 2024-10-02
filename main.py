# Import necessary dependencies
import cv2
import time
import numpy as np
import imageProcessing as imgProc
import arduinoCommunication as ard

# Constants relevant for communication with other devices
# CAMERA_IP = "http://10.0.0.152" 
# CAMERA_IP = "http://206.12.139.117"
# CAMERA_IP = "http://192.168.120.102"
CAMERA_PORT = "8080"

def main():
    # Set up the camera for object detection
    # First one is for webcam, second is for using a phone camera

    cap = None
    cap = imgProc.cameraSetup(0)
    # cap = imgProc.cameraSetup(CAMERA_IP + ":" + CAMERA_PORT + "/video")


    # Set up two-way communication with the arduino
    # arduinoIO = ard.arduinoCommunication(3)
    # arduinoIO.writeMessage("ToArduino:Hello!") # Write an initial message to start communication
    # while True:
        # Read from the COM port
        # success, message = arduinoIO.readMessage()

        # (If an unsuccessful read (arduino has not responded to us, we can wait a moment then continue)
        # if not success:
        #     time.sleep(1)
        #     continue

        # If successful, read the message and act according to the command given
        # command = message[0:3]
        # arguments = message[3:]
        # if command == "ext": # Exit from the program
        #     break
        # if command == "det": # Detect cups and return how much the robot needs to rotate to aim at the cups
        #     cupsDetected = []
        #     for i in range(5):
        #         cupsDetected += imgProc.runCupDetection()
        #         time.sleep(0.1)
            




        # time.sleep(1)


    while True:
        startTime = time.time()
        detectedCups, image = imgProc.runCupDetection(cap, drawImage = True)
        cv2.imshow('Result image', image)
        endTime = time.time()
        print("Time spent:", endTime - startTime)
        cv2.waitKey(30)
    return

main()