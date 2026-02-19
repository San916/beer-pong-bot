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
CAMERA_FOV = 70
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
COM_PORT = 3

def returnAngleToRotate(centerX):
    yaw_radians = centerX - CAMERA_WIDTH / 2
    yaw_radians = 2 * yaw_radians * np.tan(CAMERA_FOV / 2)
    yaw_radians = yaw_radians / CAMERA_WIDTH
    yaw_radians = np.atan(yaw_radians)
    yaw_degrees = 180 * yaw_radians / np.pi
    return yaw_degrees

def main():
    # Set up the camera for object detection
    # First one is for webcam, second is for using a phone camera

    cap = None
    cap = imgProc.cameraSetup(0)
    # cap = imgProc.cameraSetup(CAMERA_IP + ":" + CAMERA_PORT + "/video")


    # Set up two-way communication with the arduino
    input("Press enter to begin...")
    arduinoIO = ard.arduinoCommunication(COM_PORT)
    while True:
        arduinoIO.writeMessage("ToArduino:Hello!") # Write an initial message to start communication

        # Read from the COM port
        print("Reading message...")
        success, message = arduinoIO.readMessage()

        # (If an unsuccessful read (arduino has not responded to us, we can wait a moment then continue)
        if not success:
            time.sleep(1)
            continue

        # If successful, read the message and act according to the command given
        print("Got message:", message)
        command = message[0:3]
        arguments = message[3:]
        if command == "ext": # Exit from the program
            break
        if command == "det": # Detect cups and return how much the robot needs to rotate to aim at the cups
            print("Detecting cups:")
            cupsDetected, image = imgProc.runCupDetection(cap, CAMERA_WIDTH, CAMERA_HEIGHT, returnImage = True)

            cv2.imshow('Result image', image)
            cv2.waitKey(30)
            
            if (len(cupsDetected) == 0):
                arduinoIO.writeMessage("ToArduino:NoCupsDetected!")
                time.sleep(1)
                continue

            _, red = cupsDetected[0]
            (centerX, _), _, _ = red
            yaw = round(returnAngleToRotate(centerX))
            arduinoIO.writeMessage(f"ToArduino:Rotate:{yaw}")

        time.sleep(1)

    # while True:
    #     startTime = time.time()
    #     detectedCups, image = imgProc.runCupDetection(cap, drawImage = True)
    #     cv2.imshow('Result image', image)
    #     endTime = time.time()
    #     print("Time spent:", endTime - startTime)
    #     cv2.waitKey(30)
    return

main()