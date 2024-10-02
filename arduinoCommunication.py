# Import necessary dependencies
import serial

class arduinoCommunication:
    # Serial port object
    serialPort = None

    # Init our object
    def __init__(self, portNum):
        self.serialPort = serial.Serial(port = "COM" + str(portNum), baudrate = 9600)
        self.serialPort.flush()

    # Write the given message into the port
    def writeMessage(self, message):
        if not self.serialPort: raise Exception("Serial port object not initialized!")
        self.serialPort.write(bytes(message, "utf-8"))

    # Read the port, and if there is a message from the arduino, return the message
    def readMessage(self):
        message = self.serialPort.readline()
        message = message.decode("utf-8")
        if not message.find("ToComputer:") == -1:
            return (True, message[11:])
        return (False, message)

# Every message sent between the arduino and computer will be comprised of:
# A header
# A three character long command
# Arguments for the command taking up the rest of the message
# 
# Arduino to Computer will have header: "ToComputer:"
# Computer to Arduino will have header: "ToArduino:"