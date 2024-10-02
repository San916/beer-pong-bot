# Import necessary dependencies
import cv2
import numpy as np

# Constants relevant for image processing
CONTOUR_AREA_THRESHOLD = 250
BOUNDNARY_THICKNESS = 2
DEFECT_THRESHOLD = 512
GRAYYSCALE_VARIANCE_THRESHOLD = 13.0
GRAYSCALE_THRESHOLD = 120

# Given an image, detect the cups in the image and return the detected cups in the form of a list of rectangles
def detectCups(originalImage, drawImage = False):
    # ---------------------------------------
    # Preprocessing
    # ---------------------------------------

    blurred = cv2.GaussianBlur(originalImage, (7, 7), 0.5)
    np.uint8(blurred)

    # ---------------------------------------
    # Extract red sections from the image
    # ---------------------------------------

    filteredRed = filterRed(blurred, 150, 100, 100)
    b, g, r = cv2.split(filteredRed)
    r = np.uint8(r)

    # Fill in any enclosed spaces
    r = fillImage(r)


    # Remove noise from the image
    erodeKernel = np.ones((5, 5), dtype = np.uint8)
    dilateKernel = np.ones((9, 11), dtype = np.uint8)
    r = cv2.erode(r, erodeKernel)
    r = cv2.dilate(r, dilateKernel)
    r = cv2.dilate(r, dilateKernel)

    redContours, _ = cv2.findContours(r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filteredRed = cv2.merge((b, g, r))
    cv2.imshow('Filtered Red', filteredRed)

    # ---------------------------------------
    # Extract white sections from the image
    # ---------------------------------------
    grayScale = customGrayScale(blurred)
    normalized = cv2.normalize(grayScale, None, 0, 255, cv2.NORM_MINMAX)
    _, whitened = cv2.threshold(normalized, GRAYSCALE_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Fill in any enclosed spaces
    filteredWhite = fillImage(whitened)

    # Remove noise from the image
    ellipticalKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    filteredWhite = cv2.erode(filteredWhite, ellipticalKernel)
    filteredWhite = cv2.dilate(filteredWhite, ellipticalKernel)
    filteredWhite = cv2.erode(filteredWhite, ellipticalKernel)
    filteredWhite = cv2.dilate(filteredWhite, ellipticalKernel)

    whiteContours, _ = cv2.findContours(filteredWhite, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Whitened Image', filteredWhite)

    # ---------------------------------------
    # Combine the white and red to display to the user
    # ---------------------------------------
    b, g, r = cv2.split(filteredRed)
    cv2.bitwise_or(b, filteredWhite, b)
    cv2.bitwise_or(g, filteredWhite, g)
    cv2.bitwise_or(r, filteredWhite, r)
    combinedRedWhite = cv2.merge((b, g, r))

    cv2.imshow('Combined', combinedRedWhite)

    # ---------------------------------------
    # Detect cups using the red and white contours
    # ---------------------------------------
    
    detectedCups = lookForCups(redContours, whiteContours)
    if not drawImage: return detectedCups
    originalImage = drawCups(originalImage, detectedCups)
    return (detectedCups, originalImage)

# Apply a custom filter that keeps sufficiently red pixels
def filterRed(originalImage, minRed, maxGreen, maxBlue):
    filteredImage = np.zeros_like(originalImage)
    b, g, r = cv2.split(originalImage)

    mask = (b <= maxBlue) & (g <= maxGreen) & (r >= minRed)
    mask = mask | ((r > 75) & ((g + b) / r < 0.1) & (b / g < 1.1) & (g / b < 1.1))
    filteredImage[mask] = (0, 0, 255)

    return filteredImage

# Apply a custom grayscaling algorithm to remove all pixels with an uneven distribution of rgb values, and grayscales the rest
# Returns a single channel image
def customGrayScale(originalImage):
    # rows, cols, channels = originalImage.shape
    grayScale = np.zeros_like(originalImage)
    b, g, r = cv2.split(originalImage)

    def grayscaler(b, g, r):
        if r < GRAYSCALE_THRESHOLD or g < GRAYSCALE_THRESHOLD or b < GRAYSCALE_THRESHOLD:
            return 0

        r *= 0.33333
        g *= 0.33333
        b *= 0.33333

        grayScaleValue = r + g + b

        r *= 255 / grayScaleValue
        g *= 255 / grayScaleValue
        b *= 255 / grayScaleValue

        if (abs(r - g) <= GRAYYSCALE_VARIANCE_THRESHOLD and 
            abs(r - b) <= GRAYYSCALE_VARIANCE_THRESHOLD and 
            abs(b - g) <= GRAYYSCALE_VARIANCE_THRESHOLD):
            return grayScaleValue
        return 0

    vectorizedFunction = np.vectorize(grayscaler)
    grayScale = np.uint8(vectorizedFunction(b, g, r))

    return grayScale

# Given a single channel image, make it such that any black spots surrounded in white get filled in
def fillImage(originalImage):
    rows, cols = originalImage.shape
    filledImage = originalImage.copy()
    mask = np.zeros((rows+2, cols+2), np.uint8)
    seed = (0, 0)
    
    # Fill from four corners
    cv2.floodFill(filledImage, mask, seed, 255)
    cv2.floodFill(filledImage, mask, (cols - 1, 0), 255)
    cv2.floodFill(filledImage, mask, (0, rows - 1), 255)
    cv2.floodFill(filledImage, mask, (cols - 1, rows - 1), 255)

    cv2.bitwise_not(filledImage, filledImage)
    cv2.bitwise_or(filledImage, originalImage, filledImage)

    return filledImage

# Given all the red and white contours of an image, return all pairs of red and white contours that resemble cups, as a list of Rectangles
def lookForCups(redContours, whiteContours):
    result = []

    # Define lambda functions that are useful for getting the left/rightmost or lowest/highest points in a rectangle
    xComparator = lambda point: point[0]
    yComparator = lambda point: point[1]

    for whiteContour in whiteContours:
        if not len(whiteContour): continue
        whiteRect = cv2.minAreaRect(whiteContour)
        _, (whiteWidth, whiteHeight), _ = whiteRect
        if cv2.contourArea(whiteContour) < CONTOUR_AREA_THRESHOLD: continue
        elif whiteWidth * 6 < whiteHeight: continue # If the white rectangle is too long

        whiteRectPoints = cv2.boxPoints(whiteRect)

        lowestWhitePoint = min(whiteRectPoints, key = yComparator)[1]
        highestWhitePoint = max(whiteRectPoints, key = yComparator)[1]
        leftMostWhitePoint = min(whiteRectPoints, key = xComparator)[0]
        rightMostWhitePoint = max(whiteRectPoints, key = xComparator)[0]

        for redContour in redContours:
            if not len(redContour): continue
            redRect = cv2.minAreaRect(redContour)
            _, (redWidth, redHeight), _ = redRect
            if cv2.contourArea(redContour) < CONTOUR_AREA_THRESHOLD: continue

            redRectPoints = cv2.boxPoints(redRect)

            lowestRedPoint = min(redRectPoints, key = yComparator)[1]
            highestRedPoint = max(redRectPoints, key = yComparator)[1]
            leftMostRedPoint = min(redRectPoints, key = xComparator)[0]
            rightMostRedPoint = max(redRectPoints, key = xComparator)[0]

            if lowestRedPoint > highestWhitePoint + 15 or lowestRedPoint < lowestWhitePoint:
                continue
            elif 2 * (highestRedPoint - lowestRedPoint) < 3 * (highestWhitePoint - lowestWhitePoint):
                continue
            elif leftMostRedPoint > rightMostWhitePoint or rightMostRedPoint < leftMostWhitePoint:
                continue
            elif (abs(leftMostRedPoint - leftMostWhitePoint) > abs(leftMostRedPoint - rightMostWhitePoint) or 
                abs(rightMostRedPoint - leftMostWhitePoint) < abs(rightMostRedPoint - rightMostWhitePoint) or
                abs(leftMostWhitePoint - leftMostRedPoint) > abs(leftMostWhitePoint - rightMostRedPoint) or
                abs(rightMostWhitePoint - leftMostRedPoint) < abs(rightMostWhitePoint - rightMostRedPoint)):
                continue
            elif redWidth * 3 < redHeight:
                continue

            result.append((whiteRect, redRect))
    return result

# Draw the cups given onto the image given, then return the image
def drawCups(originalImage, cups):
    for cup in cups:
        white = np.int32(cv2.boxPoints(cup[0])) 
        red = np.int32(cv2.boxPoints(cup[1]))
        cv2.drawContours(originalImage, [white], 0, (0, 255, 0), 2)
        cv2.drawContours(originalImage, [red], 0, (255, 0, 0), 2)
    return originalImage

# Setup the VideoCapture object
def cameraSetup(url):
    # Pretty sure VideoCapture doesnt raise an exception when something went wrong, so be mindful of an error in the form of:
    #     "Connection to {url} failed: Error no. _ occured"
    try:
        return cv2.VideoCapture(url)
    except Exception as e:
        print(e.message)

# Gets a snapshot of the VideoCapture feed and runs cup detection once on that image
def runCupDetection(cap, drawImage = False):
    ret, frame = cap.read()
    if not ret:
        raise Exception("VideoCapture object couldnt get an image!")
    frame = cv2.resize(frame, (640, 480)) 
    return detectCups(frame, drawImage)
