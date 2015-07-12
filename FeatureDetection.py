import cv2

# Load different object classifiers from a file
faceCascade = cv2.CascadeClassifier('C:/Python27/Lib/site-packages/SimpleCV/Features/HaarCascades/face.xml')
eyeCascade = cv2.CascadeClassifier('C:/Python27/Lib/site-packages/SimpleCV/Features/HaarCascades/eye.xml')
noseCascade = cv2.CascadeClassifier('C:/Python27/Lib/site-packages/SimpleCV/Features/HaarCascades/nose.xml')
mouthCascade = cv2.CascadeClassifier('C:/Python27/Lib/site-packages/SimpleCV/Features/HaarCascades/mouth.xml')

# Load image as matrix.  Convert from BGR to grayscale color space.
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect ATT_Faces in grayscale image.  Return list of rectangle values for each detected face.
# Each rectangle is specified as a list with 4 items [x,y,w,h]
faces = faceCascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    # Draw rectangle around detected face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 204, 0), thickness=3)

    # Narrow focus to region of interest within face rectangle
    roiGray = gray[y:y+h, x:x+w]
    roiColor = img[y:y+h, x:x+w]

    # Detect eyes in grayscale image.  Return list of rectangles.
    # Then draw rectangle box around detected object in BGR image.
    eyes = eyeCascade.detectMultiScale(roiGray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roiColor, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Detect noses.  Same process as eyes.
    nose = noseCascade.detectMultiScale(roiGray)
    for (nx, ny, nw, nh) in nose:
        cv2.rectangle(roiColor, (nx, ny), (nx+nw, ny+nh), (255, 0, 0), 2)

# Display the BGR image with drawn rectangles.
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
