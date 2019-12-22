import os
import cv2

# This function can be used during sign-up of a user
def create_dataset():
    # The file containing the pretrained classifier 
    haar_file = 'haarcascade_frontalface_default.xml'

    # All the faces data will be present this folder 
    dataset = './dataset'

    if not os.path.exists(dataset):
        os.mkdir(dataset)
    sub_data = input("Enter your username: ")

    # Use the username as path name
    path = os.path.join(dataset, sub_data) 

    # Add a verfication for this step
    if not os.path.exists(path):
        os.mkdir(path)
        
    # Image to be resized to this shape
    (width, height) = (224, 224)     
    
    # Make the cascade classifier object
    face_cascade = cv2.CascadeClassifier(haar_file) 
    webcam = cv2.VideoCapture(0)  

    # The program loops until it has 30 images of the face. 
    count = 0
    while count < 20:
        # Read from the webcam
        (_, im) = webcam.read()
        
        # Convert to grayscale
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
        # Detect the face
        faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
        
        face_resize = None
        for (x, y, w, h) in faces:
            # The classifier seemed to scrap the chin and hair. Adjustments made to accomodate those.
            face = im[y-60 : y+h+60, x-20 : x+w+20] 
            face_resize = cv2.resize(face, (width, height)) 
            cv2.imwrite('% s/% s.png' % (path, count), face_resize) 
        count += 1

        cv2.imshow('OpenCV', im) 
        key = cv2.waitKey(100) 
        if key == 27: 
            break

# Call this function whenever you need to create a dataset of the person's images
create_dataset()
