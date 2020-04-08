import os
from os.path import join as j
import numpy as np
import matplotlib.pyplot as plt



def detect_face(img):
    # The file containing the pretrained classifier 
    haar_file = 'haarcascade_frontalface_default.xml'
  
    # Image to be resized to this shape
    (width, height) = (224, 224)     
    
    # Make the cascade classifier object
    face_cascade = cv2.CascadeClassifier(haar_file)
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Detect the face
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) 

    face_resize = None
    for (x, y, w, h) in faces:
        # The classifier seemed to scrap the chin and hair. Adjustments made to accomodate those.
        face = img[y-60:y + h + 60, x-20:x + w+20] 
        face_resize = cv2.resize(face, (width, height))
    
    return face_resize


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def verify_face(img, username):
    dir_ = 'dataset'
    NUM_CLASSES = len(os.listdir(dir_))
    
    # The directory containing the user's photos
    pos_dir = username
    P = os.listdir(os.path.join(dir_, pos_dir))[-1]

    neg_dirs = [] #os.listdir(dir_)[np.random.randint(NUM_CLASSES)]

    for i in range(3):
        neg_dir = os.listdir(dir_)[np.random.randint(NUM_CLASSES)]
        while neg_dir == pos_dir or neg_dir in neg_dirs:
            neg_dir = os.listdir(dir_)[np.random.randint(NUM_CLASSES)]
        neg_dirs.append(neg_dir)
            

    P = plt.imread(j(dir_, pos_dir, os.listdir(j(dir_, pos_dir)))[-1])
    N1 = plt.imread(dir_ +'/'+ neg_dirs[0] +'/'+ os.listdir(dir_+'/'+neg_dirs[0])[-1])
    N2 = plt.imread(dir_ +'/'+ neg_dirs[1] +'/'+ os.listdir(dir_+'/'+neg_dirs[1])[-1])
    N3 = plt.imread(dir_ +'/'+ neg_dirs[2] +'/'+ os.listdir(dir_+'/'+neg_dirs[2])[-1])

    P = cv2.resize(P, (224,224))
    N1 = cv2.resize(N1, (224,224))
    N2 = cv2.resize(N2, (224,224))
    N3 = cv2.resize(N3, (224,224))
    
    A = np.reshape(img, (1,224,224,3))
    P = np.reshape(P, (1,224,224,3))
    N1, N2, N3 = [np.reshape(N, (1,224,224,3)) for N in [N1, N2, N3]]


    req_model = load_req_model('<path to model.h5>')

    enc_anc   = req_model.predict(A)
    enc_pos   = req_model.predict(P)
    enc_neg_1 = req_model.predict(N1)
    enc_neg_2 = req_model.predict(N2)
    enc_neg_3 = req_model.predict(N3)

    
    # Normalizing the encodings to avoid large values
    maxm = np.max(enc_anc)
    enc_anc = enc_anc/maxm
    enc_pos = enc_pos/maxm
    enc_neg_1, enc_neg_2, enc_neg_3 = [enc/maxm for enc in [enc_neg_1, enc_neg_2, enc_neg_3]]

    positive_loss = mean_squared_error(enc_anc, enc_pos).numpy()
    negative_losses = [mean_squared_error(enc_anc, enc_neg).numpy() 
                        for enc_neg in [enc_neg_1, enc_neg_2, enc_neg_3]]

    # flag becomes false if the match is unsuccessful
    flag = True
    for neg_loss in negative_losses:
        if positive_loss > neg_loss:
            flag = False
    
    return flag



# Here it is assumed that some app has provided with a webcam click called 'img' at the time of login.
face_detected = detect_face(img)  
# The username is provided at the time of login.
flag = verify_face(face_detected, username)
