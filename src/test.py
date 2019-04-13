import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/weights.h5")
print("Loaded model from disk")
	
#------------------------------
#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions, foldername):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    pred = "/".join(foldername.split('/')[:-1]) + '/' + foldername.split('/')[-2] + '_pred.png'
    plt.savefig(pred)
    plt.show()
    
## Loading the face cascade
path = os.getcwd()
face_cascade = cv2.CascadeClassifier('haarcascades/frontal_face_default.xml') 
output_folder = path + '/Output/'

input_folder = path + '/testing/'
filename = '2.jpg'

foldername = output_folder + str(filename.split('.')[0]) + '/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
if not os.path.exists(foldername):
    os.makedirs(foldername)

## Loading the image
img = cv2.imread(input_folder + filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detects faces of different sizes in the input image 
faces = face_cascade.detectMultiScale(gray, 1.2, 4) 
if len(faces) >=1:
    print('Face found')
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
    #    cv2.rectangle(img,(x-20,y-20),(x+w+20,y+h+20),(255,255,0),2)  
        roi_gray = gray[y-20:y+h+20, x-20:x+w+20] 
        roi_color = img[y-20:y+h+20, x-20:x+w+20] 
    
        gray_image = cv2.resize(roi_gray, (48,48))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x = image.img_to_array(gray_image)
        x = np.expand_dims(x, axis = 0)
        
        x /= 255
        
        custom = loaded_model.predict(x)
        cv2.imshow("Processed Image", roi_color)
        cv2.waitKey()
        file = foldername + str(filename.split('.')[0]) + '.png'
        cv2.imwrite(file, roi_color)
        emotion_analysis(custom[0], foldername)
else:
    print('Face not detected')
cv2.destroyAllWindows()
#------------------------------
