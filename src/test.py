import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

#variables
num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral

# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/weights.h5")
print("Loaded model from disk")
	

##############-----------------------######################
#------------------------------
"""
#overall evaluation
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])
"""
#------------------------------
#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
#    cv2.imshow("Result", new)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
    plt.show()
#------------------------------
#make prediction for custom image out of test set

#img = image.load_img("testing/1.jpg", color_mode = "grayscale", target_size=(48, 48))
img = cv2.imread("testing/2.jpg")
img = cv2.resize(img, (48,48))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = loaded_model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

cv2.imshow("Processed Image", x)
cv2.waitKey()
cv2.destroyAllWindows()
#------------------------------