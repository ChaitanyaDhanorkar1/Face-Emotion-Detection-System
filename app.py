from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import os

# Loading the model
model = tf.keras.models.model_from_json(open("fer.json", "r").read()) 
# Loading the model weights  
model.load_weights( 'fer.h5' )                              
face_haar_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,'haarcascade_frontalface_default.xml'))
camera = cv2.VideoCapture(0)
app = Flask(__name__)

def gen_frames():                                      
    while True:
        # Used to capture frame by frame.
        success,frame = camera.read()
        if not success:
            break
        else:
            gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)  
            for (x,y,w,h) in faces_detected:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
                roi_gray=gray_img[y:y+w,x:x+h]          
                roi_gray=cv2.resize(roi_gray,(48,48))  
                img_pixels = tf.keras.utils.img_to_array(roi_gray)  
                img_pixels = np.expand_dims(img_pixels, axis = 0)  
                img_pixels /= 255  
                predictions = model.predict(img_pixels)  
                max_index = np.argmax(predictions[0])  
                emotions = ['fear', 'disgust', 'happy', 'angry', 'sad', 'surprise', 'neutral']  
                predicted_emotion = emotions[max_index]  
                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
        
            ret, buffer = cv2.imencode('.jpg', frame)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
