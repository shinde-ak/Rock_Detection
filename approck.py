from __future__ import division , print_function
import sys
import os 
import glob
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

from tensorflow.keras.models import load_model
from tensorflow.keras import backend
from tensorflow.keras import backend




import tensorflow as tf
global graph
graph = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
tf.compat.v1.keras.backend.set_session(graph)
from skimage.transform import resize
from flask import Flask,request,render_template,redirect,url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
app = Flask(__name__)
MODEL_PATH = 'models/rock_classification.h5'
model = load_model(MODEL_PATH)
@app.route('/',methods = ["GET"] )
def index():
    return render_template("base.html")
@app.route('/predict',methods = ["GET","POST"])
def upload():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        print("current path: ",basepath)
        file_path = os.path.join(basepath,"uploads",secure_filename(f.filename))
        f.save(file_path)
        print("joined path: ",file_path)
        img = image.load_img(file_path,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
       
        with graph.as_default():
            prediction_class = model.predict(x)
            classes_x=np.argmax(prediction_class,axis=1)
            print("prediction",prediction_class)
        
        i=classes_x.flatten()
        print(i)
        index = ['Igneous rock','Metamorpic rock','Sedimentary rock']
        if(str(index[i[0]])=="Igneous rock"):
            text="the predicted class is : " + str(index[i[0]])
        elif (str(index[i[0]])=="Metamorpic rock"):
            text ="the predicted class is : " + str(index[i[0]])
        elif(str(index[i[0]])=="Sedimentary rock"):
            text="the predicted class is: " + str(index[i[0]])
    return text 
if __name__ == "__main__":
    # app.run(debug = False ,threaded= False)
    app.run(host='127.0.0.1',port=5001)

 
