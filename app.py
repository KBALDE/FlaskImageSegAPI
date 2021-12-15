
from flask import Flask, render_template, request

import urllib.request
import json
import os
import ssl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import imageio
import ast


app=Flask(__name__)

# some variable s
# hide them later in a file like init file

url = 'http://4086ed70-8ad9-4c25-9f18-1a821627531d.eastus2.azurecontainer.io/score'
api_key = 'EylgUu1MzEW2AboZEUxFjcWeNDqYwToF' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}


def get_img_path_file(img_id):
    """
    could change the path_root to be transportable folder
    containing the images
    """
    #path_root='./gtFine/val_images/'
    path_root='./static/images/'

    return str(path_root+img_id+'.png')


def read_raw_image_file(image_file):
    
    target_size=(224, 224)
    
    img = imageio.imread(image_file)
    img = img[:,:, :3]                 # RETRIEVE the three first channels 
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, target_size )  
    img_array= np.asarray(img)    

    return img_array

def display_prod(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    plt.savefig('./static/images/image_pred.png') # saved here

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl,
                                                                           '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) 

def prediction_call_service(image_id):
    
    img_array=read_raw_image_file(get_img_path_file(image_id))
    
    # convert to tensor and add a dim
    sample_image=tf.constant(img_array)[tf.newaxis, ...]
    
    data= {"data": sample_image.numpy().tolist()}
    
    body = str.encode(json.dumps(data))
    
    req = urllib.request.Request(url, body, headers)
    
    try:
        response = urllib.request.urlopen(req)
    
        result = response.read()
        pred = ast.literal_eval(ast.literal_eval(result.decode("utf-8")))
        pred_mask = np.argmax(tf.constant(pred), axis=3)[..., tf.newaxis]
        
        #print("Input Image and Predicted Mask")

        display_list=[sample_image[0], pred_mask[0]]
        
        fig = plt.figure(figsize=(15, 15))

        title = ['Input Image', 'Predicted Mask']
        

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.savefig('./static/images/image_pred.png') # saved here
        plt.close()
        return fig
    

        #display_prod([sample_image[0], pred_mask[0]]) # save an image somewhere

        #print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
    
        # Print the headers - they include the requert ID and the timestamp, 
        # which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))





# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        image_id = request.form.get('image_id')

    pred_image_mask = prediction_call_service(image_id)
    
    return render_template('index.html', name = 'prediction for image segmentation', prediction = image_id, img_path = './static/images/image_pred.png')
        #image_local_url ='./static/images/image_pred.png')
     #prediction = result, img_path = image_url)


