from module import TubeletEmbedding, PositionalEncoder
from model import NN
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from threading import Thread
import os

tf.config.list_physical_devices('GPU')

INPUT_SHAPE = (64, 128, 128, 3)
NUM_CLASSES = 2

LEARNING_RATE = 1e-4

PATCH_SIZE = (16, 32, 32)

LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8
    
model = NN(
        tubelet_embedder=TubeletEmbedding(embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
        input_shape=INPUT_SHAPE,
        transformer_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        embed_dim=PROJECTION_DIM,
        layer_norm_eps=LAYER_NORM_EPS,
        num_classes=NUM_CLASSES,
    )

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
            )

model.load_weights('model/model_1210_4.h5')

def prediction():
    global pred_var
    global org
    global color
    k=0
    run=True
    while run:
        try:
            frames=np.load('frame'+str(k)+'.npy')
            image=tf.image.per_image_standardization(frames)
            image=tf.reshape(image,shape=(1,64,128,128,3))
            pred=np.argmax(model.predict(image), axis=1)
            if pred == 1:
                pred_var='VIOLENCE'
                color = (0, 0, 255)
                org = (70, 50)    
            else:
                pred_var='NO VIOLENCE'
                color = (255, 0, 0)
                org = (50, 50)  
            os.remove('frame'+str(k)+'.npy')
            k=k+1
        except:
            continue    
    cv2.destroyAllWindows()


def load_video():
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 1
    thickness = 4
    s=0
    vid = cv2.VideoCapture('video5.mp4')
    frames=[]
    success=True
    while success :
        success,frame= vid.read()
        image = frame
        if success==False:
            break
        frame = cv2.resize(frame,(128,128), interpolation=cv2.INTER_AREA)
        frame = np.reshape(frame, (128,128,3))
        image = cv2.putText(image, pred_var, org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Video',image )
        frames.append(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        if len(frames)==64:
            np.save('frame'+str(s)+'.npy',np.array(frames))
            frames=[]
            s=s+1
    vid.release()
    cv2.destroyAllWindows()

pred_var='LOADING!!!'
org = (50, 50) 
color = (0, 255, 0)

t2= Thread(target = load_video)
t1= Thread(target = prediction)
t1.daemon = True

t2.start()
t1.start() 

cv2.destroyAllWindows()