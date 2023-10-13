import os
import cv2
import skvideo.io  
import numpy as np

def video_center_crop(video,numframes):
    f,_,_,_ = video.shape
    start = f//2 - numframes//2
    end = start + numframes
    return video[start:end, :, :, :]

Fight_path = 'RWF-2000/Fight'
NonFight_path = 'RWF-2000/NonFight'
Fight=[]
NonFight=[]

for filename in os.listdir(Fight_path):
    f = os.path.join(Fight_path, filename)
    video = skvideo.io.vread(f)
    frames=[]
    for i in range(video.shape[0]):
      frame = cv2.resize(video[i], (128,128), interpolation=cv2.INTER_CUBIC)
      frames.append(frame)
    video = np.asarray(frames)
    video = video_center_crop(video, 64)
    Fight.append(video)

for filename in os.listdir(NonFight_path):
    f = os.path.join(NonFight_path, filename)    
    video = skvideo.io.vread(f)  
    frames=[]
    for i in range(video.shape[0]):
      frame = cv2.resize(video[i], (128,128), interpolation=cv2.INTER_CUBIC)
      frames.append(frame)
    video = np.asarray(frames)
    video = video_center_crop(video, 64)
    NonFight.append(video)

np.save('npy_data/Fight.npy', np.array(Fight))
np.save('npy_data/NonFight.npy', np.array(NonFight))