import torch, cv2, face_recognition
import torch.nn as nn
import numpy as np

from time import time
from sort import Sort

from PIL import Image

from torchvision import transforms
from model import vgg16


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# load model from checkpoint path
ckpt_path="epoch=87-step=19800.ckpt"
model=vgg16.load_from_checkpoint(ckpt_path)
print(f"VGG16 model: {sum(p.numel() for p in model.parameters())/1e6} million parameters")

# for running inference
transform_test=transforms.Compose([transforms.TenCrop(44),
                                   transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])

# list of emotions
emotions = {0: ['Angry', (0,0,255), (255,255,255)],
            1: ['Disgust', (0,102,0), (255,255,255)],
            2: ['Fear', (255,255,153), (0,51,51)],
            3: ['Happy', (153,0,153), (255,255,255)],
            4: ['Sad', (255,0,0), (255,255,255)],
            5: ['Surprise', (0,255,0), (255,255,255)],
            6: ['Neutral', (160,160,160), (255,255,255)]}

class EmotionDetection:
    def __init__(self,capture_index):
        
        self.capture_index=capture_index
        self.model=model
        self.emotions=emotions
    
    @torch.no_grad()
    def predict(self,frame):
        # return 2 lists: (1) [bbox,score] with bbox=[left,top,width,height]
        #                 (2) [predictions] = emotions facial regions
        
        detections=[]  
        predictions=[] 
        
        self.model.eval()
        face_locations=face_recognition.face_locations(frame)

        # give predictions to facial regions 
        for (top,right,bottom,left) in face_locations:

            # extract facial region
            face=frame[top:bottom,left:right,:]

            # convert to gray scale and resize
            gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            gray=cv2.resize(gray,(48,48)).astype(np.uint8)

            # randomly crop the face image into 10 subimages
            inputs=transform_test(Image.fromarray(gray))           #[10,1,44,44]

            # make prediction
            pred=model(inputs)                                     # [10,7]
            pred=pred.mean(axis=0)                                 # [7,]

            scores=torch.nn.functional.softmax(pred,dim=-1)
            pred=scores.argmax().item()

            score=scores[pred].item()

            # append result to detection list
            merged_detection=np.array([left,top,right,bottom,score])

            detections.append(merged_detection)
            predictions.append(pred)

        return np.array(detections), np.array(predictions)

    def draw_boxes(self,frame,bboxes,preds):
        # draw boxes around facial regions
        for bbox,pred in zip(bboxes,preds):
            
            left,top,right,bottom=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            
            # draw rectangle and put texts around the faces
            cv2.rectangle(frame,(left,top), (right,bottom), self.emotions[pred][1],2)
            cv2.rectangle(frame,(left,top-50), (right,top), self.emotions[pred][1],-1)
            cv2.putText(frame, f'{self.emotions[pred][0]}', (left,top-5), 0, 1.5, self.emotions[pred][2],2)
            
        return frame
    
    def __call__(self):
        cap=cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        
        # initiate sort tracker
        # max_age= max number of frames that are allowed to loose the object
        # min_hits= min number of frames that are required to initiate a tracker
        sort=Sort(max_age=30,min_hits=5,iou_threshold=0.3)
        
        while True:
            success,frame=cap.read()
            assert success
            
            # resize frame by 1/4 
            small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            detections, predictions = self.predict(small_frame)
            
            # times 4 to get back original positions in detections
            # detections = list of [top,left,bottom,right,score]
            for i in range(len(detections)):
                detections[i][:4]*=4
            
            # sort tracking 
            # handle the case where no detection is obtained
            if len(detections)==0:
                detections=np.empty((0,5))
            
            # new positions+scores for bounding box [bbox_new,score_new]
            results=sort.update(detections)
            bboxes=results[:,:-1]
            frame=self.draw_boxes(frame,bboxes,predictions)
            
            cv2.imshow('Emotion Detection - Tracked', frame)
            
            if cv2.waitKey(1) & 0xFF==27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
detector=EmotionDetection(capture_index=0)
detector()