import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mlp
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics



train_X = np.load('trainCells.npy')
train_y = np.load('trainLabel.npy')

test_X = np.load('testCells.npy')
test_y = np.load('testLabel.npy')

eval_X = np.load('evalCells.npy')
eval_y = np.load('evalLabel.npy')

low=127
t1=0
TrainX=[]
alpha=1.5
beta=0.0
for img in train_X:
    
    blurred = (cv2.GaussianBlur(img,(5,5),0)*255).astype('uint8')
    # blurred = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    ret, threshold = cv2.threshold(blurred,low,255,t1)
    
    contours,j=cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    area=[]
    for j in contours:
        area.append(cv2.contourArea(j))
    area.sort(reverse = True)
    
    area1=np.array([0,0,0,0,0]).astype(np.float)
    for i in range(min(len(area),5)):
        area1[i]=area[i]
    area1=area1[1:]
    TrainX.append(area1)
TrainX=np.array(TrainX)

TestX=[]
for img in test_X:
    
    blurred = (cv2.GaussianBlur(img,(5,5),0)*255).astype('uint8')
    #blurred = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    ret, threshold = cv2.threshold(blurred,low,255,t1)
    
    contours,j=cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    area=[]
    for j in contours:
        area.append(cv2.contourArea(j))
    area.sort(reverse = True)
    
    area1=np.array([0,0,0,0,0]).astype(np.float)
    for i in range(min(len(area),5)):
        area1[i]=area[i]
    area1=area1[1:]
    TestX.append(area1)
TestX=np.array(TestX)

model = RandomForestClassifier(n_estimators=100,max_depth=5)
model.fit(TrainX,train_y)
joblib.dump(model,"rf_malaria_100_4")

predictions = model.predict(TestX)

print(metrics.classification_report(predictions,test_y))
print(model.score(TestX,test_y))