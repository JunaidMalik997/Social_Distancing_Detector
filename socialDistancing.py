import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_people(frame, net, personIdx=0):
    (H, W)=frame.shape[:2]
    results= []

    blob=cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    last_layer=net.getUnconnectedOutLayersNames()
    layer_out=net.forward(last_layer)

    boxes=[]
    confidences=[]
    centroids=[]

    for output in layer_out:
        for detection in output:
            score=detection[5:]
            class_id=np.argmax(score)
            confidence=score[class_id]

            if class_id==personIdx and confidence>0.3:
                center_x=int(detection[0]*W)
                center_y=int(detection[1]*H)

                w=int(detection[2]*W)
                h=int(detection[3]*H)

                x=int(center_x-(w/2))
                y=int(center_y-(h/2))

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                centroids.append((center_x, center_y))

    indexes=cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

    if len(indexes)>0:
        for i in indexes.flatten():
            (x,y)=(boxes[i][0], boxes[i][1])
            (w,h)=(boxes[i][2], boxes[i][3])

            r=(confidences[i], (x,y,x+w,y+h), centroids[i])
            results.append(r)

    return results