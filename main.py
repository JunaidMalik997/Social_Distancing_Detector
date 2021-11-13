import argparse
import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist
from socialDistancing import detect_people

ap=argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, default='', help='path to (optional) input video file')
ap.add_argument('-o', '--output', type=str, default='', help='path to (optional) output video file')
ap.add_argument('-d', '--display', type=int, default=1, help='whether or not output frame should be displayed')
args=vars(ap.parse_args())

print('[INFO] loading YOLO from disk...')
net=cv2.dnn.readNetFromDarknet('yolov3-spp.cfg', 'yolov3-spp.weights')
with open('coco.names', 'r') as f:
    classes=[line.strip() for line in f.readlines()]

print('[INFO] accessing video stream...')
vs=cv2.VideoCapture(args['input'] if args['input'] else 0)
writer=None

while True:
    (grabbed, frame)=vs.read()

    if not grabbed:
        break

    frame=imutils.resize(frame, width=700)
    results= detect_people(frame, net, classes.index('person'))

    #intialize the set of indexes that violate the minimum social distance
    violate=set()

    if len(results)>=2:
        centroids= np.array([r[2] for r in results])
        D=dist.cdist(centroids, centroids, metric='euclidean')

        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                if D[i,j]<50:
                    violate.add(i)
                    violate.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY)=bbox
        (cX, cY)=centroid
        color=(0,255,0)

        if i in violate:
            color=(0,0,255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    text=f'Social Distancing Violations: {len(violate)}'
    cv2.putText(frame, text, (10, frame.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,255), 3)

    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # if an output video file path has been supplied and the video
    # writer has not been initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,
                                 (frame.shape[1], frame.shape[0]), True)

        # if the video writer is not None, write the frame to the output
        # video file
    if writer is not None:
        writer.write(frame)