# Social_Distancing_Detector

Used YOLO pre-trained model trained on COCO dataset. That pre-trained model also contains `person` class. So, we made sure that we only detect person class while performing Social Distancing Detector since COCO dataset is based on 80 classes. We specified the index of Label person accordingly to our YOLO pre-trained model and its corresponding config file.

# Various Parameter:

confidence_value for bounding box= 0.3

NMS_Threshold Value= 0.3       (to suppress weak detections)

Computed Euclidean Distance between the bounding boxes and if that Distance is less than 50 pixels, consider it as a Social DIstancing violation. Red bounding box is used for that, otherwise Green bounding box is shown for compliance with Social Distance.

# Downloading YOLO pre-trained weights

YOLO pre-trained weights can be downloaded from the following link: 

`https://pjreddie.com/darknet/yolo/`

Config File is already provided in the Github repo.

# How to run the Program:

To run the program run the following command in your command prompt by going to the specific directory where the files are placed:

`python main.py --input pedestrians.mp4 --output output.avi`

# Output:

https://user-images.githubusercontent.com/58310295/141609218-ecf679fd-4933-48fc-b758-7e4d77ab8dcb.mp4

