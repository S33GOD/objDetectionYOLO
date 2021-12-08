import cv2 as cv
import numpy as np
import glob
import random
from keras import backend as K


net = cv.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")   # loading net

classes = ["Ship"]   # set class names

images_path = glob.glob(r"images/*.jpg")  # select all paths of images

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

y_true = [len(images_path), 768, 768, 1]
y_pred = y_true
def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[0,1])
  union = K.sum(y_true, axis=[0,1]) + K.sum(y_pred, axis=[0,1])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice


random.shuffle(images_path)

for img_path in images_path:

    img = cv.imread(img_path)
    height, width, channels = img.shape

    blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)



    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(dice_coef(blob, blob))



    font = cv.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 0, 255)
            conf = str(round(confidences[i], 2))
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, label, (x, y), font, 3, color, 2)
            cv.putText(img, conf, (x+120, y), font, 1, color, 2)

    cv.imshow("Image", img)
    key = cv.waitKey(0)

cv.destroyAllWindows()
