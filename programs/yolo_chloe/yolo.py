import cv2 as cv
from cv2.dnn import NMSBoxes
import numpy as np
import time

'''
img = cv.imread('images/image1.jpg')
cv.imshow('window', img)
cv.waitKey(1)
'''

# Load names of classes and get random colors
classes = open('yolo2/coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Donne le fichier de config et le fichier avec les poids pour le modèle puis charge le réseau
net = cv.dnn.readNetFromDarknet('yolo2/yolov3.cfg', 'yolo2/yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# couche de sortie 
ln = net.getLayerNames()
try:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    # si get...Layers retourne tableau 1D when CUDA isn't available
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


def yolo_function(img):
    t0 = time.time()

    #contruction du blob à partir de l'image 
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

    #On donne l'objet blob en entrée du réseau + calcul du temps 
    net.setInput(blob)
    outputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Desenhe apenas as detecções com mais de 50% de confiança
    indices = cv.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    for i in indices:
        (x, y, w, h) = boxes[i]
        color = [int(c) for c in colors[classIDs[i]]]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
        cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    t1 = time.time()
    print('time=', t1-t0)

    cv.imshow('window', img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()