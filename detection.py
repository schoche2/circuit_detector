import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from skimage import io, color, morphology
import tensorflow as tf
model = tf.keras.models.load_model('/Users/martin/Desktop/LIAT/circuit_detector/models/2000_250_4_l.keras')


FIG_SIZE = 20
FIG_SIZE = (FIG_SIZE,FIG_SIZE)

img_raw = cv2.imread('/Users/martin/Desktop/LIAT/circuit_detector/dataset/circuits/20210311_144631.jpg')
img_raw = cv2.resize(img_raw, (1000, 750), interpolation=4)
img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

BLOCK = 27 #@param {type:"slider", min:1, max:27, step:2}
C = 10 #@param {type:"slider", min:0, max:30, step:1}

img = cv2.GaussianBlur(img,(5,5),0)

imgTres = cv2.adaptiveThreshold(img,255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY
                                ,BLOCK,C)

img_tres = cv2.bitwise_not(imgTres)/255

imgSkel = morphology.skeletonize(img_tres)
imgThin = morphology.thin(imgTres < 0.5)

CLOSE_SIZE = 64

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))

imgSkel.dtype = 'uint8'
img_close = cv2.bitwise_not(imgSkel)


img_close = np.copy(imgSkel)
img_close = cv2.morphologyEx(img_close, cv2.MORPH_CLOSE, kernel, iterations=1)


kernel = np.ones((3,3),np.uint8)
img_blob = cv2.erode(img_close,kernel,iterations = 1)


img_cont = img_raw.copy()

# contour detection
contours5, heirachy = cv2.findContours(img_blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_conts = np.zeros_like(img_raw)
img_conts = cv2.drawContours(cv2.cvtColor(img_blob*255, cv2.COLOR_GRAY2BGR), contours5, -1, (0,0,255), 2)

'''
cv2.imshow("Contours", img_conts)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# draw bounding boxes around detection components (blobs)
# returns tuple of new image with boudning boxes, and component locations

def detect_components(img_raw):

    img_cont = img_raw.copy()

    # contour detection
    contours5, heirachy = cv2.findContours(img_blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    components = []

    for c in contours5:
        x, y, w, h = cv2.boundingRect(c)

        # threshold for 50x50 structuring element
        if(w > CLOSE_SIZE or h > CLOSE_SIZE):
            # only include if box meets this threshold
            components.append(np.array((y-10, h+20, x-10, w+20)))

            # rect = cv2.rectangle(img_cont, (x, y), (x + w, y + h), (0,255,0), 3)

    return components


def predict_objects(img_raw):

    class_names = ['diode', 'resistor', 'inductor', 'capacitor', 'power']
    components = detect_components(img_raw)

    component_objects = {}

    for i in range(len(components)):
        y,h,x,w = components[i]

        # expanding aspect ratio to square
        if h > w:
            square = (y, y+h, x - h//2 + w//2, x + h//2 + w//2) # expand width to size of height
            orientation = "v"
        else:
            square = (y - w//2 + h//2, y + w//2 + h//2, x, x+w) # expand height to size of width
            orientation = "h"

        # expanding aspect ratio to square
        y,h,x,w = square
        crop = img[y:h, x:w]

        #v2.imshow('square', crop)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #cv2.imwrite(f'/Users/martin/Desktop/LTAI/CircuitNet/dataset/circuits/square_{i}.jpg', crop)


        img_square = tf.keras.preprocessing.image.img_to_array(crop)

        try:
            img_square = cv2.resize(crop, (128, 128))
        except:
            continue

        img_square = np.expand_dims(img_square, axis=0)
        img_square = np.vstack([img_square])

        prediction = model.predict(img_square)

        # output of softmax -> gives class with highest probaility
        classes = np.argmax(prediction, axis=-1)

        # maps highest probability class with classname
        class_prediction = class_names[int(classes)]

        component_objects.update({i + 1 :
                                    {
                                        'type': class_prediction,
                                        'location' : square,
                                        'orientation': orientation
                                    }
                                  })

    return component_objects


def draw_components(img_raw):

    img = img_raw.copy()

    try:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    except:
        pass

    components = predict_objects(img)

    FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
    FONT_SCALE = 0.7
    COLOR = (0,255,0)


    for i in components:
        y,h,x,w = components[i]['location']

        img_overlay = np.ones_like(img[y:h,x:w])*255

        # draw bounding-boxes around image
        # cv2.rectangle(img, (x, y), (w, h), COLOR, 2)

        cv2.rectangle(img_overlay, (0, 0), (w, h), (0, 255, 0), -1)
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)

        img[y:h,x:w] = cv2.addWeighted(img[y:h,x:w], 1, img_overlay, 0.1, 0)

        # get predicted label for component
        component = components[i]['type']
        orientation = components[i]['orientation']

        # add labels above bounding boxes
        cv2.putText(img, f'{component},{orientation}', (x, y-12), FONT_FACE, FONT_SCALE, COLOR, 1)

    return img

cv2.imshow('hey', draw_components(img_raw))
cv2.waitKey(0)
cv2.destroyAllWindows()