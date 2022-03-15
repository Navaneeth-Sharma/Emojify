import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


foreground_pika1 = Image.open("src/filters/imgs/pikachu_ears.png")


def get_pikachu_ear(img, size, x, y, foreground=foreground_pika1):

    foreground = foreground.resize((size, size))
    background = Image.fromarray(img)
    background.paste(foreground, (x - 50, y - 200), foreground)

    img = cv2.cvtColor(np.array(background), cv2.COLOR_BGR2RGB)
    return img


foreground_emoji = Image.open("src/filters/imgs/cry_2.png")


def get_cry_emoji(img, size, x, y, foreground=foreground_emoji):

    foreground = foreground.resize((size, size))
    background = Image.fromarray(img)
    background.paste(foreground, (x - 10 , y -20 ), foreground)

    img = cv2.cvtColor(np.array(background), cv2.COLOR_BGR2RGB)
    return img


eye_cascade = cv2.CascadeClassifier("src/models/frontalEyes35x16.xml")


def get_coolglass(img):
    eye = eye_cascade.detectMultiScale(img)[0]
    eye_x, eye_y, eye_w, eye_h = eye

    glasses = plt.imread("src/filters/imgs/cool_glass.png")

    for i in range(glasses.shape[1]):
        for j in range(glasses.shape[0]):
            if (glasses[i, j, 3] > 0):
                img[eye_y + i - 70, eye_x + j - 30, :] = glasses[i, j, :-1]

    return img
