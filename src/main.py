import torch
import cv2
import dlib
from model_v1 import EmotionClassifier
from filters import filters


device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "src/models/model.pth"
IMAGE_SIZE = 48

file = "data/Data/test/happy/PrivateTest_218533.jpg"

img = cv2.imread(file)


def get_label(img, IMAGE_SIZE=48, map_location='cpu'):
    '''
    @param img: numpy array of the image
    @param IMAGE_SIZE: Size of the Image for model
    @map_location: device name cpu or cuda

    '''
    model = EmotionClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=map_location))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img/255.
    imgs = torch.Tensor(img).to(device)
    imgs = imgs.view(-1, 1, 48, 48)

    model.eval()
    op = model(imgs)

    label_dict = {'angry': 1, 'disgust': 3, 'fear': 4, 'happy': 5, 'neutral': 0, 'sad': 6, 'surprise': 2}
    u_lable_dict = dict((v, k) for k, v in label_dict.items())

    return u_lable_dict[int(op.argmax())]


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("src/models/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

cnt_happy = 10
cnt_angry = 10
cnt_neutral = 10
cnt_sad = 10

while True:
    success, img = cap.read()
    try:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector(imgGray)

        for face in faces:
            x, y = face.left(), face.top()
            w, h = face.right(), face.bottom()

            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2

            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r)

            label = get_label(img[ny:ny+nr, nx:nx+nr])

            if label == "happy":
                if cnt_happy<=0:
                    cnt_happy=10
                
                if cnt_angry>0:
                    cnt_angry-=1
                    continue

                if cnt_sad>0:
                    cnt_sad-=1
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                img = filters.get_pikachu_ear(img, nr, x, y)
                

            elif label == "angry" or label == "surprise":
                if cnt_angry<=0:
                    cnt_angry=10
                if cnt_happy>0:
                    cnt_happy-=1
                    continue
                if cnt_sad>0:
                    cnt_sad-=1
                    continue
                img = filters.get_coolglass(img)
            
            elif label == "neutral":
                if cnt_happy>0:
                    cnt_happy-=1
                    continue

                if cnt_angry>0:
                    cnt_angry-=1
                    continue

                if cnt_sad>0:
                    cnt_sad-=1
                    continue

                pass
            else:
                if cnt_sad<=0:
                    cnt_sad=10

                if cnt_happy>0:
                    cnt_happy-=1
                    continue
                if cnt_angry>0:
                    cnt_angry-=1
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                img = filters.get_cry_emoji(img, nr, x, y)

            img = cv2.rectangle(img, (x,y), (w,h), (0,255,0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    except Exception as e:
        print("The error is :", e)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

