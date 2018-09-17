import cv2
import numpy


# get haar casscade  (trainer file)
trainerfile= "haarcascade_frontalface_default.xml"

image="fa.png"

trainer=cv2.CascadeClassifier(trainerfile)

img= cv2.imread(image)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#detect face
face= trainer.detectMultiScale(
  gray,
  scaleFactor=1.1,
  minNeighbors=5,
  minSize=(30,30),
  flags=cv2.cv.CV_HAAR_SCALE_IMAGE
 )
print("faces {0} found:".format(len(face)))
#draw Rectangle
for (x,y,w,h) in face:
    cv2.rectangle(img,(x, y),(x+w ,y+h),(255,0,0),2)

cv2.imshow("face", img)
cv2.waitKey(0)
