import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')

img = cv2.imread('assets/img/face.jpg')
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
faces = detector(gray)

distance = 0

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=4)

    landmarks = predictor(image=gray, box=face)

    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

    eye1 = landmarks.part(39)
    eye2 = landmarks.part(42)

    cv2.circle(img=img, center=(eye1.x, eye1.y), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.circle(img=img, center=(eye2.x, eye2.y), radius=3, color=(0, 0, 255), thickness=-1)

    distance = ((eye1.x - eye2.x) ** 2 + (eye1.y - eye2.y) ** 2) ** 0.5

cv2.putText(img=img, text='La distancia entre los ojos es {}'.format(distance), org=(50, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

print('The distance between eyes is {}'.format(distance))

cv2.imshow(winname='Face', mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()
