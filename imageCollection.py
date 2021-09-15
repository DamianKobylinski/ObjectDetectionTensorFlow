import cv2
import uuid
import os
import time

labels = ['thumbsup', 'thumbsdown']
number_imgs = 3

IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')

if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        os.system("mkdir -p " + str(IMAGES_PATH))
    if os.name == 'nt':
        os.system("mkdir " + str(IMAGES_PATH))
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.system("mkdir " + str(path))

for label in labels:
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')

if not os.path.exists(LABELIMG_PATH):
    os.system("mkdir " + str(LABELIMG_PATH))
    os.system("git clone https://github.com/tzutalin/labelImg" + str(LABELIMG_PATH))

if os.name == 'posix':
    os.system("make qt5py3")
if os.name =='nt':
    os.system("cd " + str(LABELIMG_PATH) + " && pyrcc5 -o libs/resources.py resources.qrc")

os.system("cd " + str(LABELIMG_PATH) + " && python labelImg.py")

TRAIN_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'train')
TEST_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'test')
ARCHIVE_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'archive.tar.gz')

os.system("tar -czf " + str(ARCHIVE_PATH) + " " + str(TRAIN_PATH) + " " + str(TEST_PATH))