import os
import sys
import bz2
# from keras.utils import get_file
from facial_expression_dataset.face_alignment import image_align
from facial_expression_dataset.landmarks_detector import LandmarksDetector
import cv2

# LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
LANDMARKS_MODEL_URL = "./shape_predictor_68_face_landmarks.dat"


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def draw_landmark_face(img,dlib_68_landmarks,save_name):
    for i,point in enumerate(dlib_68_landmarks):
        # def circle(img, center, radius, color, thickness=None, lineType=None,shift=None):
        cv2.circle(img,center=point,radius=1,color=(0,0,255),thickness=-1)
        cv2.putText(img,"{}".format(i),point,cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255))
        cv2.imwrite(save_name,img)
    return img


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py  ./raw_images ./aligned_images
    """

    landmarks_model_path = LANDMARKS_MODEL_URL

    RAW_IMAGES_DIR = "./data/affectnet/"
    ALIGNED_IMAGES_DIR = "./data/affectnet/images/"

    # ALIGNED_IMAGES_DIR = sys.argv[2]

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    # for img_name in list1:

    for img_name in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        # print(raw_img_path)

        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = '%s.jpg' % (os.path.splitext(img_name)[0])
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            print(aligned_face_path)
            os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
            image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=1024)

            draw = 0
            if draw:
                img = cv2.imread(raw_img_path)
                landmark_save_path = os.path.join(ALIGNED_IMAGES_DIR, 'landmarks_' + face_img_name)
                draw_landmark_face(img, face_landmarks, save_name=landmark_save_path)


