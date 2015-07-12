import cv2
import argparse
from PrepareData import ConstructNPndArray


class DetectFeature:
    def __init__(self, mat_img, gray_img):
        self.mat_img = mat_img
        self.gray_img = gray_img

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier('C:/Python27/Lib/site-packages/SimpleCV/Features/HaarCascades/face.xml')
        detected_faces = face_cascade.detectMultiScale(image=self.gray_img, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in detected_faces:
            cv2.rectangle(self.mat_img, (x, y), (x+w, y+h), (255, 204, 0), thickness=3)
        return self.mat_img

    def display_image(self):
        mat_img = self.detect_face()
        cv2.imshow('mat_img', mat_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self):
        mat_img = self.detect_face()
        cv2.imwrite('C:\Users\eshouppe\Documents\FinalImage.png', mat_img)


# Parse system arguments.  img_ flag indicates best method to create array
parser = argparse.ArgumentParser(description='select method to create array')
parser.add_argument('--img0', dest='url_imread')
parser.add_argument('--img1', dest='url_video')
parser.add_argument('--imgX', dest='url_check')
args = parser.parse_args()

if args.url_imread:
    new_matrix = ConstructNPndArray(args.url_imread)
    matrix_image, grayscale_image = new_matrix.imread_matrix_grayscale()

elif args.url_video:
    new_matrix = ConstructNPndArray(args.url_video)
    matrix_image, grayscale_image = new_matrix.video_matrix_grayscale()

elif args.url_check:
    new_matrix = ConstructNPndArray(args.url_check)
    flag = new_matrix.check_imread()
    if 'numpy' in str(flag):
        matrix_image, grayscale_image = new_matrix.imread_matrix_grayscale()
    else:
        flag = new_matrix.check_video()
        if flag is True:
            matrix_image, grayscale_image = new_matrix.video_matrix_grayscale()


display_detect_face = DetectFeature(matrix_image, grayscale_image)
display_detect_face.display_image()
