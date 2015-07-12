import cv2


# Construct numpy ndarray for use in cascade classifier
class ConstructNPndArray:
    def __init__(self, img):
        self.img = img

    def imread_matrix_grayscale(self):
        mat_img = cv2.imread(self.img)
        gray_img = cv2.cvtColor(mat_img, cv2.COLOR_BGR2GRAY)
        return mat_img, gray_img

    def video_matrix_grayscale(self):
        request = cv2.VideoCapture()
        request.open(self.img)
        if request.isOpened():
            ret, mat_img = request.read()
            gray_img = cv2.cvtColor(mat_img, cv2.COLOR_BGR2GRAY)
        return mat_img, gray_img

    def check_imread(self):
        mat_img = cv2.imread(self.img)
        return type(mat_img)

    def check_video(self):
        request = cv2.VideoCapture()
        request.open(self.img)
        if request.isOpened():
            ret, mat_img = request.read()
            if mat_img is None:
                return False
            else:
                return True
