import cv2, os
import numpy as np

class BLUR_GENERATE:
    def __init__(self, degree=30, save_dir=None):
        self.degree = degree
        self.save_dir = save_dir
        self.__random_degree()

    def __random_degree(self):
        self.motion_degree = np.random.randint(1, self.degree)
        self.gaussian_degree = self.degree - self.motion_degree
        if self.gaussian_degree % 2 == 0:
            self.gaussian_degree += 1

    def motion_blur(self, image, angle=20):
        image = np.array(image)
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((self.motion_degree/2, self.motion_degree/2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(self.motion_degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (self.motion_degree, self.motion_degree))

        motion_blur_kernel = motion_blur_kernel / self.motion_degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred

    def gaussian_blur(self, image):
        # 高斯核必须为奇数
        blurred = cv2.GaussianBlur(image, ksize=(self.gaussian_degree, self.gaussian_degree), sigmaX=30, sigmaY=0)
        return blurred

    def blur_pipeline(self, img_path):
        self.__random_degree()
        image = cv2.imread(img_path)
        image = self.motion_blur(image)
        image = self.gaussian_blur(image)
        print(self.motion_degree, self.gaussian_degree)
        if self.save_dir != None:
            save_path = os.path.join(self.save_dir, 'blur_{}'.format(img_path.split('\\')[-1]))
            cv2.imwrite(save_path, image)
        return image


def main():
    degree = 18
    save_dir = r'D:\Desktop\shishuai.yan\Desktop\git_code\tf_keras_classifier\imgs\clear_2\train\1'
    bg = BLUR_GENERATE(degree, save_dir)
    img_dir = r'D:\Desktop\shishuai.yan\Desktop\git_code\tf_keras_classifier\imgs\clear_2\train\0'
    img_list = np.array(os.listdir(img_dir))
    choices_num = len(os.listdir(img_dir)) - len(os.listdir(save_dir))
    choices_img = np.random.choice(img_list, choices_num, replace=False)
    print(choices_num)
    for img_name in choices_img:
        img_path = os.path.join(img_dir, img_name)
        bg.blur_pipeline(img_path)

if __name__ == '__main__':
    bg = BLUR_GENERATE(18)
    image = bg.blur_pipeline(r'D:\Desktop\shishuai.yan\Desktop\0.jpg')
    cv2.imshow('0', image)
    cv2.waitKey()
    # main()
