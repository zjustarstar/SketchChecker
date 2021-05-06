import cv2
import os
import numpy as np


def get_contour_pixel_number(img, points):
    """统计轮廓内部的像素个数
    Parameters:
        Input:
            img: 输入图片
            points: 多边形的顶点坐标
        Return:
            返回轮廓内部像素点个数
    """
    polygon = points  # 这里是多边形的顶点坐标
    im = np.zeros(img.shape[:2], dtype="uint8")
    polygon_mask = cv2.fillPoly(im, polygon, 255)
    pixel_numbers = np.sum(np.greater(polygon_mask, 0))

    return pixel_numbers


def thin_line_detection(src_img_dir, dst_img_dir, area_threshold, binary_threshold):
    """检测细线
    Parameters:
        Input:
            src_img_dir: 输入图片路径
            dst_img_dir: 输出图片路径
            area_threshold: 轮廓中最小像素个数
            binary_threshold：二值化阈值
        Output:
            红色填充的图片和黑色填充的图片
    """
    for file in os.listdir(src_img_dir):

        src_img_name = os.path.join(src_img_dir, file)
        dst_img_name = os.path.join(dst_img_dir, file)

        img = cv2.imread(src_img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义结构元素的形状和大小
        binary = cv2.erode(binary, kernel)  # 腐蚀
        binary = cv2.dilate(binary, kernel)  # 膨胀
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        count = 0  # 统计不符合要求的区域个数
        # 计算每个轮廓
        for i in contours:
            area = get_contour_pixel_number(binary, [i])
            if area <= area_threshold:
                count += 1
                cv2.fillPoly(img, [i], (0, 0, 255))
        print("二值化阈值:"+str(binary_threshold), "  图片:", src_img_name, "  面积小于"+str(area_threshold)+"的区域个数为:", count)

        cv2.imwrite(dst_img_name, img)


if __name__ == '__main__':
    src_img_dir = "thinLineDetection/src"  # 源目录
    dst_img_dir = "thinLineDetection/dst"  # 目标目录

    pixel_threshold = 10  # 轮廓内部的像素点个数阈值
    binary_threshold = 50  # 二值化阈值
    thin_line_detection(src_img_dir, dst_img_dir, pixel_threshold, binary_threshold)



