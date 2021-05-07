import cv2
import os
import numpy as np

# pixel_threshold = 10  # 轮廓内部的像素点个数阈值
# binary_threshold = 50  # 二值化阈值


def inverse_white(path):
    # 输入为png的路径
    # img = cv2.imread(path, -1)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    if img.shape[2] == 3:
        return img

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][3] == 0:
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255

    imgnew = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for k in range(3):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                imgnew[i][j][k] = img[i][j][k]
    return imgnew


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


def thin_line_detection(file, outpath, binary_threshold):
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
    (filepath, filename) = os.path.split(file)
    (onlyfilename, extension) = os.path.splitext(filename)
    dst_img_name = os.path.join(outpath, onlyfilename + "_tddst" + extension)

    if file.endswith("png"):
        img = inverse_white(file)
    else:
        img = cv2.imread(file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义结构元素的形状和大小
    binary1 = cv2.dilate(binary, kernel)  # 膨胀
    binary1 = cv2.erode(binary1, kernel)  # 腐蚀
    binary1 = cv2.erode(binary1, kernel)  # 腐蚀
    res = binary1 - binary

    res[res < 128] = 0  # 经过上述操作后可能存在值为1、2的像素点, 过滤掉

    none_zero_indexes = list(zip(*np.nonzero(res)))  # 获取roi区域非0元素的坐标
    for i in range(len(none_zero_indexes)):
        img[none_zero_indexes[i][0], none_zero_indexes[i][1], 0] = 0
        img[none_zero_indexes[i][0], none_zero_indexes[i][1], 1] = 0
        img[none_zero_indexes[i][0], none_zero_indexes[i][1], 2] = 255

    # cv2.imencode(extension, img)[1].tofile(dst_img_name)

    return img


