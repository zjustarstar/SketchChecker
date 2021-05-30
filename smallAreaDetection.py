import cv2
import os
import numpy as np
import copy


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


def small_area_detection(file, img, maker_img, outpath, area_threshold, binary_threshold, debug=False):
    """检测小面积区域
    Parameters:
        Input:
            file: 输入图片(带路径)
            img: 输入的待处理的图片
            maker_img: 做了标记的图，检测结果写在该图上
            area_threshold: 轮廓中最小像素个数
            binary_threshold：二值化阈值
            debug: 调试模式下，会生成一些中间结果图
        Output:
            红色填充的图片和黑色填充的图片。将做了标记的图返回
    """

    (filepath, filename) = os.path.split(file)
    (onlyfilename, extension) = os.path.splitext(filename)

    img_copy = copy.deepcopy(img)
    if len(img.shape) < 3:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #ret, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义结构元素的形状和大小
    #binary = cv2.dilate(binary, kernel)  # 膨胀, 去掉一些很小的区域

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    count = 0  # 统计不符合要求的区域个数
    # 计算每个轮廓
    for i in contours:
        area = get_contour_pixel_number(binary, [i])
        if 3 < area <= area_threshold:
            count += 1
            cv2.fillPoly(maker_img, [i], (0, 0, 255))
            cv2.fillPoly(img_copy, [i], 0)

    print("二值化阈值:"+str(ret), "  图片:", filename, "  面积小于"+str(area_threshold)+"的区域个数为:", count)

    if debug:
        temp_img = os.path.join(outpath, onlyfilename + "_sd_binary" + extension)
        cv2.imencode(extension, binary)[1].tofile(temp_img)
        mid_img_name = os.path.join(outpath, onlyfilename + "_sd_mid" + "_" + str(count) + extension)
        cv2.imencode(extension, maker_img)[1].tofile(mid_img_name)

    return maker_img



