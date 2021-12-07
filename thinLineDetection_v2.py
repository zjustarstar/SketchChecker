import cv2
import os
import numpy as np

CIRCLE_MASK = cv2.imread('circle_mask.png')
CIRCLE_MASK = CIRCLE_MASK[:, :, 0]

THRESHOLD = 9
SUSPICIOUS_THRESHOLD = 10

THRESHOLD_ANGLE = 1
STEP_LINE = 120
STEP_ANGlE = 4
STEP_PIXEL = 1
SHAPE = (2048, 2048)


#
# 将输入图像内容视为一段倾斜线段
# 将直线旋转至竖直角度后，计算其宽度。
# 由于抗锯齿会使倾斜的线段边缘增添一层灰像素，
# 于是记录其旋转的角度，减去由于抗锯齿算法造成的偏差。
#
def analyze_width_round_img(img):
    max_black_wid = 0
    max_black_angle = 0
    line_width = 0
    for angle in range(0, 180, STEP_ANGlE):
        rows, cols = img.shape
        rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), angle, 1)
        rotated_img = cv2.warpAffine(img, rotate, (cols, rows))
        font_black_wid, end_black_wid = get_black_wid(rotated_img)
        black_wid = font_black_wid + end_black_wid
        if black_wid > max_black_wid:
            max_black_wid = black_wid
            max_black_angle = angle
            line_width = img.shape[0] - max_black_wid
            if line_width < THRESHOLD:
                break

    return line_width - THRESHOLD_ANGLE * np.abs(np.sin(2 * max_black_angle / 180 * 3.1415926))


#
# get_black_wid
# 输入图像内容为竖直线段
# 计算图像的第二维向量和，
# 求和结果的维数减去前导零和后导零的个数即为宽度。
#
# [ 0 0 1 1 0 0,
#   0 0 1 1 0 0,
#   0 0 1 1 0 0,
#   0 0 1 1 0 0,
#   0 0 1 1 0 0,
#   0 0 1 1 0 0 ]
#
# 如上矩阵所示，第二维求和后结果为[0,0,6,6,0,0],
# 求和结果的维数为 6 ，前导零和后导零个数均为 2
# 相减得到线段宽度为 2
def get_black_wid(img):
    font_black_wid = 0
    end_black_wid = 0
    for i in range(int(img.shape[1])):
        if np.sum(img[:, i]) == 0:
            font_black_wid += 1
        else:
            break

    for i in range(int(img.shape[1])):
        if np.sum(img[:, img.shape[1] - i - 1]) == 0:
            end_black_wid += 1
        else:
            break
    return font_black_wid, end_black_wid


def mask_circle(img):
    img = np.bitwise_and(img, CIRCLE_MASK)
    return img


#
# analyze_width
# 为了旋转后获取真实的宽度，要过滤旋转后会影响宽度判断的像素
# 于是把图像用一个圆形mask处理后，输入analyze_width_round_img获取宽度
#
def analyze_width(img):
    img = mask_circle(img)
    line_width = analyze_width_round_img(img)
    return line_width


def output(img, points):
    r = 255 - img
    g = 255 - img
    b = 255 - img
    o_r = int(np.max(SHAPE) / 200)
    i_r = int(np.max(SHAPE) / 300)
    for x, center in points:
        g[x - o_r:x - i_r, center - o_r:center + o_r] = 0
        g[x + i_r:x + o_r, center - o_r:center + o_r] = 0
        g[x - o_r:x + o_r, center - o_r:center - i_r] = 0
        g[x - o_r:x + o_r, center + i_r:center + o_r] = 0
    # imgShow(cv2.merge([r, g, b]))
    # cv2.imwrite(OUTPUT_PREFIX + filename, cv2.merge([r, g, b]))
    return cv2.merge([r, g, b])


#
# get_suspicious_points
# 该函数通过一张图片获取图中可能为细线的可疑点，方法是分两次，横向和纵向遍历像素，
# 遇到黑像素视为起点，离开黑像素视为重点，若起点与终点距离大于阈值，设两者中间点为可疑点
#
def get_suspicious_points(img):
    img_height, img_width = img.shape
    BLACK_PIXEL = 0   # 黑色像素值
    suspicious_points = []
    for x in (range(100, img_height - 100, STEP_LINE)):
        start = 0
        end = 0
        centers = []
        for y in range(100, img_width - 100, STEP_PIXEL):
            if img[x, y] > BLACK_PIXEL and start == 0:
                start = y
            elif img[x, y] == 0 and start != 0:
                end = y
                center = int((start + end) / 2)
                if (end - start) < SUSPICIOUS_THRESHOLD:
                    suspicious_points.append([x, center])
                start = 0
                end = 0

    for y in (range(100, img_width - 100, STEP_LINE)):
        start = 0
        end = 0
        centers = []
        for x in range(100, img_height - 100, STEP_PIXEL):
            if img[x, y] > BLACK_PIXEL and start == 0:
                start = x
            elif img[x, y] == 0 and start != 0:
                end = x
                center = int((start + end) / 2)
                if (end - start) < SUSPICIOUS_THRESHOLD:
                    suspicious_points.append([center, y])
                start = 0
                end = 0
    return suspicious_points


#
# get_points
# 在可疑点附近截取30*30的子图，输入到analyze_width函数判断真实宽度
#
def get_points(img, suspicious_points):
    points = []
    for x, y in suspicious_points:
        line_width = analyze_width(img[x - 15:x + 15, y - 15:y + 15])
        if 0 < line_width < THRESHOLD:
            points.append([x, y])
    return points


#
# 算法大步骤分为两步，先求可疑点，
# 再从可疑点中筛选确切点
def thin_line_detection(file, img, out_path, debug=False, delta=0):
    '''
    :param file: 输入的原文件名
    :param img: 输入图像
    :param out_path: 输出路径
    :param debug: 为true时,会输出一些信息
    :param delta: 增大或者缩小细线粗细的阈值。为正时，线的阈值增加，将有更多的线被检测到。
                                              为负时，线的阈值降低，将有更少的线被检测到.
    :return:
    '''
    global THRESHOLD
    global SHAPE
    global STEP_LINE
    global SUSPICIOUS_THRESHOLD

    img = 255 - img[:, :, 0]
    SHAPE = img.shape
    # 10.3是经验值
    THRESHOLD = np.sqrt(np.max(img.shape)) / 10.3 + delta
    print('line_threshold=', THRESHOLD)
    SUSPICIOUS_THRESHOLD = int(THRESHOLD * 1.4)
    STEP_LINE = int(np.max(img.shape) / 100)

    suspicious_points = get_suspicious_points(img)
    points = get_points(img, suspicious_points)

    if debug:
        print('suspicious_points num:', len(suspicious_points))
        print('thin points num:', len(points))

    return output(img, points), len(points)
    # return img
