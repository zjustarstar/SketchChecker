import cv2
import os
import numpy as np


THRESHOLD = 9
SUSPICIOUS_THRESHOLD = 10

THRESHOLD_ANGLE = 1
STEP_LINE = 120
STEP_ANGlE = 1
STEP_PIXEL = 1
BOX_SIZE = 10   # 截取的小块的大小
SHAPE = (2048, 2048)

CIRCLE_MASK = cv2.imread('circle_mask.png')
CIRCLE_MASK = cv2.resize(CIRCLE_MASK, (BOX_SIZE*2, BOX_SIZE*2))
CIRCLE_MASK = CIRCLE_MASK[:, :, 0]


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
            delta = THRESHOLD_ANGLE * np.abs(np.sin(2 * max_black_angle / 180 * 3.1415926))

            if line_width-delta < THRESHOLD:
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
def analyze_width(img):
    img = mask_circle(img)
    line_width = analyze_width_round_img(img)
    return line_width


def output(img, points):
    clr = (255, 0, 255)
    w = 45
    for p in points:
        y, x = p[0], p[1]
        cv2.rectangle(img, (x-w, y+w), (x+w, y-w), clr, 8)
    return img


#
# get_suspicious_points
# 该函数通过一张图片获取图中可能为细线的可疑点，方法是分两次，横向和纵向遍历像素，
# 遇到黑像素视为起点，离开黑像素视为重点，若起点与终点距离大于阈值，设两者中间点为可疑点
#
def get_suspicious_points(img):
    img_height, img_width = img.shape
    BLACK_PIXEL = 0   # 黑色像素值
    MARGIN = 50        # 边界
    suspicious_points = []
    confirm_points = []
    for x in (range(MARGIN, img_height - MARGIN, STEP_LINE)):
        start = 0
        end = 0
        for y in range(MARGIN, img_width - MARGIN, STEP_PIXEL):
            if img[x, y] > BLACK_PIXEL and start == 0:
                start = y
            elif img[x, y] == 0 and start != 0:
                end = y
                center = int((start + end) / 2)
                if THRESHOLD < (end - start) < SUSPICIOUS_THRESHOLD:
                    suspicious_points.append([x, center, end-start])
                if (end - start) <= THRESHOLD:
                    confirm_points.append([x, center, end-start])
                start = 0
                end = 0

    for y in (range(MARGIN, img_width - MARGIN, STEP_LINE)):
        start = 0
        end = 0
        for x in range(MARGIN, img_height - MARGIN, STEP_PIXEL):
            if img[x, y] > BLACK_PIXEL and start == 0:
                start = x
            elif img[x, y] == 0 and start != 0:
                end = x
                center = int((start + end) / 2)
                if THRESHOLD < (end - start) < SUSPICIOUS_THRESHOLD:
                    suspicious_points.append([center, y, end-start])
                if (end - start) <= THRESHOLD:
                    confirm_points.append([center, y, end - start])
                start = 0
                end = 0

    return confirm_points, suspicious_points


#
# get_points
# 在可疑点附近截取一定大小的子图，输入到analyze_width函数判断真实宽度
def get_points(img, suspicious_points):
    points = []
    for p in suspicious_points:
        x, y = p[0], p[1]
        line_width = analyze_width(img[x - BOX_SIZE:x + BOX_SIZE, y - BOX_SIZE:y + BOX_SIZE])
        if 0 < line_width < THRESHOLD:
            points.append([x, y, round(line_width, 2)])

    return points


def isPointInRect(x, y, rect):
    '''
    判断点是否在某个rect内
    :param x,y:
    :param rect: x0, x1, y0, y1
    :return:
    '''
    if rect[0] < x < rect[1] and rect[2] < y < rect[3]:
        return True
    else:
        return False


def remSinglePt(pts):
    '''
    去除孤立点:周边一定范围内一个点都没有的点
    :param pts: 输入的点
    :return: 去除孤立点后的点，以及去除了多少个点
    '''
    Scale = 2
    newpt = []
    for i in range(len(pts)):
        y, x = pts[i][0], pts[i][1]
        rect = [x-Scale*STEP_LINE, x+Scale*STEP_LINE, y-Scale*STEP_LINE, y+Scale*STEP_LINE]
        num = 0
        for p in pts:
            if isPointInRect(p[1], p[0], rect):
                num = num + 1

        # 超过一个相邻点,保留
        if num > 1:
          newpt.append(pts[i])

    return newpt, len(pts)-len(newpt)


# 测试用，查看某些具体的线为何检测不到
# p[0]是y
def test_filter(points):
    new_pts = []
    for p in points:
        if 0 < p[0] < 8534 * 0.2 and 0 < p[1] < 8534 * 0.4:
            new_pts.append(p)
    return new_pts

#
# 算法大步骤分为两步，先求可疑点，
# 再从可疑点中筛选确切点
def thin_line_detection(file, img, out_path, debug=False, delta=0, isWallPaper=False):
    '''
    :param file: 输入的原文件名
    :param img: 输入图像
    :param out_path: 输出路径
    :param debug: 为true时,会输出一些信息
    :param delta: 增大或者缩小细线粗细的阈值。为正时，线的阈值增加，将有更多的线被检测到。
                                              为负时，线的阈值降低，将有更少的线被检测到.
    :param isWallPaper: 是否是壁纸类图，壁纸一般是比较长比较窄的长方形图.普通图阈值2px,
                         壁纸类是1个像素;
    :return:
    '''
    global THRESHOLD
    global SHAPE
    global STEP_LINE
    global SUSPICIOUS_THRESHOLD

    ori_img = img
    img = 255 - img[:, :, 0]
    SHAPE = img.shape
    # 1是经验值
    THRESHOLD = 8 + delta
    STEP_LINE = int(np.max(img.shape) / 100)
    if isWallPaper:
        THRESHOLD = 3 + delta
        STEP_LINE = 10
    print('line_threshold=', THRESHOLD)
    SUSPICIOUS_THRESHOLD = int(THRESHOLD * 1.4)

    confirm_points, suspicious_points = get_suspicious_points(img)
    points = get_points(img, suspicious_points)
    print('suspicious_points num:', len(suspicious_points))
    print('point from suspicious:', len(points))
    # 用于测试
    # points = test_filter(points)
    points, del_num = remSinglePt(points+confirm_points)
    print('remove single point num:', del_num)

    if debug:
        print('suspicious_points num:', len(suspicious_points))
        print('thin points num:', len(points))
        for i in range(len(points)):
            print(points[i], end='')
            # 每行显示10个点
            if i % 10 == 0:
                print('\n')
        print("\n")

    # 显示最终结果
    return output(ori_img, points), len(points)
