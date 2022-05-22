import cv2
import torch
import time
import os
import copy
import numpy as np
import train.thinline as cnnChecker


THRESHOLD = 9
SUSPICIOUS_THRESHOLD = 10

THRESHOLD_ANGLE = 1
STEP_LINE = 120
STEP_ANGlE = 1
STEP_PIXEL = 1
BOX_SIZE = 9   # 截取的小块的大小
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
        # 记录各个角度中黑色线段最窄的时候(对应black_wid最大)
        if black_wid > max_black_wid:
            max_black_wid = black_wid
            max_black_angle = angle
            line_width = img.shape[0] - max_black_wid
            # 1.08是经验值;
            delta = THRESHOLD_ANGLE * np.abs(np.sin(2 * max_black_angle / 180 * 3.1415926)) * 1.08

            if line_width-delta < THRESHOLD:
                break

    return line_width - THRESHOLD_ANGLE * np.abs(np.sin(2 * max_black_angle / 180 * 3.1415926)) * 1.08


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


# analyze_width
# 为了旋转后获取真实的宽度，要过滤旋转后会影响宽度判断的像素
# 于是把图像用一个圆形mask处理后，输入analyze_width_round_img获取宽度
def analyze_width(img):
    img = mask_circle(img)
    line_width = analyze_width_round_img(img)
    return line_width


def output_wallpaper(model, img, gray, points):
    width = int(np.max(img.shape) / 400)
    thick = int(np.max(img.shape) / 1000)

    h, w = img.shape[0], img.shape[1]
    final_pts = 0
    radius = 24
    for i in range(len(points)):
        pt = points[i]

        # # 越界检测
        # if pt[1] - width < 0 or pt[0]-width < 0 or \
        #         pt[1]+width > w-1 or pt[0]+width > h-1:
        #     continue

        # 越界检测
        if pt[1] - radius < 0 or pt[0] - radius < 0 or \
                pt[1] + radius > w - 1 or pt[0] + radius > h - 1:
            continue

        left, right = max(pt[1] - radius, 0), min(pt[1] + radius, w - 1)
        up, dw = max(pt[0] - radius, 0), min(pt[0] + radius, h - 1)

        # 不同类型的点，选用不同的颜色
        roi_img = gray[up:dw, left:right]
        patch_type, confidence = cnnChecker.patchCheck(model, roi_img)

        # 超出区域范围
        if patch_type == -1:
            continue

        y, x = pt[0], pt[1]
        clr = (0, 0, 255)
        # # broken
        # if patch_type == 0 and confidence > 0.9:
        #     clr = (255, 0, 0)
        # # # thick
        # elif patch_type == 1:
        #     clr = (0, 255, 0)
        if patch_type == 1:
            continue
        # 这里还有误判，所以将置信度高的剔除
        elif patch_type == 0 and confidence > 0.9:
            continue
        # thin1
        elif patch_type == 2:
            clr = (0, 0, 255)
        elif patch_type == 3:
            clr = (255, 0, 255)
        cv2.rectangle(img, (x - width, y + width), (x + width, y - width), clr, thick)

        # y, x = pt[0], pt[1]
        # clr = (0, 0, 255)
        # cv2.rectangle(img, (x - width, y + width), (x + width, y - width), clr, thick)

        final_pts += 1

    return img, final_pts


def output(model, img, gray, points):
    # width = int(np.max(img.shape) / 250)
    # thick = int(np.max(img.shape) / 1000)
    width = 40
    thick = 8

    radius = 24
    h, w = gray.shape[0], gray.shape[1]
    final_pts = 0
    for i in range(len(points)):
        pt = points[i]

        # 越界检测
        if pt[1] - radius < 0 or pt[0]-radius < 0 or \
                pt[1]+radius > w-1 or pt[0]+radius > h-1:
            continue

        left, right = max(pt[1] - radius, 0), min(pt[1] + radius, w - 1)
        up, dw = max(pt[0] - radius, 0), min(pt[0] + radius, h - 1)
        # if abs(up - dw) < radius - 1 or abs(left - right) < radius - 1:
        #     continue

        # 不同类型的点，选用不同的颜色
        roi_img = gray[up:dw, left:right]
        patch_type, confidence = cnnChecker.patchCheck(model, roi_img)

        # 超出区域范围
        if patch_type == -1:
            continue

        y, x = pt[0], pt[1]
        clr = (125, 125, 125)
        # broken
        # if patch_type == 0:
        #     clr = (255, 0, 0)
        # # thick
        # elif patch_type == 1:
        #     clr = (0, 255, 0)
        if patch_type <= 1:
            continue
        # thin1
        elif patch_type == 2:
            clr = (0, 0, 255)
        elif patch_type == 3:
            clr = (255, 0, 255)
        cv2.rectangle(img, (x-width, y+width), (x+width, y-width), clr, thick)

        final_pts += 1

    return img, final_pts


#
# get_suspicious_points
# 该函数通过一张图片获取图中可能为细线的可疑点，方法是分两次，横向和纵向遍历像素，
# 遇到黑像素视为起点，离开黑像素视为重点，若起点与终点距离大于阈值，设两者中间点为可疑点
# isWallPaper是否是墙纸图，该图不是正方形的.
def get_suspicious_points(img, isWallPaper):
    img_height, img_width = img.shape
    BLACK_PIXEL = 0   # 黑色像素值
    MARGIN = 50        # 边界
    if isWallPaper:
        MARGIN = 10
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
def thin_line_detection(file, img, out_path, debug=False, delta=0, zoomratio=4, isWallPaper=False):
    '''
    :param file: 输入的原文件名
    :param img: 输入图像
    :param out_path: 输出路径
    :param debug: 为true时,会输出一些信息
    :param delta: 增大或者缩小细线粗细的阈值。为正时，线的阈值增加，将有更多的线被检测到。
                                              为负时，线的阈值降低，将有更少的线被检测到.
    :param zoomratio: pdf->png的缩放比例
    :param isWallPaper: 是否是壁纸类图，壁纸一般是比较长比较窄的长方形图.普通图阈值2px,
                         壁纸类是1个像素;
    :return:
    '''
    global THRESHOLD
    global SHAPE
    global STEP_LINE
    global SUSPICIOUS_THRESHOLD

    # 加载网络..
    if not os.path.exists("models\\thinline.pth"):
        print("fail to load model")
        return None, 0

    model = cnnChecker.myLeNet()
    model.load_state_dict(torch.load("models\\thinline.pth"))

    ori_img = copy.deepcopy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # save_specified_region(img)

    img = 255 - img[:, :, 0]
    SHAPE = img.shape
    # 正方形的原pdf尺寸多为2048，发现此时参数为8最佳
    THRESHOLD = 2*int(zoomratio) + delta
    STEP_LINE = int(np.max(img.shape) / 100)
    if isWallPaper:
        THRESHOLD = int(zoomratio) + delta
        STEP_LINE = 30
    print('line_threshold=', THRESHOLD)
    # 乘以根号2(1.414)，表示最倾斜时的情况
    SUSPICIOUS_THRESHOLD = int(THRESHOLD * 1.4)

    confirm_points, suspicious_points = get_suspicious_points(img, isWallPaper)
    points = get_points(img, suspicious_points)
    print('suspicious_points num:', len(suspicious_points))
    print('point from suspicious:', len(points))
    # 用于测试
    # points = test_filter(points)
    # 在suspicious point中去除单独点
    points, del_num = remSinglePt(points)

    points = points + confirm_points
    # 对于wallpaper，孤立点去除相对比较鲁棒
    # 而且更加密集，可以在confirm点中去除
    if isWallPaper:
        points, del_num = remSinglePt(points)
        print('remove single point num:', del_num)

    # 保存临时结果,用于训练
    # save_temp_img(gray, points, 24)

    if debug:
        print('suspicious_points num:', len(suspicious_points))
        print('thin points num:', len(points))
        for i in range(len(points)):
            print(points[i], end='')
            # 每行显示10个点
            if i % 10 == 0:
                print('\n')
        print("\n")

    if isWallPaper:
        return output_wallpaper(model, ori_img, gray, points)

    # 显示最终结果
    return output(model, ori_img, gray, points)


# 保存指定区域的图像，用于测试
def save_specified_region(img):
    left,top = 1100, 4400
    right, bottom = 1400, 4900
    roi_img = img[top:bottom, left:right]
    cv2.imwrite("temp.png", roi_img)


# 保存临时图像
def save_temp_img(img, points, radius):
    w, h = img.shape[0], img.shape[1]
    for i in range(len(points)):
        pt = points[i]
        left, right = max(pt[1] - radius, 0), min(pt[1] + radius, w - 1)
        up, dw = max(pt[0] - radius, radius), min(pt[0] + radius, h - 1)

        if abs(up - dw) < radius - 1 or abs(left - right) < radius - 1:
            return

        roi_img = img[up:dw, left:right]

        timestamp = time.strftime("_%Y%m%d_%H%M%S_", time.localtime())
        filename = "temp\\" + timestamp + str(i) + ".jpg"
        try:
            cv2.imwrite(filename, roi_img)
        except Exception as e:
            print(e)
