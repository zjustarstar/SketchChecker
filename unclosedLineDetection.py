import cv2
import os
import copy
import time
import torch
import numpy as np
from tqdm import tqdm
from skimage import morphology
import train.brokenline as cnnChecker

# binary_threshold = 128  # 二值化阈值
# solid_window_size = 7  # 判断是否是实心区域的窗口大小
# search_num = 20  # 沿着边缘节点反向往回搜索的次数 这个值越高，未闭合的线头检测的越干净


def is_solid_area(threshold_img, skeleton, x, y, window_size=7):
    """对于类似实心圆这样的区域进行骨架提取后也是一条线，因此可能存在误判.
       我们在原图二值化后的结果上检查该点window_size连通区域中白色点的个数,如果数目等于(window_size*window_size - 1)则一定是误判
        Parameters:
            Input:
                threshold_img: 原图二值化反转后的结果
                skeleton: 骨架图
                x,y : 坐标值, x为行, y为列
                window_size: 连通区域大小,数值尽量选用奇数
           Return:
               True or False
    """
    # 第一种情况
    height, width = threshold_img.shape[0: 2]
    if (x-window_size//2) < 0 or (x+window_size//2) > height-1 or (y-window_size//2) < 0 or (y+window_size//2) > width-1:
        return True

    roi = threshold_img[x - window_size//2: x - window_size//2 + window_size, y - window_size//2: y - window_size//2 + window_size]
    pixelNumbers = np.sum(np.greater(roi, 0)) - 1

    if pixelNumbers == window_size*window_size-1:
        return True

    # 第二种情况: 骨架化以后，变成了很短长度的一个小区域, 容易误判
    # window = 4
    # pt_num = 0  # 小范围区域，统计4*4边界上的骨架点
    # row_start = max(0, x-window)
    # row_end = min(x+window+1, height)
    # col_start = max(0, y-window)
    # col_end = min(y+window+1, width)
    # for row in range(row_start, row_end):
    #     for col in range(col_start, col_end):
    #         # 统计边界上的骨架点数
    #         if row==row_start or row==row_end-1 or col==col_start or col==col_end-1:
    #             if skeleton[col][row] == 255:
    #                 pt_num += 1
    # if pt_num == 0:
    #     return True

    return False


def calculate_surrouding_pixels(skeleton, x, y):
    """计算8连通区域像素点的个数
    Parameters:
        Input:
            skeleton: 细化后的图片
            x,y : 坐标值, x为行, y为列
       Return:
           8连通区域中非背景元素坐标值
    """
    roi = skeleton[x-1: x+2, y-1: y+2]
    none_zero_indexes = list(zip(*np.nonzero(roi)))  # 获取roi区域非0元素的坐标
    none_zero_points = []
    for i in range(len(none_zero_indexes)):
        none_zero_points.append((x+none_zero_indexes[i][0]-1, y+none_zero_indexes[i][1]-1))
    none_zero_points.remove((x, y))
    return none_zero_points


def is_correct_line_head(x, y, point_neighbors, num=10):
    """以（x, y)为起点，沿着线的方向查找num次，如果在此过程中发现某个点有除上个点以外的2个邻居，则（x,y)不符合要求
        Parameters:
            Input:
                x,y : 坐标值, x为行, y为列
                point_neighbors：存放所有点对应邻居的字典, key为(x,y),value是其对应的邻居坐标
                num: 搜索次数
           Return:
               True or False
    """
    # (x,y)没有邻居, 完全孤立, 符合要求
    if len(point_neighbors[(x, y)]) == 0:
        return True
    else:
        cur_point = (x, y)  # 当前的点
        previous_points = []  # 记录所有走过的点
        for i in range(num):
            previous_points.append(cur_point)
            if i == 0:
                cur_point = point_neighbors[cur_point][0]
            else:
                # print(cur_point, i)
                neighbors = point_neighbors[cur_point]
                if neighbors is None:  # 找到边界点，直接返回
                    return True
                neighbors.remove(previous_points[-2])
                # 发现某个点有除上个点以外的2个邻居，不符合要求
                if len(neighbors) >= 2:
                    return False
                elif len(neighbors) == 0:
                    return True
                cur_point = neighbors[0]

    return True


def get_unclosed_pixel_points(binary, skeleton, solid_window_size, search_num):
    """获取未闭合的像素点坐标(边界点不用检测)
    Parameters:
        Input:
            binary: 二值化后的图片
            skeleton: 细化后的图片
            solid_window_size: 判断是否是实心区域的窗口大小
            search_num: 沿着边缘节点反向往回搜索的次数
       Return:
            points: 符合要求的点构成的数组
    """

    height, width = skeleton.shape[0: 2]

    point_neighbors = {}  # key为每个点的坐标：（x, y)  value为一个list,list里面存放所有邻居的坐标
    isolated_points = []  # 存储符合要求的点
    none_zero_point_lists = list(zip(*np.nonzero(skeleton)))  # 获取骨架图中非0元素的点

    # 第一次过滤
    for i in tqdm(range(len(none_zero_point_lists))):
        x, y = none_zero_point_lists[i][0], none_zero_point_lists[i][1]
        if x not in range(1, height-1) or y not in range(1, width-1):
            point_neighbors[(x, y)] = None
            continue
        # 计算x,y 八连通区域的非0像素
        neighbors = calculate_surrouding_pixels(skeleton, x, y)
        point_neighbors[(x, y)] = neighbors

        # 找出八连通区域邻居个数小于等于1的点
        if len(neighbors) == 1:
            # 如果是类似于实心圆的封闭图形,排除
            if is_solid_area(binary, skeleton, x, y, solid_window_size):
                continue
            else:
                isolated_points.append((x, y))
        # # 孤立点,去掉，否则可能影响后续计算
        elif len(neighbors) == 0:
            binary[x][y] = 0

    temp_isolated_points = isolated_points[:]

    # 第二次过滤 的点往线条方向反搜索
    for i in range(len(isolated_points)):
        x, y = isolated_points[i][0], isolated_points[i][1]
        if not is_correct_line_head(x, y, point_neighbors, search_num):
            temp_isolated_points.remove((x, y))

    return temp_isolated_points


# 对于小区域中有单个黑点的四周全白的，直接去掉，变为白色
def rm_single_pt(binary_img):
    cols, rows = binary_img.shape[0], binary_img.shape[1]
    binary_img[binary_img==255] = 1
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            # 一个黑点四周全是白的，则变为白的
            if binary_img[c][r] == 0:
                sum = binary_img[c-1][r] + binary_img[c+1][r] + binary_img[c][r-1] + binary_img[c][r+1]
                if sum == 4:
                    binary_img[c][r] = 1
    binary_img[binary_img==1] = 255
    return binary_img


# 根据该点周边的信息来判断是否是未闭合的线
def check_candidate_regions(points, binary):
    if len(points) == 0:
        return

    new_points = copy.deepcopy(points)
    binary[binary == 1] = 255

    # 一定的半径范围内
    radius = 8

    h, w = binary.shape[0], binary.shape[1]
    for pt in points:
        left, right = max(pt[1]-radius, 0), min(pt[1]+radius, w-1)
        up, dw = max(pt[0]-radius, radius), min(pt[0]+radius, h-1)
        roi_img = binary[up:dw, left:right]
        # 合法性判断
        if up >= dw or left >= right:
            continue

        #计算连通区域
        contours, _ = cv2.findContours(roi_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        regions1 = len(contours)
        # if pt[1] == 804:
        #     cv2.imwrite("roi1.jpg", roi_img)
            # print("regions1: c1={},c2={}".format(cv2.contourArea(contours[0]), cv2.contourArea(contours[1])))
        # 膨胀后再次计算连通区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素的形状和大小
        roi_img1 = cv2.dilate(roi_img, kernel)  # 膨胀
        roi_img1 = rm_single_pt(roi_img1)
        contours, _ = cv2.findContours(roi_img1, cv2.RETR_LIST, cv2. CHAIN_APPROX_NONE)
        regions2 = len(contours)
        # if pt[1] == 804:
        #     cv2.imwrite("roi2.jpg", roi_img1)
        # 如果连通区域不会减少，则认为不存在未闭合区域
        # 如果只有一个连通区域，也认为不存在未闭合区域
        if regions1 == regions2 or regions1 == 1:
            # print("pt={}, regions1={}, regions2={}".format(pt, regions1, regions2))
            new_points.remove(pt)

    return new_points


# 如果有两个点非常近，则从队列中删除一个
def rm_dup_pts(points):
    # 间隔阈值
    dist = 10
    new_points = copy.deepcopy(points)
    for i in range(len(points)-1):
        pt1 = points[i]
        for j in range(i+1, len(points)):
            pt2 = points[j]
            d = [abs(pt1[0]-pt2[0]), abs(pt1[1]-pt2[1])]
            if max(d) <= dist:
                if pt2 in new_points:
                    new_points.remove(pt2)
    return new_points


def unclosed_line_detection(file, img, outpath, is_color_sketch=False, debug=False):
    """
    Parameters:
        Input:
            src_img_dir: 输入图片路径及文件名
            img: 输入的图片
            is_color_sketch：是否是彩色线框图,即使只有1条线用了彩色，也算彩色线框图
            debug: 调试模式,在该模式下，可以生成一些中间结果图
        Output:
            骨架图和原图的标注结果
    """
    model_name = "models\\brokenline.pth"
    (filepath, filename) = os.path.split(file)
    (onlyfilename, extension) = os.path.splitext(filename)

    # 加载判别网络
    if not os.path.exists(model_name):
        print("fail to load brokenline model")
        return None, 0

    # 加载网络..
    model = cnnChecker.myLeNet()
    model.load_state_dict(torch.load(model_name))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(onlyfilename + ".jpg", gray)

    # 彩色线框图和黑白线框图采用不同阈值
    threshold = 128
    if is_color_sketch:
        threshold = 220
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)  # 二值化处理

    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)  # 骨架提取
    skeleton = skeleton0.astype(np.uint8) * 255
    # 用于寻找未闭合点：判断是否是实心区域的窗口大小
    solid_window_size = 7
    # 用于寻找未闭合点：沿着边缘节点反向往回搜索的次数
    search_num = 10
    points = get_unclosed_pixel_points(binary, skeleton, solid_window_size, search_num)  # 获取符合要求的点

    # 去掉非常近的重复点.比如一条直线中间断开,断开的两个点很近,就会都被列入
    points = rm_dup_pts(points)

    # 去掉一些不可能的点
    print(points)
    points = check_candidate_regions(points, binary)
    print(points)

    len_points = 0
    real_pt = 0
    if points is not None:
        len_points = len(points)
        for i in range(len(points)):
            cv2.circle(skeleton, (points[i][1], points[i][0]), 13, 255, 1)

            pt = points[i]
            radius = 14
            h, w = gray.shape[0], gray.shape[1]
            left, right = max(pt[1] - radius, 0), min(pt[1] + radius, w - 1)
            up, dw = max(pt[0] - radius, radius), min(pt[0] + radius, h - 1)
            if abs(up - dw) < radius - 1 or abs(left - right) < radius - 1:
                continue

            # 不同类型的点，选用不同的颜色
            roi_img = gray[up:dw, left:right]
            patch_type, confidence = cnnChecker.patchCheck(model, roi_img)
            # print("patch type:%d, confidence=%.2f" % (patch_type, confidence))
            # 边界区域
            if patch_type == -1:
                continue

            clr = (0, 0, 255)
            # # thinline
            # if patch_type == 2:
            #     clr = (0, 255, 0)
            # # multi-lines
            # elif patch_type == 0:
            #     clr = (255, 0, 0)
            # # point
            # elif patch_type == 1:
            #     clr = (255, 255, 0)
            if patch_type < 3:
                continue
            real_pt += 1
            cv2.circle(img, (points[i][1], points[i][0]), 15, clr, 2)

            # save_temp_img(gray, points[i], 14, i)


    if debug:
        binary[binary == 1] = 255
        binary_name = os.path.join(outpath, onlyfilename + "_uc_binary" + extension)
        cv2.imencode(extension, binary)[1].tofile(binary_name)
        mid_img_name = os.path.join(outpath, onlyfilename + "_uc_skeleton" + extension)
        cv2.imencode(extension, skeleton)[1].tofile(mid_img_name)

    print("  图片:", filename, " 未筛选前个数:", len_points, "  筛选后不闭合点的个数为：", real_pt)

    return img, real_pt


# 保存临时图像
def save_temp_img(img, pt, radius, i):
    h, w = img.shape[0], img.shape[1]
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

