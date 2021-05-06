import cv2
import os
import numpy as np
from skimage import morphology


def is_solid_area(threshold_img, x, y, window_size=7):
    """对于类似实心圆这样的区域进行骨架提取后也是一条线，因此可能存在误判.
       我们在原图二值化后的结果上检查该点window_size连通区域中白色点的个数,如果数目等于(window_size*window_size - 1)则一定是误判
        Parameters:
            Input:
                threshold_img: 原图二值化反转后的结果
                x,y : 坐标值, x为行, y为列
                window_size: 连通区域大小,数值尽量选用奇数
           Return:
               True or False
    """
    height, width = threshold_img.shape[0: 2]
    if (x-window_size//2) < 0 or (x+window_size//2) > height-1 or (y-window_size//2) < 0 or (y+window_size//2) > width-1:
        return True

    roi = threshold_img[x - window_size//2: x - window_size//2 + window_size, y - window_size//2: y - window_size//2 + window_size]
    pixelNumbers = np.sum(np.greater(roi, 0)) - 1

    if pixelNumbers == window_size*window_size-1:
        return True
    else:
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
                neighbors = point_neighbors[cur_point]
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
    for i in range(len(none_zero_point_lists)):
        x, y = none_zero_point_lists[i][0], none_zero_point_lists[i][1]
        if x not in range(1, height-1) or y not in range(1, width-1):
            continue
        neighbors = calculate_surrouding_pixels(skeleton, x, y)
        point_neighbors[(x, y)] = neighbors

        # 找出八连通区域邻居个数小于等于1的点
        if len(neighbors) <= 1:
            # 如果是类似于实心圆的封闭图形,排除
            if is_solid_area(binary, x, y, solid_window_size):
                continue
            else:
                isolated_points.append((x, y))

    temp_isolated_points = isolated_points[:]

    # 第二次过滤 的点往线条方向反搜索
    for i in range(len(isolated_points)):
        x, y = isolated_points[i][0], isolated_points[i][1]
        if not is_correct_line_head(x, y, point_neighbors, search_num):
            temp_isolated_points.remove((x, y))

    return temp_isolated_points


def unclosed_line_detection(src_img_dir, thin_img_dir, dst_img_dir, binary_threshold=128, solid_window_size=7, search_num=10):
    """
    Parameters:
        Input:
            src_img_dir: 输入图片路径
            thin_img_dir: 输出细化图片路径
            dst_img_dir: 输出图片路径
            area_threshold: 轮廓中最小像素个数
            binary_threshold：二值化阈值
            solid_window_size: 判断是否是实心区域的窗口大小
            search_num: 沿着边缘节点反向往回搜索的次数

        Output:
            骨架图和原图的标注结果
    """

    for file in os.listdir(src_img_dir):
        src_img_name = os.path.join(src_img_dir, file)
        thin_img_name = os.path.join(thin_img_dir, file)
        dst_img_name = os.path.join(dst_img_dir, file)

        img = cv2.imread(src_img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY_INV)  # 二值化处理
        binary[binary == 255] = 1
        skeleton0 = morphology.skeletonize(binary)  # 骨架提取
        skeleton = skeleton0.astype(np.uint8) * 255
        points = get_unclosed_pixel_points(binary, skeleton, solid_window_size, search_num)  # 获取符合要求的点

        for i in range(len(points)):
            cv2.circle(skeleton, (points[i][1], points[i][0]), 10, 255, 2)
            cv2.circle(img, (points[i][1], points[i][0]), 10, (0, 0, 255), 2)
            # cv2.circle(img, (points[i][1], points[i][0]), 1, (0, 0, 255), -1)

        cv2.imwrite(thin_img_name, skeleton)
        cv2.imwrite(dst_img_name, img)

        print("  图片:", src_img_name, "  不闭合点的个数为：", len(points))


if __name__ == '__main__':
    src_img_dir = "/home/cgim/wushukai/code/LeXin/LineDetection/unclosedLineDetection/src"  # 源目录
    thin_img_dir = "/home/cgim/wushukai/code/LeXin/LineDetection/unclosedLineDetection/thin"  # 细化图片目录
    dst_img_dir = "/home/cgim/wushukai/code/LeXin/LineDetection/unclosedLineDetection/dst"  # 目标目录

    binary_threshold = 128  # 二值化阈值
    solid_window_size = 7  # 判断是否是实心区域的窗口大小
    search_num = 20  # 沿着边缘节点反向往回搜索的次数 这个值越高，未闭合的线头检测的越干净
    unclosed_line_detection(src_img_dir, thin_img_dir, dst_img_dir, binary_threshold, solid_window_size, search_num)



