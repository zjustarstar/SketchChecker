import glob
import os
import cv2
import time
import math
import copy
import numpy as np
import smallAreaDetection as sd
import thinLineDetection as td
import unclosedLineDetection as ud


# 设为true时，会生成一些中间结果，方便调试程序
debug = False
# 检测功能的开关
ENABLE_SMALL_AREA = False       # 小区域检测
ENABLE_UNCLOSED_LINE = True     # 未闭合线头检测
ENABLE_THIN_LINE = False        # 过细的线检测


def inverse_white(path):
    # 输入为png的路径
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


input_path = "F:\\PythonProj\\SketchChecker\\testimage\\"
output_folder = "result\\"
# input_path = "/home/cgim/wushukai/code/LeXin/SketchChecker-0506/testimage"
# output_folder = "result"
# input_path = "/home/cgim/wushukai/code/LeXin/LineDetection/thinLineDetection/src"
# output_folder = "result"

# 在当前目录自动生成用于保存的文件夹
if not os.path.exists(os.path.join(input_path, output_folder)):
    os.makedirs(os.path.join(input_path, output_folder))

imgfile1 = glob.glob(os.path.join(input_path, "*.png"))
imgfile2 = glob.glob(os.path.join(input_path, "*.jpg"))
imgfile = imgfile1 + imgfile2
totalfile = len(imgfile)

i = 0
output_path = os.path.join(input_path, output_folder)
tstart = time.time()
for f in imgfile:
    i = i+1
    (filepath, filename) = os.path.split(f)
    (shotname, extension) = os.path.splitext(filename)
    print("当前正在处理 %d/%d :%s" % (i, totalfile, filename))

    # 读取图像
    if f.endswith("png"):
        img = inverse_white(f)
    else:
        img = cv2.imread(f)
    maker_img = copy.deepcopy(img)

    # 细线化检测
    if ENABLE_THIN_LINE:
        print("开始细线化检测")
        maker_img = td.thin_line_detection(f, img, output_path, debug)

    # 小区域检测
    if ENABLE_SMALL_AREA:
        print("开始小区域检测")
        maker_img = sd.small_area_detection(f, img, maker_img, output_path, 20, 50, debug)

    # 未闭合线头检测
    if ENABLE_UNCLOSED_LINE:
        print("开始未闭合线头检测")
        maker_img = ud.unclosed_line_detection(f, img, maker_img, output_path, 128, 7, 20, debug)

    # 保存最终的maker图
    dst_img_name = os.path.join(output_path, shotname + "_final" + extension)
    cv2.imencode(extension, maker_img)[1].tofile(dst_img_name)

tend = time.time()

timespan = tend - tstart
hour = math.floor(timespan / 3600)
m = math.floor((timespan - hour * 3600) / 60)
sec = math.floor(timespan - hour * 3600 - m * 60)
print("finish. 一共%d张图片，耗时:%d时%d分%d秒" % (totalfile, hour, m, sec))
