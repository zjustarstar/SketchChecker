import glob
import os
import cv2
import time
import math
import copy
import numpy as np
from PIL import Image
import imgConverter as img_converter
import smallAreaDetection as sd
import thinLineDetection_v2 as td
import unclosedLineDetection as ud

# 设为true时，会生成一些中间结果，方便调试程序
debug = False
# 是否是彩色线框图
IS_COLOR_SKETCH = True
# 检测功能的开关
ENABLE_SMALL_AREA = False  # 小区域检测
ENABLE_UNCLOSED_LINE = False  # 未闭合线头检测
ENABLE_THIN_LINE = True  # 过细的线检测

input_path = "F:\\PythonProj\\SketchChecker\\testimage\\"
output_folder = "result\\"

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
    i = i + 1
    (filepath, filename) = os.path.split(f)
    (shotname, extension) = os.path.splitext(filename)
    print("当前正在处理 %d/%d :%s" % (i, totalfile, filename))

    img = Image.open(f)
    newimg = img_converter.alpha_composite_with_color(img).convert('RGB')
    img = cv2.cvtColor(np.asarray(newimg), cv2.COLOR_RGB2BGR)
    # newimg.save(shotname+".jpg")
    maker_img = copy.deepcopy(img)
    uc_num = 0
    # 细线化检测
    if ENABLE_THIN_LINE:
        print("开始细线化检测")
        # delta控制线的粗细阈值,增减单元建议0.1。为正时，线的阈值增加，将有更多的线被检测到。
        # 为负时，线的阈值降低，将有更少的线被检测到.
        maker_img = td.thin_line_detection(f, img, output_path, debug, delta=0)

    # 小区域检测
    if ENABLE_SMALL_AREA:
        print("开始小区域检测")
        maker_img = sd.small_area_detection(f, img, maker_img, output_path, 20, 50, debug)

    # 未闭合线头检测
    if ENABLE_UNCLOSED_LINE:
        print("开始未闭合线头检测")
        maker_img, uc_num = ud.unclosed_line_detection(f, img, maker_img, output_path, IS_COLOR_SKETCH, debug)

    # 保存最终的maker图, 并在结果图中标示uc的个数
    dst_img_name = os.path.join(output_path, shotname + "_uc_" + str(uc_num) + extension)
    cv2.imencode(extension, maker_img)[1].tofile(dst_img_name)

tend = time.time()

timespan = tend - tstart
hour = math.floor(timespan / 3600)
m = math.floor((timespan - hour * 3600) / 60)
sec = math.floor(timespan - hour * 3600 - m * 60)
print("finish. 一共%d张图片，耗时:%d时%d分%d秒" % (totalfile, hour, m, sec))
