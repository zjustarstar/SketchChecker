import glob
import os
import smallAreaDetection as sd
import thinLineDetection as td
import unclosedLineDetection as ud


input_path = "F:\\PythonProj\\SketchChecker\\testimage\\"
output_folder = "result\\"

# 在当前目录自动生成用于保存的文件夹
if not os.path.exists(input_path+output_folder):
    os.makedirs(input_path+output_folder)

imgfile1 = glob.glob(input_path + "*.png")
imgfile2 = glob.glob(input_path + "*.jpg")
imgfile = imgfile1 + imgfile2
totalfile = len(imgfile)
i = 0
for f in imgfile:
    i = i+1
    (filepath, filename) = os.path.split(f)
    (shotname, extension) = os.path.splitext(filename)
    print("当前正在处理 %d/%d :%s" % (i, totalfile, filename))

    # 小区域检测
    print("开始小区域检测")

    pixelThreshold = 50  # 轮廓内部的像素点个数阈值
    binaryThreshold = 10  # 二值化阈值
    sd.small_area_detection(f, input_path+output_folder, pixelThreshold, binaryThreshold)

print("finish")
