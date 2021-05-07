import glob
import os
import smallAreaDetection as sd
import thinLineDetection as td
import unclosedLineDetection as ud


# input_path = "F:\\PythonProj\\SketchChecker\\testimage\\"
# output_folder = "result\\"
input_path = "/home/cgim/wushukai/code/LeXin/SketchChecker-0506/testimage"
output_folder = "result"
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
for f in imgfile:
    i = i+1
    (filepath, filename) = os.path.split(f)
    (shotname, extension) = os.path.splitext(filename)
    print("当前正在处理 %d/%d :%s" % (i, totalfile, filename))

    # 细线化检测
    print("开始细线化检测")
    maker_img = td.thin_line_detection(f, os.path.join(input_path, output_folder), 50)

    # 小区域检测
    print("开始小区域检测")
    maker_img2 = sd.small_area_detection(f, maker_img, os.path.join(input_path, output_folder), 20, 50)

    print("开始未闭合线头检测")
    ud.unclosed_line_detection(f, maker_img2, os.path.join(input_path, output_folder), 128, 7, 20)

print("finish")
