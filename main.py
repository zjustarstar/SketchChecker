import glob
import os
import cv2
import smallAreaDetection as sd
import thinLineDetection as td
import unclosedLineDetection as ud

# 设为true时，会生成一些中间结果，方便调试程序
debug = False

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
for f in imgfile:
    i = i+1
    (filepath, filename) = os.path.split(f)
    (shotname, extension) = os.path.splitext(filename)
    print("当前正在处理 %d/%d :%s" % (i, totalfile, filename))

    # 细线化检测
    print("开始细线化检测")
    maker_img = td.thin_line_detection(f, output_path, 50, debug)

    # 小区域检测
    print("开始小区域检测")
    maker_img = sd.small_area_detection(f, maker_img, output_path, 20, 50, debug)
    #
    print("开始未闭合线头检测")
    maker_img = ud.unclosed_line_detection(f, maker_img, output_path, 128, 7, 20, debug)

    # 保存最终的maker图
    dst_img_name = os.path.join(output_path, shotname + "_final" + extension)
    cv2.imencode(extension, maker_img)[1].tofile(dst_img_name)

print("finish")
