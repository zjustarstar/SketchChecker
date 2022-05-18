import glob
import os
import time
import math
import main_process as mp


input_path = "F:\\PythonProj\\SketchChecker\\testpdf\\"
output_folder = os.path.join(input_path, "result")

# 在当前目录自动生成用于保存的文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

imgfile = glob.glob(os.path.join(input_path, "*.pdf"))
totalfile = len(imgfile)

i = 0
tstart = time.time()
for f in imgfile:
    i = i + 1

    # convert to png image
    print("start to convert pdf to png image")
    pngImg, isWallPaper = mp.pdf2image(f)
    (filepath, filename) = os.path.split(pngImg)
    (shotname, extension) = os.path.splitext(filename)
    print("当前正在处理 %d/%d :%s" % (i, totalfile, filename))
    mp.main_checker(pngImg, output_folder, isWallPaper)

    # 将临时生成的png图像删除
    if os.path.exists(pngImg):
        os.remove(pngImg)

tend = time.time()
timespan = tend - tstart
hour = math.floor(timespan / 3600)
m = math.floor((timespan - hour * 3600) / 60)
sec = math.floor(timespan - hour * 3600 - m * 60)
print("finish. 一共%d张图片，耗时:%d时%d分%d秒" % (totalfile, hour, m, sec))
