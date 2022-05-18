import os
import cv2
import math
import copy
import numpy as np
from PIL import Image
import imgConverter as img_converter
import thinLineDetection_v2 as td
import unclosedLineDetection as ud
import fitz


# 实际生产线上使用的图的分辨率
PRODUCT_IMG_SIZE = 2048

# 设为true时，会生成一些中间结果，方便调试程序
debug = False
# 是否是彩色线框图
IS_COLOR_SKETCH = False

# 检测功能的开关
ENABLE_UNCLOSED_LINE = False  # 未闭合线头检测
ENABLE_THIN_LINE = True # 过细的线检测


def pdf2image(pdfPath):
    square_size = 8534
    wallpaper_size = 3125
    # 是正方形的还是wallpaper
    isSquare = True

    (filepath, filename) = os.path.split(pdfPath)
    (shotname, extension) = os.path.splitext(filename)

    # 打开PDF文件
    pdf = fitz.open(pdfPath)
    # 逐页读取PDF
    for pg in range(0, pdf.pageCount):
        page = pdf[pg]
        r = page.rect
        print("pdf content size = {}".format(r))
        zoom_ratio = 1.0
        if int(r.width) == int(r.height):
            zoom_ratio = square_size / r.width
        else:
            zoom_ratio = wallpaper_size / r.width
            isSquare = False

        # 设置缩放和旋转系数
        trans = fitz.Matrix(zoom_ratio, zoom_ratio)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        # 开始写图像,保存在当前文件夹中
        save_file = filepath+"\\"+shotname+".png"
        pm.save(save_file)

    pdf.close()
    return save_file, isSquare


def main_checker(imgPath, outpath, isWallPaper):
    '''
    根据输入的图像检测断点和细线
    :param imgPath: 输入的图像的路径
    :param outpath: 保存检测结果的路径
    :param isWallPaper: 输入的图像是正方形图还是墙纸图，后者长宽不一样
    :return:
    '''
    (filepath, filename) = os.path.split(imgPath)
    (shotname, extension) = os.path.splitext(filename)

    # 创建细线检测和断点检测结果图的路径
    blpath = os.path.join(outpath, "broken_line_result")
    tlpath = os.path.join(outpath, "thin_line_result")
    if not os.path.exists(blpath):
        os.makedirs(blpath)
    if not os.path.exists(tlpath):
        os.makedirs(tlpath)

    img = Image.open(imgPath)
    # test = img.crop((550, 1300, 1200, 1900))
    # test.save("tt.png")
    if len(img.getbands()) == 4:
        newimg = img_converter.alpha_composite_with_color(img).convert('RGB')
    elif len(img.getbands()) == 3:
        newimg = img
    else:
        print("image format error")
    img = cv2.cvtColor(np.asarray(newimg), cv2.COLOR_RGB2BGR)

    # 结果文件中间名
    middle_name = ''
    maker_img = copy.deepcopy(img)

    dst_img_thin = dst_img_uc = " "
    # 细线化检测
    if ENABLE_THIN_LINE:
        print("开始细线化检测")
        # delta控制线的粗细阈值,增减单元建议0.05。为正时，线的阈值增加，将有更多的线被检测到。
        # 为负时，线的阈值降低，将有更少的线被检测到.
        # isWallPaper表示输入的是墙纸图.
        maker_img, pt_num = td.thin_line_detection(imgPath, img, tlpath, False, delta=0, isWallPaper=isWallPaper)
        middle_name = "_" + str(pt_num)

        # 保存最终的maker图, 并在结果图中标示uc的个数
        dst_img_thin = os.path.join(tlpath, shotname + middle_name + extension)
        print(dst_img_thin)
        cv2.imencode(extension, maker_img)[1].tofile(dst_img_thin)

    # 未闭合线头检测
    if ENABLE_UNCLOSED_LINE:
        print("开始未闭合线头检测")
        # 实际使用的图片分辨率是2048
        if img.shape[0] > PRODUCT_IMG_SIZE or img.shape[1] > PRODUCT_IMG_SIZE:
            ratio = min(img.shape[0], img.shape[1]) / PRODUCT_IMG_SIZE
            newh, neww = math.floor(img.shape[0] / ratio), math.floor(img.shape[1] / ratio)
            input_img = cv2.resize(img, (neww, newh))
            maker_img = cv2.resize(maker_img, (neww, newh))
        maker_img, uc_num = ud.unclosed_line_detection(imgPath, input_img, maker_img, blpath, IS_COLOR_SKETCH, debug)
        middle_name = middle_name + "_" + str(uc_num)

        # 保存最终的maker图, 并在结果图中标示uc的个数
        dst_img_uc = os.path.join(blpath, shotname + middle_name + extension)
        print(dst_img_uc)
        cv2.imencode(extension, maker_img)[1].tofile(dst_img_uc)

    return dst_img_thin, dst_img_uc


