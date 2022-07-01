import os
import cv2
import math
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


def pdf2image(pdfPath):
    square_size = 8534
    wallpaper_size = 3125
    # 是正方形的还是wallpaper
    isWallPaper = False

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
            isWallPaper = True

        # 设置缩放和旋转系数
        trans = fitz.Matrix(zoom_ratio, zoom_ratio)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        # 开始写图像,保存在当前文件夹中
        save_file = filepath+"\\"+shotname+".png"
        pm.save(save_file)

    pdf.close()
    return save_file, isWallPaper, zoom_ratio


# 转为2k,用于未闭合区域检测
def pdf2image_2k(pdfPath):
    square_size = PRODUCT_IMG_SIZE

    (filepath, filename) = os.path.split(pdfPath)
    (shotname, extension) = os.path.splitext(filename)

    # 打开PDF文件
    pdf = fitz.open(pdfPath)
    # 逐页读取PDF
    for pg in range(0, pdf.pageCount):
        page = pdf[pg]
        r = page.rect

        zoom_ratio = square_size / r.width

        # 设置缩放和旋转系数
        trans = fitz.Matrix(zoom_ratio, zoom_ratio)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        # 开始写图像,保存在当前文件夹中
        save_file = filepath+"\\"+shotname+"_uc.png"
        pm.save(save_file)

    pdf.close()
    return save_file


def main_checker_thinline(imgPath, outpath, isWallPaper, zoom_ratio):
    '''
    根据输入的图像检测断点和细线
    :param imgPath: 输入的图像的路径
    :param outpath: 保存检测结果的路径
    :param isWallPaper: 输入的图像是正方形图还是墙纸图，后者长宽不一样
    :param zoom_ratio: pdf->png图像的缩放比例
    :return: 返回细线检测处理结果, 以及检测到的细线点个数
    '''
    (filepath, filename) = os.path.split(imgPath)
    (shotname, extension) = os.path.splitext(filename)

    # 创建细线检测和断点检测结果图的路径
    tlpath = os.path.join(outpath, "thin_line_result")
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

    # 细线化检测
    print("开始细线化检测")
    # delta控制线的粗细阈值,增减单元建议0.05。为正时，线的阈值增加，将有更多的线被检测到。
    # 为负时，线的阈值降低，将有更少的线被检测到.
    # isWallPaper表示输入的是墙纸图.
    maker_img, pt_num = td.thin_line_detection(imgPath, img, tlpath, False,
                                               delta=0, zoomratio=zoom_ratio, isWallPaper=isWallPaper)
    if maker_img is None:
        return None

    middle_name = "_" + str(pt_num)

    # 保存最终的maker图, 并在结果图中标示uc的个数
    dst_img_thin = os.path.join(tlpath, shotname + middle_name + extension)
    print(dst_img_thin)
    cv2.imencode(extension, maker_img)[1].tofile(dst_img_thin)

    return dst_img_thin, pt_num


def main_checker_brokenline(imgPath, outpath, thinpoint_num):
    '''
    根据输入的图像检测断点和细线
    :param imgPath: 输入的图像的路径
    :param outpath: 保存检测结果的路径
    :param thinpoint_num: 检测到的细线点的个数;这个数用于区分很多细线的原图，以及细线有限的原图。
    这两类图需要采用不同的二值化算法参数
    :return: 返回断点检测处理结果
    '''
    (filepath, filename) = os.path.split(imgPath)
    (shotname, extension) = os.path.splitext(filename)

    # 创建细线检测和断点检测结果图的路径
    blpath = os.path.join(outpath, "broken_line_result")
    if not os.path.exists(blpath):
        os.makedirs(blpath)

    img = Image.open(imgPath)
    if len(img.getbands()) == 4:
        newimg = img_converter.alpha_composite_with_color(img).convert('RGB')
    elif len(img.getbands()) == 3:
        newimg = img
    else:
        print("image format error")
    img = cv2.cvtColor(np.asarray(newimg), cv2.COLOR_RGB2BGR)

    print("开始未闭合线头检测")
    input_img = img
    # 实际使用的图片分辨率是2048
    if img.shape[0] > PRODUCT_IMG_SIZE or img.shape[1] > PRODUCT_IMG_SIZE:
        ratio = min(img.shape[0], img.shape[1]) / PRODUCT_IMG_SIZE
        newh, neww = math.floor(img.shape[0] / ratio), math.floor(img.shape[1] / ratio)
        input_img = cv2.resize(img, (neww, newh))
    maker_img, uc_num = ud.unclosed_line_detection(imgPath, input_img, blpath, thinpoint_num, debug=False)
    middle_name = "_" + str(uc_num)

    if maker_img is None:
        return None

    # 保存最终的maker图, 并在结果图中标示uc的个数
    dst_img_uc = os.path.join(blpath, shotname + middle_name + extension)
    print(dst_img_uc)
    cv2.imencode(extension, maker_img)[1].tofile(dst_img_uc)

    return dst_img_uc


