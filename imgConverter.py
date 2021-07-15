# 将RGBA的PNG图转为RGB图

import cv2
import numpy as np
from PIL import Image


def alpha_composite(front, back):
    """Alpha composite two RGBA images.
    Source: http://stackoverflow.com/a/9166671/284318
    Keyword Arguments:
    front -- PIL RGBA Image object
    back -- PIL RGBA Image object
    """
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result


def alpha_composite_with_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA image with a single color image of the
    specified color and the same size as the original image.

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)


def inverse_white(path):
    # 输入为png的路径
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    if img.shape[2] == 3:
        return img

    # 将png透明背景改为255全白
    img2 = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    a = img2[:, 3] == 0
    img2[a, 0:3] = 255

    imgnew = img2[:, 0:3]
    imgnew = np.reshape(imgnew, (img.shape[0], img.shape[1], 3))

    return imgnew

