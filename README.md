线框图质量检测程序

## 运行环境
    python 3.7  
## 库
    PIL
    opencv-python
    scikit-image
## 主程序
- main.py
- 该文件通过读取*input_path*变量定义的文件夹下的所有图像，进行线框图质量的检测。
并将不符合标准的检测结果以特殊颜色的方框、圆圈等进行标识。结果文件存放在*output_folder*
变量定义的文件夹中。现阶段，仅支持线框图中**未闭合线段**的检测。检测结果如下所示：
![输入输出示意图](https://github.com/zjustarstar/SketchChecker/blob/main/result/flowchart.jpg)

## 使用说明
- 彩色图与黑白图略有不同。如果处理的是彩色线框图，需要定义*IS_COLOR_SKETCH*参数为True。即使只有部分
线框是彩色，也作为彩色图处理。
- 检测的主函数是*unclosed_line_detection*,该函数返回两个参数，分别是结果图，以及检测到的未闭合的点的数量
uc_num。
- 结果文件以：原图文件名_uc_闭合点个数.png 命名。

## 效果说明
    我们设计算法时，尽量保证不漏检，容忍存在误检。黑白图的检测效果还是不错的，但是仍然存在以下可能的误检。
- 线段靠近造成的误检。两条线段的端点靠的很近，造成了误检。
![线段靠近造成的误检](https://github.com/zjustarstar/SketchChecker/blob/main/result/closelines.jpg)
- 短线段造成的误检。线框图中有些很短的正常的线，也可能造成误检。
![短线段造成的误检](https://github.com/zjustarstar/SketchChecker/blob/main/result/shortlines.jpg)
- 小圆点造成的误检。由于算法的原因，某些线框图中的小圆点也会造成误检。
![小圆点造成的误检](https://github.com/zjustarstar/SketchChecker/blob/main/result/smalldots.jpg)
- 太薄的线造成的误检。由于算法的原因，某些线框太薄太细（可能仅1像素宽)，也可能造成误检。
![太薄的线造成的误检](https://github.com/zjustarstar/SketchChecker/blob/main/result/thinlines.jpg)
    除了黑白图，彩色图也可能造成误检。彩色图的最大问题在于部分颜色（比如黄色）特别浅，很容易被遗漏。比如下面
的彩色线框图颜色就很浅。
![彩色线框图](https://github.com/zjustarstar/SketchChecker/blob/main/result/thincolor.jpg)
  

    