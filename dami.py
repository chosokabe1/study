import os 
import cv2
import time
import random
import argparse
import glob
import numpy as np
def remove_objects(img, lower_size=None, upper_size=None):
  # find all objects
  nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
  sizes = stats[1:, -1]
  _img = np.zeros((labels.shape))
  # process all objects, label=0 is background, objects are started from 1
  for i in range(1, nlabels):
    # remove small objects
    if (lower_size is not None) and (upper_size is not None):
      if lower_size < sizes[i - 1] and sizes[i - 1] < upper_size:
        _img[labels == i] = 255
    elif (lower_size is not None) and (upper_size is None):
      if lower_size < sizes[i - 1]:
        _img[labels == i] = 255
    elif (lower_size is None) and (upper_size is not None):
      if sizes[i - 1] < upper_size:
        _img[labels == i] = 255
  return _img
def anaume(img):
  kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
  th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_ellipse)
  th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_ellipse)
  th_clean = remove_objects(th, lower_size=30000, upper_size=None)
  th_clean = th_clean.astype(np.uint8)
  th_clean = cv2.morphologyEx(th_clean, cv2.MORPH_CLOSE, kernel_ellipse, iterations=2)
  th_clean_not = cv2.bitwise_not(th_clean)
  th_clean_not_clean = remove_objects(th_clean_not, lower_size=None, upper_size=10000)
  th_clean_not_clean = th_clean_not_clean.astype(np.uint8)
  anaume_img = cv2.bitwise_or(th_clean, th_clean_not_clean)
  return anaume_img

def output_width(img_path):
  img = cv2.imread(img_path)
  anaume_img = anaume(img)
  # skeleton_img = cv2.ximgproc.thinning(anaume_img, thinningType=cv2.ximgproc.THINNING_GUOHALL)
  # cv2.imwrite(os.path.join('data/221221_autodetect/anaume', os.path.splitext(os.path.basename(img_path))[0] + '_skeleton.bmp'), skeleton_img)
  dist = cv2.distanceTransform(anaume_img, cv2.DIST_L2, 5)
  max_distance = np.amax(dist)
  # print(f'test{max_distance}')
  output_width = max_distance * 2
  return output_width
def output_area(img_path):
  img = cv2.imread(img_path)
  anaume_img = anaume(img)
  cv2.imwrite(os.path.join('data/221221_autodetect/anaume',os.path.basename(img_path)), anaume_img)
  output_area = np.count_nonzero(anaume_img)
  return output_area
def output_probability_of_having_egg(img_path):
  output = 0.99999
  return output
def output_overall_length(img_path):
  overall_length = 9999
  return overall_length
def output_probability_of_several(img_path):
  output = 0.99999
  return output

def output_life_judge(img1_path, img2_path):
  output = "TRUE or FALSE"
  return output

parser = argparse.ArgumentParser(description='顕微鏡で撮像した画像から卵持ちか否かを判定')
parser.add_argument('-f', '--file_path', default='./file_name.bmp', type=str, help='ファイル名を指定')
args = parser.parse_args()
#ファイル名　○-○_20221215.bmp
#1つ目の○　個体番号
#2つ目の○　時間番号

number_of_organism = 5
number_of_flame = 2

img_paths = []
for file_path in glob.glob(os.path.join(args.file_path, "*")):
  img_paths.append(file_path)

for i in range(0, number_of_organism * number_of_flame, number_of_flame):

  probability_of_having_egg = output_probability_of_having_egg(img_paths[i])
  print(probability_of_having_egg)

  overall_length = output_overall_length(img_paths[i])
  print(overall_length)

  width = output_width(img_paths[i])
  print(width)

  area = output_area(img_paths[i])
  print(area)

  probability_of_several = output_probability_of_several(img_paths[i])
  print(probability_of_several)

  life_judge = output_life_judge(img_paths[i], img_paths[i+1])
  print(life_judge)


# if os.path.isfile(args.file_path):
#   image = cv2.imread(args.file_path)          # 画像読込
# #   time.sleep(0.05)                            # ダミープログラム遅延
# #   print("result :", random.randint(0, 1000)/10)
# else:
#   print('ファイルが見つかりませんでした')
