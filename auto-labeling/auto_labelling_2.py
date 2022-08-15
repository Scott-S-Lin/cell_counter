#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:33:28 2022

@author: root

https://ithelp.ithome.com.tw/articles/10269138?sc=rss.iron
"""

import cv2
import darknet
import time
import auto_labelling_3

def image_detection(image_path, network, class_names, class_colors, thresh):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

# 指定網路結構配置檔 yolov3.cfg，自建數據集設定檔 obj.data，事先訓練好的權重檔 yolov3.backup
network, class_names, class_colors = darknet.load_network(
    "./cfg/scott/yolo-obj-scott.cfg",
    "./cfg/scott/obj_cell.data",
    "./weights/yolo-obj-scott_last.weights",
    1
)



train_txt = "/home/oem/ScottLin/project/darknet/data/train_cell_0715.txt"


file1 = open(train_txt, 'r')
Lines = file1.readlines()
for image_name in Lines:
        print("image=", image_name.strip())
        prev_time = time.time() # 用來計算辨識一張圖片的時間
        print('predicting...', prev_time)
        # 進行影像辨識，回傳畫出方塊框的圖片以及辨識結果，辨識結果包含標籤、置信度以及方塊框的座標
        image, detections = image_detection(
                image_name.strip(), network, class_names, class_colors, 0.25
                )
        # 印出標籤、置信度以及方塊框的座標
        darknet.print_detections(detections, '--ext_output')
        auto_labelling_3.save_annotations(image_name, image, detections, class_names)
        # 顯示辨識時間
        print('predict end=',(time.time() - prev_time),"\n\n")
        # 將結果圖片寫成檔案 result.jpg
        cv2.imwrite('result.jpg', image)