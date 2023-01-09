#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Commands.PythonCommandBase import PythonCommand, ImageProcPythonCommand
from Commands.Keys import KeyPress, Button, Direction, Stick, Hat
from logging import getLogger, DEBUG, NullHandler

import subprocess
import os
import sys
import random
import string
import cv2
from PIL import Image, ImageOps
import numpy as np
import time
import datetime
import winsound
import configparser
import threading
import pyperclip
import json
import requests
import io

import tkinter as tk
from tkinter import ttk

# カーソル位置を確認しながら行うボックス操作(画像認識使用)
class BoxOperationMethods(ImageProcPythonCommand):
    NAME = 'SV_BOX操作(画像認識)'
    PRESS_BUTTON_DURATION=0.1
    PRESS_BUTTON_WAIT=0.5


    def __init__(self, cam, gui=None):
        super().__init__(cam)
        self.cam = cam
        self.gui = gui
        # loggerを有効にする
        self._logger = getLogger(__name__)
        self._logger.addHandler(NullHandler())
        self._logger.setLevel(DEBUG)
        self._logger.propagate = True



    def do(self):
        self.testPerformance2()
        

    # カーソルが正しく認識できているかテスト
    # できていないなら画像を差し替える
    def testPerformance1(self):
        cnt=0
        while(cnt<10):
            self.calcCoordinatesInBox()
            time.sleep(1)
            cnt+=1


    # カーソルが操作ができているかテスト
    # testPerformance1ができているなら必要ない？
    def testPerformance2(self):
        testlist=[(0,0), (1,1), (2,2), (6,0), (3,3), (6,1), (4,5), (6,2)]
        for tl in testlist:
            print("次の座標へ移動します:"+str(tl))
            self.setPositionInBox((tl))
            print("移動完了")
            time.sleep(3)
        

    # mask付きテンプレートマッチング
    # テンプレート画像の他にマスク画像が必要になるが、透過処理が可能
    # 返り値はtupleで、[0]に検知結果、[1]に検知した座標を格納する
    # 参考url:
    # https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html
    def isContainTemplateWithMask(self, 
        template_path, mask_path, threshold=0.7, use_gray=True, show_value=False,
        area=[], tmp_area=[], mask_area=[]):

        # Read a current image
        src = self.camera.readFrame()
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if use_gray else src

        # Read a template image
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE if use_gray else cv2.IMREAD_COLOR)
        w, h = template.shape[1], template.shape[0]

        # Read a mask image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE if use_gray else cv2.IMREAD_COLOR)

        # mask未対応のためCCOEFFからCCORRに変更
        method = cv2.TM_CCORR_NORMED
        res = cv2.matchTemplate(src, template, method, None, mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if show_value:
        	print(template_path + ' ZNCC value: ' + str(max_val))

        if max_val > threshold:
            if use_gray:
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(src, top_left, bottom_right, (255, 0, 255), 2)
            return (True, max_loc)
        else:
            return (False, max_loc)


    # boxを開いた状態で使用
    # box内の現在の座標(y,x)を計算する
    # 座標((0,0):ボックス名, (1-6,0):手持ち, (1-5,1-6):ボックス, (6,1):いちらん, (6,2):検索)
    def calcCoordinatesInBox(self):
        INF=10007
        PIXS_IN_BOX = np.array([
            [[526,61], [INF,INF], [INF,INF], [INF,INF], [INF,INF], [INF,INF], [INF,INF]],
            [[188,127], [316,127], [400,127], [484,127], [568,127], [652,127], [736,127]],
            [[188,211], [316,211], [400,211], [484,211], [568,211], [652,211], [736,211]],
            [[188,295], [316,295], [400,295], [484,295], [568,295], [652,295], [736,295]],
            [[188,379], [316,379], [400,379], [484,379], [568,379], [652,379], [736,379]],
            [[188,463], [316,463], [400,463], [484,463], [568,463], [652,463], [736,463]],
            [[188,547], [388,575], [662,575], [INF,INF], [INF,INF], [INF,INF], [INF,INF]]
        ])

        image_dir=r"./Template/SV/SV_BoxOperationMethods_Images/"
        template_path=image_dir+r"cursor.png"
        mask_path=image_dir+r"cursor_mask.png"
        match_result=self.isContainTemplateWithMask(template_path, mask_path, use_gray=False)

        if(match_result[0]):
            self._logger.debug("cursor is detected.")
            distances=np.abs(PIXS_IN_BOX-match_result[1]).sum(axis=2)
            detect_index=np.unravel_index(distances.argmin(), distances.shape)
            self._logger.debug("pixs:" + str(match_result[1]) + ", coordinates in box:" + str(detect_index))
            res=(True,detect_index)
        else:
            self._logger.debug("cursor is not found.")
            res=(False,(-1,-1))
        return res


    # 指定した分だけカーソルを進める
    def moveCursor(self, diff_y, diff_x):
        while(not 0==diff_y):
            if(0<diff_y):
                self.press(Hat.BTM,self.PRESS_BUTTON_DURATION,self.PRESS_BUTTON_WAIT)
                diff_y-=1
            else:
                self.press(Hat.TOP,self.PRESS_BUTTON_DURATION,self.PRESS_BUTTON_WAIT)
                diff_y+=1

        while(not 0==diff_x):
            if(0<diff_x):
                self.press(Hat.RIGHT,self.PRESS_BUTTON_DURATION,self.PRESS_BUTTON_WAIT)
                diff_x-=1
            else:
                self.press(Hat.LEFT,self.PRESS_BUTTON_DURATION,self.PRESS_BUTTON_WAIT)
                diff_x+=1
        return


    # boxを開いた状態で使用
    # box内の指定座標へ移動する
    def setPositionInBox(self, target_coordinates):
        cnt=0
        while(True):
            if(10<=cnt):
                self._logger.debug("cursol is not found.")
                return False

            res_coordinates_in_box=self.calcCoordinatesInBox()
            if(res_coordinates_in_box[0]):
               break 
            cnt+=1

        nowPosition=res_coordinates_in_box[1]
        while(nowPosition!=target_coordinates):
            if(0==target_coordinates[0]):
                self.moveCursor(0, 1-nowPosition[1]) 
                self.moveCursor(target_coordinates[0]-nowPosition[0], 0) 
            else:
                self.moveCursor(target_coordinates[0]-nowPosition[0], target_coordinates[1]-nowPosition[1]) 
            nowPosition=self.calcCoordinatesInBox()[1]
        return


    # box一覧を開いた状態で使用
    # 現在選択しているboxを計算する
    # 返り値は0-34の整数値(0:手持ち, 1-32:各ボックス, 33:検索の横にあるボックス, 34:検索)
    def calcCoordinatesSwitchBox(self):
        return


    # boxを開いた状態で使用
    # 指定のboxへ移動する
    def setBox(self, goalBoxIndex):
        return



