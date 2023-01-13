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
    THRESHOLD_TEMPLATE_MATCHING=0.9


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
        

    # boxを開いた画面で使用
    # 現在のカーソル座標を20回出力
    # Controllerボタンで位置を動かして動作確認
    def testPerformance1(self):
        cnt=0
        while(cnt<20):
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
        

    # box一覧画面でカーソル座標が正しく計算できているかテスト
    # できていないなら画像を差し替える
    def testPerformance3(self):
        cnt=0
        while(cnt<20):
            self.calcCoordinatesSwitchBox()
            time.sleep(1)
            cnt+=1
    
    def testPerformance4():
        cnt=0
        while(cnt<20):
            # boxを開いた状態か確認
            image_dir=r"./Template/SV/SV_BoxOperationMethods_Images/"
            template_is_box_path=image_dir+r"box_ichiran_normal.png"
            mask_is_box_path=image_dir+r"box_ichiran_normal_mask.png"
            template_is_box_selected_path=image_dir+r"box_ichiran_selected.png"
            res=self.isContainTemplateWithMask(template_is_box_path, mask_is_box_path, use_gray=False)
            if(not res[0]):
                print("boxが開かれていません")
            else:
                print("boxが開かれています")

            # box一覧を開いた状態か確認
            image_dir=r"./Template/SV/SV_BoxOperationMethods_Images/"
            template_is_switch_box_path=image_dir+r"switch_box_temochi.png"
            mask_is_switch_box_path=image_dir+r"switch_box_temochi_mask.png"
            res=self.isContainTemplateWithMask(template_is_switch_box_path, mask_is_switch_box_path, use_gray=False)
            if(not res[0]):
                print("box一覧が開かれていません")
            else:
                print("box一覧が開かれています")

            self.calcCoordinatesSwitchBox()
            time.sleep(2)
            cnt+=1
 


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
        self._logger.debug(max_val)

        if show_value:
        	print(template_path + ' ZNCC value: ' + str(max_val))

        if(max_val > threshold):
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

        # 操作時に一瞬カーソルが消えることがあるため、何度かチェックする
        cnt=0
        while(cnt<10):
            match_result=self.isContainTemplateWithMask(template_path, mask_path, use_gray=False)
            if(match_result[0]):
                break
            cnt+=1

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
        return 0


    # boxを開いた状態で使用
    # box内の指定座標へ移動する
    # 座標((0,0):ボックス名, (1-6,0):手持ち, (1-5,1-6):ボックス, (6,1):いちらん, (6,2):検索)
    def setPositionInBox(self, target_coordinates):
        # boxを開いた状態か確認
        image_dir=r"./Template/SV/SV_BoxOperationMethods_Images/"
        template_is_box_path=image_dir+r"box_ichiran_normal.png"
        mask_is_box_path=image_dir+r"box_ichiran_normal_mask.png"
        template_is_box_selected_path=image_dir+r"box_ichiran_selected.png"
        res=self.isContainTemplateWithMask(template_is_box_path, mask_is_box_path, use_gray=False)
        if(not res[0]):
            if(not self.isContainTemplate(template_is_box_selected_path, THRESHOLD_TEMPLATE_MATCHING)):
                print("boxが開かれていません")
                return 1

        # 初期座標計算
        cnt=0
        while(True):
            if(10<=cnt):
                self._logger.debug("cursol is not found.")
                return 1

            res_coordinates_in_box=self.calcCoordinatesInBox()
            if(res_coordinates_in_box[0]):
               break
            cnt+=1

        # カーソル移動
        # 指定座標へ到着したか確認して終了
        nowPosition=res_coordinates_in_box[1]
        while(nowPosition!=target_coordinates):
            if(0==target_coordinates[0]):
                self.moveCursor(0, 1-nowPosition[1])
                self.moveCursor(target_coordinates[0]-nowPosition[0], 0)
            else:
                #TODO: 最小経路を通るよう修正
                #self.moveCursor(target_coordinates[0]-nowPosition[0], target_coordinates[1]-nowPosition[1]) 
                self.moveCursor(target_coordinates[0]-nowPosition[0], target_coordinates[1]-nowPosition[1]) 
            nowPosition=self.calcCoordinatesInBox()[1]
        return 0


    # box一覧を開いた状態で使用
    # box一覧内の現在座標(y,x)を計算する
    # 座標((0,0):手持ち, (1-4,0-7):各ボックス, (5,0):いちらん, (5,1):検索)
    def calcCoordinatesSwitchBox(self):
        INF=10007
        PIXS_SWITCH_BOX = np.array([
            [[889,55], [INF,INF], [INF,INF], [INF,INF], [INF,INF], [INF,INF], [INF,INF], [INF,INF]],
            [[618,130], [698,130], [778,130], [858,130], [938,130], [1018,130], [1098,130], [1178,130]],
            [[618,210], [698,210], [778,210], [858,210], [938,210], [1018,210], [1098,210], [1178,210]],
            [[618,290], [698,290], [778,290], [858,290], [938,290], [1018,290], [1098,290], [1178,290]],
            [[618,370], [698,370], [778,370], [858,370], [938,370], [1018,370], [1098,370], [1178,370]],
            [[760,570], [1033,570], [INF,INF], [INF,INF], [INF,INF], [INF,INF], [INF,INF], [INF,INF]]
        ])

        image_dir=r"./Template/SV/SV_BoxOperationMethods_Images/"
        template_path=image_dir+r"cursor.png"
        mask_path=image_dir+r"cursor_mask.png"

        # 操作時に一瞬カーソルが消えることがあるため、何度かチェックする
        cnt=0
        while(cnt<10):
            match_result=self.isContainTemplateWithMask(template_path, mask_path, use_gray=False)
            if(match_result[0]):
                break
            cnt+=1

        if(match_result[0]):
            self._logger.debug("cursor is detected.")
            distances=np.abs(PIXS_SWITCH_BOX-match_result[1]).sum(axis=2)
            detect_index=np.unravel_index(distances.argmin(), distances.shape)
            #self._logger.debug("pixs:" + str(match_result[1]) + ", coordinates in switch box:" + str(detect_index))
            print("pixs:" + str(match_result[1]) + ", coordinates in switch box:" + str(detect_index))
            res=(True,detect_index)
        else:
            self._logger.debug("cursor is not found.")
            res=(False,(-1,-1))
        return res




    # boxを開いた状態で使用
    # 指定のboxへ移動する
    def setBox(self, goalBoxIndex):
        # box一覧を開く
        self.setPositionInBox((6,1))
        self.press(Button.A,self.PRESS_BUTTON_DURATION,self.PRESS_BUTTON_WAIT)
        time.sleep(1)

        # box一覧を開いた状態か確認
        image_dir=r"./Template/SV/SV_BoxOperationMethods_Images/"
        template_is_switch_box_path=image_dir+r"switch_box_temochi.png"
        mask_is_switch_box_path=image_dir+r"switch_box_temochi_mask.png"
        res=self.isContainTemplateWithMask(template_is_switch_box_path, mask_is_switch_box_path, use_gray=False)
        if(not res[0]):
            print("box一覧が開かれていません")
            return 1


        return



