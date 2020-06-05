import cv2
import pyocr
import pyocr.builders
import os
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw

path_tesseract = "C:\\Program Files\\Tesseract-OCR"
if path_tesseract not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + path_tesseract


tools = pyocr.get_available_tools()
tool = tools[0]

def ocr(img,Width,Height):
    
    #OCRで読みたい領域を切り出す
    dst = img[100:Height-200,100:Width-200]
    PIL_Image=Image.fromarray(dst)
    text = tool.image_to_string(
        PIL_Image,
        lang='jpn',
        builder=pyocr.builders.TextBuilder()
    )


    #空白削除
    import re
    text = re.sub('([あ-んア-ン一-龥ー])\s+((?=[あ-んア-ン一-龥ー]))',
              r'\1\2', text)

    if(text != ""):
        print(text)

def OcrTest():
    cap = cv2.VideoCapture(0)

#動画書き出し用のオブジェクト
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
 
    fps = 15.0
    size = (640, 360)
    writer = cv2.VideoWriter('out1.m4v', fmt, fps, size)

#face5
    face_cascade_file5 = "haarcascade_frontalface_default.xml"
    face_cascade_5 = cv2.CascadeClassifier(face_cascade_file5)
  
#画像
    anime_file = "Bakuga.png"
    anime_face = cv2.imread(anime_file)
 
#画像を貼り付ける
    def anime_face_func(img, rect):
        (x1, y1, x2, y2) = rect
        w = x2 - x1
        h = y2 - y1
        img_face = cv2.resize(anime_face, (w, h))
       
        img2 = img.copy()
        img2[y1:y2, x1:x2] = img_face
        return img2

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Height, Width = frame.shape[:2]
        img = cv2.resize(frame,(int(Width),int(Height)))
        faces = face_cascade_5.detectMultiScale(gray, 1.1, 2)
        
        for (x, y, w, h) in faces:
                img = anime_face_func(img, (x, y, x+w, y+h))
        writer.write(img)
        cv2.imshow('img', img)

        # OCRで読み取りたい領域を赤枠で囲む
        cv2.rectangle(img, (100, 100), (Width-200, Height-200), (0, 0, 255), 10)
        ocr(img, Width,Height)
        edimg = img

        # 文字を追加
             # へッター
        cv2.putText(img, "YUI-OCR PROTO.", (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255,0), 5, cv2.LINE_AA) 

           #OCR結果 
        dst = img[100:Height-200,100:Width-200]
        PIL_Image=Image.fromarray(edimg)
        text = tool.image_to_string(
                PIL_Image,
                lang='jpn',
                builder=pyocr.builders.TextBuilder(tesseract_layout=11)
        )

        fontpath ='C:\Windows\Fonts\HGRPP1.TTC'   # Windows10 だと C:\Windows\Fonts\ 以下にフォントがあります。
        font = ImageFont.truetype(fontpath, 24)         # フォントサイズ
        draw = ImageDraw.Draw(PIL_Image)             # drawインスタンスを生成
        draw.text((30,420), text, font=font, fill=(255,255,255,0))

            # フッター
        font2 = ImageFont.truetype(fontpath, 36)         # フォントサイズ
        list_gobi = ["かもしれない", "かな～?", "に似てる","じゃない？", "...読めない","字は綺麗に！","読めないー"]
        list_random = random.choice(list_gobi)
        draw.text((410,420), list_random, font=font2, fill=(255,255,0,0))
        draw.text((0,420), "<", font=font2, fill=(255,255,0,0))
        draw.text((390,420), ">", font=font2, fill=(255,255,0,0))

        img = np.array(PIL_Image)              
        
        cv2.imshow("Edited Image", img)

        key = cv2.waitKey(300)
        if key == 27:
            break
        
    writer.release()
    cap.release()
    cv2.destroyAllWindows() 


OcrTest()


