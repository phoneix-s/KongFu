# 武术动作评分系统 & 武术双人PK小游戏

## 项目简介

本项目共实现了两项任务：

- 武术动作评分系统：检测人体关键点，识别人物的武术动作，并根据标准程度进行打分（包括图片评分、视频评分、实时评分）。
- 武术双人PK小游戏：开启摄像头实时检测画面中的双方玩家，玩家通过自己的动作控制游戏人物的动作，击败对手，获得胜利。

## 效果演示

**武术动作评分系统**

<video src="./img/评分.mp4"></video>

**武术双人PK小游戏**

<video src="./img/PK.mp4"></video>





其中主要的人体关键点检测部分使用了两种工具：

- 商汤科技的SensetimeSDK
- openmmlab中的MMdetection和MMpose



以下具体说明运行方法：

## 武术动作评分系统

### 1、运行环境说明

环境需求：

- scikit-learn
- pyqt5
- numpy 
- joblib
- matplotlib
- opencv-python

以及商汤的SensetimeSDK或openmmlab的mmpose和mmdet。

### 2、如何运行

直接运行`main.py`即可。

```cmd
python main.py
```

### 3、运行结果

运行`main.py`后会出现以下界面：

<img src=".\img\武术1.png" alt="1" style="zoom:60%;" />



左侧图片为标准动作图片，默认为弓步，在右侧操作界面中点击马步/仆步可以切换动作并显示在左侧窗口中。

点击训练模型可以读取`gongbu`,`mabu`等文件夹中的图片并进行训练，获得基于逻辑回归的模型。

第二步可以选择基于规则/机器学习的评分标准，基于规则的方法实际上是中心化之后对比当前关键点坐标和标准动作关键点坐标，基于机器学习的方法是根据之前训练的模型对当前图片进行预测。

第三步，可以选择图片评分、录像评分、实时评分三种，点击对应按钮即可。注：实时评分需调用电脑摄像头，若有多个摄像头需填写相机序号，若笔记本电脑只有自带摄像头则可不填相机序号。

## 武术双人PK小游戏

### 1、运行环境说明

环境需求：

- numpy 
- pygame
- matplotlib
- opencv-python

以及商汤的SensetimeSDK或openmmlab的mmpose和mmdet。

### 2、如何运行

直接运行`main.py`即可。

```cmd
python main.py
```

### 3、运行结果

运行`main.py`后会出现以下界面：

<img src=".\img\PK1.png" alt="1" style="zoom:60%;" />

按照画面中提示进行PK对战即可。
