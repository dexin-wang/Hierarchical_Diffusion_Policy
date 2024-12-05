"""
绘制操作可控的action轨迹
"""

import cv2
import numpy as np
import yaml
import math


class ColorWrap:
    """
    将颜色范围均匀划分为num份
    """
    def __init__(self, color1, color2, num) -> None:
        """
        color: bgr
        """
        self.color1 = color1
        self.color2 = color2
        self.db = (color2[0] - color1[0]) / num
        self.dg = (color2[1] - color1[1]) / num
        self.dr = (color2[2] - color1[2]) / num
    
    def color(self, n):
        return (
            self.color1[0]+n*self.db,
            self.color1[1]+n*self.dg,
            self.color1[2]+n*self.dr,
            )


    
img_path = 'paper_results/pushT_init_2.png'
img = cv2.imread(img_path)
# cv2.imshow('img', img)

results_action_path = 'paper_results/control/pushT_dp_43_action.npy'
# 读取轨迹数据
actions = np.load(results_action_path)[6:, :30]

# 绘制轨迹
for trajectory in actions:
    colorWrapL = ColorWrap((9, 196, 243), (0, 0, 255), trajectory.shape[0])
    colorWrapR = ColorWrap((225, 0, 0), (0, 225, 0), trajectory.shape[0])
    for s in range(trajectory.shape[0]):
        colorL = colorWrapL.color(s)
        colorR = colorWrapR.color(s)
        cv2.circle(img, 
                   center=(int(trajectory[s, 0]), int(trajectory[s, 1])), 
                   radius=1, color=colorL, thickness=-1)
        if s < trajectory.shape[0]-1:
            cv2.line(img, 
                     (int(trajectory[s, 0]), int(trajectory[s, 1])), 
                     (int(trajectory[s+1, 0]), int(trajectory[s+1, 1])), 
                     color=colorL, thickness=2)

cv2.imshow('img', img)
cv2.waitKey()