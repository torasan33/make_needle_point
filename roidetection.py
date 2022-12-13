import numpy as np

import torch

import net

class RoiDetection():
    def __init__(self, ct_img):
        self.ct_img = ct_img
        self.weight_path = "weights/ex_new_data_test_cpu.pth"

        self.needle_roi = []
        self.grip_roi = []

    def cut_img(self):
        "画像をカットして32x32画像と対応座標をリストで返す"
        IMG, point = [], []
        X, Y = self.ct_img.shape[1]-16, self.ct_img.shape[0]-16
        for y in range(0, Y, int(32/2)):
            for x in range(0, X, int(32/2)):
                cuted_img = np.array([l[x:x+32] for l in self.ct_img[y:y+32]])
                #画面が真っ黒と端の画像はcontinue
                if cuted_img.max() == cuted_img.min():
                    continue
                elif (x == 0) or (x >= 511-32) or (y == 0) or (y >= 511-32):
                    continue
                #type変換
                if np.array(cuted_img).max() < 0 and np.array(cuted_img).min() < 0:
                    cuted_img = cuted_img + abs(np.array(cuted_img).min())
                    cuted_img = cuted_img / abs(np.array(cuted_img).max())
                elif np.array(cuted_img).max() >= 0 and np.array(cuted_img).min() < 0:
                    cuted_img = cuted_img + abs(np.array(cuted_img).min())
                    cuted_img = cuted_img / abs(np.array(cuted_img).max())
                elif np.array(cuted_img).max() >= 0 and np.array(cuted_img).min() >=0:
                    cuted_img = cuted_img / np.array(cuted_img).max()
                IMG.append([cuted_img])
                point.append([x, y])
        return IMG, point


    def inference2D(self, model_state_dict):
        IMG, point = self.cut_img()
        X = torch.tensor(np.array(IMG).reshape(len(IMG), 1, 32, 32), dtype= torch.float32)
        Y = model_state_dict(X)
        for i in range(len(Y)):
            if 1 <= Y[i][1]:
                self.needle_roi.append(point[i])
            elif 1 == Y[i][2]:                #しきい値
                self.grip_roi.append(point[i])
        
    def roi_detection(self):
        model_state_dict = net.Deep_Net()
        model_state_dict.load_state_dict(torch.load(self.weight_path), strict=False)
        self.inference2D(model_state_dict)
        
        return self.needle_roi, self.grip_roi
        

