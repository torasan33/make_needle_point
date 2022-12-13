import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk

import pydicom
import numpy as np
from matplotlib import pyplot as plt
import os
import glob

import roidetection

class Model():
    def __init__(self):
        self.np_img = None
        self.roi_img = None
        self.tk_img = None
        self.ct_img = None
        self.roi_tk_img = None
        self.ratio = 0
        self.CANVAS_SIZE = 512

    def roi_detection(self):
        #ROI検出
        RoiDetection = roidetection.RoiDetection(self.ct_img)
        needle_roi, grip_roi = RoiDetection.roi_detection()
        if needle_roi == []:
            return 0
        xmax, ymax, self.xmin, self.ymin = np.array(needle_roi).max(axis=0)[0]+32, np.array(needle_roi).max(axis=0)[1]+32, np.array(needle_roi).min(axis=0)[0], np.array(needle_roi).min(axis=0)[1]
        self.roi_img = self.np_img[self.ymin:ymax, self.xmin:xmax]
        
        if xmax - self.xmin < ymax - self.ymin:   self.ratio = int(self.CANVAS_SIZE / (ymax - self.ymin))
        else:   self.ratio = int(self.CANVAS_SIZE / (xmax - self.xmin))
        self.roi_tk_img = Image.fromarray(self.roi_img)
        self.roi_tk_img = self.roi_tk_img.resize((int((xmax-self.xmin) * self.ratio), int((ymax-self.ymin) * self.ratio)), Image.BILINEAR)
        self.roi_tk_img = ImageTk.PhotoImage(self.roi_tk_img)
        
    def write_output(self, out_list, name):
        f = open("output/" + name + ".txt", "a")
        f.write(str(out_list))
        f.close()

    def read_tkimg(self, path):
        ds = pydicom.dcmread(path)
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        wc = 400
        ww = 2000

        ri = ds.RescaleIntercept
        rs = ds.RescaleSlope
        self.ct_img = ds.pixel_array

        img = self.ct_img * rs+ri
        max = wc + ww / 2
        min = wc - ww / 2
        img = 255*(img - min) / (max - min)
        self.np_img = np.clip(img,0,255)
        self.tk_img = Image.fromarray(self.np_img)
        self.tk_img = ImageTk.PhotoImage(self.tk_img)

    def make_line_points(self, x0, y0, x1, y1):
        #直線作成
        return_list = []
        if (y0 > y1):   x0, y0, x1, y1 = x1, y1, x0, y0
        if (abs(y0-y1) > abs(x0-x1)):
            if (x0 == x1):
                for y in range(y0, y1+1):
                    return_list.append([x0, y])
            elif (x0 < x1):
                a, b = (y0-y1)/(x0-x1), y0-x0*(y0-y1)/(x0-x1)
                now_x = x0
                for y in range(y0, y1+1):
                    target_x = (y-b)/a
                    if abs(now_x - target_x) < abs((now_x+1) - target_x):
                        return_list.append([now_x, y])
                    else:
                        now_x += 1
                        return_list.append([now_x, y])
            elif (x0 > x1):
                a, b = (y0-y1)/(x0-x1), y0-x0*(y0-y1)/(x0-x1)
                now_x = x0
                for y in range(y0, y1+1):
                    target_x = (y-b)/a
                    if abs(now_x - target_x) < abs((now_x-1) - target_x):
                        return_list.append([now_x, y])
                    else:
                        now_x -= 1
                        return_list.append([now_x, y])
        elif (abs(y0-y1) < abs(x0-x1)):
            if (x0 < x1):
                a, b = (y0-y1)/(x0-x1), y0-x0*(y0-y1)/(x0-x1)
                now_y = y0
                for x in range(x0, x1+1):
                    target_y = a*x+b
                    if abs(now_y - target_y) < abs((now_y+1) - target_y):
                        return_list.append([x, now_y])
                    else:
                        now_y += 1
                        return_list.append([x, now_y])
            elif (x0 > x1):
                a, b = (y0-y1)/(x0-x1), y0-x0*(y0-y1)/(x0-x1)
                now_y = y0
                for x in range(x0, x1-1, -1):
                    target_y = a*x+b
                    if abs(now_y - target_y) < abs((now_y+1) - target_y):
                        return_list.append([x, now_y])
                    else:
                        now_y += 1
                        return_list.append([x, now_y])
        return return_list

    def reset(self):
        self.np_img = None
        self.roi_img = None
        self.tk_img = None
        self.ct_img = None
        self.roi_tk_img = None
        self.ratio = 0

class View():
    def __init__(self, root, model):
        self.SHOW_ROI = 1
        self.SHOW_ORIGIN = 0

        self.master = root
        self.model = model
        # アプリ内のウィジェットを作成
        self.create_widgets()

    def create_widgets(self):
        #左大枠：img_frame
        self.img_frame = tk.Frame(self.master, width=530, height=600)
        self.img_frame.grid(column=0, row=0, rowspan=4, padx=20, pady=10)

        self.img_canvas = tk.Canvas(self.img_frame, width=self.model.CANVAS_SIZE, height=self.model.CANVAS_SIZE, bg='gray')
        self.img_canvas.grid(column=0, row=0, pady=10)
        self.message = tk.StringVar()
        self.file_infor = tk.Label(self.img_frame, textvariable=self.message, width=70, height=2, anchor=tk.W)
        self.file_infor.grid(column=0, row=1)

        #右大枠：button_frame
        self.button_frame = tk.Frame(self.master)
        self.button_frame.grid(column=1, row=0, rowspan=4, pady=10)

        #labelフレーム
        self.label_frame = tk.Frame(self.button_frame)
        self.label_frame.grid(column=0, row=0, padx=15, pady=5)
        self.now_mouse_point = tk.StringVar()
        self.now_mouse_point_infor = tk.Label(self.label_frame, textvariable=self.now_mouse_point, width=70, height=2, anchor=tk.W)
        self.now_mouse_point_infor.grid(column=0, row=0)
        self.now_points = tk.StringVar()
        self.now_points_infor = tk.Label(self.label_frame, textvariable=self.now_points, width=70, height=2, anchor=tk.W)
        self.now_points_infor.grid(column=0, row=1)

        #ファイル関連frame
        self.file_frame = tk.LabelFrame(self.button_frame, text="file select", width=270, height=30)
        self.file_frame.grid(column=0, row=1, padx=15, pady=5, sticky=tk.W)
        self.file_load_button = tk.Button(self.file_frame, text="file select", width=20)
        self.file_load_button.grid(column=0, row=1, padx=3, sticky=tk.W)
        self.file_back_button = tk.Button(self.file_frame, text="beck", width=10)
        self.file_back_button.grid(column=1, row=1, padx=2)
        self.file_next_button = tk.Button(self.file_frame, text="next", width=10)
        self.file_next_button.grid(column=2, row=1, padx=3)
        self.roi_show_button = tk.Button(self.file_frame, text="ROI show", width=10)
        self.roi_show_button.grid(column=3, row=1, padx=3)


        #points追加関連frame
        self.points_add_frame = tk.LabelFrame(self.button_frame, text="Points add", width=270, height=70, bg='gray90')
        self.points_add_frame.grid(column=0, row=2, padx=15, pady=5, sticky=tk.W)

        self.xy_label = tk.Label(self.points_add_frame, text="x\ny", font=("10"), width=2, height=2, anchor=tk.W)
        self.xy_label.grid(column=0, row=0, rowspan=2)

        self.point1_x = tk.Entry(self.points_add_frame, width=10)
        self.point1_x.grid(column=1, row=0, padx=2)
        self.point1_y = tk.Entry(self.points_add_frame, width=10)
        self.point1_y.grid(column=1, row=1, padx=2)

        self.to_label = tk.Label(self.points_add_frame, text="to\nto", font=("10"), width=2, height=2)
        self.to_label.grid(column=2, row=0, rowspan=2)

        self.point2_x = tk.Entry(self.points_add_frame, width=10)
        self.point2_x.grid(column=3, row=0, padx=2)
        self.point2_y = tk.Entry(self.points_add_frame, width=10)
        self.point2_y.grid(column=3, row=1, padx=2)

        self.points_add_button = tk.Button(self.points_add_frame, text="add", width=10)
        self.points_add_button.grid(column=4, row=1, padx=40, pady=2)

        #points削除関連frame
        self.point_delete_frame = tk.LabelFrame(self.button_frame, text="Point Delete", width=350, height=70, bg='gray90')
        self.point_delete_frame.grid(column=0, row=3, padx=15, pady=5, sticky=tk.W)

        self.xy_delete_label = tk.Label(self.point_delete_frame, text="x\ny", font=("10"), width=2, height=2, anchor=tk.W)
        self.xy_delete_label.grid(column=0, row=0, rowspan=2)
        self.point_delete_x = tk.Entry(self.point_delete_frame, width=10)
        self.point_delete_x.grid(column=1, row=0, pady=0, padx=2)
        self.point_delete_y = tk.Entry(self.point_delete_frame, width=10)
        self.point_delete_y.grid(column=1, row=1, pady=2, padx=2)

        self.point_delete_button = tk.Button(self.point_delete_frame, text="delete", width=10)
        self.point_delete_button.grid(column=4, row=1, padx=2, pady=2, sticky=tk.E)
        
        self.point_reset_button = tk.Button(self.point_delete_frame, text="point reset", width=10)
        self.point_reset_button.grid(column=5, row=1, padx=2, pady=2, sticky=tk.E)

        #otherボタンframe
        self.other_button_frame = tk.Frame(self.button_frame, width=350, height=70, bg='gray90')
        self.other_button_frame.grid(column=0, row=4, padx=15, pady=5, sticky=tk.W)
        #plt.確認ボタン
        self.plt_confirm_button = tk.Button(self.other_button_frame, text="matplotlib confirm", width=15)
        self.plt_confirm_button.grid(column=0, row=4, padx=2, pady=2, sticky=tk.W)
        #テキストファイル出力ボタン
        self.out_put_button = tk.Button(self.other_button_frame, text="output", width=10)
        self.out_put_button.grid(column=1, row=4, padx=2, pady=2, sticky=tk.W)

    def select_file(self):
        full_path = tkinter.filedialog.askopenfilename(initialdir=".")
        return full_path

    def draw_image(self, type):
        objs = self.img_canvas.find_withtag("image")        #キャンバスの画像削除
        for obj in objs:
            self.img_canvas.delete(obj)
        if (self.model.tk_img is not None) and (type == self.SHOW_ORIGIN):                   #画像があれば表示
            self.img_canvas.create_image(self.model.tk_img.width()//2, self.model.tk_img.height()//2, image=self.model.tk_img, tag='image')
        elif (self.model.roi_tk_img is not None) and (type == self.SHOW_ROI):
            self.img_canvas.create_image(self.model.roi_tk_img.width()//2, self.model.roi_tk_img.height()//2, image=self.model.roi_tk_img, tag='image')

    def draw_message(self, message):
        self.message.set(message)
    def draw_now_points(self, now_points):
        self.now_points.set(now_points)
    def draw_now_mouse_point(self, mouse_point):
        self.now_mouse_point.set(mouse_point)
    def draw_points(self, all_points, roi_flag):
        for point in all_points:
            if roi_flag == 0:
                self.img_canvas.create_text(point[0], point[1], text=".", fill="red", font=(10))
            elif roi_flag == 1:
                self.img_canvas.create_text(self.model.xmin + int(point[0]/self.model.ratio), self.model.ymin + int(point[1]/self.model.ratio), text=".", fill="red", font=(10))
                print(self.model.xmin + int(point[0]/self.model.ratio))


    def plt_confirm(self, out_list):
        if out_list != []:
            plt.scatter(np.array(out_list)[:, 0], np.array(out_list)[:, 1], color="r", marker=".")
        plt.imshow(self.model.np_img, cmap="gray")
        plt.show()

class Controller():
    INTERVAL = 50

    def __init__(self, root, model, view):
        self.master = root
        self.model = model
        self.view = view
     
        self.set_events()
        self.message = "file none"
        self.now_points_message = 'points none'
        self.file_name, self.full_path = None, None
        self.dir_path = None
        self.out_list = []
        self.now_mouse_point_message = "mouse point"
        self.roi_flag = 0
        self.now_file_no = 0

    def set_events(self):
        #画像表示，ファイル関係
        self.view.file_load_button['command'] = self.push_file_load_button
        self.view.roi_show_button['command'] = self.push_roi_show_button
        self.view.file_back_button['command'] = self.push_file_back_button
        self.view.file_next_button['command'] = self.push_file_next_button
        #ポイント関係
        self.view.points_add_button['command'] = self.push_points_add_button
        self.view.point_delete_button['command'] = self.push_point_delete_button 
        self.view.point_reset_button['command'] = self.push_point_reset_button
        #otherボタン関係
        self.view.out_put_button['command'] = self.push_out_put_button 
        self.view.plt_confirm_button['command'] = self.push_plt_confirm_button 
        #canvas内関係
        self.view.img_canvas.bind('<Motion>', self.pickup_position)
        self.view.img_canvas.bind('<Button-1>', self.pickup_point)
        self.view.img_canvas.bind('<Button-3>', self.delete_pickup_point)
        #master操作
        self.master.bind("<KeyPress-m>", self.push_file_load_button)
        self.master.bind("<KeyPress-s>", self.push_file_back_button)
        self.master.bind("<KeyPress-f>", self.push_file_next_button)
        self.master.bind("<KeyPress-r>", self.push_roi_show_button)
        self.master.bind("<KeyPress-a>", self.push_points_add_button)
        self.master.bind("<KeyPress-p>", self.push_plt_confirm_button)
        self.master.bind("<KeyPress-o>", self.push_out_put_button)

        self.master.after(Controller.INTERVAL, self.timer)
        
    def timer(self):
        self.master.after(Controller.INTERVAL, self.timer)
        self.view.draw_message(self.message)
        self.view.draw_now_points(self.now_points_message)
        self.view.draw_now_mouse_point(self.now_mouse_point_message)
        
    def push_file_load_button(self, event=None):
        "ファイル選択ボタン処理"
        full_path = self.view.select_file()
        if full_path != None:
            self.full_path = full_path
            self.model.reset()
            self.dir_path, self.file_name = os.path.dirname(self.full_path), os.path.basename(self.full_path).split('.', 1)[0]
            self.model.read_tkimg(self.full_path)
            self.message = "file:" + str(self.full_path)
            self.view.draw_image(self.view.SHOW_ORIGIN)
            self.roi_flag = 0
            self.now_file_no = glob.glob(os.path.join(self.dir_path, '*')).index(os.path.join(self.dir_path, self.file_name))

    def push_file_back_button(self, event=None):
        "ファイル一つ戻るボタン処理"
        if self.now_file_no > 0:
            self.now_file_no -= 1
            self.full_path = glob.glob(os.path.join(self.dir_path, '*'))[self.now_file_no]
            self.file_name = os.path.basename(self.full_path).split('.', 1)[0]
            self.model.reset()
            self.model.read_tkimg(self.full_path)
            self.message = "file:" + str(self.full_path)
            self.view.draw_image(self.view.SHOW_ORIGIN)
            self.roi_flag = 0
        
    def push_file_next_button(self, event=None):
        "ファイル一つ進むボタン処理"
        if self.now_file_no < len(glob.glob(os.path.join(self.dir_path, '*'))):
            self.now_file_no += 1
            self.full_path = glob.glob(os.path.join(self.dir_path, '*'))[self.now_file_no]
            self.file_name = os.path.basename(self.full_path).split('.', 1)[0]
            self.model.reset()
            self.model.read_tkimg(self.full_path)
            self.message = "file:" + str(self.full_path)
            self.view.draw_image(self.view.SHOW_ORIGIN)
            self.roi_flag = 0

    def push_points_add_button(self, event=None):
        "addボタン処理"
        point1_x, point1_y = int(self.view.point1_x.get()), int(self.view.point1_y.get())
        point2_x, point2_y = int(self.view.point2_x.get()), int(self.view.point2_y.get())
        self.view.point1_x.delete(0, tk.END)
        self.view.point1_y.delete(0, tk.END)
        self.view.point2_x.delete(0, tk.END)
        self.view.point2_y.delete(0, tk.END)
        print(point1_x, point1_y, point2_x, point2_y)

        for point in (self.model.make_line_points(point1_x, point1_y, point2_x, point2_y)):
            if point not in self.out_list:  self.out_list.append(point)
        self.out_list.sort()
        self.view.draw_points(self.out_list, self.roi_flag)
        self.now_points_message = 'now_points: x1:' + str(point1_x) + 'y1:' + str(point1_y) + 'x2:' + str(point2_x) + 'y2:' + str(point2_y)


    def push_point_delete_button(self):
        "deleteボタン処理"
        delete_point_x, delete_point_y = int(self.view.point_delete_x.get()), int(self.view.point_delete_x.get())
        self.view.point_delete_x.delete(0, tk.END)
        self.view.point_delete_y.delete(0, tk.END)
        if [delete_point_x, delete_point_y] in self.out_list:
            self.out_list.remove([int(delete_point_x), int(delete_point_y)])
        self.now_points_message = 'now_points:' + str(self.out_list)

    def push_point_reset_button(self):
        "リセットボタン処理"
        self.out_list = []
        self.now_points_message = 'now_points:' + str(self.out_list)
        self.view.img_canvas.delete('all')
        if self.model.tk_img != None:   self.view.draw_image(self.view.SHOW_ORIGIN)

    def push_plt_confirm_button(self, event=None):
        'matplot確認用ボタン処理'
        self.view.plt_confirm(self.out_list)

    def push_out_put_button(self):
        "outputボタン処理"
        self.model.write_output(self.out_list, str(self.file_name))
        self.now_points_message = 'output complete'
        self.out_list = []

    def push_roi_show_button(self, event=None):
        "roi表示ボタン処理"
        if self.roi_flag == 0:
            if self.model.roi_tk_img == None:
                self.model.roi_detection()
                if self.model.roi_tk_img == None:
                    self.view.img_canvas.create_text(self.model.CANVAS_SIZE/2, self.model.CANVAS_SIZE/2, text="roi is not detected", font=(24), fill="white")
            self.view.draw_image(self.view.SHOW_ROI)
            self.roi_flag = 1
        elif self.roi_flag == 1:
            self.view.draw_image(self.view.SHOW_ORIGIN)
            self.roi_flag = 0
        

    def pickup_position(self, event):
        #canvas内のマウスポイント追従
        if self.roi_flag == 0:
            self.now_mouse_point_message = 'mouse point: x : ' + str(event.x+1) + ' y : ' + str(event.y+1)
        elif self.roi_flag == 1:
            self.now_mouse_point_message = 'mouse point: x : ' + str(int(((event.x+1)/self.model.ratio)) + self.model.xmin) + ' y : ' + str(int(((event.y+1)/self.model.ratio)) + self.model.ymin)

    def pickup_point(self, event):
        #canvas内左クリックでポイント追加
        if (view.point1_x.get() == '') and (self.roi_flag == 0):
            self.view.point1_x.delete(0, tk.END)
            self.view.point1_x.insert(tk.END, (event.x+1))
            self.view.point1_y.delete(0, tk.END)
            self.view.point1_y.insert(tk.END, (event.y+1))
        elif (view.point1_x.get() == '') and (self.roi_flag == 1):
            self.view.point1_x.delete(0, tk.END)
            self.view.point1_x.insert(tk.END, int(((event.x+1)/self.model.ratio)) + self.model.xmin)
            self.view.point1_y.delete(0, tk.END)
            self.view.point1_y.insert(tk.END, int(((event.y+1)/self.model.ratio)) + self.model.ymin)
        elif (view.point2_x.get() == '') and (self.roi_flag == 0):
            self.view.point2_x.delete(0, tk.END)
            self.view.point2_x.insert(tk.END, (event.x+1))
            self.view.point2_y.delete(0, tk.END)
            self.view.point2_y.insert(tk.END, (event.y+1))
        elif (view.point2_x.get() == '') and (self.roi_flag == 1):
            self.view.point2_x.delete(0, tk.END)
            self.view.point2_x.insert(tk.END, int(((event.x+1)/self.model.ratio)) + self.model.xmin)
            self.view.point2_y.delete(0, tk.END)
            self.view.point2_y.insert(tk.END, int(((event.y+1)/self.model.ratio)) + self.model.ymin)
        
    def delete_pickup_point(self, event):
        #canvas内右クリックで前選択のポイント削除
        if (view.point2_x.get() == '') == False:
            self.view.point2_x.delete(0, tk.END)
            self.view.point2_y.delete(0, tk.END)
        elif (view.point1_x.get() == '') == False:
            self.view.point1_x.delete(0, tk.END)
            self.view.point1_y.delete(0, tk.END)

root = tk.Tk()
root.title("make needle points")
root.geometry("1000x600")

model = Model()
view = View(root, model)
controller = Controller(root, model, view)

root.mainloop()
