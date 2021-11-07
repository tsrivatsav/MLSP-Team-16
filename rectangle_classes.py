# https://stackoverflow.com/questions/50301946/matplotlib-imshow-doesnt-display-numpy-ones-array
# https://medium.com/@enriqueav/how-to-create-video-animations-using-python-and-opencv-881b18e41397
# https://stackoverflow.com/questions/62880911/generate-video-from-numpy-arrays-with-opencv

import numpy as np
import matplotlib.pyplot as plt 
import cv2 

class rectangle:
    def __init__(self, r, c, h, w, lr=0, ud=0):
        self.row = r
        self.col = c
        self.hieght = h
        self.width = w
        self.shift_lr = lr
        self.shift_ud = ud
        self.rect = self.draw_rect(lr,ud)

    def draw_rect(self, s_lr, s_ud):
        xl = round(self.col/2 - self.width/2) + s_lr 
        xr = round(self.col/2 + self.width/2) + s_lr
        yd = round(self.row/2 + self.hieght/2) + s_ud 
        yu = round(self.row/2 - self.hieght/2) + s_ud

        if xl < 0:
            xl = 0
        if xr > self.col:
            xr = self.col-1
        if xl > xr:
            xl = xr

        if yu < 0:
            yu = 0
        if yd > self.row:
            yd = self.row-1
        if yu > yd:
            yu = yd
        
        if xl != xr and yu != yd:
            rect = np.ones((self.row,self.col))
            rect[yu:yd,xl:xr] = 0
        if xl == xr and yu != yd:
            rect = np.ones((self.row,self.col))
            rect[yu:yd,xl] = 0
        if xl != xr and yu == yd:
            rect = np.ones((self.row,self.col))
            rect[yu,xl:xr] = 0
        if xl == xr and yu == yd:
            rect = np.ones((self.row,self.col))     
       
        return rect

    def show_rect(self):
        plt.imshow(self.rect, cmap="gray", vmin=0, vmax=1)


class moving_rect(rectangle):
    def __init__(self, r, c, h, w, num_frames, d, v=0, lr=0, ud=0, FPS=None, write_vid_flag = False):
        self.velocity = v
        self.direction = d
        rectangle.__init__(self, r, c, h, w, lr, ud)
        self.moving_rect = self.draw_moving_rect(num_frames, FPS, write_vid_flag)
    
    def draw_moving_rect(self,num_frames, FPS=None, write_vid_flag = False):

        if type(self.velocity) != list:
            frame_stack = self.draw_contant_vel(num_frames, FPS, write_vid_flag)
        else:
            print("in deveploment")

        return frame_stack
        
    def draw_contant_vel(self, num_frames, FPS=None, write_vid_flag = False):
        dx, dy = 0, 0
        rect_vid = list()

        out = cv2.VideoWriter('sq.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (self.row, self.col), False)

        if write_vid_flag:
            
            for n in range(num_frames):
                d_lr = self.direction[0]
                d_ud = self.direction[1]
                dx += round(self.velocity*(n-1)*d_lr) + self.shift_lr
                dy += round(self.velocity*(n-1)*d_ud) + self.shift_ud
                new_rect = self.draw_rect(dx, dy)
                data = np.random.randint(0,256, (self.row, self.col), dtype='uint8')
                data = 255*new_rect
                out.write(data.astype('uint8'))
                rect_vid.append(new_rect)

            out.release()
            
        
        else:

            for n in range(num_frames):
                d_lr = self.direction[0]
                d_ud = self.direction[1]
                dx += round(self.velocity*(n-1)*d_lr)
                dy += round(self.velocity*(n-1)*d_ud)
                new_rect = self.draw_rect(dx, dy)

                rect_vid.append(new_rect)

        return np.array(rect_vid)



def Make_SQ_Vid():

    row = 256
    col = 256
    sq_hw = 10 # pix hieght & width 
    shift_lr = -10
    shift_ud = 0

    num_f = 100
    direction = [1,0]
    vel = 1 # pix/frame

    obj = moving_rect(row,col,sq_hw,sq_hw,num_f, direction, vel, lr=shift_lr, ud=shift_ud, FPS=10, write_vid_flag=True)
    obj.moving_rect.shape
    return