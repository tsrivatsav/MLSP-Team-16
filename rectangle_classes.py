# https://stackoverflow.com/questions/50301946/matplotlib-imshow-doesnt-display-numpy-ones-array
# https://medium.com/@enriqueav/how-to-create-video-animations-using-python-and-opencv-881b18e41397
# https://stackoverflow.com/questions/62880911/generate-video-from-numpy-arrays-with-opencv

import numpy as np
import matplotlib.pyplot as plt 
import cv2 

class rectangle:
    def __init__(self, row, col, height, width):
        self.row = row
        self.col = col
        self.height = height
        self.width = width

    def draw_rect(self, pos):
        if pos[0] < 0:
            pos[0] = 0
        elif pos[0] + self.height > self.row:
            pos[0] = self.row - self.height
        
        if pos[1] < 0:
            pos[1] = 0
        elif pos[1] + self.width > self.col:
            pos[1] = self.col - self.width
        
        frame = np.ones((self.row,self.col))*255
        frame[pos[0]:pos[0]+self.height, pos[1]:pos[1]+self.width] = 0

        return frame
    
    def draw_roming_rectangle(self, max_speed=10, max_acc=0.5, num_frames=1800, fps=30, write_vid_flag = False):
        vid = []
        pos_list = []
        vel_list = []

        out = cv2.VideoWriter('sq.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.row, self.col), False)

        pos = np.random.randint((self.row - self.height, self.col - self.width))
        
        vel = np.random.random(2)
        vel = vel/np.linalg.norm(vel)
        
        acc = np.random.random(2)
        acc = acc/np.linalg.norm(acc) * (np.random.random()/2 + 0.5) * max_acc

        pos_list.append(pos)
        vel_list.append(vel)

        turn_counter = 0
        turn_threshold = np.random.randint(10, 20)
        
        for _ in range(num_frames-1):
            pos = (pos.astype(float) + vel).astype(int)
            vel += acc
            if np.linalg.norm(vel) > max_speed:
                vel = vel / np.linalg.norm(vel) * max_speed

            if pos[0] < 0:
                pos[0] = 0
                vel[0] *= -1
                acc[0] *= -1
            elif pos[0] + self.height > self.row:
                pos[0] = self.row - self.height
                vel[0] *= -1
                acc[0] *= -1
            
            if pos[1] < 0:
                pos[1] = 0
                vel[1] *= -1
                acc[1] *= -1
            elif pos[1] + self.width > self.col:
                pos[1] = self.col - self.width
                vel[1] *= -1
                acc[1] *= -1

            if turn_counter >= turn_threshold:
                acc = np.random.random(2)
                acc = acc / np.linalg.norm(acc) * (np.random.random()/2 + 0.5) * max_acc
                turn_counter = 0
                turn_threshold = np.random.randint(10, 20)
            else:
                turn_counter += 1

            frame = self.draw_rect(pos)
            vid.append(frame)
            pos_list.append(pos)
            vel_list.append(vel)
            if write_vid_flag:
                out.write(frame.astype('uint8'))

        out.release()
            
        return np.array(vid), np.array(pos_list), np.array(vel_list)

    def show_rect(self):
        plt.imshow(self.rect, cmap="gray", vmin=0, vmax=1)


def Make_SQ_Vid():

    row = 256
    col = 256
    sq_hw = 10 # rect height & width 

    obj = rectangle(row, col, sq_hw, sq_hw)
    obj.draw_roming_rectangle(write_vid_flag=True)
    return

if __name__ == '__main__':
    Make_SQ_Vid()