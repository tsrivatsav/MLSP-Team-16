import numpy as np
import matplotlib.pyplot as plt 

from scipy.signal import convolve
from seg_mask_class import seg_mask
from optical_flow_class import optical_flow
from kalman_filter_class import start_kalman_obj

class Object_Tracker():
    def __init__(self, set_zero_int_flag = False):
        
        self.zero_flag = set_zero_int_flag 
        
        self.OF_OBJ = optical_flow()
        self.KF_OBJ = None
        
        self.seg_mask = None
        self.of_mapx = None
        self.of_mapy = None
        
        self.center_obs = None
        self.pred_state = None
        self.RT_FRAME = None
        
        self.of_bkx = None
        self.of_bky = None
        
        
    def Get_Seg_Mask(self, I):
        Ix, Iy, It = self.grad_Image(I)
        # image Segmetizer
        seg_obj = seg_mask()
        self.seg_mask = seg_obj.obj_det(Ix, Iy) 
        return self.seg_mask

    def grad_Image(self, I):
        I_stack = self.pad_I(I)
        k = np.array([-1/2, 0, 1/2])
        kx = k.reshape(1,1,-1)
        ky = k.reshape(1,-1,1)
        kt = k.reshape(-1,1,1)

        Ix = convolve(I_stack, kx, mode = "valid") 
        Ix = Ix[1:-1,1:-1,:]
        Iy = convolve(I_stack, ky, mode = "valid")
        Iy = Iy[1:-1,:,1:-1]
        It = convolve(I_stack, kt, mode = "valid")
        It = It[:,1:-1,1:-1]
                
        return Ix, Iy, It
    
    def pad_I(self, I_stack):
        return np.pad(I_stack, (1), mode= "edge")
    
    
    def window_function(self, img):
        as2 = (img).reshape(2,2,-1)
        as2 = (np.swapaxes(as2, 0,2).reshape(-1,2)).T
        as2 = as2.reshape(-1,2,2)
        as2 = np.swapaxes(as2, 2,1)
        return as2
    
    # https://stackoverflow.com/questions/27175400/how-to-find-the-index-of-a-value-in-2d-array-in-python
    def cal_centroid(self):
        m_mask = np.abs(self.of_mapx) + np.abs(self.of_mapy)
        if np.sum(m_mask) > 0:    
            centroid = np.mean(np.argwhere(
                            m_mask > np.mean(m_mask)),
                            axis = 0)
            
            centroid = centroid.reshape(-1,1)
        else:
            centroid = None  
        return centroid
    
    
    def Start_Obj_Tracking(self):
        self.KF_OBJ = start_kalman_obj(self.center_obs, set_zero = self.zero_flag)        
        self.pred_state = self.KF_OBJ.state    
        return  
     
    # https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
    def RUN_MOV_TRACK(self, VID, plot_flag = True):
        plt.ion()
        fig = plt.figure()
                
        # Vid[numfram, row, col] => assume gray 
        n_frame, row, col = VID.shape
                        
        for fnum in range(0,n_frame):
            
            save_str = str(fnum)+".png"
            
            # calulated Optical Flow
            I = VID[fnum:fnum+2,:,:]
            Ix, Iy, It = self.grad_Image(I)
            # optical flow cal
            of_x, of_y = \
                    self.OF_OBJ.LK_Optical_Flow(
                        Ix[0,:,:], 
                        Iy[0,:,:], 
                        It[0,:,:]
                        )
                    
            self.of_mapx = of_x.reshape(row//2, col//2)
            self.of_mapy = of_y.reshape(row//2, col//2)    
            # calulated Centroid      
            self.center_obs = self.cal_centroid()
            
            if fnum == 0:
                # init tracking
                self.Start_Obj_Tracking()
            else:
                #run kalman filter
                self.pred_state, _ = self.KF_OBJ.run(self.center_obs)
            
            print(np.round(self.pred_state))
            print()
            if self.center_obs is not None:
                print(np.round(self.center_obs))
            
            if plot_flag:
                plt.imshow(downsample(VID[fnum,:,:],2), cmap = "gray")
                
                plt.scatter(
                        self.pred_state[2],
                        self.pred_state[0],
                        s=30,
                        facecolor='none',
                        edgecolor='black')
                
                if self.center_obs is not None:
                    plt.scatter(
                        self.center_obs[1], 
                        self.center_obs[0], 
                        s=25, 
                        color = "red")
                
                plt.savefig(save_str) 
                plt.show()
                plt.pause(0.0001)
    
    ##
    #
    ##
    def INIT_RT(self, Buffer, ds = 1):
        _, row, col = Buffer.shape
        self.RT_FRAME = Buffer[-1,:,:]
        Ix, Iy, It = self.grad_Image(Buffer)
        # optical flow cal
        of_x, of_y = \
                self.OF_OBJ.LK_Optical_Flow(
                    Ix[0,:,:], 
                    Iy[0,:,:], 
                    It[0,:,:]
                    )
                    
        self.of_bkx = of_x.reshape(row//2, col//2)
        self.of_bky = of_y.reshape(row//2, col//2)  
        self.of_mapx = of_x.reshape(row//2, col//2)
        self.of_mapy = of_x.reshape(row//2, col//2)
        # calulated Centroid      
        self.center_obs = np.array([[row//2], [col//2]]) 
         
        # init tracking
        self.Start_Obj_Tracking()
        
        return
                  
    def RUN_MOV_TRACK_RT(self, FRAME, ds = 1, update=True):
                 
        # FRAME[row, col] => assume gray 
        row, col = downsample(FRAME,ds).shape
        
        I = list()
        I.append(np.expand_dims(
                downsample(self.RT_FRAME,ds),
                axis=0))
        I.append(np.expand_dims(
            downsample(FRAME,ds),
            axis=0))              
        I = np.vstack(I) 
         
        if update:
            # calulated Optical Flow       
            Ix, Iy, It = self.grad_Image(I)
            # optical flow cal
            of_x, of_y = \
                    self.OF_OBJ.LK_Optical_Flow(
                        Ix[0,:,:], 
                        Iy[0,:,:], 
                        It[0,:,:]
                        )
                             
            self.of_mapx = \
                of_x.reshape(row//2, col//2) 
            self.of_mapy = \
                of_y.reshape(row//2, col//2) 
            # calulated Centroid      
            center_run = self.cal_centroid()
            
            if center_run is not None:
                center_run = 2*ds*center_run 
                self.center_obs = center_run               
            
        else:
            center_run = None

        #run kalman filter
        self.pred_state, _ = self.KF_OBJ.run(center_run)
        
        opt_flow_center = (round(self.center_obs[1][0]), 
                           round(self.center_obs[0][0]))
        kalman_center = (round(self.pred_state[2][0]), 
                         round(self.pred_state[0][0]))
        
        return opt_flow_center, kalman_center
                  
           
                
                
##
# down sample
# https://stackoverflow.com/questions/34231244/downsampling-a-2d-numpy-array-in-python/34232507
##
def downsample(img_arr, factor):
    return img_arr[::factor, ::factor]