import numpy as np
from scipy.signal import convolve
from seg_mask_class import seg_mask
from optical_flow_class import optical_flow

class Object_Tracker():
    def __init__(self):
        self.seg_mask = None
        self.of_mapx = None
        self.of_mapy = None
        
    def Track_Obj(self, I):
        Ix, Iy, It = self.grad_Image(I)
        
        # image Segmetizer
        seg_obj = seg_mask()
        self.seg_mask = seg_obj.obj_det(Ix, Iy)
        
        # optical flow cal
        of_obj = optical_flow()
        self.of_mapx, self.of_mapy = \
                of_obj.LK_Optical_Flow(
                    Ix[0,:,:], 
                    Iy[0,:,:], 
                    It[0,:,:]
                    )        
        return      

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
