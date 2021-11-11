import numpy as np
from scipy.signal import convolve

class optical_flow():
    def __init__(self):
        pass
    
    def LK_Optical_Flow(self, Ix, Iy, It):
        Itwn = -1*self.window_function(It)
        Itwn = np.swapaxes(Itwn.reshape(-1,1,4), 0,2)
        
        Ix0 = self.window_function(Ix)
        Ix0v = np.swapaxes(Ix0.reshape(-1,1,4), 0,2)
        Iy0 = self.window_function(Iy)
        Iy0v = np.swapaxes(Iy0.reshape(-1,1,4), 0,2)
        
        A = np.hstack([Ix0v,Iy0v])
        vloop = loop_cal(A, Itwn)
        # invesigate direction from calculation
        vx = vloop[:,0,:]
        vy = vloop[:,1,:]
                
        return vx, vy

    def pad_I(self, I_stack):
        return np.pad(I_stack, (1), mode= "edge")

    def grad_Image(self, I):
        I_stack = self.pad_I(I)
        # I_stack = I
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

    def window_function(self, img, win_size=2):
        as2 = (img).reshape(win_size,win_size,-1)
        as2 = (np.swapaxes(as2, 0,2).reshape(-1,win_size)).T
        as2 = as2.reshape(-1,win_size,win_size)
        as2 = np.swapaxes(as2, 2,1)
        return as2
 

def elm_dot(A, B):
    x = np.tensordot(
        A, B, axes=((1,0)) 
        )
    x = x.transpose(0,2,1,3)
    r,c,s,_ = x.shape
    di = np.diag_indices(s)
    return x[:,[0],di[0],di[1]], x[:,[1],di[0],di[1]]


# element wise maxtric mult 
# Improve this func!
def loop_cal(A, Itn):
    _,_,s = A.shape 
    v_list = list()
    for i in range(s):
        v_list.append(np.linalg.pinv(A[:,:,i])@Itn[:,:,i])
        
    return np.array(v_list)