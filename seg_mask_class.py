import numpy as np

class seg_mask():
    def __init__(self):
        pass
    
    def obj_det(self,Ix, Iy):
        _, row, col = Ix.shape
        Ix1 = Ix[0,:,:].reshape(1,-1)
        Iy1 = Iy[0,:,:].reshape(1,-1)
        I1 = (Iy1.T@Iy1)+(Ix1.T@Ix1)
        di = np.diag_indices(row*col)
        return (I1[di[0],di[1]]).reshape(row,col)
    