# https://machinelearningspace.com/object-tracking-python/
# https://www.cs.toronto.edu/~jepson/csc2503/tracking.pdf
# https://www.kalmanfilter.net/alphabeta.html
# https://towardsdatascience.com/sensor-fusion-part-2-kalman-filter-code-78b82c63dcd
# https://cs229.stanford.edu/proj2007/Kim-ObjectTrackingInAVideoSequence.pdf



import numpy as np
from numpy.linalg import inv

class kalman_filter():
    def __init__(self, t_eps, t_gamma, m_eps = 0, m_gamma = 0):
        # state (mean & covar)
        self.state = None
        self.R = None
                
        #p(s_t+1|st)
        self.A_mat = None       
        self.theta_eps = t_eps
        self.mean_eps = m_eps
        
        #p(ot|st)
        self.B_mat = None
        self.theta_gamma = t_gamma
        self.mean_gamma = m_gamma
        return
    
    # https://stackoverflow.com/questions/10198747/how-can-i-simultaneously-select-all-odd-rows-and-all-even-columns-of-an-array   
    def init_func(self, init_state, init_R, dt, mode=0):
        self.state = init_state
        self.R = init_R
        
        # Constant Velocity
        if mode == 0:
            self.A_mat = np.array(
                [[1, dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1]]
                )
            
            teps_temp = np.eye(self.A_mat.shape[0])
            teps_temp[::2] = teps_temp[::2]*0
            teps_temp[1::2] = teps_temp[1::2]*self.theta_eps
            self.theta_eps = teps_temp
            
            
            # Obs Postion
            self.B_mat = np.array(
                [[1, 0, 0, 0],
                [0, 0, 1, 0]]         
                )
            
            tgamma_temp = np.eye(self.B_mat.shape[0])*self.theta_gamma
            self.theta_gamma = tgamma_temp   
            
        else:
            print("Work In Progress")
            
        return
        
    def K_gain(self, p_R, lam = 0.0):
        k_t = self.B_mat@p_R@self.B_mat.T 
        k_t = k_t + self.theta_gamma
        reg = lam*np.eye(k_t.shape[0])
        k_t = k_t+reg
        k_t = inv(k_t)
        k_gain = p_R@(self.B_mat.T)@k_t
        return k_gain
    
    def pred_func(self):
        pred_state = self.A_mat@self.state + self.mean_eps
        pred_R = self.theta_eps + self.A_mat@self.R@self.A_mat.T
        return pred_state, pred_R
    
    def update_func(self, obs, p_state, p_R):
        def new_state(kg, obs, p_state):
            dns = obs - self.B_mat@p_state - self.mean_gamma
            new_state = p_state + kg@dns
            return new_state
        
        def new_R(kg, p_R):
            k_gain1 = kg@self.B_mat
            k_gain2 = (np.eye(k_gain1.shape[0]) - k_gain1)
            new_R = k_gain2@p_R
            return new_R
        
        k_gain = self.K_gain(p_R)     
        return new_state(k_gain, obs, p_state), new_R(k_gain, p_R)
    
    def run_init(self, obs, init_state, init_R, dt, mode=0):
        self.init_func(init_state=init_state, init_R=init_R, dt=dt, mode=mode)
        self.state, self.R = self.update_func(obs, init_state, init_R)
        return
        
    def run(self, new_obs=None):
        ps, pR = self.pred_func()
        if new_obs is not None:
            self.state, self.R = self.update_func(new_obs, ps, pR)
        else:
            self.state, self.R = ps, pR
        return self.state, self.R
    
    
def start_kalman_obj(obs0, dt = 1, t_eps=1, t_gamma=1, set_zero = False):
    s0 = np.zeros((4,1))
    
    if not set_zero:
        s0[0] = obs0[0]
        s0[2] = obs0[1]
    
    R0 = np.zeros(
        (len(s0),len(s0))
        )

    obj_kf = kalman_filter(t_eps, t_gamma)
    obj_kf.run_init(obs0, s0, R0, dt)
    obj_kf.run(obs0)
    return obj_kf