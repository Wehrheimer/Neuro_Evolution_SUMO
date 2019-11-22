import numpy as np


class SUMO:
    """Class for all SUMO related variables and methods"""
    def __init__(self, Route_Ego, Route_Preceding, Route_Preceding2, timestep=0.2, net='SUMO_net_test_2.net.xml'):
        self.sim = {'duration': 100000., 'timestep': timestep, 'net': net}
        self.v_profile = []
        self.a_set = np.zeros((100000, 1), dtype=np.float32)
        self.a_real = np.zeros((100000, 1), dtype=np.float32)
        self.step = 0
        self.v_ego = np.zeros((100000, 1), dtype=np.float32)
        self.v_prec = np.zeros((100000, 1), dtype=np.float32)
        self.headway = np.zeros((100000, 1), dtype=np.float32)
        self.mode_ego = np.zeros((100000, 1), dtype=np.float32)
        self.distance = np.zeros((100000, 1), dtype=np.float32)
        #self.ID_prec_previous = 'none'
        self.Collision = False
        self.RouteEnd = False
        self.endstate = False
        self.a_hybrid_a = np.zeros((100000, 1), dtype=np.float32)
        self.v_set = np.zeros((100000, 1), dtype=np.float32)
        self.v_set_nonoise = np.zeros((100000, 1), dtype=np.float32)

    def init_vars_episode(self):
        self.step = 0
        self.v_ego[:] = 0.
        self.v_prec[:] = 0.
        self.headway[:] = 0.
        self.mode_ego[:] = 0.
        self.distance[:] = 0.
        self.Collision = False
        self.RouteEnd = False
        self.endstate = False
        self.a_set[:] = 0.
        self.a_real[:] = 0.
        self.a_hybrid_a[:] = 0.
        self.v_set[:] = 0.

    def create_v_profile_prec(self, w=0.06, a=10/3.6, c=30/3.6, profile='sin'):
        """Creates the velocity profile for the preceding car (currently only the sinusodial profile implemented)
        w: frequency of speed profile
        a: amplitude of speed profile
        c: offset of speed profile
        """
        t = np.arange(1., self.sim['duration'])
        if profile == 'sin':
            self.v_profile = a*np.sin(w*t)+c
        else:
            print('No valid velocity profile for preceding car selected!')

    def create_v_profile_emerg_stop(self):
        a_brake = -7
        a_accel = 2.5
        self.v_profile = 10. * np.ones(int(self.sim['duration']))
        for jj in range(300, int(self.sim['duration'])):
            self.v_profile[jj] = self.v_profile[jj - 1] + a_brake * self.sim['timestep']
            if self.v_profile[jj] < 0:
                self.v_profile[jj] = 0
        for jj in range(600, int(self.sim['duration'])):
            self.v_profile[jj] = self.v_profile[jj - 1] + a_accel * self.sim['timestep']
        self.v_profile = np.clip(self.v_profile, None, 20.)

    def postproc_v(self):
        self.v_prec[self.v_prec < 0] = 0
        self.v_prec = self.v_prec[:, 0]
        self.v_prec[0] = self.v_prec[1]
        self.a_car = np.gradient(self.v_prec) / self.sim['timestep']
        self.time = np.arange(0., self.v_prec.size) * self.sim['timestep']
        self.v_ego[self.v_ego < 0] = 0
        self.v_ego = self.v_ego[:, 0]
        self.v_ego[0] = self.v_ego[1]
        self.a_ego = np.gradient(self.v_ego) / self.sim['timestep']

class features():
    """Input variables (features) for the controller"""
    v_ego = np.NaN
    v_prec = np.NaN
    distance = np.NaN
    headway = np.NaN
    distance_TLS = np.NaN
    v_allowed = np.NaN