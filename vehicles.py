import scipy.io
from scipy import interpolate
import numpy as np


class Vehicle():
    """TODO: Description"""
    def __init__(self, ego, ID, RouteID, Route, powertrain_concept=None, battery_SOC_start=0.7, mass=1200,c_rolling=0.0075,c_w=0.3,front_area=2.2,rot_inertia=1.1,rolling_radius=0.3):
        self.ID = ID
        self.RouteID = RouteID
        self.Route = Route
        self.end = False
        self.depart_speed = 0
        if ego:
            self.mass = mass
            self.c_rolling = c_rolling
            self.c_w = c_w
            self.front_area = front_area
            self.rot_inertia = rot_inertia
            self.rolling_radius = rolling_radius
            self.ID_prec = 'init'
            self.ID_prec_previous = 'init'
            self.fuel_cons = np.zeros((100000, 1), dtype=np.float32)
            self.fuel_cons_ECMS = np.zeros((100000, 1), dtype=np.float32)
            self.fuel_cons_ECMS_per_s = np.zeros((100000, 1), dtype=np.float32)
            self.density_fuel = 748.  # density super in g/l
            self.concept = powertrain_concept
            if powertrain_concept == 'ICEV':
                self.architecture = {'ICE': 1, 'EM': 0, 'TM': 1, 'eTM': 0, 'kombTM': 0, 'battery': 0}
            elif powertrain_concept == 'BEV':
                self.architecture = {'ICE': 0, 'EM': 1, 'TM': 0, 'eTM': 1, 'kombTM': 0, 'battery': 1}
            elif powertrain_concept == 'P2HEV':
                self.architecture = {'ICE': 1, 'EM': 1, 'TM': 0, 'eTM': 1, 'kombTM': 1, 'battery': 1}
            else:
                self.architecture = []
            self.n_starting = 2000  # speed of ICE when starting with ICEV
            self.ICE = scipy.io.loadmat('files/ICE8.mat')  # loads the matlab struct as a python dictionary with numpy array entries
            self.ICE['M_max_interp_fnc'] = interpolate.interp1d(np.reshape(self.ICE['grid_speed'], (-1,)), self.ICE['max_torque'], axis=0, bounds_error=False)
            self.ICE['bsfc_interp_fnc'] = interpolate.interp2d(np.reshape(self.ICE['grid_torque'], (-1,)), np.reshape(self.ICE['grid_speed'], (-1,)), self.ICE['bsfc'], bounds_error=False, fill_value=np.nan)
            self.EM = scipy.io.loadmat('files/EM.mat')  # EM.mat = DE-REX EM -- loads the matlab struct as a python dictionary with numpy array entries
            self.EM['M_max_interp_fnc'] = interpolate.interp1d(np.reshape(self.EM['grid_speed'], (-1,)), self.EM['max_torque'], bounds_error=False)
            #self.EM['eta_interp_fnc'] = interpolate.interp2d(np.reshape(self.EM['grid_torque'], (-1,)), np.reshape(self.EM['grid_speed'], (-1,)), self.EM['efficiency'])
            self.EM['eta_interp_fnc'] = interpolate.RectBivariateSpline(np.reshape(self.EM['grid_speed'], (-1,)), np.reshape(self.EM['grid_torque'], (-1,)),
                                                             self.EM['efficiency'])
            # TODO: implement arbitrary transmission parametrisation
            self.kombTM = {'ratios': np.array([15.3, 9.0, 5.7, 4.0, 3.1, 2.5]), 'n_gears': 6, 'eta': 0.96, 'mass': 110.}
            self.eTM = {'ratios': np.array([10.]), 'n_gears': 1, 'eta': 0.995, 'mass': 15.}
            self.TM = {'ratios': np.array([1]), 'n_gears': 1, 'eta': 1, 'mass': 0.}
            self.number_gear_comb = self.kombTM['n_gears'] * self.eTM['n_gears'] * self.TM['n_gears']
            kombTM_gears_mtrx, eTM_gears_mtrx, TM_gears_mtrx = np.meshgrid(self.kombTM['ratios'], self.eTM['ratios'], self.TM['ratios'])
            self.kombTM['gears_vector'] = np.ravel(kombTM_gears_mtrx)
            self.eTM['gears_vector'] = np.ravel(eTM_gears_mtrx)
            self.TM['gears_vector'] = np.ravel(TM_gears_mtrx)
            self.battery = {'SOC': battery_SOC_start, 'SOC_target': battery_SOC_start, 'eta': 0.95, 'capacity': 3600000.*20}  # only charge sustaining mode for hybrids, identical charge and discharge efficiency of the battery


