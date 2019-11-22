import numpy as np
from scipy import interpolate


class ACC_Controller:
    """Calculate the set acceleration based on the input features and control variables
    (integral of error, previous error)"""
    def __init__(self, v_set, h_set, a_set_max=2., a_set_min=-7., timestep=0.2):
        self.v_set = v_set  # in m/s -- for constant speed
        self.h_set = h_set  # in s -- headway
        self.lmbda = 0.15  # controller parameter
        self.L_des = 3  # in m -- desired stationary spacing at v=0
        self.R_sensor_max = 150  # in m -- parameter for switching trajectory dimensioning: max range of (radar) sensor to detect vehicles
        self.D = 0.5  # in m/s^2 -- parameter for switching trajectory dimensioning: coasting deceleration
        self.mode = []
        self.mode_switching = []
        self.mode_switching_next = []
        self.a_set_max = a_set_max
        self.a_set_min = a_set_min

        self.timestep = timestep
        self.previous_error = 0
        self.mode_previous = 0
        self.integral_error = 0
        self.CS_k_p = 0.2  # constant speed P-factor
        self.CS_k_i = 0.00  # 0.02  - constant speed I-factor
        self.CS_k_d = 0  # constant speed D-factor
        self.CH_k_p = 0.05  # constant headway P-factor
        self.CH_k_d = 0.00  # 0.01  - constant headway D-factor
        self.CH_k_i = 0  # constant headway I-factor
        self.VS_k_p = 0.4  # variable speed P-factor  -- 0.3
        self.VS_k_i = 0  # variable speed I-factor -- 0
        self.v_slope = 10 # constant design velocity to determine the slope of the switching line (R-R_rate diagram)
        self.mode_map_2to1 = []
        self.mode_map_1to2 = []
        self.R_function = []

    def create_mode_map(self):
        """Create the mode map (mode as a function of distance and velocity difference)
        Mode1: Headway Control
        Mode2: Speed Control
        Mode1.5: Switching Trajectory
        Creates two Maps depending on the previous mode to enable a hysteresis --> to prevent mode oscillations from mode 1 to mode 2"""
        R_final = self.L_des + self.h_set * self.v_slope  # v_slope: constant design velocity to determine the slope of the switching line
        R_p_slope_calc = -np.sqrt((self.R_sensor_max - R_final) * 2 * self.D)
        T = (-self.R_sensor_max + R_final) / (0 - R_p_slope_calc)  # slope of switching line constant speed -> constant headway
        R_p_map = np.arange(-50., 50.)
        R_map = np.arange(0, 1000)
        self.mode_map_2to1 = np.ones((len(R_map), len(R_p_map)))  # mode map to use when in mode 2
        self.mode_map_1to2 = np.ones((len(R_map), len(R_p_map)))  # mode map to use when in mode 1
        R_swl = T * R_p_map + R_final
        for jj in range(len(R_p_map)):
            self.mode_map_2to1[R_map > R_swl[jj], jj] = 2
        for jj in range(len(R_p_map)):
            self.mode_map_1to2[R_map > R_swl[jj] + 15, jj] = 2  # hysteresis to prevent mode oscillations from mode 1 <-> mode 2
        self.mode_map_2to1 = interpolate.RectBivariateSpline(R_map, R_p_map, self.mode_map_2to1)
        self.mode_map_1to2 = interpolate.RectBivariateSpline(R_map, R_p_map, self.mode_map_1to2)
        self.R_function = interpolate.interp1d(R_p_map, R_swl)  # switching trajectory set distance function

    def calculate_a(self, features):
        """Calculate the set acceleration based on the mode (speed or headway control), the distance to preceding vehicle and the speed difference"""
        if self.mode_previous == 2:
            self.mode = self.mode_map_2to1.ev(features.distance, features.v_prec - features.v_ego)
        else:
            self.mode = self.mode_map_1to2.ev(features.distance, features.v_prec - features.v_ego)
        self.mode = int(np.round(self.mode))
        delta = -features.distance + self.L_des + self.h_set * features.v_ego

        if self.mode == 2:  # cs: constant speed
            if self.mode_previous != 2:  # reset integral error when switching from headway control to speed control
                self.integral_error = 0
            error_cs = features.v_ego - self.v_set
            self.integral_error = self.integral_error + error_cs * self.timestep
            derivative = 1/self.timestep * (error_cs - self.previous_error)
            a_set = -self.CS_k_p * error_cs - self.CS_k_i * self.integral_error + self.CS_k_d * derivative  # a: output control factor (set acceleration)
            self.previous_error = error_cs
            self.mode_switching_next = 2
        elif self.mode == 1:  # constant headway or switching trajectory
            if abs(features.v_ego - features.v_prec) <= 2:  # constant headway
                a_set = -1/self.h_set * (features.v_ego - features.v_prec + self.lmbda * delta)
                self.mode_switching_next = 1  # parameter to stop switching from constant headway to switching trajectory
            elif abs(features.v_ego - features.v_prec) >= 2 and self.mode_switching == 1:  # constant headway
                a_set = -1 / self.h_set * (features.v_ego - features.v_prec + self.lmbda * delta)
                self.mode_switching_next = 1
            elif self.mode_switching != 1:  # switching trajectory
                R = self.R_function(features.v_prec - features.v_ego)  # switching trajectory set distance
                error_sw = features.distance - R
                a_set = self.CH_k_p * error_sw + self.CH_k_d * (error_sw - self.previous_error) / self.timestep
                self.mode_switching_next = 1.5
                self.mode = 1.5
                self.previous_error = error_sw
        # limit set variable values
        if a_set > self.a_set_max:
            a_set = self.a_set_max
        elif a_set < self.a_set_min:
            a_set = self.a_set_min
        self.mode_switching = self.mode_switching_next
        self.mode_previous = self.mode
        return a_set

    def calc_a_P(self, features, v_set):
        self.v_set = v_set
        error_PI = self.v_set - features.v_ego
        self.integral_error += error_PI * self.timestep
        a_set = self.VS_k_p * error_PI + self.VS_k_i * self.integral_error
        a_set = np.clip(a_set, self.a_set_min, self.a_set_max)
        return a_set

    def reset_integral_error(self):
        self.integral_error = 0
