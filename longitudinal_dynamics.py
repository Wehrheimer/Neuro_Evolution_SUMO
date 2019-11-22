import numpy as np
from helper_functions import *
from scipy import interpolate
from numpy import matlib
from copy import copy
from SUMO import features


class Longitudinal_dynamics():
    """Class for the longitudinal dynamics. Consists of relevant variables for dynamics and the methods to calculate
     wheel demand / operating strategy and the simulated transfer function for the lower level controller dynamics a_real = f(a_set)"""
    def __init__(self, tau=0.5, g=9.81, rho=1.2041):
        self.slope = 0  # road slope in rad
        self.g = g  # gravitational constant in m/s^2
        self.rho = rho  # air density in kg/m^3
        self.M_trac = np.zeros((100000,))
        self.n_trac = np.zeros((100000,))
        self.a_previous = 0.
        self.a_real = np.zeros((100000, 1))
        self.a_set = np.zeros((100000, 1))
        self.M_ICE_opt = np.zeros((100000,))
        self.n_ICE_opt = np.zeros((100000,))
        self.E_ICE = np.zeros((100000,))
        self.M_EM_opt = np.zeros((100000,))
        self.n_EM_opt = np.zeros((100000,))
        self.E_EM = np.zeros((100000,))
        self.kombTM_gear_opt = np.zeros((100000,))
        self.eTM_gear_opt = np.zeros((100000,))
        self.TM_gear_opt = np.zeros((100000,))
        self.tau = tau  # time constant for low level controller lag in s
        self.v_real_next = []
        self.fuel_cons_per_100km = []
        self.fuel_cons_per_s = []

    def wheel_demand(self, v, vehicle, step):
        """Calculate the traction demand at the wheels by calculating driving resistances"""
        # rolling resistance force
        F_r = vehicle.c_rolling * vehicle.mass * self.g * np.cos(self.slope)  # in N
        # slope force
        F_slope = vehicle.mass * self.g * np.sin(self.slope)  # in N
        # inertial force
        F_i = vehicle.rot_inertia * vehicle.mass * self.a_real  # in N
        # drag force
        F_d = vehicle.c_w * vehicle.front_area * self.rho/2 * np.power(v,2)  # in N
        # total traction force,torque,speed
        F_trac = F_r + F_slope + F_i + F_d  # in N
        self.M_trac[step] = F_trac * vehicle.rolling_radius  # in Nm
        self.n_trac[step] = 60/(2*np.pi) * v / vehicle.rolling_radius  # in 1/min

    def low_lev_controller(self, a_set, timestep):
        """PT1 behaviour to model low level controller (e.g. powertrain inertia)"""
        self.a_real = (a_set + self.tau * self.a_previous / timestep) / (1 + self.tau/timestep)

    def reset_variables(self):
            self.M_trac[:] = 0.
            self.n_trac[:] = 0.
            self.a_previous = 0
            self.a_real = 0.
            self.a_set[:] = 0.
            self.M_ICE_opt[:] = 0.
            self.n_ICE_opt[:] = 0.
            self.E_ICE[:] = 0.
            self.M_EM_opt[:] = 0.
            self.n_EM_opt[:] = 0.
            self.E_EM[:] = 0.
            self.kombTM_gear_opt[:] = 0
            self.eTM_gear_opt[:] = 0
            self.TM_gear_opt[:] = 0

    def operating_strategy(self, timestep, vehicle, s0, kp, step, M_ICE_reso=200):
        """Calculate the operating point of the energy converters of 1 timestep by choosing the gear (and the torque split for hybrid vehicles)
        The operating strategy implemented is the A-ECMS, choosing locally optimal operating points to minimize an equivalent fuel mass flow rate
        Input are the demanded torque and speed at the wheels"""
        ICE_breaking = False
        clutch_open = False
        if self.n_trac[step] == 0:
            kombTM_gear_opt = 0
            eTM_gear_opt = 0
            TM_gear_opt = 0
            M_ICE_opt = 0
            n_ICE_opt = 0 + strcmp('ICEV', vehicle.concept) * vehicle.ICE['speed_idle'][0]
            M_EM_opt = 0
            n_EM_opt = 0
            E_ICE = 0 + strcmp('ICEV', vehicle.concept) * vehicle.ICE['consumption_idle'][0] * vehicle.ICE['lhv'][0] * timestep
            E_EM = 0
        else:
            s = s0 + kp * (vehicle.battery['SOC_target'] - vehicle.battery['SOC'])

            """ICE"""
            if vehicle.architecture['ICE'] == 1:
                n_ICE = self.n_trac[step] * vehicle.kombTM['gears_vector'] * vehicle.TM['gears_vector']
                if strcmp('ICEV', vehicle.concept):
                    M_ICE_reso = 1
                    if np.amax(n_ICE) < np.amin(vehicle.ICE['grid_speed']):
                        if self.M_trac[step] > 0:
                            n_ICE = np.ones((vehicle.number_gear_comb,)) * vehicle.n_starting
                        else:
                            n_ICE = np.ones((vehicle.number_gear_comb,)) * vehicle.ICE['speed_idle'][
                                0]  # vehicle decelerating to zero below min possible ICE speed --> idle speed
                            clutch_open = True
                    if self.M_trac[step] < 0:  # modelling of mechanical braking, torque of ICE during mechanical braking is zero (clutch is closed)
                        M_ICE = np.zeros((vehicle.number_gear_comb, 1))
                        ICE_breaking = True
                    else:
                        M_ICE = 1 / vehicle.TM['eta'] * 1 / vehicle.kombTM['eta'] * self.M_trac[step] * 1 / (
                                    vehicle.kombTM['gears_vector'] * vehicle.TM['gears_vector'])
                    M_ICE_mtrx = np.reshape(M_ICE, (-1, 1))  # (vehicle.number_gear_comb x 1 ) matrix
                    n_ICE_mtrx = np.reshape(n_ICE, (-1, 1))  # (vehicle.number_gear_comb x 1 ) matrix

                    """definition of penalty matrix for M_ICE > M_ICE_max"""
                    M_ICE_max = vehicle.ICE['M_max_interp_fnc'](np.reshape(n_ICE, (-1,)))
                    M_ICE_max[np.isnan(M_ICE_max)] = 0
                    # M_ICE_max_mtrx = matlib.repmat(M_ICE_max, 1, np.size(M_ICE))
                    M_ICE_diff = M_ICE_mtrx - M_ICE_max
                    M_ICE_penalty_vector = np.ones((np.size(M_ICE_diff), 1))
                    M_ICE_penalty_vector[M_ICE_diff > 0] = 1000

                    """penalty when speed of ICE larger than max possible ICE speed"""
                    M_ICE_penalty_vector[n_ICE_mtrx > np.amax(vehicle.ICE['grid_speed'])] = M_ICE_penalty_vector[n_ICE_mtrx > np.amax(
                        vehicle.ICE['grid_speed'])] + 2000
                    M_ICE_penalty_matrix = M_ICE_penalty_vector  # (vehicle.number_gear_comb x 1) matrix
                    M_ICE_penalty_matrix_constant = np.zeros(M_ICE_mtrx.shape)

                    """bsfc value for operating points M_ICE / n_ICE"""
                    bsfc = np.zeros(np.shape(M_ICE_mtrx))
                    for ii in range(np.size(M_ICE_mtrx)):
                        bsfc[ii, 0] = vehicle.ICE['bsfc_interp_fnc'](M_ICE[ii], n_ICE[ii])
                    # bsfc = copy(vehicle.ICE['bsfc_interp_fnc'](M_ICE, n_ICE).diagonal().reshape((-1, 1)))
                    bsfc[n_ICE == vehicle.ICE['speed_idle'][0]] = 0
                else:
                    M_ICE = np.linspace(0., np.amax(vehicle.ICE['max_torque']),
                                        M_ICE_reso)  # M_ICE varied in (M_ICE_reso)-number of steps from 0 to max M_ICE
                    M_ICE_mtrx = matlib.repmat(M_ICE, np.size(n_ICE), 1)
                    n_ICE_mtrx = matlib.repmat(np.reshape(n_ICE, (-1, 1)), 1, np.size(M_ICE))

                    """penalty matrix for M_ICE > M_ICE_max and n_ICE > n_ICE_max"""
                    M_ICE_max = vehicle.ICE['M_max_interp_fnc'](n_ICE)
                    M_ICE_max_mtrx = matlib.repmat(M_ICE_max, 1, np.size(M_ICE))
                    M_ICE_diff = M_ICE_mtrx - M_ICE_max_mtrx

                    M_ICE_penalty_matrix = np.ones(M_ICE_diff.shape)
                    M_ICE_penalty_matrix[M_ICE_diff > 0] = 1000
                    M_ICE_penalty_matrix[n_ICE_mtrx > np.amax(vehicle.ICE['grid_speed'])] = M_ICE_penalty_matrix[n_ICE_mtrx > np.amax(
                        vehicle.ICE['grid_speed'])] + 2000
                    M_ICE_penalty_matrix_constant = np.zeros((M_ICE_diff.shape))
                    M_ICE_penalty_matrix_constant[n_ICE_mtrx > np.amax(vehicle.ICE['grid_speed'])] = 2000

                    """bsfc for operating points M_ICE/n_ICE"""
                    bsfc = vehicle.ICE['bsfc_interp_fnc'](M_ICE, n_ICE)
                    bsfc[:, 0] = 0  # overwrite Nans from interpolation at M_ICE = 0
            else:
                s = 1
                M_ICE = np.zeros((1, M_ICE_reso))
                M_ICE_mtrx = np.zeros((vehicle.number_gear_comb, M_ICE_reso))

            """EDM"""
            if vehicle.architecture['EM'] == 1:
                n_EM = self.n_trac[step] * vehicle.kombTM['gears_vector'] * vehicle.eTM['gears_vector']
                n_EM_mtrx = matlib.repmat(n_EM.reshape((-1, 1)), 1, np.size(M_ICE))
                M_trac_vector = self.M_trac[step] * np.ones((np.size(M_ICE),))

                """M_EM, the difference between M_trac and M_ICE"""
                if self.M_trac[step] < 0:
                    M_EM = np.outer(vehicle.kombTM['eta'] * vehicle.eTM['eta'] / (vehicle.eTM['gears_vector'] * vehicle.kombTM['gears_vector']),
                                    M_trac_vector) - np.outer(
                        1 / (vehicle.eTM['eta'] * vehicle.eTM['gears_vector'] * (1 / (vehicle.TM['eta'] * vehicle.TM['gears_vector']))),
                        M_ICE)  # (vehicle.number_gear_comb x M_ICE_reso) matrix
                else:
                    M_EM = np.outer(
                        1 / ((vehicle.eTM['eta'] * vehicle.eTM['gears_vector']) * (vehicle.kombTM['eta'] * vehicle.kombTM['gears_vector'])),
                        M_trac_vector) - np.outer(
                        1 / (vehicle.eTM['eta'] * vehicle.eTM['gears_vector'] * (1 / (vehicle.TM['eta'] + vehicle.TM['gears_vector']))),
                        M_ICE)  # (vehicle.number_gear_comb x M_ICE_reso) matrix

                """Penalty matrix when M_EM > M_EM_max"""
                M_EM_max = vehicle.EM['M_max_interp_fnc'](n_EM).reshape((-1, 1))
                M_EM_max_mtrx = matlib.repmat(M_EM_max, 1, M_ICE_reso)
                M_EM_penalty_mtrx = np.ones((vehicle.number_gear_comb, M_ICE_reso))
                M_EM_penalty_mtrx[M_EM > M_EM_max_mtrx] = 20000

                """Efficiency of EM for operating points M_EM/n_EM"""
                eta_EM = vehicle.EM['eta_interp_fnc'](M_EM.reshape((-1,)), n_EM_mtrx.reshape((-1,)),
                                                      grid=False)  # (vehicle.number_gear_comb x M_ICE_reso) matrix
                eta_EM = np.reshape(eta_EM, (vehicle.number_gear_comb, M_ICE_reso))
                eta_EM[n_EM_mtrx > np.amax(vehicle.EM['grid_speed'])] = np.nan
                eta_EM[abs(M_EM) > np.amax(vehicle.EM['grid_torque'])] = np.nan
                eta_EM[M_EM * n_EM_mtrx < 0] = 1 / eta_EM[M_EM * n_EM_mtrx < 0]

                """Build Matrix for Efficiency of battery"""
                eta_battery_mtrx = np.ones((vehicle.number_gear_comb, M_ICE_reso)) * vehicle.battery['eta']
                eta_battery_mtrx[M_EM * n_EM_mtrx < 0] = 1 / eta_battery_mtrx[M_EM * n_EM_mtrx < 0]
            else:
                M_EM = np.zeros((vehicle.number_gear_comb, M_ICE_reso))

            """A-ECMS Cost function"""
            if vehicle.architecture['ICE'] == 1:
                J_ICE = 1 / 3600000 * bsfc * n_ICE_mtrx * M_ICE_mtrx * M_ICE_penalty_matrix + M_ICE_penalty_matrix_constant
            else:
                J_ICE = np.zeros((vehicle.number_gear_comb, M_ICE_reso))

            if vehicle.architecture['EM'] == 1:
                J_EM = s * 1 / vehicle.ICE['lhv'][0] * n_EM_mtrx * M_EM * M_EM_penalty_mtrx * 1 / eta_EM * 1 / eta_battery_mtrx
            else:
                J_EM = np.zeros((vehicle.number_gear_comb, M_ICE_reso))
            J = J_ICE + J_EM

            # TODO: Routine when no operating point in grid
            if np.all(np.isnan(J)):
                # raise Exception('Traction demand too high for powertrain (grid)')
                self.M_ICE_opt[step] = 0
                self.n_ICE_opt[step] = 0
                self.E_ICE[step] = 10000
                self.M_EM_opt[step] = 0
                self.n_EM_opt[step] = 0
                self.E_EM[step] = 10000
                self.kombTM_gear_opt[step] = 0
                self.eTM_gear_opt[step] = 0
                self.TM_gear_opt[step] = 0
                print('OS: No operating point in grid')
                return

            """Search operating point with minimal cost"""
            gear_opt_index, M_ICE_opt_index = np.unravel_index(np.nanargmin(J), J.shape)
            kombTM_gear_opt, eTM_gear_opt, TM_gear_opt = np.unravel_index(gear_opt_index,
                                                                          (vehicle.kombTM['n_gears'], vehicle.eTM['n_gears'], vehicle.TM['n_gears']))

            M_ICE_opt = M_ICE_mtrx[gear_opt_index, M_ICE_opt_index]
            if M_ICE_opt == 0 and not ICE_breaking:  # electric-only driving
                n_ICE_opt = 0
            else:
                n_ICE_opt = n_ICE[gear_opt_index]

            if vehicle.architecture['ICE'] == 1:
                bsfc_opt = bsfc[gear_opt_index, M_ICE_opt_index]
                if n_ICE_opt == vehicle.ICE['speed_idle'][0]:
                    E_ICE = strcmp(vehicle.concept, 'ICEV') * vehicle.ICE['consumption_idle'][0] * vehicle.ICE['lhv'][0] * timestep  # in J
                else:
                    E_ICE = 2 * np.pi / (1000 * 3600 * 60) * vehicle.ICE['lhv'][0] * bsfc_opt * n_ICE_opt * M_ICE_opt * timestep  # in J
            else:
                n_ICE_opt = 0
                E_ICE = 0

            if vehicle.architecture['EM'] == 1:
                M_EM_opt = M_EM[gear_opt_index, M_ICE_opt_index]
                n_EM_opt = n_EM[gear_opt_index]
                eta_EM_opt = eta_EM[gear_opt_index, M_ICE_opt_index]
                if M_EM_opt > M_EM_max_mtrx[
                    gear_opt_index, M_ICE_opt_index]:  # TODO: Routine when all operating points above max torque curve but inside grid
                    # raise Exception('Traction demand too high for powertrain (map)')
                    self.M_ICE_opt[step] = 0
                    self.n_ICE_opt[step] = 0
                    self.E_ICE[step] = 10000
                    self.M_EM_opt[step] = 0
                    self.n_EM_opt[step] = 0
                    self.E_EM[step] = 10000
                    self.kombTM_gear_opt[step] = 0
                    self.eTM_gear_opt[step] = 0
                    self.TM_gear_opt[step] = 0
                    print('OS: All operating points above max torque curve')
                    return
                M_EM_recup_max = -M_EM_max[gear_opt_index]
                if M_EM_opt < M_EM_recup_max:
                    M_EM_opt = M_EM_recup_max
                E_EM = 2 * np.pi / 60 * n_EM_opt * M_EM_opt * 1 / eta_EM_opt * 1 / eta_battery_mtrx[gear_opt_index, M_ICE_opt_index] * timestep
            else:
                M_EM_opt = 0
                n_EM_opt = 0
                E_EM = 0
        vehicle.battery['SOC'] -= E_EM / vehicle.battery['capacity']

        self.M_ICE_opt[step] = M_ICE_opt
        self.n_ICE_opt[step] = n_ICE_opt
        self.E_ICE[step] = E_ICE
        self.M_EM_opt[step] = M_EM_opt
        self.n_EM_opt[step] = n_EM_opt
        self.E_EM[step] = E_EM
        self.kombTM_gear_opt[step] = kombTM_gear_opt
        self.eTM_gear_opt[step] = eTM_gear_opt
        self.TM_gear_opt[step] = TM_gear_opt
        if self.n_trac[step] == 0 or clutch_open:
            self.fuel_cons_per_100km = np.NaN
        else:
            self.fuel_cons_per_100km = self.E_ICE[step] * 100000. / (timestep * vehicle.ICE['lhv'] * vehicle.density_fuel * features.v_ego)  # in l/100km
        self.fuel_cons_per_s = self.E_ICE[step] / (timestep * vehicle.ICE['lhv'] * vehicle.density_fuel/1000)  # in ml/s

        # del M_ICE_opt, n_ICE_opt, E_ICE, M_EM_opt, n_EM_opt, E_EM, kombTM_gear_opt, eTM_gear_opt, TM_gear_opt










