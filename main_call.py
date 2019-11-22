import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import timeit
from longitudinal_dynamics import Longitudinal_dynamics
from DQN import DQN
from DDPG import DDPG
from acc_controller import ACC_Controller
from copy import copy
#from scipy import io, interpolate
#import tkinter as tk
from tkinter import filedialog
from vehicles import Vehicle
from SUMO import SUMO, features
from helper_functions import strcmp, DataCursor, plot_results
matplotlib.use('TkAgg')
from math import *


# import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
import traci


def plot_running_init(training):
    plt.ion()
    if training:
        fig_running, (ax_running_1, ax_running_2, ax_running_3, ax_running_4) = plt.subplots(4, 1)
        ax_running_1.set_ylabel('Cum. Reward Mean 100')
        ax_running_2.set_xlabel('Episode')
        reward_mean100 = []
        ax_running_1.plot(reward_mean100)
        fig_running.show()
        return fig_running, ax_running_1, ax_running_2, ax_running_3, ax_running_4


def plot_running(reward_mean100, episode, cum_reward_evaluation):
    reward_mean100 = reward_mean100[:episode+1]
    observed_weights = nn_controller.observed_weights[:episode+1, :]
    critic_loss = nn_controller.critic_loss[nn_controller.warmup_time:nn_controller.step_counter+1]
    ax_running_1.clear()
    ax_running_1.set_ylabel('Cum. reward (mean100)')
    ax_running_1.plot(reward_mean100)
    ax_running_2.clear()
    ax_running_2.set_ylabel('Weights and biases')
    ax_running_2.set_xlabel('Episode')
    for ii in range(np.size(observed_weights, 1)):
        ax_running_2.plot(observed_weights[:, ii])
    ax_running_3.clear()
    ax_running_3.set_ylabel('Cum. reward (evaluation)')
    ax_running_3.set_xlabel('Evaluation episode')
    ax_running_3.plot(cum_reward_evaluation[1:])
    ax_running_4.clear()
    ax_running_4.set_ylabel('Critic Loss')
    #ax_running_4.set_xlabel('Evaluation episode')
    ax_running_4.plot(critic_loss)
    ax_running_1.autoscale_view()
    ax_running_2.autoscale_view()
    ax_running_3.autoscale_view()
    ax_running_4.autoscale_view()
    fig_running.canvas.flush_events()


def calculate_features_firststep():
    state = np.zeros([1, feature_number])
    sub_ego = traci.vehicle.getSubscriptionResults(vehicle_ego.ID)

    ## TLS Distance
    if traci.constants.VAR_NEXT_TLS in sub_ego and len(sub_ego[traci.constants.VAR_NEXT_TLS]) > 0:
        features.distance_TLS = sub_ego[traci.constants.VAR_NEXT_TLS][0][2]
        features.TLS_state = sub_ego[traci.constants.VAR_NEXT_TLS][0][3]
    else:
        features.distance_TLS = 1000  # TODO: Handling when no TLS ahead
        features.TLS_state = None

    ## v_ego
    if traci.constants.VAR_SPEED in sub_ego:
        SUMO.v_ego[SUMO.step] = sub_ego[traci.constants.VAR_SPEED]
    else:
        SUMO.v_ego[SUMO.step] = 0.
    features.v_ego = SUMO.v_ego[SUMO.step]

    ## Fuel Consumption
    if traci.constants.VAR_FUELCONSUMPTION in sub_ego:
        vehicle_ego.fuel_cons[SUMO.step] = sub_ego[traci.constants.VAR_FUELCONSUMPTION]
    else:
        vehicle_ego.fuel_cons[SUMO.step] = 0.

    ## distance, v_prec
    try:
        if traci.constants.VAR_LEADER in sub_ego:
            vehicle_ego.ID_prec, features.distance = sub_ego[traci.constants.VAR_LEADER]
            SUMO.distance[SUMO.step] = features.distance
            features.distance = np.clip(features.distance, None, 250.)
            features.v_prec = traci.vehicle.getSpeed(vehicle_ego.ID_prec)
            SUMO.v_prec[SUMO.step] = features.v_prec
            if features.distance == 250:
                features.v_prec = features.v_ego
        else:
            raise TypeError
    except TypeError:
        vehicle_ego.ID_prec = 'none'
        features.distance = 250
        SUMO.distance[SUMO.step] = features.distance
        SUMO.v_prec[SUMO.step] = features.v_ego
    if features.distance == 250:
        features.v_prec = copy(features.v_ego)

    ## v_allowed
    if traci.constants.VAR_LANE_ID in sub_ego:
        features.v_allowed = traci.lane.getMaxSpeed(sub_ego[traci.constants.VAR_LANE_ID])
    else:
        features.v_allowed = 33.33  # tempo limit set to 120 km/h when no signal received, unlikely to happen

    ## correct distance, v_prec with virtual TLS vehicle
    if TLS_virt_vehicle:
        if features.TLS_state == 'y' or features.TLS_state == 'r':
            if features.distance_TLS < features.distance:
                features.distance = copy(features.distance_TLS)
                features.v_prec = 0

    ## headway
    if features.v_ego < 0.1:
        features.headway = 10000.
    else:
        features.headway = features.distance / features.v_ego
    SUMO.headway[SUMO.step] = features.headway


def get_state():
    state = np.zeros([1, feature_number])
    if controller == 'DQN' or controller == 'hybrid_a' or controller == 'DDPG' or controller == 'DDPG_v':
        """feature space: distance, v_ego, v_prec"""
        state[0, 0] = features.distance  # features.distance / 250
        state[0, 1] = features.v_ego  # features.v_ego / 25
        state[0, 2] = features.v_prec  # features.v_prec / 25
    if controller == 'DDPG_v':
        state[0, 3] = features.v_allowed  # features.v_allowed / 25
    if sample_generation:
        """distance, v_ego, v_prec state space"""
        sample_generator.state[0, 0] = features.distance
        sample_generator.state[0, 1] = features.v_ego
        sample_generator.state[0, 2] = features.v_prec
    return state


def calculate_features():
    state = np.zeros([1, feature_number])
    sub_ego = traci.vehicle.getSubscriptionResults(vehicle_ego.ID)
    if SUMO.Collision:  # collision happened!
        features.distance = 0  # set the distance of the cars after a collision to 0
        features.v_prec = SUMO.v_prec[SUMO.step]  # set the velocity of the preceding car after a collision to the value of the previous timestep
        features.v_ego = SUMO.v_ego[SUMO.step]  # set the velocity of the ego car after a collision to the value of the previous timestep
    elif SUMO.RouteEnd:
        features.distance = SUMO.distance[SUMO.step]  # set the distance of the cars after preceding vehicle ends route to previous timestep
        features.v_prec = SUMO.v_prec[
            SUMO.step]  # set the velocity of the preceding car after preceding vehicle ends route to the value of the previous timestep
        features.v_ego = SUMO.v_ego[
            SUMO.step]  # set the velocity of the ego car after preceding vehicle ends route to the value of the previous timestep
    else:
        ## TLS Distance
        if traci.constants.VAR_NEXT_TLS in sub_ego and len(sub_ego[traci.constants.VAR_NEXT_TLS]) > 0:
            features.distance_TLS = sub_ego[traci.constants.VAR_NEXT_TLS][0][2]
            features.TLS_state = sub_ego[traci.constants.VAR_NEXT_TLS][0][3]
        else:
            features.distance_TLS = 1000  # TODO: Handling when no TLS ahead
            features.TLS_state = None

        ## v_ego
        features.v_ego = sub_ego[traci.constants.VAR_SPEED]

        ## fuel_consumption
        vehicle_ego.fuel_cons[SUMO.step + 1] = sub_ego[traci.constants.VAR_FUELCONSUMPTION]  # in ml/s
        vehicle_ego.fuel_cons_ECMS[SUMO.step + 1] = dynamics_ego.fuel_cons_per_100km
        vehicle_ego.fuel_cons_ECMS_per_s[SUMO.step + 1] = dynamics_ego.fuel_cons_per_s

        ## distance, v_prec
        try:
            if traci.constants.VAR_LEADER in sub_ego:
                vehicle_ego.ID_prec, features.distance = sub_ego[traci.constants.VAR_LEADER]
                SUMO.distance[SUMO.step + 1] = features.distance
                features.distance = np.clip(features.distance, None, 250.)
                features.v_prec = traci.vehicle.getSpeed(vehicle_ego.ID_prec)
                SUMO.v_prec[SUMO.step + 1] = features.v_prec
            else:
                raise TypeError
        except TypeError:
            features.distance = 250
            SUMO.distance[SUMO.step + 1] = features.distance
            SUMO.v_prec[SUMO.step + 1] = features.v_ego
            vehicle_ego.ID_prec = 'none'
        if features.distance == 250:
            features.v_prec = copy(features.v_ego)  # when no preceding car detected OR distance > 250 (clipped), set a 'virtual velocity' = v_ego

        ## correct distance, v_prec with virtual TLS vehicle
        if TLS_virt_vehicle:
            if features.TLS_state == 'y' or features.TLS_state == 'r':
                if features.distance_TLS < features.distance:
                    features.distance = copy(features.distance_TLS)
                    features.v_prec = 0

        ## headway
        if features.v_ego < 0.1:
            features.headway = 10000.
        else:
            features.headway = features.distance / features.v_ego

        ## v_allowed
        if traci.constants.VAR_LANE_ID in sub_ego:
            features.v_allowed = traci.lane.getMaxSpeed(sub_ego[traci.constants.VAR_LANE_ID])
        else:
            features.v_allowed = 33.33  # tempo limit set to 120 km/h when no signal received, unlikely to happen

    ## plotting variables
    SUMO.headway[SUMO.step + 1] = features.headway
    SUMO.v_ego[SUMO.step + 1] = features.v_ego


def run_control():
    """Initialise simulation ======================================================================================="""
    traci.route.add(vehicle_ego.RouteID, vehicle_ego.Route)
    if vehicle2_exist:
        traci.route.add(vehicle_2.RouteID, vehicle_2.Route)
    if vehicle3_exist:
        traci.route.add(vehicle_3.RouteID, vehicle_3.Route)
    traci.vehicletype.setSpeedFactor(typeID='traffic_vehicle', factor=5.0)
    cum_reward = np.zeros((number_episodes, 1))
    best_cum_reward = -1000000
    reward_mean100 = np.zeros((number_episodes, 1))
    length_episode = np.zeros((number_episodes, 1))
    data_export = np.zeros((number_episodes, 2))
    restart_step = 0  # counter for calculating the reset timing when the simulation time gets close to 24 days
    cum_reward_evaluation = [0]  # list for cum reward of evaluation episodes
    evaluation = False
    sub_ego = {}
    if training:
        vehicle_ego.depart_speed = np.random.randint(0, 30, size=number_episodes)
    else:
        vehicle_ego.depart_speed = ego_depart_speed
    traci.trafficlight.setProgram(tlsID='junction1', programID=TLS_ID)

    """Begin episode loop =============================================================================================="""
    for episode in range(number_episodes):
        try:  # for keyboard interrupt
            """Check if total simulation time is close to 24 days ======================================================"""
            # TraCI time inputs have a maximum value of ~24days --> restart SUMO to reset time
            if np.sum(length_episode[restart_step:])*SUMO.sim['timestep'] > 2000000:
                print('Almost 24 days of simulation time reached! Restarting SUMO and continue with next episode...')
                traci.close()
                traci.start(['sumo', '-c', 'SUMO_config.sumocfg'])
                traci.route.add(vehicle_ego.RouteID, vehicle_ego.Route)
                if vehicle2_exist:
                    traci.route.add(vehicle_2.RouteID, vehicle_2.Route)
                if vehicle3_exist:
                    traci.route.add(vehicle_3.RouteID, vehicle_3.Route)
                restart_step = episode
            print('Episode: ', episode, '/', number_episodes)

            """Initialise episode =================================================================================="""
            SUMO.init_vars_episode()
            dynamics_ego.reset_variables()
            if controller == 'DQN' or controller == 'DDPG' or controller == 'hybrid_a' or controller == 'DDPG_v':
                nn_controller.reset_variables()
            if controller == 'ACC' or controller == 'hybrid_a':
                acc_controller.create_mode_map()
            if exploration_policy == 'ACC':
                explo_policy.create_mode_map()
            if (controller == 'DDPG' or controller == 'hybrid_a' or controller == 'DDPG_v') and ((episode+1) % 5 == 0):  # perform an evaluation episode (without exploration noise) every x episodes to observe the cum reward progress
                evaluation = True

            """Anmerkung: Hier werden einige Variationen des Verkehrsszenarios für meine Trainingsepisoden definiert, wenn 'training = True'
            gesetzt ist. Im Fall 'training = False' oder 'evaluation = True' (Evaluierungsepisoden unter gleichen Randbedingungen) wird immer eine
            Episode mit gleichen Randbedingungen (z.B. Geschwindigkeitsprofil vorausfahrendes Fahrzeug) gesetzt"""
            if evaluation:
                traci.vehicle.add(vehicle_ego.ID, vehicle_ego.RouteID, departSpeed='0',
                                  typeID='ego_vehicle')  # Ego vehicle
                traci.trafficlight.setPhase('junction1', 0)  # set traffic light phase to 0 for evaluation (same conditions)
            else:
                traci.vehicle.add(vehicle_ego.ID, vehicle_ego.RouteID, departSpeed=np.array2string(vehicle_ego.depart_speed[episode]), typeID='ego_vehicle')  # Ego vehicle
                if not training:
                    traci.trafficlight.setPhase('junction1', 0)
            if training and not evaluation and vehicle3_exist:
                vehicle3 = np.random.choice([True, False], p=[0.95, 0.05])
                traci.lane.setMaxSpeed('gneE01_0', np.random.choice([8.33, 13.89, 19.44, 25.]))
                traci.lane.setMaxSpeed('gneE02_0', np.random.choice([8.33, 13.89, 19.44, 25.]))
                traci.lane.setMaxSpeed('startedge_0', np.random.choice([8.33, 13.89, 19.44, 25.]))
                SUMO.create_v_profile_prec(a=SUMO.prec_train_amplitude[episode], c=SUMO.prec_train_mean[episode])
            else:
                vehicle3 = vehicle3_exist
                traci.lane.setMaxSpeed('startedge_0', 13.89)  # 13.89
                traci.lane.setMaxSpeed('gneE01_0', 19.44)  # 19.44
                traci.lane.setMaxSpeed('gneE02_0', 13.89)  # 13.89
                traci.lane.setMaxSpeed('stopedge_0', 8.33)  # 8.33
            if vehicle2_exist:
                traci.vehicle.add(vehicle_2.ID, vehicle_2.RouteID, typeID='traffic_vehicle')  # preceding vehicle 1
            if vehicle3:
                traci.vehicle.add(vehicle_3.ID, vehicle_3.RouteID, typeID='traffic_vehicle')  # preceding vehicle 2
                if training and not evaluation:
                    traci.vehicle.moveTo(vehicle_3.ID, 'gneE01_0', np.random.rand()*240.)
                else:
                    traci.vehicle.moveTo(vehicle_3.ID, 'gneE01_0', 0.)

            traci.simulationStep()  # to spawn vehicles
            if controller != 'SUMO':
                traci.vehicle.setSpeedMode(vehicle_ego.ID, 16)  # only emergency stopping at red traffic lights --> episode ends
            if vehicle2_exist:
                traci.vehicle.setSpeedMode(vehicle_2.ID, 17)
            if vehicle3:
                traci.vehicle.setSpeedMode(vehicle_3.ID, 17)

            SUMO.currentvehiclelist = traci.vehicle.getIDList()

            # SUMO subscriptions
            traci.vehicle.subscribeLeader(vehicle_ego.ID, 10000)
            traci.vehicle.subscribe(vehicle_ego.ID, [traci.constants.VAR_SPEED, traci.constants.VAR_BEST_LANES, traci.constants.VAR_FUELCONSUMPTION,
                                                     traci.constants.VAR_NEXT_TLS, traci.constants.VAR_ALLOWED_SPEED, traci.constants.VAR_LANE_ID])

            """ Run episode ======================================================================="""
            while traci.simulation.getMinExpectedNumber() > 0:  # timestep loop

                """Get state for first iteration ==================================================================="""
                if SUMO.step == 0:
                    calculate_features_firststep()
                    if controller == 'DDPG_v' or controller == 'DDPG' or controller == 'DQN':
                        nn_controller.state = get_state()

                """Controller ======================================================================================
                Hier wird die Stellgröße des Reglers (z.B. Sollbeschleunigung) in Abhängigkeit der Reglereingänge (zusammengefasst in 'features' 
                für den ACC Controller bzw. 'nn_controller.state' für die Neuronalen Regler"""
                if controller == 'ACC':
                    SUMO.a_set[SUMO.step] = acc_controller.calculate_a(features)  # Calculate set acceleration with ACC controller
                    # Save ACC mode to SUMO class
                    SUMO.mode_ego[SUMO.step] = acc_controller.mode  # ACC control modes (speed control, headway control, etc)
                elif controller == 'DQN':
                    # Calculate Q-Values for all actions
                    nn_controller.action_value = nn_controller.model.predict(nn_controller.state)  # NN inputs: distance, delta_v
                    # Choose Action according to epsilon-greedy policy
                    nn_controller.index[SUMO.step] = nn_controller.choose_action(nn_controller.action_value, episode)
                    # Calculate the set acceleration for the timestep
                    nn_controller.a_set = nn_controller.actions[nn_controller.index[SUMO.step]]
                    SUMO.a_set[SUMO.step] = nn_controller.a_set
                elif controller == 'DDPG':
                    if training and not evaluation:
                        if exploration_policy == 'ACC':
                            SUMO.a_set[SUMO.step] = explo_policy.calculate_a(features)
                            SUMO.a_set[SUMO.step] = nn_controller.add_noise(SUMO.a_set[SUMO.step])
                        else:
                            SUMO.a_set[SUMO.step] = nn_controller.choose_action(copy(nn_controller.state), network='main', noise=True)
                    else:
                        SUMO.a_set[SUMO.step] = nn_controller.choose_action(copy(nn_controller.state), network='main', noise=False)
                elif controller == 'hybrid_a':
                    acc_controller.a_set = acc_controller.calculate_a(
                        features)  # Calculate set acceleration with ACC controller
                    # Save ACC mode to SUMO class
                    SUMO.mode_ego[SUMO.step] = acc_controller.mode  # ACC control modes (speed control, headway control, etc)
                    if SUMO.step == 0:
                        nn_controller.state[0, 3] = acc_controller.a_set
                    if training and not evaluation:
                        nn_controller.a_set = nn_controller.choose_action(copy(nn_controller.state), network='main', noise=True)
                    else:
                        nn_controller.a_set = nn_controller.choose_action(copy(nn_controller.state), network='main', noise=False)
                    SUMO.a_set[SUMO.step] = acc_controller.a_set + nn_controller.k_hybrid_a * nn_controller.a_set
                    SUMO.a_hybrid_a[SUMO.step] = nn_controller.a_set
                elif controller == 'DDPG_v':
                    if training and not evaluation:
                        SUMO.v_set[SUMO.step], SUMO.v_set_nonoise[SUMO.step] = nn_controller.choose_action(copy(nn_controller.state), network='main', features=features, noise=True)
                    else:
                        SUMO.v_set[SUMO.step] = nn_controller.choose_action(copy(nn_controller.state), network='main', noise=False)
                    SUMO.a_set[SUMO.step] = acc_controller.calc_a_P(features, SUMO.v_set[SUMO.step])

                """Longitudinal Dynamics ===========================================================================
                Hier werden Längsdynamische Größen des Egofahrzeugs berechnet. beispielsweise die Realbeschleunigung aus den Stellgrößen des Reglers oder der
                Kraftstoffverbrauch für ein definiertes Fahrzeug (aktuell konventionelles Verbrennerfahrzeug mit Mehrganggetriebe). Am Ende wird als
                Schnittstelle zu SUMO eine neue Geschwindigkeit für das Egofahrzeug für den nächsten Zeitschritt gesetzt"""
                if controller == 'SUMO':
                    dynamics_ego.a_real = traci.vehicle.getAcceleration(vehicle_ego.ID)
                    dynamics_ego.wheel_demand(SUMO.v_ego[SUMO.step], vehicle_ego,
                                              SUMO.step)  # Calculate the torque and speed wheel demand of this timestep
                    dynamics_ego.operating_strategy(SUMO.sim['timestep'], vehicle_ego, s0=2, kp=10, step=SUMO.step)
                    SUMO.a_real[SUMO.step] = dynamics_ego.a_real
                else:
                    dynamics_ego.low_lev_controller(SUMO.a_set[SUMO.step], SUMO.sim['timestep'])  # Calculate a_real with a_set and a PT1 Transfer function
                    dynamics_ego.wheel_demand(SUMO.v_ego[SUMO.step], vehicle_ego, SUMO.step)  # Calculate the torque and speed wheel demand of this timestep
                    dynamics_ego.operating_strategy(SUMO.sim['timestep'], vehicle_ego, s0=2, kp=10, step=SUMO.step)
                    dynamics_ego.v_real_next = SUMO.v_ego[SUMO.step] + dynamics_ego.a_real * SUMO.sim['timestep']
                    dynamics_ego.v_real_next = np.clip(dynamics_ego.v_real_next, 0., None)
                    SUMO.a_real[SUMO.step] = dynamics_ego.a_real
                    traci.vehicle.setSpeed(vehicle_ego.ID, dynamics_ego.v_real_next)  # Set velocity of ego car for next time step

                """Control traffic ================================================================"""
                if vehicle2_exist and vehicle_2.ID in SUMO.currentvehiclelist:
                    if vehicle_2.end == False:
                        if traci.vehicle.getLaneID(vehicle_2.ID) == 'junction_out_0' and traci.vehicle.getLanePosition(vehicle_2.ID) > 90:
                            traci.vehicle.remove(vehicle_2.ID)
                            vehicle_2.end = True
                        else:
                            #traci.vehicle.setSpeed(vehicle_2.ID, SUMO.v_profile[SUMO.step])  # set velocity of preceding car 1
                            pass
                if vehicle3 and vehicle_3.ID in SUMO.currentvehiclelist:
                    traci.vehicle.setSpeed(vehicle_3.ID, SUMO.v_profile[SUMO.step])  # set velocity of preceding car 2

                """SUMO simulation step ============================================================================"""
                traci.simulationStep()
                SUMO.currentvehiclelist = traci.vehicle.getIDList()

                """Check if any of the endstate conditions is true (e.g. collision) ==================================="""
                if vehicle_ego.ID not in SUMO.currentvehiclelist:
                    SUMO.RouteEnd = True
                    SUMO.endstate = True
                    print('Route finished!')
                elif traci.simulation.getCollidingVehiclesNumber() > 0:
                    SUMO.endstate = True
                    SUMO.Collision = True
                    print('Collision!')
                elif vehicle_ego.ID in traci.simulation.getEmergencyStoppingVehiclesIDList():  # Check for ego vehicle passing red traffic light
                    SUMO.Collision = True
                    SUMO.endstate = True
                    print('Red lights passed!')
                # Set a maximum time step limit for 1 episode
                elif SUMO.step > 5000:
                    SUMO.RouteEnd = True
                    SUMO.endstate = True
                    print('Maximum time reached!')

                """get new state ==================================================================================="""
                calculate_features()
                if controller == 'DDPG_v' or controller == 'DDPG' or controller == 'DQN':
                    nn_controller.new_state = get_state()

                """Calculate Reward ================================================================================
                Dummy function"""
                reward = features.v_ego-100*np.sign(features.v_allowed-features.v_ego)-dynamics_ego.a_real*10-1/((features.v_ego*3.6-features.distance)**2)-1000*int(SUMO.Collision)
                cum_reward[episode] += reward

                """Add new sample to Experience Replay Data Base ===================================================
                Sammlung der trainigsrelevanten Größen 'state, action, new_state, reward' in einem großen Array. Ggfs. irrelevant für dich"""
                if not evaluation:
                    if training and controller == 'DQN':
                        nn_controller.add_sample(copy(nn_controller.state), nn_controller.index[SUMO.step],
                                                 copy(nn_controller.new_state), reward, SUMO.endstate, episode)

                    elif training and controller == 'DDPG':
                        nn_controller.add_sample(copy(nn_controller.state), SUMO.a_set[SUMO.step],
                                                 copy(nn_controller.new_state), reward, SUMO.endstate, episode)

                    elif training and controller == 'hybrid_a':
                        nn_controller.add_sample(copy(nn_controller.state), SUMO.a_hybrid_a[SUMO.step],
                                                 copy(nn_controller.new_state), reward, SUMO.endstate, episode)
                    elif training and controller == 'DDPG_v':
                        nn_controller.add_sample(copy(nn_controller.state), SUMO.v_set[SUMO.step],
                                                 copy(nn_controller.new_state), reward, SUMO.endstate, episode)

                """create experience replay for supervised pretraining of networks"""
                if sample_generation:
                    sample_generator.add_sample(copy(sample_generator.state), SUMO.a_set[SUMO.step],
                                                copy(sample_generator.new_state), reward, SUMO.endstate, episode)
                    sample_generator.step_counter += 1

                """Training step - update NNs =====================================================================================
                Optimierungsschritte für die gradientenbasierten RL ALgorithmen - irrelevant für dich"""
                if training and not evaluation:
                    if nn_controller.step_counter > nn_controller.warmup_time and nn_controller.step_counter % nn_controller.update_frequency == 0:
                        if controller == 'DQN' or controller == 'hybrid_a' or controller == 'DDPG' or controller == 'DDPG_v':
                            """Create a random minibatch for updating Q ==============================="""
                            nn_controller.create_minibatch()

                        if controller == 'DQN':
                            nn_controller.update_Q(epochs=1, discount_factor=0.95, reward=nn_controller.minibatch[:, 3],
                                                   state=nn_controller.minibatch[:, 0],
                                                   state_new=nn_controller.minibatch[:, 2],
                                                   a_index=nn_controller.minibatch[:, 1], endstate=nn_controller.minibatch[:, 4])

                        elif controller == 'DDPG' or controller == 'hybrid_a' or controller == 'DDPG_v':
                            nn_controller.update_critic(epochs=1, discount_factor=0.95, reward=nn_controller.minibatch[:, 3],
                                                   state=nn_controller.minibatch[:, 0],
                                                   state_new=nn_controller.minibatch[:, 2],
                                                   action=nn_controller.minibatch[:, 1], endstate=nn_controller.minibatch[:, 4])
                            action_ddpg = nn_controller.choose_action(nn_controller.minibatch[:, 0], network='main', noise=False)
                            dQda_ddpg = nn_controller.evaluate_dQda(states=nn_controller.minibatch[:, 0], actions=action_ddpg)
                            nn_controller.update_actor(states=nn_controller.minibatch[:, 0], dQda=dQda_ddpg)
                    """Update Target Network every N (freeze_rate) steps=============================================================="""
                    if nn_controller.step_counter % (nn_controller.freeze_rate * nn_controller.update_frequency) == 0:
                        nn_controller.update_target_network()
                    nn_controller.step_counter += 1

                """Prepare next timestep ==========================================================================="""
                if SUMO.Collision or SUMO.RouteEnd:  # end episode when endstate conditions are true
                    break
                dynamics_ego.a_previous = copy(dynamics_ego.a_real)  # copy --> to create an independent copy of the variable, not a reference to it
                if controller == 'DQN' or controller == 'hybrid_a' or controller == 'DDPG' or controller == 'DDPG_v':
                    nn_controller.state = copy(nn_controller.new_state)
                if sample_generation:
                    sample_generator.state = copy(sample_generator.new_state)
                vehicle_ego.ID_prec_previous = copy(vehicle_ego.ID_prec)
                SUMO.step += 1

            """End of episode - prepare next episode ======================================================================
            Reset position of vehicles by removing and (later) adding them again, call running plots and export data """
            if vehicle_ego.ID in SUMO.currentvehiclelist:
                traci.vehicle.remove(vehicle_ego.ID)
            if vehicle2_exist and vehicle_2.ID in SUMO.currentvehiclelist:
                traci.vehicle.remove(vehicle_2.ID)
            if vehicle3 and vehicle_3.ID in SUMO.currentvehiclelist:
                traci.vehicle.remove(vehicle_3.ID)
            traci.simulationStep()
            length_episode[episode] = SUMO.step
            cum_reward[episode] /= (SUMO.step/5)  # normalize cum_reward with episode length
            print('Cumulative Reward:', cum_reward[episode])
            if evaluation:
                cum_reward_evaluation.append(cum_reward[episode])
                evaluation = False
                if cum_reward[episode] > best_cum_reward:
                    nn_controller.save_models(savefile_best_actor+'_'+str(episode), savefile_best_critic+'_'+str(episode))
                    best_cum_reward = cum_reward[episode]
            if training and (controller == 'DQN' or controller == 'hybrid_a' or controller == 'DDPG' or controller == 'DDPG_v') and liveplot:
                reward_mean100[episode] = nn_controller.reward_mean_100_running(cum_reward, episode)
                nn_controller.weight_observer(episode)
                plot_running(reward_mean100, episode, cum_reward_evaluation)
            data_export[:, 0] = cum_reward[:, 0]
            data_export[:, 1] = length_episode[:, 0]
            if training:
                try:
                    if (episode+1) % 25 == 0:  # ==> save rewards every 50 episodes
                        np.savetxt(savefile_reward, data_export)
                    if (episode+1) % 25 == 0:  # save model every 50 episodes
                        nn_controller.save_models(savefile_model_actor, savefile_model_critic)
                except OSError:
                    print('File saving failed')
                    pass
            if acc_controller:
                acc_controller.reset_integral_error()
        except KeyboardInterrupt:
            print('Manual interrupt')
            break
    traci.close()
    return cum_reward

if __name__ == "__main__":

    """Input ======================================================================================================="""
    loadfile_model_actor = r'saved_models\\actor_dummy'
    loadfile_model_critic = r'saved_models\\critic_dummy'
    savefile_model_actor = r'saved_models\\actor_dummy'
    savefile_model_critic = r'saved_models\\critic_dummy'
    savefile_best_actor = r'saved_models\\best_actor_dummy'
    savefile_best_critic = r'saved_models\\best_critic_dummy'
    savefile_reward = r'saved_models\\rewards_dummy.txt'
    number_episodes = 10000  # number of episodes to run
    training = True  # when True, RL training is applied to simulation without SUMO GUI use; when False, no training is applied and SUMO GUI is used
    double_gpu = False  # only relevant for computer with two GPUs - keep on False otherwise
    device = 'cpu'  # gpu0, gpu1, cpu
    controller = 'DDPG_v'  # ACC, DDPG, hybrid_a, DDPG_v, SUMO
    TLS_virt_vehicle = True  # Red and Yellow traffic lights are considered as preceding vehicles with v=0
    TLS_ID = '0'  # program of TLS - 0(TLS with red phase), 1(TLS always green)
    feature_number = 4  # state representation (number of inputs to Neural Network) - currently distance, v_ego, v_preceding, v_allowed
    exploration_policy = 'noisy_DDPG'  # only relevant for RL training
    sample_generation = False  # only relevant for Supervised Pretraining - keep on False
    vehicle2_exist = False  # currently no meaningful trajectory / route - keep on False
    vehicle3_exist = True
    vehicle3_vprofile = 'sinus'  # 'sinus', 'emergstop'
    liveplot = False  # memory leak problem on windows when turned on
    Route_Ego = ['startedge', 'gneE01', 'gneE02', 'stopedge']
    ego_depart_speed = np.ones((number_episodes,))*0.   # specific depart speed for ego vehicle when not training
    Route_Traffic1 = ['gneE01', 'junction_out']  # for vehicle2
    Route_Traffic2 = ['gneE01', 'gneE02', 'stopedge']  # for vehicle3

    """Initialisation =============================================================================================="""
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device == 'gpu1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    elif device == 'gpu0':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    SUMO = SUMO(Route_Ego, Route_Traffic1, Route_Traffic2, timestep=0.2)
    ## create velocity profile of preceding vehicle ##
    """Hier werden bei 'training = True' unterschiedliche Geschwindigkeitsprofile für das vorausfahrende Fahrzeug definiert.
    Für 'training = False' wird ein festes sinusförmiges Profil mit Mittelwert 30 km/h und Amplitude +- 25 km/h definiert."""
    if vehicle3_exist:
        if vehicle3_vprofile == 'sinus':
            if training:  # create random sinusodial velocity profiles for training
                SUMO.prec_train_amplitude = np.random.rand(number_episodes) * 20/3.6
                SUMO.prec_train_mean = np.random.rand(number_episodes) * 20/3.6 + 10/3.6
            else:
                SUMO.prec_train_amplitude = 25/3.6  # a=25/3.6
                SUMO.prec_train_mean = 30/3.6  # c=30/3.6
                SUMO.create_v_profile_prec(a=SUMO.prec_train_amplitude, c=SUMO.prec_train_mean)
        elif vehicle3_vprofile == 'emergstop':
            SUMO.create_v_profile_emerg_stop()
        else:
            raise NameError('No valid velocity profile selected')
    vehicle_ego = Vehicle(ego=True, ID='ego', RouteID='RouteID_ego', Route=Route_Ego, powertrain_concept='ICEV')
    dynamics_ego = Longitudinal_dynamics(tau=0.5)
    if vehicle2_exist:
        vehicle_2 = Vehicle(ego=False, ID='traffic.0', RouteID='traci_route_traffic.0', Route=Route_Traffic1)
    if vehicle3_exist:
        vehicle_3 = Vehicle(ego=False, ID='traffic.1', RouteID='traci_route_traffic.1', Route=Route_Traffic2)
    acc_controller = {}
    if controller == 'ACC' or controller == 'hybrid_a' or controller == 'DDPG_v':
        acc_controller = ACC_Controller(v_set=19.44, h_set=1, a_set_min=-10.)  # instantiate ACC controller
        acc_controller.create_mode_map()
    if controller == 'DQN':
        nn_controller = DQN(number_episodes, training, double_gpu)  # instantiate DQN controller
        if os.path.isfile(loadfile_model_actor):  # load pretrained model if existing (model_actor means Q network)
            nn_controller.model.load_weights(loadfile_model_actor)
            nn_controller.target_model.load_weights(loadfile_model_actor)
            print('Previous model file found. Loading model...')
        else:
            print('No previous model file found. Initialising a new model...')
    if controller == 'DDPG' or controller == 'hybrid_a' or controller == 'DDPG_v':
        nn_controller = DDPG(number_episodes, training, feature_number, double_gpu, controller=controller)  # instantiate DDPG controller
        if exploration_policy == 'ACC':
            explo_policy = ACC_Controller(15, 1.5)  # arguments: v_set, h_set
        if os.path.isfile(loadfile_model_actor+'.h5'):
            nn_controller.actor.load_weights(loadfile_model_actor+'.h5')
            nn_controller.target_actor.load_weights(loadfile_model_actor+'.h5')
            print('Previous actor model file found. Loading model...')
        else:
            print('No previous actor model file found. Initialising a new model...')
        if os.path.isfile(loadfile_model_critic+'.h5'):
            nn_controller.target_critic.load_weights(loadfile_model_critic+'.h5')
            nn_controller.critic.load_weights(loadfile_model_critic+'.h5')
            print('Previous critic model file found. Loading model...')
        else:
            print('No previous critic model file found. Initialising a new model...')
    if sample_generation:
        """instantiate a DDPG controller to use the add_sample method for sample generation"""
        sample_generator = DDPG(number_episodes, training, feature_number)
    if training and liveplot:
        fig_running, ax_running_1, ax_running_2, ax_running_3, ax_running_4 = plot_running_init(training)


    """run simulation =============================================================================================="""
    start = timeit.default_timer()
    if training or sample_generation:
        traci.start(['sumo-gui', '-c', 'SUMO_config.sumocfg', '--no-warnings'])
    else:
        traci.start(['sumo-gui', '-c', 'SUMO_config.sumocfg'])
    cum_reward = run_control()

    """Postprocessing ==============================================================================================="""
    ### save keras model ###
    if training:
        nn_controller.save_models(savefile_model_actor, savefile_model_critic)
    ### postprocess SUMO data for plotting ###
    SUMO.postproc_v()
    stop = timeit.default_timer()
    print('Calculation time: ', stop-start)
    if training and (controller == 'DQN' or controller == 'DDPG'):
        reward_mean100 = nn_controller.reward_mean100(cum_reward)  # calculate the mean of last 100 episodes
    else:
        reward_mean100 = []
    if sample_generation:
        np.save('saved_models/dummy_generation', sample_generator.experience_batch)
    ### Plot results ###
    print('Ego vehicle mean fuel consumption:', float(np.sum(dynamics_ego.E_ICE) * 100000. / (
                np.sum(SUMO.v_ego[:SUMO.step + 1]) * SUMO.sim['timestep'] * vehicle_ego.ICE[
            'lhv'] * vehicle_ego.density_fuel)), 'l/100km')
    if False:
        plot_results(cum_reward, reward_mean100, SUMO, vehicle_ego, dynamics_ego, controller, training)
