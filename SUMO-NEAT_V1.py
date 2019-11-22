# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:04:54 2019

@author: peters
"""
from __future__ import print_function
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
from SUMO_NEAT_Population import Population
import neat
import visualize
import Send_Message_Bot2
import show_best_result as sbr
# import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
import traci
os.environ['PATH'] += os.pathsep + r'C:\Users\Daniel\Anaconda3\Library\bin\graphviz'


class trafic:
    def __init__(self):
        self.vehicle2_exist=False
        self.vehicle3_exist=False
        self.vehicle3_vprofile='sinus'

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
    sub_ego = traci.vehicle.getSubscriptionResults(trafic.vehicle_ego.ID)
#    print(sub_ego[traci.constants.VAR_NEXT_TLS])
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
        trafic.vehicle_ego.fuel_cons[SUMO.step] = sub_ego[traci.constants.VAR_FUELCONSUMPTION]
    else:
        trafic.vehicle_ego.fuel_cons[SUMO.step] = 0.

    ## distance, v_prec
    try:
        if traci.constants.VAR_LEADER in sub_ego:
            trafic.vehicle_ego.ID_prec, features.distance = sub_ego[traci.constants.VAR_LEADER]
            SUMO.distance[SUMO.step] = features.distance
            features.distance = np.clip(features.distance, None, 250.)
            features.v_prec = traci.vehicle.getSpeed(trafic.vehicle_ego.ID_prec)
            SUMO.v_prec[SUMO.step] = features.v_prec
            if features.distance == 250:
                features.v_prec = features.v_ego
        else:
            raise TypeError
    except TypeError:
        trafic.vehicle_ego.ID_prec = 'none'
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
        state = np.zeros([feature_number,1])
   
        """feature space: distance, v_ego, v_prec"""
        state[0] = features.distance/250  # features.distance / 250
        state[1] = features.v_ego/250  # features.v_ego / 25
        state[2] = features.v_prec/250  # features.v_prec / 25
        state[3] = features.v_allowed/250
        return state


def calculate_features():
    state = np.zeros([1, feature_number])
    dynamics_ego = Longitudinal_dynamics(tau=0.5)
    sub_ego = traci.vehicle.getSubscriptionResults(trafic.vehicle_ego.ID)
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
        
#        print(dynamics_ego.fuel_cons_per_100km)
        ## fuel_consumption
        trafic.vehicle_ego.fuel_cons[SUMO.step + 1] = sub_ego[traci.constants.VAR_FUELCONSUMPTION]  # in ml/s
#        trafic.vehicle_ego.fuel_cons_ECMS[SUMO.step + 1] = dynamics_ego.fuel_cons_per_100km
#        trafic.vehicle_ego.fuel_cons_ECMS_per_s[SUMO.step + 1] = dynamics_ego.fuel_cons_per_s

        ## distance, v_prec
        try:
            if traci.constants.VAR_LEADER in sub_ego:
                trafic.vehicle_ego.ID_prec, features.distance = sub_ego[traci.constants.VAR_LEADER]
                SUMO.distance[SUMO.step + 1] = features.distance
                features.distance = np.clip(features.distance, None, 250.)
                features.v_prec = traci.vehicle.getSpeed(trafic.vehicle_ego.ID_prec)
                SUMO.v_prec[SUMO.step + 1] = features.v_prec
            else:
                raise TypeError
        except TypeError:
            features.distance = 250
            SUMO.distance[SUMO.step + 1] = features.distance
            SUMO.v_prec[SUMO.step + 1] = features.v_ego
            trafic.vehicle_ego.ID_prec = 'none'
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


def eval_genomes(genomes, config, episode):
    """" SOMETHING SMETHING"""
    x=0
    for genome_id, genome in genomes:
        x+=1
        SUMO.step=0
        SUMO.Colliion = False
        SUMO.RouteEnd = False
        v_episode=[]
        a_episode=[]
        distance_episode=[]
        SUMO.init_vars_episode()
        """Anmerkung: Hier werden einige Variationen des Verkehrsszenarios für meine Trainingsepisoden definiert, wenn 'training = True'
        gesetzt ist. Im Fall 'training = False' oder 'evaluation = True' (Evaluierungsepisoden unter gleichen Randbedingungen) wird immer eine
        Episode mit gleichen Randbedingungen (z.B. Geschwindigkeitsprofil vorausfahrendes Fahrzeug) gesetzt"""
        if trafic.evaluation:
            traci.vehicle.add(trafic.vehicle_ego.ID, trafic.vehicle_ego.RouteID, departSpeed='0',
                              typeID='ego_vehicle')  # Ego vehicle
            traci.trafficlight.setPhase('junction1', 0)  # set traffic light phase to 0 for evaluation (same conditions)
        else:
            traci.vehicle.add(trafic.vehicle_ego.ID, trafic.vehicle_ego.RouteID, departSpeed=np.array2string(trafic.vehicle_ego.depart_speed[episode-1]), typeID='ego_vehicle')  # Ego vehicle
        if trafic.vehicle2_exist:
            traci.vehicle.add(trafic.vehicle_2.ID, trafic.vehicle_2.RouteID, typeID='traffic_vehicle')  # preceding vehicle 1
        if trafic.vehicle3:
            traci.vehicle.add(trafic.vehicle_3.ID, trafic.vehicle_3.RouteID, typeID='traffic_vehicle')  # preceding vehicle 2
            if trafic.training and not trafic.evaluation:
                traci.vehicle.moveTo(trafic.vehicle_3.ID, 'gneE01_0', trafic.episoden_variante)
            else:
                traci.vehicle.moveTo(trafic.vehicle_3.ID, 'gneE01_0', 0.)
   
        traci.simulationStep()  # to spawn vehicles
#                    if controller != 'SUMO':
        traci.vehicle.setSpeedMode(trafic.vehicle_ego.ID, 16)  # only emergency stopping at red traffic lights --> episode ends
        if trafic.vehicle2_exist:
            traci.vehicle.setSpeedMode(trafic.vehicle_2.ID, 17)
        if trafic.vehicle3:
            traci.vehicle.setSpeedMode(trafic.vehicle_3.ID, 17)
    
        SUMO.currentvehiclelist = traci.vehicle.getIDList()
    
        # SUMO subscriptions
        traci.vehicle.subscribeLeader(trafic.vehicle_ego.ID, 10000)
        traci.vehicle.subscribe(trafic.vehicle_ego.ID, [traci.constants.VAR_SPEED, traci.constants.VAR_BEST_LANES, traci.constants.VAR_FUELCONSUMPTION,
                                                            traci.constants.VAR_NEXT_TLS, traci.constants.VAR_ALLOWED_SPEED, traci.constants.VAR_LANE_ID])
        
            
            
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        """"Run episode ======================================================================="""
        while traci.simulation.getMinExpectedNumber() > 0:  # timestep loop

             """Get state for first iteration ==================================================================="""
             if SUMO.step == 0:
                 calculate_features_firststep()
    #             if controller == 'DDPG_v' or controller == 'DDPG' or controller == 'DQN':
    #                 nn_controller.state = get_state()
           
             """Controller ======================================================================================
             Hier wird die Stellgröße des Reglers (z.B. Sollbeschleunigung) in Abhängigkeit der Reglereingänge (zusammengefasst in 'features' 
             für den ACC Controller bzw. 'nn_controller.state' für die Neuronalen Regler"""
    #         if controller == 'ACC':
    #             SUMO.a_set[SUMO.step] = acc_controller.calculate_a(features)  # Calculate set acceleration with ACC controller
    #             # Save ACC mode to SUMO class
    #             SUMO.mode_ego[SUMO.step] = acc_controller.mode  # ACC control modes (speed control, headway control, etc)
    #         elif controller == 'DQN':
    #             # Calculate Q-Values for all actions
    #             nn_controller.action_value = nn_controller.model.predict(nn_controller.state)  # NN inputs: distance, delta_v
    #             # Choose Action according to epsilon-greedy policy
    #             nn_controller.index[SUMO.step] = nn_controller.choose_action(nn_controller.action_value, episode)
    #             # Calculate the set acceleration for the timestep
    #             nn_controller.a_set = nn_controller.actions[nn_controller.index[SUMO.step]]
    #             SUMO.a_set[SUMO.step] = nn_controller.a_set
    #         elif controller == 'DDPG':
    #             if training and not evaluation:
    #                 if exploration_policy == 'ACC':
    #                     SUMO.a_set[SUMO.step] = explo_policy.calculate_a(features)
    #                     SUMO.a_set[SUMO.step] = nn_controller.add_noise(SUMO.a_set[SUMO.step])
    #                 else:
    #                     SUMO.a_set[SUMO.step] = nn_controller.choose_action(copy(nn_controller.state), network='main', noise=True)
    #             else:
    #                 SUMO.a_set[SUMO.step] = nn_controller.choose_action(copy(nn_controller.state), network='main', noise=False)
    #         elif controller == 'hybrid_a':
    #             acc_controller.a_set = acc_controller.calculate_a(
    #                 features)  # Calculate set acceleration with ACC controller
    #             # Save ACC mode to SUMO class
    #             SUMO.mode_ego[SUMO.step] = acc_controller.mode  # ACC control modes (speed control, headway control, etc)
    #             if SUMO.step == 0:
    #                 nn_controller.state[0, 3] = acc_controller.a_set
    #             if training and not evaluation:
    #                 nn_controller.a_set = nn_controller.choose_action(copy(nn_controller.state), network='main', noise=True)
    #             else:
    #                 nn_controller.a_set = nn_controller.choose_action(copy(nn_controller.state), network='main', noise=False)
    #             SUMO.a_set[SUMO.step] = acc_controller.a_set + nn_controller.k_hybrid_a * nn_controller.a_set
    #             SUMO.a_hybrid_a[SUMO.step] = nn_controller.a_set
    #         elif controller == 'DDPG_v':
    #             if training and not evaluation:
    #                 SUMO.v_set[SUMO.step], SUMO.v_set_nonoise[SUMO.step] = nn_controller.choose_action(copy(nn_controller.state), network='main', features=features, noise=True)
    #             else:
    #                 SUMO.v_set[SUMO.step] = nn_controller.choose_action(copy(nn_controller.state), network='main', noise=False)
    #             SUMO.a_set[SUMO.step] = acc_controller.calc_a_P(features, SUMO.v_set[SUMO.step])
             
             state=get_state()
#             print(state)
#             net = neat.nn.FeedForwardNetwork.create(genome, config)
             SUMO.a_set[SUMO.step] = net.activate(state)#[0,0], state[0,1], state[0,2], state[0,3])
#             print(SUMO.a_set[SUMO.step], SUMO.step)
             """Longitudinal Dynamics ===========================================================================
             Hier werden Längsdynamische Größen des Egofahrzeugs berechnet. beispielsweise die Realbeschleunigung aus den Stellgrößen des Reglers oder der
             Kraftstoffverbrauch für ein definiertes Fahrzeug (aktuell konventionelles Verbrennerfahrzeug mit Mehrganggetriebe). Am Ende wird als
             Schnittstelle zu SUMO eine neue Geschwindigkeit für das Egofahrzeug für den nächsten Zeitschritt gesetzt"""
             dynamics_ego = Longitudinal_dynamics(tau=0.5)
#             if controller == 'SUMO':
#                 dynamics_ego.a_real = traci.vehicle.getAcceleration(vehicle_ego.ID)
#                 dynamics_ego.wheel_demand(SUMO.v_ego[SUMO.step], vehicle_ego,
#                                           SUMO.step)  # Calculate the torque and speed wheel demand of this timestep
#                 dynamics_ego.operating_strategy(SUMO.sim['timestep'], vehicle_ego, s0=2, kp=10, step=SUMO.step)
#                 SUMO.a_real[SUMO.step] = dynamics_ego.a_real
#             else:
             dynamics_ego.low_lev_controller(SUMO.a_set[SUMO.step], SUMO.sim['timestep'])  # Calculate a_real with a_set and a PT1 Transfer function
             dynamics_ego.wheel_demand(SUMO.v_ego[SUMO.step], trafic.vehicle_ego, SUMO.step)  # Calculate the torque and speed wheel demand of this timestep
             dynamics_ego.operating_strategy(SUMO.sim['timestep'], trafic.vehicle_ego, s0=2, kp=10, step=SUMO.step)
             dynamics_ego.v_real_next = SUMO.v_ego[SUMO.step] + dynamics_ego.a_real * SUMO.sim['timestep']
             dynamics_ego.v_real_next = np.clip(dynamics_ego.v_real_next, 0., None)
             SUMO.a_real[SUMO.step] = dynamics_ego.a_real
             traci.vehicle.setSpeed(trafic.vehicle_ego.ID, dynamics_ego.v_real_next)  # Set velocity of ego car for next time step
           
             """Control traffic ================================================================"""
             if trafic.vehicle2_exist and trafic.vehicle_2.ID in SUMO.currentvehiclelist:
                 if trafic.vehicle_2.end == False:
                     if traci.vehicle.getLaneID(trafic.vehicle_2.ID) == 'junction_out_0' and traci.vehicle.getLanePosition(trafic.vehicle_2.ID) > 90:
                         traci.vehicle.remove(trafic.vehicle_2.ID)
                         trafic.vehicle_2.end = True
                     else:
                         #traci.vehicle.setSpeed(vehicle_2.ID, SUMO.v_profile[SUMO.step])  # set velocity of preceding car 1
                         pass
             if trafic.vehicle3 and trafic.vehicle_3.ID in SUMO.currentvehiclelist:
                 traci.vehicle.setSpeed(trafic.vehicle_3.ID, SUMO.v_profile[SUMO.step])  # set velocity of preceding car 2
           
             """SUMO simulation step ============================================================================"""
             traci.simulationStep()
             SUMO.currentvehiclelist = traci.vehicle.getIDList()
           
             """Check if any of the endstate conditions is true (e.g. collision) ==================================="""
             if trafic.vehicle_ego.ID not in SUMO.currentvehiclelist:
                 SUMO.RouteEnd = True
                 SUMO.endstate = True
                 print([x, SUMO.step],' Route finished!')
             elif traci.simulation.getCollidingVehiclesNumber() > 0:
                 SUMO.endstate = True
                 SUMO.Collision = True
                 print([x, SUMO.step],' Collision!')
             elif trafic.vehicle_ego.ID in traci.simulation.getEmergencyStoppingVehiclesIDList():  # Check for ego vehicle passing red traffic light
                 SUMO.Collision = True
                 SUMO.endstate = True
                 print([x, SUMO.step],' Red lights passed!')
             # Set a maximum time step limit for 1 episode
             elif SUMO.step > 5000:
                 SUMO.RouteEnd = True
                 SUMO.endstate = True
                 print([x, SUMO.step],' Maximum time reached!')
           
             """get new state ==================================================================================="""
             calculate_features()
             v_episode.append(features.v_ego)
             a_episode.append(dynamics_ego.a_real)
             distance_episode.append(features.distance)
#             if controller == 'DDPG_v' or controller == 'DDPG' or controller == 'DQN':
#                 nn_controller.new_state = get_state()
             """Prepare next timestep ==========================================================================="""
             if SUMO.Collision or SUMO.RouteEnd:  # end episode when endstate conditions are true
                 break
             dynamics_ego.a_previous = copy(dynamics_ego.a_real)  # copy --> to create an independent copy of the variable, not a reference to it
#             if controller == 'DQN' or controller == 'hybrid_a' or controller == 'DDPG' or controller == 'DDPG_v':
#                 nn_controller.state = copy(nn_controller.new_state)
             if sample_generation:
                 sample_generator.state = copy(sample_generator.new_state)
             trafic.vehicle_ego.ID_prec_previous = copy(trafic.vehicle_ego.ID_prec)
#             print(SUMO.step)
             SUMO.step += 1  
             
#    print(v_episode)
        
        """Calculate Reward ================================================================================"""
        fitness = 0.1*np.sqrt(np.sum(np.square(v_episode)))-np.sum(0.5*np.sign(features.v_allowed-np.asarray(v_episode))+1)-np.mean(np.square(a_episode))-1/sum(np.square(np.asarray(v_episode)*3.6-np.asarray(distance_episode)))-10000*int(SUMO.Collision)
        genome.fitness=fitness[0].item()
        print(fitness)
        
        
        #     cum_reward[episode] += reward
        """End of episode - prepare next episode ======================================================================
            Reset position of vehicles by removing and (later) adding them again, call running plots and export data """
        if trafic.vehicle_ego.ID in SUMO.currentvehiclelist:
            traci.vehicle.remove(trafic.vehicle_ego.ID)
        if trafic.vehicle2_exist and trafic.vehicle_2.ID in SUMO.currentvehiclelist:
            traci.vehicle.remove(trafic.vehicle_2.ID)
        if trafic.vehicle3 and trafic.vehicle_3.ID in SUMO.currentvehiclelist:
            traci.vehicle.remove(trafic.vehicle_3.ID)
        traci.simulationStep()
        dynamics_ego.reset_variables()
#        print([x, SUMO.step])

def run(config_file,vehicle_ego, SUMO):
       

    """load the config, create a population, evolve and show the result"""
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, trafic, SUMO, number_episodes)
#    p.stop()

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
#    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
#    for xi, xo in zip(xor_inputs, xor_outputs):
#        output = winner_net.activate(xi)
#        print(
#            "input {!r}, expected output {!r}, got {!r}".format(xi, xo, output)
#            )

    if visualize is not None:
        node_names = {-1: 'distance', -2: 'v_ego',-3:'v_prec', -4:'v_allowed', 0: 'a_set'}
        visualize.draw_net(config, winner, True, node_names=node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
    return winner

if __name__ == "__main__":

    """Input ======================================================================================================="""
    loadfile_model_actor = r'saved_models\\actor_dummy'
    loadfile_model_critic = r'saved_models\\critic_dummy'
    savefile_model_actor = r'saved_models\\actor_dummy'
    savefile_model_critic = r'saved_models\\critic_dummy'
    savefile_best_actor = r'saved_models\\best_actor_dummy'
    savefile_best_critic = r'saved_models\\best_critic_dummy'
    savefile_reward = r'saved_models\\rewards_dummy.txt'
    number_episodes = 2  # number of episodes to run
    trafic.training = True  # when True, RL training is applied to simulation without SUMO GUI use; when False, no training is applied and SUMO GUI is used
    trafic.evaluation = False
    double_gpu = False  # only relevant for computer with two GPUs - keep on False otherwise
    device = 'cpu'  # gpu0, gpu1, cpu
#  controller = 'DDPG_v'  # ACC, DDPG, hybrid_a, DDPG_v, SUMO
    TLS_virt_vehicle = True  # Red and Yellow traffic lights are considered as preceding vehicles with v=0
    TLS_ID = '0'  # program of TLS - 0(TLS with red phase), 1(TLS always green)
    feature_number = 4  # state representation (number of inputs to Neural Network) - currently distance, v_ego, v_preceding, v_allowed
#    exploration_policy = 'noisy_DDPG'  # only relevant for RL training
    sample_generation = False  # only relevant for Supervised Pretraining - keep on False
    
    trafic.vehicle2_exist = False  # currently no meaningful trajectory / route - keep on False
    trafic.vehicle3_exist = True
    trafic.vehicle3_vprofile = 'sinus'  # 'sinus', 'emergstop'
    liveplot = False  # memory leak problem on windows when turned on
    trafic.Route_Ego = ['startedge', 'gneE01', 'gneE02', 'stopedge']
    trafic.ego_depart_speed = np.ones((number_episodes,))*0.   # specific depart speed for ego vehicle when not training
    trafic.Route_Traffic1 = ['gneE01', 'junction_out']  # for vehicle2
    trafic.Route_Traffic2 = ['gneE01', 'gneE02', 'stopedge']  # for vehicle3
    
    
    """Input NEAT ================================================================================================="""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_SUMO')
    """Initialisation =============================================================================================="""
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device == 'gpu1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    elif device == 'gpu0':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    SUMO = SUMO(trafic.Route_Ego, trafic.Route_Traffic1, trafic.Route_Traffic2, timestep=0.2)
    ## create velocity profile of preceding vehicle ##
    """Hier werden bei 'training = True' unterschiedliche Geschwindigkeitsprofile für das vorausfahrende Fahrzeug definiert.
    Für 'training = False' wird ein festes sinusförmiges Profil mit Mittelwert 30 km/h und Amplitude +- 25 km/h definiert."""
    if trafic.vehicle3_exist:
        if trafic.vehicle3_vprofile == 'sinus':
            if trafic.training:  # create random sinusodial velocity profiles for training
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
    trafic.vehicle_ego = Vehicle(ego=True, ID='ego', RouteID='RouteID_ego', Route=trafic.Route_Ego, powertrain_concept='ICEV')
    trafic.dynamics_ego = Longitudinal_dynamics(tau=0.5)
    if trafic.vehicle2_exist:
        trafic.vehicle_2 = Vehicle(ego=False, ID='traffic.0', RouteID='traci_route_traffic.0', Route=trafic.Route_Traffic1)
    if trafic.vehicle3_exist:
        trafic.vehicle_3 = Vehicle(ego=False, ID='traffic.1', RouteID='traci_route_traffic.1', Route=trafic.Route_Traffic2)
    acc_controller = {}
#    if controller == 'ACC' or controller == 'hybrid_a' or controller == 'DDPG_v':
#        acc_controller = ACC_Controller(v_set=19.44, h_set=1, a_set_min=-10.)  # instantiate ACC controller
#        acc_controller.create_mode_map()
#    if controller == 'DQN':
#        nn_controller = DQN(number_episodes, training, double_gpu)  # instantiate DQN controller
#        if os.path.isfile(loadfile_model_actor):  # load pretrained model if existing (model_actor means Q network)
#            nn_controller.model.load_weights(loadfile_model_actor)
#            nn_controller.target_model.load_weights(loadfile_model_actor)
#            print('Previous model file found. Loading model...')
#        else:
#            print('No previous model file found. Initialising a new model...')
#    if controller == 'DDPG' or controller == 'hybrid_a' or controller == 'DDPG_v':
#        nn_controller = DDPG(number_episodes, training, feature_number, double_gpu, controller=controller)  # instantiate DDPG controller
#        if exploration_policy == 'ACC':
#            explo_policy = ACC_Controller(15, 1.5)  # arguments: v_set, h_set
#        if os.path.isfile(loadfile_model_actor+'.h5'):
#            nn_controller.actor.load_weights(loadfile_model_actor+'.h5')
#            nn_controller.target_actor.load_weights(loadfile_model_actor+'.h5')
#            print('Previous actor model file found. Loading model...')
#        else:
#            print('No previous actor model file found. Initialising a new model...')
#        if os.path.isfile(loadfile_model_critic+'.h5'):
#            nn_controller.target_critic.load_weights(loadfile_model_critic+'.h5')
#            nn_controller.critic.load_weights(loadfile_model_critic+'.h5')
#            print('Previous critic model file found. Loading model...')
#        else:
#            print('No previous critic model file found. Initialising a new model...')
    if sample_generation:
        """instantiate a DDPG controller to use the add_sample method for sample generation"""
        sample_generator = DDPG(number_episodes, training, feature_number)
    if trafic.training and liveplot:
        fig_running, ax_running_1, ax_running_2, ax_running_3, ax_running_4 = plot_running_init(training)


    """run simulation =============================================================================================="""
    start = timeit.default_timer()
    if trafic.training or sample_generation:
        traci.start(['sumo-gui', '-c', 'SUMO_config.sumocfg', '--no-warnings'])
    else:
        traci.start(['sumo-gui', '-c', 'SUMO_config.sumocfg'])
    best_genome=run(config_path, trafic, SUMO)
    andIsendtomyself('Geschafft!')
    traci.close
    #cum_reward = run_control()

    """Postprocessing ==============================================================================================="""
#    ### save keras model ###
#    if training:
#        nn_controller.save_models(savefile_model_actor, savefile_model_critic)
    ### postprocess SUMO data for plotting ###
    SUMO.postproc_v()
    stop = timeit.default_timer()
    print('Calculation time: ', stop-start)
    
#    if training and (controller == 'DQN' or controller == 'DDPG'):
#        reward_mean100 = nn_controller.reward_mean100(cum_reward)  # calculate the mean of last 100 episodes
#    else:
#        reward_mean100 = []
##    if sample_generation:
##        np.save('saved_models/dummy_generation', sample_generator.experience_batch)
#    ### Plot results ###
#    print('Ego vehicle mean fuel consumption:', float(np.sum(dynamics_ego.E_ICE) * 100000. / (
#                np.sum(SUMO.v_ego[:SUMO.step + 1]) * SUMO.sim['timestep'] * vehicle_ego.ICE[
#            'lhv'] * trafic.vehicle_ego.density_fuel)), 'l/100km')
#    if False:
#        plot_results(cum_reward, reward_mean100, SUMO, vehicle_ego, dynamics_ego, controller, training)
