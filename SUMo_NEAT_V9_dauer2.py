# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:10:35 2019

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
from SUMO_NEAT_Population_V3 import Population
import neat
from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
from itertools import repeat
import visualize
from Send_Message_Bot2 import andIsendtomyself
import show_best_result as sbr
from multiprocessing  import Pool
import multiprocessing
from datetime import datetime
import pickle
import logging
# import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
import traci
import eval_genomes_file
os.environ['PATH'] += os.pathsep + r'C:\Users\Daniel\Anaconda3\Library\bin\graphviz'


class trafic:
    def __init__(p):
        p.vehicle2_exist=False
        p.vehicle3_exist=False
        p.vehicle3_vprofile='sinus'

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


def calculate_features_firststep(sim, SUMO):
    try:
        simulation=traci.getConnection(sim)
        traci.switch(sim)
    #    state = np.zeros([1, feature_number])
        sub_ego = simulation.vehicle.getSubscriptionResults(trafic.vehicle_ego.ID)
    #    print(sub_ego[traci.constants.VAR_NEXT_TLS])
        ## TLS Distance
        if traci.constants.VAR_NEXT_TLS in sub_ego and len(sub_ego[traci.constants.VAR_NEXT_TLS]) > 0:
            features.distance_TLS = sub_ego[traci.constants.VAR_NEXT_TLS][0][2]
            features.TLS_state = sub_ego[traci.constants.VAR_NEXT_TLS][0][3]
        else:
            features.distance_TLS = 1000  # TODO: Handling when no TLS ahead
            features.TLS_state = None
        errorat=1
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
        errorat=2
        ## distance, v_prec
        try:
            if traci.constants.VAR_LEADER in sub_ego:
                trafic.vehicle_ego.ID_prec, features.distance = sub_ego[traci.constants.VAR_LEADER]
                SUMO.distance[SUMO.step] = features.distance
                features.distance = np.clip(features.distance, None, 250.)
                features.v_prec = simulation.vehicle.getSpeed(trafic.vehicle_ego.ID_prec)
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
        errorat=3
        ## v_allowed
        if traci.constants.VAR_LANE_ID in sub_ego:
            features.v_allowed = simulation.lane.getMaxSpeed(sub_ego[traci.constants.VAR_LANE_ID])
        else:
            features.v_allowed = 33.33  # tempo limit set to 120 km/h when no signal received, unlikely to happen
    
        ## correct distance, v_prec with virtual TLS vehicle
        if trafic.TLS_virt_vehicle:
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
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
#        fitness=timestep
        raise Exception('My error! firststep', errorat, message)

def get_state(state):
#        state = np.zeros([feature_number,1])
    try:
        """feature space: distance, v_ego, v_prec"""
        state[0] = features.distance/250  # features.distance / 250
        state[1] = features.v_ego/250  # features.v_ego / 25
        state[2] = features.v_prec/250  # features.v_prec / 25
        state[3] = features.v_allowed/250
        return state
    except Exception as ex:
        template = "An exception of type {0} occurred in get_state. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
#        fitness=timestep
        raise Exception('My error9!', message)

def calculate_features(sim, SUMO):
    try:
        simulation=traci.getConnection(sim)
        traci.switch(sim)
        sub_ego = simulation.vehicle.getSubscriptionResults(trafic.vehicle_ego.ID)
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
                    features.v_prec = simulation.vehicle.getSpeed(trafic.vehicle_ego.ID_prec)
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
            if trafic.TLS_virt_vehicle:
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
                features.v_allowed = simulation.lane.getMaxSpeed(sub_ego[traci.constants.VAR_LANE_ID])
            else:
                features.v_allowed = 33.33  # tempo limit set to 120 km/h when no signal received, unlikely to happen
    
        ## plotting variables
        SUMO.headway[SUMO.step + 1] = features.headway
        SUMO.v_ego[SUMO.step + 1] = features.v_ego
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
#        fitness=timestep
        raise Exception('My error! Calc features', message)
    

def eval_genomes(genome_id, genome, config, episode, trafic):
    """" SOMETHING SMETHING"""
    error=False
    error_return=10000
    timestep=0
    try:
        [simname, SUMO, error] = SimInitializer(trafic, episode)
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        if error==True:
            
            return error_return
#        fitness=timestep
        # raise Exception('Error SimInitializer',  message)
   
    if error==True:
        return error_return
    
    try:
        x=0
        #    state_empty=np.zeros([feature_number,1])
        #    for genome_id, genome in genomes:
        x+=1
        SUMO.step=0
        SUMO.Colliion = False
        SUMO.RouteEnd = False
        v_episode=[]
        a_episode=[]
        distance_episode=[]
        timestep=1 
        process_param= multiprocessing.Process()
        sim=simname#process_param.name 
        timestep=[sim, simname]
        simulation=traci.getConnection(simname)
        timestep=1.5
        SUMO.init_vars_episode()
        """Anmerkung: Hier werden einige Variationen des Verkehrsszenarios für meine Trainingsepisoden definiert, wenn 'training = True'
        gesetzt ist. Im Fall 'training = False' oder 'evaluation = True' (Evaluierungsepisoden unter gleichen Randbedingungen) wird immer eine
        Episode mit gleichen Randbedingungen (z.B. Geschwindigkeitsprofil vorausfahrendes Fahrzeug) gesetzt"""
        if trafic.evaluation:
            simulation.vehicle.add(trafic.vehicle_ego.ID, trafic.vehicle_ego.RouteID, departSpeed='0',
                              typeID='ego_vehicle')  # Ego vehicle
            simulation.trafficlight.setPhase('junction1', 0)  # set traffic light phase to 0 for evaluation (same conditions)
        else:
            simulation.vehicle.add(trafic.vehicle_ego.ID, trafic.vehicle_ego.RouteID, departSpeed=np.array2string(trafic.vehicle_ego.depart_speed[episode-1]), typeID='ego_vehicle')  # Ego vehicle
        if trafic.vehicle2_exist:
            simulation.vehicle.add(trafic.vehicle_2.ID, trafic.vehicle_2.RouteID, typeID='traffic_vehicle')  # preceding vehicle 1
        timestep=1.7
        if trafic.vehicle3:
            simulation.vehicle.add(trafic.vehicle_3.ID, trafic.vehicle_3.RouteID, typeID='traffic_vehicle')  # preceding vehicle 2
            if trafic.training and not trafic.evaluation:
                simulation.vehicle.moveTo(trafic.vehicle_3.ID, 'gneE01_0', trafic.episoden_variante)
            else:
                simulation.vehicle.moveTo(trafic.vehicle_3.ID, 'gneE01_0', 0.)
        timestep=2
        simulation.simulationStep()  # to spawn vehicles
    #                    if controller != 'SUMO':
        simulation.vehicle.setSpeedMode(trafic.vehicle_ego.ID, 16)  # only emergency stopping at red traffic lights --> episode ends
        if trafic.vehicle2_exist:
            simulation.vehicle.setSpeedMode(trafic.vehicle_2.ID, 17)
        if trafic.vehicle3:
            simulation.vehicle.setSpeedMode(trafic.vehicle_3.ID, 17)
    
        SUMO.currentvehiclelist = simulation.vehicle.getIDList()
    
        # SUMO subscriptions
        simulation.vehicle.subscribeLeader(trafic.vehicle_ego.ID, 10000)
        simulation.vehicle.subscribe(trafic.vehicle_ego.ID, [traci.constants.VAR_SPEED, traci.constants.VAR_BEST_LANES, traci.constants.VAR_FUELCONSUMPTION,
                                                            traci.constants.VAR_NEXT_TLS, traci.constants.VAR_ALLOWED_SPEED, traci.constants.VAR_LANE_ID])
        timestep=3
            
        dynamics_ego = Longitudinal_dynamics(tau=0.5)    
        timestep=3.1
        net = neat.nn.FeedForwardNetwork.create(genome[1], config)

        """"Run episode ======================================================================="""
        while simulation.simulation.getMinExpectedNumber() > 0:  # timestep loop
    
             """Get state for first iteration ==================================================================="""
             if SUMO.step == 0:
                 calculate_features_firststep(sim, SUMO)
    #             if controller == 'DDPG_v' or controller == 'DDPG' or controller == 'DQN':
    #                 nn_controller.state = get_state()
             timestep=3.3
             """Controller ======================================================================================
             Hier wird die Stellgröße des Reglers (z.B. Sollbeschleunigung) in Abhängigkeit der Reglereingänge (zusammengefasst in 'features' 
             für den ACC Controller bzw. 'nn_controller.state' für die Neuronalen Regler"""

             state=get_state(trafic.state_empty)
             timestep=4
    #             print(state)
    #             net = neat.nn.FeedForwardNetwork.create(genome, config)
             SUMO.a_set[SUMO.step] = net.activate(state) #[0,0], state[0,1], state[0,2], state[0,3])
             SUMO.a_set[SUMO.step] = 10*SUMO.a_set[SUMO.step]#-10 #[0,0], state[0,1], state[0,2], state[0,3])
#             print(SUMO.a_set[SUMO.step], SUMO.step)
             timestep=5
             """Longitudinal Dynamics ===========================================================================
             Hier werden Längsdynamische Größen des Egofahrzeugs berechnet. beispielsweise die Realbeschleunigung aus den Stellgrößen des Reglers oder der
             Kraftstoffverbrauch für ein definiertes Fahrzeug (aktuell konventionelles Verbrennerfahrzeug mit Mehrganggetriebe). Am Ende wird als
             Schnittstelle zu SUMO eine neue Geschwindigkeit für das Egofahrzeug für den nächsten Zeitschritt gesetzt"""
             dynamics_ego = Longitudinal_dynamics(tau=0.5)
             dynamics_ego.low_lev_controller(SUMO.a_set[SUMO.step], SUMO.sim['timestep'])  # Calculate a_real with a_set and a PT1 Transfer function
             dynamics_ego.wheel_demand(SUMO.v_ego[SUMO.step], trafic.vehicle_ego, SUMO.step)  # Calculate the torque and speed wheel demand of this timestep
             dynamics_ego.operating_strategy(SUMO.sim['timestep'], trafic.vehicle_ego, s0=2, kp=10, step=SUMO.step)
             dynamics_ego.v_real_next = SUMO.v_ego[SUMO.step] + dynamics_ego.a_real * SUMO.sim['timestep']
             dynamics_ego.v_real_next = np.clip(dynamics_ego.v_real_next, 0., None)
             SUMO.a_real[SUMO.step] = dynamics_ego.a_real
             simulation.vehicle.setSpeed(trafic.vehicle_ego.ID, dynamics_ego.v_real_next)  # Set velocity of ego car for next time step
             timestep=6
             """Control traffic ================================================================"""
             if trafic.vehicle2_exist and trafic.vehicle_2.ID in SUMO.currentvehiclelist:
                 if trafic.vehicle_2.end == False:
                     if simulation.vehicle.getLaneID(trafic.vehicle_2.ID) == 'junction_out_0' and simulation.vehicle.getLanePosition(trafic.vehicle_2.ID) > 90:
                         simulation.vehicle.remove(trafic.vehicle_2.ID)
                         trafic.vehicle_2.end = True
                     else:
                         #traci.vehicle.setSpeed(vehicle_2.ID, SUMO.v_profile[SUMO.step])  # set velocity of preceding car 1
                         pass
             if trafic.vehicle3 and trafic.vehicle_3.ID in SUMO.currentvehiclelist:
                 simulation.vehicle.setSpeed(trafic.vehicle_3.ID, SUMO.v_profile[SUMO.step])  # set velocity of preceding car 2
           
             """SUMO simulation step ============================================================================"""
             simulation.simulationStep()
             SUMO.currentvehiclelist = traci.vehicle.getIDList()
             timestep=7
             """Check if any of the endstate conditions is true (e.g. collision) ==================================="""
             if trafic.vehicle_ego.ID not in SUMO.currentvehiclelist:
                 SUMO.RouteEnd = True
                 SUMO.endstate = True
                 # print([x, SUMO.step],' Route finished!')
                 
             elif simulation.simulation.getCollidingVehiclesNumber() > 0:
                 SUMO.endstate = True
                 SUMO.Collision = True
                  # print([x, SUMO.step],' Collision!')
                 
             elif trafic.vehicle_ego.ID in simulation.simulation.getEmergencyStoppingVehiclesIDList():  # Check for ego vehicle passing red traffic light
                 SUMO.Collision = True
                 SUMO.endstate = True
                 # print([x, SUMO.step],' Red lights passed!')
             # Set a maximum time step limit for 1 episode
             elif SUMO.step > 5000:
                 SUMO.RouteEnd = True
                 SUMO.endstate = True
                 # print([x, SUMO.step],' Maximum time reached!')
                 
             timestep=7.5
             """get new state ==================================================================================="""
             calculate_features(sim, SUMO)
             v_episode.append(features.v_ego)
             a_episode.append(dynamics_ego.a_real)
             distance_episode.append(features.distance)
    #             
             """Prepare next timestep ==========================================================================="""
             if SUMO.Collision or SUMO.RouteEnd:  # end episode when endstate conditions are true
                 timestep=9
                 break
             dynamics_ego.a_previous = copy(dynamics_ego.a_real)  # copy --> to create an independent copy of the variable, not a reference to it
             trafic.vehicle_ego.ID_prec_previous = copy(trafic.vehicle_ego.ID_prec)
             SUMO.step += 1  
             timestep=8
        
        """Calculate Reward ================================================================================"""
#        fitness = np.sqrt(len(v_episode))*np.sqrt(np.sum(np.square(np.asarray(v_episode)/250)))+np.sum(0.5*np.sign(features.v_allowed/250-np.asarray(v_episode)/250)+1) -0.0001*np.sum(np.square(a_episode))-1/sum(np.square(np.asarray(v_episode)/250*3.6-np.asarray(distance_episode)/250))-100000*int(SUMO.Collision) +100000*int(SUMO.RouteEnd)
        verbrauch=(10*(np.mean(trafic.vehicle_ego.fuel_cons)/0.05)-1)#*50#*100
        travelled_distance=sum([c*0.2 for c in v_episode])
        travelled_distance_divisor = 1/1800 if travelled_distance==0 else travelled_distance
        abstand=np.sum([1 for ii in range(len(v_episode)) if distance_episode[ii]<2*v_episode[ii]*3.6 and distance_episode[ii]>v_episode[ii]*3.6])
        crash=int(SUMO.Collision)
        #route_ende=int(SUMO.RouteEnd and not SUMO.maxtime)
        speed=np.mean(v_episode)/features.v_allowed if np.mean(v_episode)/features.v_allowed<1 else 0
        over_speed_limit=sum([1 for jj in range(len(v_episode)) if v_episode[jj]>features.v_allowed])
        evil_acc=sum([1 for kk in range(len(a_episode)) if abs(a_episode[kk].item())>4]) 
        travelled_distance_divisor = evil_acc+1 if travelled_distance==0 else travelled_distance
        
        fitness=travelled_distance/1800+ abstand/SUMO.step-crash+speed-10*over_speed_limit/SUMO.step-evil_acc/travelled_distance_divisor-0.5*verbrauch
        
        
        genome[1].fitness=fitness[0].item()
        # print(fitness)
        output=fitness[0].item()
        #     cum_reward[episode] += reward
        """End of episode - prepare next episode ======================================================================
            Reset position of vehicles by removing and (later) adding them again, call running plots and export data """
        traci.close()
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        output=10000
        traci.close()
        # raise Exception('My error eval genome!', timestep, message)
    return output


def SimInitializer(trafic, episode):
   error=False
   
   try:
        timestep=1
        trafic.number_episodes=100000
        feature_number=4
        trafic.training=True
        sample_generation=False
        trafic.TLS_virt_vehicle = True  # Red and Yellow traffic lights are considered as preceding vehicles with v=0
        trafic.TLS_ID = ['0','1','2','3','4'] 
        trafic.evaluation=False
        
        
        trafic.vehicle2_exist = False  # currently no meaningful trajectory / route - keep on False
        trafic.vehicle3_exist = True
        trafic.vehicle3_vprofile = 'sinus'  # 'sinus', 'emergstop'
        liveplot = False  # memory leak problem on windows when turned on
        trafic.Route_Ego = ['startedge', 'gneE01', 'gneE02', 'stopedge']
        trafic.ego_depart_speed = np.ones((trafic.number_episodes,))*0.   # specific depart speed for ego vehicle when not training
        trafic.Route_Traffic1 = ['gneE01', 'junction_out']  # for vehicle2
        trafic.Route_Traffic2 = ['gneE01', 'gneE02', 'stopedge']  # for vehicle3
        trafic.state_empty=np.zeros([feature_number,1])
        timestep=2
        
        np.random.seed(42+episode)
        
        SUMO2 = SUMO(trafic.Route_Ego, trafic.Route_Traffic1, trafic.Route_Traffic2, timestep=0.2)
        ## create velocity profile of preceding vehicle ##
        """Hier werden bei 'training = True' unterschiedliche Geschwindigkeitsprofile für das vorausfahrende Fahrzeug definiert.
        Für 'training = False' wird ein festes sinusförmiges Profil mit Mittelwert 30 km/h und Amplitude +- 25 km/h definiert."""
        if trafic.vehicle3_exist:
            if trafic.vehicle3_vprofile == 'sinus':
                if trafic.training:  # create random sinusodial velocity profiles for training
                    SUMO2.prec_train_amplitude = np.random.rand(trafic.number_episodes) * 20/3.6
                    SUMO2.prec_train_mean = np.random.rand(trafic.number_episodes) * 20/3.6 + 10/3.6
                else:
                    SUMO2.prec_train_amplitude = 25/3.6  # a=25/3.6
                    SUMO2.prec_train_mean = 30/3.6  # c=30/3.6
                    SUMO2.create_v_profile_prec(a=SUMO.prec_train_amplitude, c=SUMO.prec_train_mean)
            elif vehicle3_vprofile == 'emergstop':
                SUMO2.create_v_profile_emerg_stop()
            else:
                raise NameError('No valid velocity profile selected')
                
        trafic.vehicle_ego = Vehicle(ego=True, ID='ego', RouteID='RouteID_ego', Route=trafic.Route_Ego, powertrain_concept='ICEV')
        trafic.dynamics_ego = Longitudinal_dynamics(tau=0.5)
        if trafic.vehicle2_exist:
            trafic.vehicle_2 = Vehicle(ego=False, ID='traffic.0', RouteID='traci_route_traffic.0', Route=trafic.Route_Traffic1)
        if trafic.vehicle3_exist:
            trafic.vehicle_3 = Vehicle(ego=False, ID='traffic.1', RouteID='traci_route_traffic.1', Route=trafic.Route_Traffic2)
        acc_controller = {}
        timestep=3
#       if trafic.training and liveplot:
#            fig_running, ax_running_1, ax_running_2, ax_running_3, ax_running_4 = plot_running_init(training)
            
        process_param= multiprocessing.Process()
#        print(process_param.name)
        traci.start(['sumo', '-c', 'SUMO_config.sumocfg', '--no-warnings'], label=str(process_param.name))#, label=sim)
        
        simulation=traci.getConnection(process_param.name)
        simulation.route.add(trafic.vehicle_ego.RouteID, trafic.vehicle_ego.Route)
        
        if trafic.vehicle2_exist:
               simulation.route.add(trafic.vehicle_2.RouteID, trafic.vehicle_2.Route)
                     
        if trafic.vehicle3_exist:
              simulation.route.add(trafic.vehicle_3.RouteID, trafic.vehicle_3.Route)
              simulation.vehicletype.setSpeedFactor(typeID='traffic_vehicle', factor=5.0)
              length_episode = np.zeros((trafic.number_episodes, 1))
              restart_step = 0  # counter for calculating the reset timing when the simulation time gets close to 24 days
              evaluation = False
           
        if trafic.training:
              trafic.vehicle_ego.depart_speed = np.random.randint(0, 30, size=trafic.number_episodes)
        else:
              trafic.vehicle_ego.depart_speed = ego_depart_speed
        simulation.trafficlight.setProgram(tlsID='junction1', programID=np.random.choice(trafic.TLS_ID))
        timestep=4+episode
        if not trafic.training:
               simulation.trafficlight.setPhase('junction1', 0)
        if trafic.training and not trafic.evaluation and trafic.vehicle3_exist:
            trafic.vehicle3 = np.random.choice([True, False], p=[0.95, 0.05])
            simulation.lane.setMaxSpeed('gneE01_0', np.random.choice([8.33, 13.89, 19.44, 25.]))
            simulation.lane.setMaxSpeed('gneE02_0', np.random.choice([8.33, 13.89, 19.44, 25.]))
            simulation.lane.setMaxSpeed('startedge_0', np.random.choice([8.33, 13.89, 19.44, 25.]))
            SUMO2.create_v_profile_prec(a=SUMO2.prec_train_amplitude[episode-1], c=SUMO2.prec_train_mean[episode-1])
        else:
            trafic.vehicle3 = vehicle3_exist
            simulation.lane.setMaxSpeed('startedge_0', 13.89)  # 13.89
            simulation.lane.setMaxSpeed('gneE01_0', 19.44)  # 19.44
            simulation.lane.setMaxSpeed('gneE02_0', 13.89)  # 13.89
            simulation.lane.setMaxSpeed('stopedge_0', 8.33)  # 8.33
           
        trafic.episoden_variante=np.random.rand()*240.
        
        return process_param.name, SUMO2, error
    
   except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        error=True
        traci.close()
#        fitness=timestep
        # raise Exception('Error SimInitializer internal', timestep, process_param.name,  message)
  # return process_param.name, SUMO2, error     
        
if __name__ == "__main__":
    
    """Input ======================================================================================================="""
    loadfile_model_actor = r'saved_models\\actor_dummy'
    loadfile_model_critic = r'saved_models\\critic_dummy'
    savefile_model_actor = r'saved_models\\actor_dummy'
    savefile_model_critic = r'saved_models\\critic_dummy'
    savefile_best_actor = r'saved_models\\best_actor_dummy'
    savefile_best_critic = r'saved_models\\best_critic_dummy'
    savefile_reward = r'saved_models\\rewards_dummy.txt'
    number_episodes = 100000  # number of episodes to run
    trafic.training = True  # when True, RL training is applied to simulation without SUMO GUI use; when False, no training is applied and SUMO GUI is used
    trafic.evaluation = False
    double_gpu = False  # only relevant for computer with two GPUs - keep on False otherwise
    device = 'cpu'  # gpu0, gpu1, cpu
    trafic.TLS_virt_vehicle = True  # Red and Yellow traffic lights are considered as preceding vehicles with v=0
    trafic.TLS_ID = '0'  # program of TLS - 0(TLS with red phase), 1(TLS always green)
    feature_number = 4  # state representation (number of inputs to Neural Network) - currently distance, v_ego, v_preceding, v_allowed
    sample_generation = False  # only relevant for Supervised Pretraining - keep on False
    
    trafic.number_episodes=number_episodes
    """Input NEAT ================================================================================================="""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_SUMO_lang')
    """Initialisation =============================================================================================="""
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device == 'gpu1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    elif device == 'gpu0':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    

    """run simulation =============================================================================================="""
    start = timeit.default_timer()
#       
    """load the config, create a population, evolve and show the result"""
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    """
    Runs NEAT's genetic algorithm for at most n generations.  If n
    is None, run until solution is found or extinction occurs.
   
    The user-provided fitness_function must take only two arguments:
        1. The population as a list of (genome id, genome) tuples.
        2. The current configuration object.
   
    The return value of the fitness function is ignored, but it must assign
    a Python float to the `fitness` member of each genome.
   
    The fitness function is free to maintain external state, perform
    evaluations in parallel, etc.
   
    It is assumed that fitness_function does not modify the list of genomes,
    the genomes themselves (apart from updating the fitness member),
    or the configuration object.
    """
    send_iterator=0
    save_iterator=0
    
    if p.config.no_fitness_termination and (number_episodes is None):
        raise RuntimeError("Cannot have no generational limit with no fitness termination")

    """Initialise simulation ======================================================================================="""
    pool=Pool(processes=os.cpu_count()-6, maxtasksperchild=200)#os.cpu_count())
    
    now=datetime.now()
    nowstr=now.strftime('%Y%m%d%H%M%S')
    format_string='{:0'+str(len(str(number_episodes)))+'.0f}'
    
    nn=1   
    k = 0
    while number_episodes is None or k < number_episodes:
           k += 1
           episode=k
           error=True
           error_ref=True
           
           try:  # for keyboard interrupt     
               print('\n Episode: ', episode, '/', number_episodes)
               send_iterator+=1
               save_iterator+=1
               if send_iterator==30:
                   msg='Episode: '+ str(episode)+ '/'+ str(number_episodes)
                   andIsendtomyself(msg)
                   send_iterator=0
               
#                   
               p.reporters.start_generation(p.generation)
#                
               y=0
               sim_id=[]
               for sims in range(1,len(p.population)+1):
                   if y==os.cpu_count()-1:
                       y=0
                   else:
                       y+=1
                   sim_id.append(y)
                   
               pop_input=list(iteritems(p.population))
               while error:
                   error_count=0
                   try:
                       results=pool.starmap(eval_genomes, zip(sim_id, pop_input, repeat(p.config), repeat(episode), repeat(trafic)))
                       for fitness in results:
                            if fitness==10000:
                                error_count+=1                             
                   except Exception as ex:
                       template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                       message = template.format(type(ex).__name__, ex.args)
                       print(message)
                       andIsendtomyself(message+' Episode '+str(episode))
                       error_count+=1
                       
                   if error_count!=0:
                       print('An Error occured. Restarting episode')
                       andIsendtomyself(str(error_count)+' Error occured. Restarting episode' + str(episode)+'.')                       
                   else:
                       error=False
#               results=pool.starmap(fitness_function, zip(repeat(p.population[1],10), repeat(p.config), repeat(episode)))
#               print('hallo')
#               results=eval_genomes(1, p.population[1], p.config, episode, trafic)
#               resultlist=results.astype(float)
#               print(resultlist)
               nn=0
               for fitness in results:
                   p.population[pop_input[nn][0]].fitness=fitness
                   nn+=1
               # Gather and report statistics.
               best = None
               for g in itervalues(p.population):
                   if best is None or g.fitness > best.fitness:
                       best = g
#                   print(best.fitness, best.size(),p.species.get_species_id(best.key),best.key)
               p.reporters.post_evaluate(p.config, p.population, p.species, best)
               
               if p.best_genome is not None:
                   while error_ref:
                       error_count=0
                       try:
                           ref_result=pool.starmap(eval_genomes, zip(repeat(1), repeat(list([0, p.best_genome]),5), repeat(p.config), [1,2,3,4,5], repeat(trafic)))
                           for fitness in ref_result:
                               if fitness==10000:
                                   error_count+=1
                           best_current_gen=pool.starmap(eval_genomes, zip(repeat(1), repeat(list([0, best]),5), repeat(p.config), [1,2,3,4,5], repeat(trafic)))
                           for fitness in best_current_gen:
                               if fitness==10000:
                                   error_count+=1
                       except Exception as ex:
                           template = "An exception of type {0} occurred in reference episodes. Arguments:\n{1!r}"
                           message = template.format(type(ex).__name__, ex.args)
                           print(message)
                           andIsendtomyself(message+' Episode '+str(episode))
                           error_count+=1
                       if error_count!=0:
                           print('An Error occured in reference episodes. Restarting episode')
                           andIsendtomyself(str(error_count)+' Error occured in reference episodes. Restarting episode' + str(episode)+'.')                       
                       else:
                           error_ref=False   
##                   print(ref_fitness[0], type(ref_fitness[0]))
#                   if not np.isnan(ref_fitness):
#                       p.best_genome.fitness=ref_fitness[0].item()
               # Track the best genome ever seen.
               #if p.best_genome is None or best.fitness > p.best_genome.fitness:
               if p.best_genome is None or np.mean(best_current_gen) > np.mean(ref_result):
                   p.best_genome = best
          
               if not p.config.no_fitness_termination:
                   # End if the fitness threshold is reached.
                   fv = p.fitness_criterion(g.fitness for g in itervalues(p.population))
                   if fv >= p.config.fitness_threshold:
                       p.reporters.found_solution(p.config, p.generation, best)
                       break
          
               # Create the next generation from the current generation.
               p.population = p.reproduction.reproduce(p.config, p.species,
                                                             p.config.pop_size, p.generation)
          
               # Check for complete extinction.
               if not p.species.species:
                   p.reporters.complete_extinction()
          
                   # If requested by the user, create a completely new population,
                   # otherwise raise an exception.
                   if p.config.reset_on_extinction:
                       p.population = p.reproduction.create_new(p.config.genome_type,
                                                                      p.config.genome_config,
                                                                      p.config.pop_size)
                   else:
                       raise CompleteExtinctionException()
          
               # Divide the new population into species.
               p.species.speciate(p.config, p.population, p.generation)
          
               p.reporters.end_generation(p.config, p.population, p.species)
          
               p.generation += 1
          
               if p.config.no_fitness_termination:
                      p.reporters.found_solution(p.config, p.generation, p.best_genome)
              
               if save_iterator==50:
                   save_iterator=0
                   data={p.best_genome, stats}
                   with open('saved models\\'+'best_genome_neat_'+nowstr+'_'+format_string.format(episode)+'-'+format_string.format(number_episodes) , 'wb') as f:
                       pickle.dump(data, f)
                       
           except KeyboardInterrupt:
               print('Manual interrupt')
               break
           
    pool.close()
#   
    data={p.best_genome, stats}
    with open('saved models\\'+'best_genome_neat_'+nowstr+'_'+format_string.format(episode)+'-'+format_string.format(number_episodes) , 'wb') as f:
        pickle.dump(data, f)
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(p.best_genome))

    # Show output of the most fit genome against training data.
    print('\nOutput:')


    if visualize is not None:
        node_names = {-1: 'distance', -2: 'v_ego',-3:'v_prec', -4:'v_allowed', 0: 'a_set'}
        visualize.draw_net(config, p.best_genome, True, filename='saved models\\'+'best_genome_net'+nowstr+'.svg', node_names=node_names)
        visualize.plot_stats(stats, ylog=False, view=True, filename='saved models\\'+'best_genome_stats'+nowstr+'.svg',)
        visualize.plot_species(stats, view=True, filename='saved models\\'+'best_genome_species'+nowstr+'.svg')

    andIsendtomyself('Geschafft!')

    """Postprocessing ==============================================================================================="""
    stop = timeit.default_timer()
    print('Calculation time: ', stop-start)
    