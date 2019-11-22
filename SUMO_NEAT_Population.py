"""Implements the core evolution algorithm."""
from __future__ import print_function

import sys
import os
import numpy as np
from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
from Send_Message_Bot2 import andIsendtomyself
import pickle
from datetime import datetime
import cProfile
#from SUMO import SUMO, features
# import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
import traci
import show_best_result as sbr
os.environ['PATH'] += os.pathsep + r'C:\Users\peters\.conda\envs\NEAT\Library\bin\graphviz'

class CompleteExtinctionException(Exception):
    pass



class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def run(self, fitness_function, trafic, SUMO, n=None):
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
        number_episodes=n
        send_iterator=0
        
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        """Initialise simulation ======================================================================================="""
        traci.route.add(trafic.vehicle_ego.RouteID, trafic.vehicle_ego.Route)
        if trafic.vehicle2_exist:
               traci.route.add(trafic.vehicle_2.RouteID, trafic.vehicle_2.Route)
                     
        if trafic.vehicle3_exist:
              traci.route.add(trafic.vehicle_3.RouteID, trafic.vehicle_3.Route)
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
              
#        traci.vehicle.setSpeedMode(trafic.vehicle_ego.ID, 16)             
        if trafic.training:
              trafic.vehicle_ego.depart_speed = np.random.randint(0, 30, size=number_episodes)
        else:
              trafic.vehicle_ego.depart_speed = ego_depart_speed
              traci.trafficlight.setProgram(tlsID='junction1', programID=TLS_ID)

       

           
        k = 0
        while n is None or k < n:
               k += 1
               episode=k
               try:  # for keyboard interrupt
                   """Check if total simulation time is close to 24 days ======================================================"""
                   # TraCI time inputs have a maximum value of ~24days --> restart SUMO to reset time
                   if np.sum(length_episode[restart_step:])*SUMO.sim['timestep'] > 2000000:
                       print('Almost 24 days of simulation time reached! Restarting SUMO and continue with next episode...')
                       traci.close()
                       traci.start(['sumo', '-c', 'SUMO_config.sumocfg'])
                       traci.route.add(trafic.vehicle_ego.RouteID, trafic.vehicle_ego.Route)
                       if trafic.vehicle2_exist:
                           traci.route.add(trafic.vehicle_2.RouteID, trafic.vehicle_2.Route)
                       if trafic.vehicle3_exist:
                           traci.route.add(trafic.vehicle_3.RouteID, trafic.vehicle_3.Route)
                       restart_step = episode
                   print('Episode: ', episode, '/', number_episodes)
                   send_iterator+=1
                   if send_iterator==10:
                       msg='Episode: '+ str(episode)+ '/'+ str(number_episodes)
                       andIsendtomyself(msg)
                       send_iterator=0
       
                   """Initialise episode =================================================================================="""
#                   SUMO.init_vars_episode()
#                   dynamics_ego.reset_variables()
#                   if controller == 'DQN' or controller == 'DDPG' or controller == 'hybrid_a' or controller == 'DDPG_v':
#                       nn_controller.reset_variables()
#                   if controller == 'ACC' or controller == 'hybrid_a':
#                       acc_controller.create_mode_map()
#                   if exploration_policy == 'ACC':
#                       explo_policy.create_mode_map()
#                   if (controller == 'DDPG' or controller == 'hybrid_a' or controller == 'DDPG_v') and ((episode+1) % 5 == 0):  # perform an evaluation episode (without exploration noise) every x episodes to observe the cum reward progress
#                       evaluation = True
       
                   """Anmerkung: Hier werden einige Variationen des Verkehrsszenarios für meine Trainingsepisoden definiert, wenn 'training = True'
                   gesetzt ist. Im Fall 'training = False' oder 'evaluation = True' (Evaluierungsepisoden unter gleichen Randbedingungen) wird immer eine
#                   Episode mit gleichen Randbedingungen (z.B. Geschwindigkeitsprofil vorausfahrendes Fahrzeug) gesetzt"""
#                   if trafic.evaluation:
#                       traci.vehicle.add(trafic.vehicle_ego.ID, trafic.vehicle_ego.RouteID, departSpeed='0',
#                                         typeID='ego_vehicle')  # Ego vehicle
#                       traci.trafficlight.setPhase('junction1', 0)  # set traffic light phase to 0 for evaluation (same conditions)
#                   else:
#                       traci.vehicle.add(trafic.vehicle_ego.ID, trafic.vehicle_ego.RouteID, departSpeed=np.array2string(trafic.vehicle_ego.depart_speed[episode]), typeID='ego_vehicle')  # Ego vehicle
                   if not trafic.training:
                           traci.trafficlight.setPhase('junction1', 0)
                   if trafic.training and not evaluation and trafic.vehicle3_exist:
                       trafic.vehicle3 = np.random.choice([True, False], p=[0.95, 0.05])
                       traci.lane.setMaxSpeed('gneE01_0', np.random.choice([8.33, 13.89, 19.44, 25.]))
                       traci.lane.setMaxSpeed('gneE02_0', np.random.choice([8.33, 13.89, 19.44, 25.]))
                       traci.lane.setMaxSpeed('startedge_0', np.random.choice([8.33, 13.89, 19.44, 25.]))
                       SUMO.create_v_profile_prec(a=SUMO.prec_train_amplitude[episode-1], c=SUMO.prec_train_mean[episode-1])
                   else:
                       trafic.vehicle3 = vehicle3_exist
                       traci.lane.setMaxSpeed('startedge_0', 13.89)  # 13.89
                       traci.lane.setMaxSpeed('gneE01_0', 19.44)  # 19.44
                       traci.lane.setMaxSpeed('gneE02_0', 13.89)  # 13.89
                       traci.lane.setMaxSpeed('stopedge_0', 8.33)  # 8.33
                       
                   trafic.episoden_variante=np.random.rand()*240.
#                   if trafic.vehicle2_exist:
#                       traci.vehicle.add(vehicle_2.ID, vehicle_2.RouteID, typeID='traffic_vehicle')  # preceding vehicle 1
#                   if trafic.vehicle3:
#                       traci.vehicle.add(trafic.vehicle_3.ID, trafic.vehicle_3.RouteID, typeID='traffic_vehicle')  # preceding vehicle 2
#                       if trafic.training and not evaluation:
#                           traci.vehicle.moveTo(trafic.vehicle_3.ID, 'gneE01_0', np.random.rand()*240.)
#                       else:
#                           traci.vehicle.moveTo(trafic.vehicle_3.ID, 'gneE01_0', 0.)
#       
#                   traci.simulationStep()  # to spawn vehicles
##                   if controller != 'SUMO':
##                       traci.vehicle.setSpeedMode(trafic.vehicle_ego.ID, 16)  # only emergency stopping at red traffic lights --> episode ends
#                   if trafic.vehicle2_exist:
#                       traci.vehicle.setSpeedMode(trafic.vehicle_2.ID, 17)
#                   if trafic.vehicle3:
#                       traci.vehicle.setSpeedMode(trafic.vehicle_3.ID, 17)
#       
#                   SUMO.currentvehiclelist = traci.vehicle.getIDList()
#       
#                   # SUMO subscriptions
#                   traci.vehicle.subscribeLeader(trafic.vehicle_ego.ID, 10000)
#                   traci.vehicle.subscribe(trafic.vehicle_ego.ID, [traci.constants.VAR_SPEED, traci.constants.VAR_BEST_LANES, traci.constants.VAR_FUELCONSUMPTION,
#                                                            traci.constants.VAR_NEXT_TLS, traci.constants.VAR_ALLOWED_SPEED, traci.constants.VAR_LANE_ID])
#                   
                   self.reporters.start_generation(self.generation)
#                   print(self.population[49+k])
                   # Evaluate all genomes using the user-provided function.
                   cProfile.runctx('fitness_function(list(iteritems(self.population)), self.config, episode)', globals(), locals())
#                   print(self.fitness)
                   # Gather and report statistics.
                   best = None
                   for g in itervalues(self.population):
                       if best is None or g.fitness > best.fitness:
                           best = g
#                   print(best.fitness, best.size(),self.species.get_species_id(best.key),best.key)
                   self.reporters.post_evaluate(self.config, self.population, self.species, best)
              
                   # Track the best genome ever seen.
                   if self.best_genome is None or best.fitness > self.best_genome.fitness:
                       self.best_genome = best
              
                   if not self.config.no_fitness_termination:
                       # End if the fitness threshold is reached.
                       fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                       if fv >= self.config.fitness_threshold:
                           self.reporters.found_solution(self.config, self.generation, best)
                           break
              
                   # Create the next generation from the current generation.
                   self.population = self.reproduction.reproduce(self.config, self.species,
                                                                 self.config.pop_size, self.generation)
              
                   # Check for complete extinction.
                   if not self.species.species:
                       self.reporters.complete_extinction()
              
                       # If requested by the user, create a completely new population,
                       # otherwise raise an exception.
                       if self.config.reset_on_extinction:
                           self.population = self.reproduction.create_new(self.config.genome_type,
                                                                          self.config.genome_config,
                                                                          self.config.pop_size)
                       else:
                           raise CompleteExtinctionException()
              
                   # Divide the new population into species.
                   self.species.speciate(self.config, self.population, self.generation)
              
                   self.reporters.end_generation(self.config, self.population, self.species)
              
                   self.generation += 1
              
                   if self.config.no_fitness_termination:
                          self.reporters.found_solution(self.config, self.generation, self.best_genome)
#                   
#                   print('Cumulative Reward:', cum_reward[episode])
#                   if evaluation:
#                       cum_reward_evaluation.append(cum_reward[episode])
#                       evaluation = False
#                       if cum_reward[episode] > best_cum_reward:
#                           nn_controller.save_models(savefile_best_actor+'_'+str(episode), savefile_best_critic+'_'+str(episode))
#                           best_cum_reward = cum_reward[episode]
#                           
#                   if training and (controller == 'DQN' or controller == 'hybrid_a' or controller == 'DDPG' or controller == 'DDPG_v') and liveplot:
#                       reward_mean100[episode] = nn_controller.reward_mean_100_running(cum_reward, episode)
#                       nn_controller.weight_observer(episode)
#                       plot_running(reward_mean100, episode, cum_reward_evaluation)
#                   data_export[:, 0] = cum_reward[:, 0]
#                   data_export[:, 1] = length_episode[:, 0]
#                   if training:
#                       try:
#                           if (episode+1) % 25 == 0:  # ==> save rewards every 50 episodes
#                               np.savetxt(savefile_reward, data_export)
#                           if (episode+1) % 25 == 0:  # save model every 50 episodes
#                               nn_controller.save_models(savefile_model_actor, savefile_model_critic)
#                               
#                       except OSError:
#                           print('File saving failed')
#                           pass
#                           
#                   if acc_controller:
#                       acc_controller.reset_integral_error()
#                       
               except KeyboardInterrupt:
                   print('Manual interrupt')
                   break
        traci.close()
#        traci.start(['sumo-gui', '-c', 'SUMO_config.sumocfg'])
#        sbr.result(self.best_genome, self.config,  trafic)
#        sbr.eval_genomes(self.best_genome, self.config, 0, SUMO, trafic)
        now=datetime.now()
        nowstr=now.strftime('%Y%m%d%H%M%S')
        with open('H:\\MT\\Python\\NEAT und SUMO\\saved models\\'+'best_genome_neat'+nowstr , 'wb') as f:
            pickle.dump(self.best_genome, f)
        return self.best_genome
