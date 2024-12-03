# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveQLerningAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}




class OffensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        # number of food left
        features['successor_score'] = -len(self.get_food(successor).as_list())

        #distance to the nearest food
        food_list = self.get_food(successor).as_list()
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # distance to the safe zon
        carrying_food = successor.get_agent_state(self.index).num_carrying
        if carrying_food > 0:
            mid_x = game_state.data.layout.width // 2
            if self.red:
                safe_boundary = [(mid_x - 1, y) for y in range(game_state.data.layout.height)
                                 if not game_state.has_wall(mid_x - 1, y)]
            else:
                safe_boundary = [(mid_x, y) for y in range(game_state.data.layout.height)
                                 if not game_state.has_wall(mid_x, y)]
            min_safe_distance = min([self.get_maze_distance(my_pos, safe) for safe in safe_boundary])
            features['distance_to_safe_zone'] = min_safe_distance
        else:
            features['distance_to_safe_zone'] = 0

        # decider whether to escape via capsule or safe zone)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible_enemies = [a for a in enemies if a.get_position() is not None and not a.is_pacman]
        non_scared_enemies = [a for a in visible_enemies if a.scared_timer == 0]

        if carrying_food > 0 and len(non_scared_enemies) > 0:
            # Closest enemy
            enemy_distances = [self.get_maze_distance(my_pos, enemy.get_position()) for enemy in non_scared_enemies]
            closest_enemy_distance = min(enemy_distances)
            features['distance_to_enemy'] = closest_enemy_distance

            capsules = self.get_capsules(successor)
            min_capsule_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules]) if capsules else float('inf')
            enemy_to_capsule = min([self.get_maze_distance(enemy.get_position(), capsule)
                                    for capsule in capsules for enemy in non_scared_enemies]) if capsules else float('inf')

            enemy_to_safe_zone = min([self.get_maze_distance(enemy.get_position(), safe)
                                      for safe in safe_boundary for enemy in non_scared_enemies])

            safe_zone_risk = enemy_to_safe_zone - features['distance_to_safe_zone']
            capsule_risk = enemy_to_capsule - min_capsule_distance

            # Decide based on the safer option
            if capsules and capsule_risk < safe_zone_risk and min_capsule_distance < closest_enemy_distance:
                features['use_capsule'] = 1
                features['distance_to_capsule'] = min_capsule_distance
            else:
                features['use_capsule'] = 0
                features['distance_to_capsule'] = 0
        else:
            features['distance_to_capsule'] = 0
            features['use_capsule'] = 0

        # penalize stopping unless absolutely necessary
        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0

        # avoid defensive enemy agents
        if len(visible_enemies) > 0:
            min_enemy_distance = min([self.get_maze_distance(my_pos, enemy.get_position()) for enemy in visible_enemies])
            features['distance_to_defensive_enemy'] = min_enemy_distance
        else:
            features['distance_to_defensive_enemy'] = 0

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,           # pprioritize eating food
            'distance_to_food': -1,          # go towards food
            'distance_to_safe_zone': -10,    # returning to safety
            'distance_to_enemy': 5,          # avoid enemies
            'distance_to_capsule': -50,      # go towards capsules if safer
            'use_capsule': 1000,             #prefer capsules when safer
            'stop': -1000,                   #strongly avoid stopping
            'distance_to_defensive_enemy': 10 # avoid defensive agents
        }

    def choose_action(self, game_state):
        """
        Overriding to ensure movement in all cases.
        """
        actions = game_state.get_legal_actions(self.index)

        # evaluate each action
        values = [self.evaluate(game_state, action) for action in actions]
        max_value = max(values)
        best_actions = [action for action, value in zip(actions, values) if value == max_value]

        #svoid stopping
        if len(best_actions) > 1 and Directions.STOP in best_actions:
            best_actions.remove(Directions.STOP)

        return random.choice(best_actions)


class MinimaxDefensiveAgent(CaptureAgent):
    """
    A defensive agent that uses the Minimax algorithm to make decisions,
    taking into account the positions of the offensive agent and the opposite team.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # computes whether we're on defense or offense
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # Compute distance to the nearest offensive agent
        offensive_agents = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if offensive_agents:
            dists = [self.get_maze_distance(my_pos, agent.get_position()) for agent in offensive_agents]
            features['distance_to_offensive_agent'] = min(dists)
        else:
            features['distance_to_offensive_agent'] = 0

        # Compute distance to the nearest ghost
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if ghosts:
            dists = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
            features['distance_to_ghost'] = min(dists)
        else:
            features['distance_to_ghost'] = 0

        #compute distance to the nearest capsule
        capsules = self.get_capsules_you_are_defending(successor)
        if capsules:
            dists = [self.get_maze_distance(my_pos, capsule) for capsule in capsules]
            features['distance_to_capsule'] = min(dists)
        else:
            features['distance_to_capsule'] = 0

        #compute the number of food items left
        food_list = self.get_food_you_are_defending(successor).as_list()
        features['num_food_left'] = len(food_list)

        #compute distance to the home boundary
        mid_x = (successor.data.layout.width // 2) - 1 if self.red else (successor.data.layout.width // 2)
        home_positions = [(mid_x, y) for y in range(successor.data.layout.height) if not successor.has_wall(mid_x, y)]
        dists = [self.get_maze_distance(my_pos, pos) for pos in home_positions]
        features['distance_to_home'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'distance_to_offensive_agent': -10,
            'distance_to_ghost': 1,
            'distance_to_capsule': -5,
            'num_food_left': -2,
            'distance_to_home': -1,
            'stop': -100,
            'reverse': -2
        }

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def minimax(self, game_state, depth, agent_index):
        if depth == 0 or game_state.is_over():
            return self.evaluate(game_state, Directions.STOP)

        if agent_index == self.index:
            return self.max_value(game_state, depth)
        else:
            return self.min_value(game_state, depth, agent_index)

    def max_value(self, game_state, depth):
        v = float('-inf')
        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            v = max(v, self.minimax(successor, depth - 1, self.get_opponents(game_state)[0]))
        return v

    def min_value(self, game_state, depth, agent_index):
        v = float('inf')
        for action in game_state.get_legal_actions(agent_index):
            successor = game_state.generate_successor(agent_index, action)
            if agent_index == self.get_opponents(game_state)[-1]:
                v = min(v, self.minimax(successor, depth - 1, self.index))
            else:
                v = min(v, self.minimax(successor, depth, self.get_opponents(game_state)[self.get_opponents(game_state).index(agent_index) + 1]))
        return v

    def get_action(self, game_state):
        best_action = Directions.STOP
        best_value = float('-inf')
        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            value = self.minimax(successor, 2, self.get_opponents(game_state)[0])
            if value > best_value:
                best_value = value
                best_action = action
        return best_action


