
import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
       
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
       
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
      
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
      
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    This offensive agent focuses on collecting food and returning it to its home base, however, when a ghost is persuing him
    he tries to escape while avoiding going into places where there is a dead end. Moreover it tries to have capsules as a 
    tool to avoid getting killed, and once the capsule is eaten, he ignores the enemy ghosts (so that it focuses on scoring),
    but if the ghosts are too close, he tries to eat them.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()

        # Position and food carrying status
        my_pos = successor.get_agent_state(self.index).get_position()
        carried_food = successor.get_agent_state(self.index).num_carrying
        features['successor_score'] = -len(food_list)

        # Distance to nearest food
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Distance to the home base
        if carried_food > 0:
            home_positions = self.get_home_positions(successor)
            min_home_distance = min([self.get_maze_distance(my_pos, home) for home in home_positions])
            features['distance_to_home'] = min_home_distance

        # Prevent getting eaten by ghosts or even persue scared ghosts only if they are close
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        scared_ghosts = [g for g in ghosts if g.scared_timer > 0]
        normal_ghosts = [g for g in ghosts if g.scared_timer <= 1]

        if scared_ghosts:
            for ghost in scared_ghosts:
                ghost_distance = self.get_maze_distance(my_pos, ghost.get_position())
                if ghost_distance == 1:
                    features['eat_scared_ghost'] = 1  # Encourage eating scared ghosts
                if ghost.scared_timer <= 1:
                    features['ghost_near_expiry'] = 1  # Avoid ghosts close to recovery
        if normal_ghosts:
            normal_ghost_distances = [self.get_maze_distance(my_pos, g.get_position()) for g in normal_ghosts]
            closest_normal_ghost = min(normal_ghost_distances)
            features['ghost_distance'] = closest_normal_ghost

            if closest_normal_ghost <= 5:
                features['is_being_chased'] = 1
            if closest_normal_ghost <= 2:
                features['danger_zone'] = 1

        # Detect potential dead ends
        if self.is_dead_end(successor):
            features['dead_end'] = 1

        # Use capsules to avoid getting killed
        capsules = self.get_capsules(successor)
        if capsules:
            min_capsule_distance = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
            features['distance_to_capsule'] = min_capsule_distance

        return features

    def get_weights(self, game_state, action):
        """
        Assign weights to features based on priorities.
        """
        return {
            'successor_score': 100,
            'distance_to_food': -1,
            'distance_to_home': -2,
            'ghost_distance': 2,
            'is_being_chased': -500,
            'danger_zone': -1000,
            'distance_to_capsule': -5,
            'dead_end': -100,
            'eat_scared_ghost': 50,
            'ghost_near_expiry': -300,
        }

    def is_dead_end(self, game_state):
        """
        Tries to determine if the current position leads to a dead end, i believe it doesn't work that well.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        legal_actions = game_state.get_legal_actions(self.index)
        # Check for tight corridors
        num_open_paths = sum(
            1 for action in legal_actions
            if self.get_successor(game_state, action).get_legal_actions(self.index)
        )
        return num_open_paths <= 1

    def get_home_positions(self, game_state):
        """
        Identify all positions on the agent's home side, that helps the agent find a fastest path to return home,
        If we only looked at our starting position, mabye the agent would find a worst path. We detect the middle of the 
        field.
        """
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        mid_x = width // 2
        if self.red: 
            return [(x, y) for x in range(mid_x) for y in range(height) if not walls[x][y]]
        else:  
            return [(x, y) for x in range(mid_x, width) for y in range(height) if not walls[x][y]]

    def choose_action(self, game_state):
        """
        Modify the action selection to account for scared ghosts and their timers.
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Prioritize escape
        for action in best_actions:
            successor = self.get_successor(game_state, action)
            my_pos = successor.get_agent_state(self.index).get_position()
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
            normal_ghosts = [g for g in ghosts if g.scared_timer <= 1]

            if normal_ghosts:
                ghost_distances = [self.get_maze_distance(my_pos, g.get_position()) for g in normal_ghosts]
                if min(ghost_distances) <= 2:  # Immediate danger
                    safe_actions = [a for a in actions if a != Directions.STOP and not self.is_dead_end(self.get_successor(game_state, a))]
                    if safe_actions:
                        return random.choice(safe_actions)

        return random.choice(best_actions)

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    While he hasn't seen the enemy pacman, it wanders around each food position (like that we avoid going into useless paths
    where the enemy pacman wont be). Once it sees the pacman it persues it. If the pacman has eaten a capsule and we are scared,
    we escape trying to avoid dead ends to not get killed.
    """
    
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.food_positions = self.get_food_you_are_defending(game_state).as_list()
        self.food_index = 0
        self.visited_food = set()

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # If no invaders are visible, move towards the next food position
        if len(invaders) == 0:
            if not self.food_positions:
                self.food_positions = self.get_food_you_are_defending(successor).as_list()
                self.visited_food = set()
                self.food_index = 0

            # Get the current target food position
            target_food = self.food_positions[self.food_index]
            distance_to_food = self.get_maze_distance(my_pos, target_food)
            features['distance_to_food'] = distance_to_food

            # Check if the agent has reached the current target food
            if my_pos == target_food:
                self.visited_food.add(target_food)
                self.food_index = (self.food_index + 1) % len(self.food_positions)
                while self.food_positions[self.food_index] in self.visited_food:
                    self.food_index = (self.food_index + 1) % len(self.food_positions)

        if action == Directions.STOP: 
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000, 
            'on_defense': 100, 
            'invader_distance': -10, 
            'distance_to_food': -1,  
            'stop': -100, 
            'reverse': -2
        }
    
    def is_dead_end(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        legal_actions = game_state.get_legal_actions(self.index)
        # Check for tight corridors
        num_open_paths = sum(
            1 for action in legal_actions
            if self.get_successor(game_state, action).get_legal_actions(self.index)
        )
        return num_open_paths <= 1

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a). If the agent is scared, it will try to escape.
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Check if the agent is scared
        my_state = game_state.get_agent_state(self.index)
        if my_state.scared_timer > 1:
            # Find the nearest non-scared enemy ghost
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            non_scared_ghosts = [a for a in enemies if not a.is_pacman and a.scared_timer <= 1 and a.get_position() is not None]
            
            if non_scared_ghosts:
                # Calculate distances to non-scared ghosts
                ghost_distances = [self.get_maze_distance(my_state.get_position(), g.get_position()) for g in non_scared_ghosts]
                closest_ghost_distance = min(ghost_distances)
                
                # If a non-scared ghost is too close, try to escape
                if closest_ghost_distance <= 5:
                    escape_actions = [a for a in actions if a != Directions.STOP and not self.is_dead_end(self.get_successor(game_state, a))]
                    if escape_actions:
                        # Choose the action that maximizes distance from the closest ghost
                        escape_values = [self.get_maze_distance(self.get_successor(game_state, a).get_agent_state(self.index).get_position(), non_scared_ghosts[0].get_position()) for a in escape_actions]
                        best_escape = max(escape_values)
                        best_escape_actions = [a for a, v in zip(escape_actions, escape_values) if v == best_escape]
                        return random.choice(best_escape_actions)

        return random.choice(best_actions)
