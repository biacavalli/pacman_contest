# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
import time

from capture_agents import CaptureAgent
from game import Directions
from game import Actions
from util import nearest_point
# Removed: from pacmanbuster import HybridInference 

#################
# MCTS Implementation #
#################

class MCTSNode:
    def __init__(self, state, agent_index, parent=None, action=None):
        self.state = state
        self.agent_index = agent_index
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        legal = self.state.get_legal_actions(self.agent_index)
        return len(self.children) == len(legal)

    def best_child(self, c_param=1.41):
        choices = [
            (child.value / (child.visits + 1e-6) +
             c_param * ((2 * self.visits) ** 0.5 / (child.visits + 1e-6)),
             child)
            for child in self.children
        ]
        return max(choices, key=lambda x: x[0])[1]


def simulate_game(state, agent_index, dead_ends=set(), depth_limit=20, is_offensive=True):
    """
    Rollout simulation.
    Includes logic for Dead Ends and Scared Survival Mode.
    """
    current_state = state
    initial_score = state.get_score()
    
    # Pre-calculate home boundary x-coordinate
    layout_width = current_state.data.layout.width
    home_x_max = (layout_width // 2) - 1 if agent_index % 2 == 0 else (layout_width // 2)
    
    for _ in range(depth_limit):
        actions = current_state.get_legal_actions(agent_index)
        if not actions:
            break
        
        best_action = None
        best_score = -float('inf')
        
        # Get simplified data for heuristic
        my_pos = current_state.get_agent_position(agent_index)
        if agent_index % 2 == 0: 
            food_list = current_state.get_blue_food().as_list()
        else: 
            food_list = current_state.get_red_food().as_list()

        for action in actions:
            successor = current_state.generate_successor(agent_index, action)
            new_pos = successor.get_agent_position(agent_index)
            new_state = successor.get_agent_state(agent_index)
            
            if is_offensive:
                # ---------------- OFFENSIVE SIMULATION ----------------
                score_gained = successor.get_score() - initial_score
                if agent_index % 2 != 0: 
                    score_gained = -score_gained
                
                carrying_reward = new_state.num_carrying * 2.0
                
                min_food_dist = 999
                if food_list:
                    min_food_dist = min([util.manhattan_distance(new_pos, f) for f in food_list])
                
                dist_to_home = abs(new_pos[0] - home_x_max)
                
                if new_state.num_carrying > 0:
                    heuristic = (score_gained * 1000) + (carrying_reward * 10) - (dist_to_home * 5)
                else:
                    heuristic = (score_gained * 100) - (min_food_dist * 2)

            else:
                # ---------------- DEFENSIVE SIMULATION ----------------
                
                # Get enemy info
                enemies = [successor.get_agent_state(i) for i in range(successor.get_num_agents()) 
                           if i % 2 != agent_index % 2]
                invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
                
                # Check if we are scared
                my_scared = new_state.scared_timer > 0
                
                # ================= SCARED / SURVIVAL MODE =================
                if my_scared:
                    # 1. IMMEDIATE DEATH CHECK
                    # If an invader is too close, this state is terrible.
                    min_dist_to_invader = 9999
                    if invaders:
                        min_dist_to_invader = min([util.manhattan_distance(new_pos, e.get_position()) for e in invaders])
                    
                    if min_dist_to_invader <= 1:
                        heuristic = -float('inf') # We got eaten!
                    else:
                        # 2. RUN AWAY (Maximize Distance)
                        # We ignore score here. Survival is more important than defending food.
                        heuristic = min_dist_to_invader * 200 
                        
                        # 3. DEAD ENDS ARE FATAL WHEN SCARED
                        if new_pos in dead_ends:
                            heuristic -= 100000 # Absolute prohibition on dead ends when scared
                
                # ================= NORMAL DEFENSE MODE =================
                else:
                    score_change = initial_score - successor.get_score()
                    if agent_index % 2 != 0: 
                        score_change = -score_change
                    
                    if invaders:
                        # Chase invaders
                        min_invader_dist = min([util.manhattan_distance(new_pos, e.get_position()) for e in invaders])
                        heuristic = (score_change * 100) - (min_invader_dist * 10)
                        
                        # Dead End Logic (Aggressive):
                        # Enter dead end ONLY if invader is inside
                        if new_pos in dead_ends:
                            invader_trapped = False
                            for inv in invaders:
                                if inv.get_position() == new_pos:
                                    invader_trapped = True
                                    break
                            
                            if not invader_trapped:
                                heuristic -= 10000 # Don't enter useless dead ends
                    else:
                        heuristic = score_change * 100
                        # Avoid dead ends when patrolling
                        if new_pos in dead_ends:
                            heuristic -= 1000

            if heuristic > best_score:
                best_score = heuristic
                best_action = action
        
        # Execute best action in simulation
        if random.random() < 0.9 and best_action: # Increased greediness for stability
            current_state = current_state.generate_successor(agent_index, best_action)
        elif actions:
            current_state = current_state.generate_successor(agent_index, random.choice(actions))
            
    return current_state.get_score()


def run_mcts(root_state, agent_index, dead_ends=set(), is_offensive=True, time_limit=0.07):
    start_t = time.time()
    root = MCTSNode(root_state, agent_index)
    
    while time.time() - start_t < time_limit:
        node = root
        
        # Selection
        while node.children and node.is_fully_expanded():
            node = node.best_child()
        
        # Expansion
        if not node.is_fully_expanded():
            legal = node.state.get_legal_actions(agent_index)
            tried_actions = [c.action for c in node.children]
            for a in legal:
                if a not in tried_actions:
                    new_state = node.state.generate_successor(agent_index, a)
                    child = MCTSNode(new_state, agent_index, parent=node, action=a)
                    node.children.append(child)
                    node = child
                    break
        
        # Simulation
        reward = simulate_game(node.state, agent_index, dead_ends=dead_ends, depth_limit=12, is_offensive=is_offensive)
        
        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    if root.children:
        best = max(root.children, key=lambda n: n.visits)
        return best.action
    
    return Directions.STOP


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
        return random.choice(actions) if actions else Directions.STOP

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # This is a hack for Pacman's movement when it passes a corner
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
    def choose_action(self, game_state):
        action = run_mcts(game_state, self.index, is_offensive=True, time_limit=0.9)
        if action == Directions.STOP:
            actions = game_state.get_legal_actions(self.index)
            if actions:
                return random.choice([a for a in actions if a != Directions.STOP])
        return action

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        food = self.get_food(successor).as_list()
        features['successor_score'] = -len(food)

        if food:
            dist_to_food = min([self.get_maze_distance(my_pos, f) for f in food])
            features['distance_to_food'] = dist_to_food

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [g for g in enemies if not g.is_pacman and g.get_position() is not None and g.scared_timer == 0]

        if ghosts:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            features['ghost_distance'] = min(dists)

        capsules = self.get_capsules(successor)
        if capsules:
            features['capsule_distance'] = min([self.get_maze_distance(my_pos, c) for c in capsules])

        carried = successor.get_agent_state(self.index).num_carrying
        if carried > 1:
            mid = self.start
            features['return_home'] = self.get_maze_distance(my_pos, mid)

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'distance_to_food': -1,
            'ghost_distance': 3,
            'capsule_distance': -3,
            'return_home': 3
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    def find_dead_ends(self, game_state):
        layout = game_state.data.layout
        walls = layout.walls
        width, height = layout.width, layout.height
        non_walls = []
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    non_walls.append((x, y))
        dead_ends = set()
        changed = True
        while changed:
            changed = False
            for pos in non_walls:
                if pos in dead_ends: continue
                x, y = pos
                neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                valid_neighbors = [n for n in neighbors if not walls[n[0]][n[1]] and n not in dead_ends]
                # A dead end is a non-wall tile with at most one non-wall neighbor
                if len(valid_neighbors) <= 1: 
                    dead_ends.add(pos)
                    changed = True
        return dead_ends

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        # Removed: self.trackers = {} initialization
        # Removed: Loop to initialize HybridInference trackers
        self.dead_ends = self.find_dead_ends(game_state)

    def choose_action(self, game_state):
        # Removed: Tracker observation and elapse_time calls
        # for tracker in self.trackers.values():
        #     tracker.observe(self, game_state)
        #     tracker.elapse_time(game_state)
        
        # Check scared status for MCTS tuning
        my_state = game_state.get_agent_state(self.index)
        
        # Run MCTS
        action = run_mcts(game_state, self.index, 
                          dead_ends=self.dead_ends, 
                          is_offensive=False, 
                          time_limit=0.9)
        
        if action == Directions.STOP:
            actions = game_state.get_legal_actions(self.index)
            if actions:
                return random.choice([a for a in actions if a != Directions.STOP])
        return action

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        scared_timer = my_state.scared_timer
        features['on_defense'] = 0 if my_state.is_pacman else 1

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        features['num_invaders'] = len(visible)

        # Inferred positions logic has been simplified/removed, 
        # so we only rely on visible invaders for targets.
        inferred_positions = [] 
        
        if scared_timer > 0:
            targets = [e.get_position() for e in visible] # Inferred positions are ignored when scared
            if targets:
                dists = [self.get_maze_distance(my_pos, t) for t in targets]
                features['scared_distance'] = -min(dists)
            
            if my_pos in self.dead_ends:
                features['in_dead_end'] = 1
                
            if action == Directions.STOP:
                features['stop'] = 1
        else:
            targets = [e.get_position() for e in visible] # Inferred positions are now completely ignored
            if targets:
                dists = [self.get_maze_distance(my_pos, t) for t in targets]
                features['invader_distance'] = min(dists)
                border_x = successor.data.layout.width // 2
                for t in targets:
                    if my_pos[0] > border_x and t[0] <= border_x:
                        features['block'] = 1
            else:
                features['patrol'] = self.compute_patrol_priority(successor, my_pos)

            if my_pos in self.dead_ends:
                features['in_dead_end'] = 1
            if action == Directions.STOP:
                features['stop'] = 1
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1

        return features

    def compute_patrol_priority(self, game_state, my_pos):
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        mid_x = width // 2
        valid_border_targets = []
        for y in range(1, height - 1):
            pos = (mid_x, y)
            if not game_state.has_wall(pos[0], pos[1]) and pos not in self.dead_ends:
                valid_border_targets.append(pos)

        if not valid_border_targets: return 0 

        food_defending = self.get_food_you_are_guarding(game_state).as_list()
        if food_defending:
            # Sort border targets by proximity to the food we are guarding
            valid_border_targets.sort(key=lambda p: min(self.get_maze_distance(p, f) for f in food_defending))

        # Return distance to the best patrol spot
        return min(self.get_maze_distance(my_pos, p) for p in valid_border_targets)

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)
        if my_state.scared_timer > 0:
            return {
                'scared_distance': 200, # Highly prioritize running away
                'in_dead_end': -5000, 	# Massive penalty for dead ends
                'stop': -100,
                'on_defense': 50
            }
        else:
            return {
                'num_invaders': -1000,
                'on_defense': 100,
                'invader_distance': -10,
                'block': 20,
                'patrol': -2,
                'in_dead_end': -200,
                'stop': -100,
                'reverse': -2
            }
