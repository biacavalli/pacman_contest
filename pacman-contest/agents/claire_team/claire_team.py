




import random
import time
from contest import util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
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


####################
# Monte Carlo Node #
####################

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


def simulate_game(state, agent_index, depth_limit=20, is_offensive=True):
    """
    Rollout now understands that carrying food home = points.
    """
    current_state = state
    initial_score = state.get_score()
    initial_carrying = state.get_agent_state(agent_index).num_carrying
    
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
        if agent_index % 2 == 0: food_list = current_state.get_blue_food().as_list()
        else: food_list = current_state.get_red_food().as_list()

        for action in actions:
            successor = current_state.generate_successor(agent_index, action)
            new_pos = successor.get_agent_position(agent_index)
            new_state = successor.get_agent_state(agent_index)
            
            if is_offensive:
                # 1. ACTUAL SCORE (Deposited points)
                score_gained = successor.get_score() - initial_score
                if agent_index % 2 != 0: score_gained = -score_gained
                
                # 2. POTENTIAL SCORE (Carrying)
                # If we are carrying a lot, we must value safety (not dying) highly
                carrying_reward = new_state.num_carrying * 2.0
                
                # 3. DISTANCE TO FOOD (Hunting)
                min_food_dist = 999
                if food_list:
                    min_food_dist = min([util.manhattan_distance(new_pos, f) for f in food_list])
                
                # 4. DISTANCE HOME (Retreating)
                # Calculate distance to center line
                dist_to_home = abs(new_pos[0] - home_x_max)
                
                # DYNAMIC HEURISTIC
                # If carrying score, priority shifts to returning home
                if new_state.num_carrying > 0:
                    # Huge reward for depositing (score_gained), penalty for being far from home
                    heuristic = (score_gained * 1000) + (carrying_reward * 10) - (dist_to_home * 5)
                else:
                    # Standard hunting
                    heuristic = (score_gained * 100) - (min_food_dist * 2)

            else:
                # Defensive heuristic (simple)
                score_change = initial_score - successor.get_score()
                if agent_index % 2 != 0: score_change = -score_change
                heuristic = score_change * 100
            
            if heuristic > best_score:
                best_score = heuristic
                best_action = action
        
        if random.random() < 0.8 and best_action:
            current_state = current_state.generate_successor(agent_index, best_action)
        else:
            current_state = current_state.generate_successor(agent_index, random.choice(actions))
            
    return current_state.get_score()


def run_mcts(root_state, agent_index, is_offensive=True, time_limit=0.07):
    start_t = time.time()
    root = MCTSNode(root_state, agent_index)
    
    # Loop until time runs out
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
        reward = simulate_game(node.state, agent_index, depth_limit=10, is_offensive=is_offensive)
        
        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    if root.children:
        # Pick the most visited node (robust child)
        best = max(root.children, key=lambda n: n.visits)
        return best.action
    
    return Directions.STOP




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
    """
    A reflex agent that seeks food but knows when to run home.
    """
    
    def get_boundary(self, game_state):
        """
        Finds the list of boundary coordinates on the home side.
        """
        layout = game_state.data.layout
        width = layout.width
        height = layout.height
        
        # Red team (even index) home is on the left (x < width/2)
        # Blue team (odd index) home is on the right (x >= width/2)
        if self.index % 2 == 0:
            x = (width // 2) - 1 
        else:
            x = (width // 2)
            
        boundary_nodes = []
        for y in range(height):
            if not layout.walls[x][y]:
                boundary_nodes.append((x, y))
        return boundary_nodes

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # 1. SURVIVAL CHECK
        # If we have food and a ghost is close, panic and use MCTS
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if ghosts:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            # If carrying food and ghost is within 5 steps, calculate safe path
            if my_state.num_carrying > 0 and min(dists) < 6:
                return run_mcts(game_state, self.index, is_offensive=True, time_limit=0.07)
            # If trapped
            if min(dists) < 4 and len(game_state.get_legal_actions(self.index)) <= 2:
                 return run_mcts(game_state, self.index, is_offensive=True, time_limit=0.07)

        # 2. RETREAT LOGIC (The logic you were missing)
        # If we have a lot of food, we might want to prioritize MCTS to find the safest way home
        # rather than the quickest way, but Reflex is usually fine for navigation.
        
        return super().choose_action(game_state)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food(successor).as_list()
        
        # --- Feature 1: Food ---
        features['successor_score'] = -len(food_list)
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # --- Feature 2: Returning Home ---
        # We calculate distance to the nearest boundary point
        boundary_nodes = self.get_boundary(game_state)
        dist_to_home = min([self.get_maze_distance(my_pos, b) for b in boundary_nodes])
        features['distance_to_home'] = dist_to_home
        
        # --- Feature 3: Ghosts ---
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if len(ghosts) > 0:
            min_ghost_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])
            if min_ghost_dist < 5:
                features['ghost_distance'] = min_ghost_dist
        
        # --- Feature 4: Mechanics ---
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        # Current status
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying
        
        # Base weights (Hunting mode)
        weights = {
            'successor_score': 100,
            'distance_to_food': -1,
            'distance_to_home': 0,      # Don't care about home yet
            'ghost_distance': 200,      # Run from ghosts
            'stop': -100,
            'reverse': -2
        }
        
        # --- DYNAMIC WEIGHTS ADJUSTMENT ---
        
        # 1. The "Greedy" Threshold
        # If we have collected 2 or more dots, start thinking about home.
        if carrying >= 2:
            # We add a small pull towards home.
            # This allows us to eat food if it's on the way, but generally drift back.
            weights['distance_to_home'] = -2 

        # 2. The "Deposit" Mode
        # If we have 5+ dots, OR the game is almost over, stop hunting and RUN.
        # Also if the nearest food is very far away, just go home and bank the points.
        food_left = len(self.get_food(game_state).as_list())
        
        if carrying >= 5 or (food_left <= 2):
            weights['distance_to_food'] = 0     # Ignore food
            weights['successor_score'] = 0      # Ignore score count
            weights['distance_to_home'] = -10   # SPRINT HOME
            
        return weights
    




class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


