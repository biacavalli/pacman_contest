




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
    Improved Offensive Agent: Hunts Scared Ghosts, Fixes Looping, and Safe Retreat.
    """
    
    def __init__(self, index):
        ReflexCaptureAgent.__init__(self, index)
        self.retreating = False 

    def register_initial_state(self, game_state):
        ReflexCaptureAgent.register_initial_state(self, game_state)
        self.dead_ends = self.find_dead_ends(game_state)

    def find_dead_ends(self, game_state):
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        
        # 'all_dead_ends' is used for calculation (we need to know where the tips are 
        # to calculate the necks of the tunnels)
        all_dead_ends = set()
        
        # 'dangerous_dead_ends' is what we return. We will exclude the tips (depth 1).
        dangerous_dead_ends = set()
        
        candidates = []
        for x in range(width):
            for y in range(height):
                if not walls[x][y]: candidates.append((x,y))

        changed = True
        depth = 0 # Track how deep we are in the tunnel
        
        while changed:
            changed = False
            depth += 1
            new_dead_ends = set()
            
            for pos in candidates:
                if pos in all_dead_ends: continue
                
                x, y = pos
                neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                valid_exits = 0
                for nx, ny in neighbors:
                    # A neighbor is a valid exit if it is not a wall 
                    # AND it has not been marked as a dead end in a previous pass
                    if not walls[nx][ny] and (nx, ny) not in all_dead_ends:
                        valid_exits += 1
                
                if valid_exits <= 1:
                    new_dead_ends.add(pos)
                    changed = True
            
            # Update working set so the next pass can find the next layer of the tunnel
            all_dead_ends.update(new_dead_ends)
            
            # --- THE FIX ---
            # Only mark as "Dangerous" if depth > 1.
            # Depth 1 = The single tile at the very end (3 walls). Safe to step in and out.
            # Depth 2+ = The corridor leading to it. Dangerous to enter.
            if depth > 1:
                dangerous_dead_ends.update(new_dead_ends)
                
        return dangerous_dead_ends

    def get_safe_distance_to_home(self, game_state, my_pos, dangerous_positions):
        layout = game_state.data.layout
        width = layout.width
        start_pos = (int(my_pos[0]), int(my_pos[1]))
        queue = [(start_pos, 0)]
        visited = set([start_pos])
        
        if self.index % 2 == 0: home_x = (width // 2) - 1
        else: home_x = (width // 2)

        while queue:
            curr, dist = queue.pop(0)
            if (self.index % 2 == 0 and curr[0] <= home_x) or \
               (self.index % 2 != 0 and curr[0] >= home_x):
                return dist

            x, y = curr
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = int(x + dx), int(y + dy)
                next_pos = (nx, ny)
                if not layout.walls[nx][ny]:
                    if next_pos not in visited and next_pos not in dangerous_positions:
                        visited.add(next_pos)
                        queue.append((next_pos, dist + 1))
        return 9999 

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying
        food_left = len(self.get_food(game_state).as_list())
        
        # --- NEW LOGIC: Check for Scared Ghosts ---
        # If there is a scared ghost nearby, CANCEL RETREAT immediately.
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        scared_ghosts = [a for a in enemies if a.is_pacman is False and a.get_position() and a.scared_timer > 5]
        
        if scared_ghosts:
            self.retreating = False
        else:
            # Normal Retreat Logic
            if not self.retreating:
                if carrying >= 5 or (carrying > 0 and food_left <= 2):
                    self.retreating = True
            else:
                if carrying == 0:
                    self.retreating = False

        # Emergency MCTS override (Only for ACTIVE ghosts)
        active_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() and a.scared_timer <= 5]
        if active_ghosts and carrying > 0:
             dists = [self.get_maze_distance(my_state.get_position(), g.get_position()) for g in active_ghosts]
             if min(dists) <= 4:
                 return run_mcts(game_state, self.index, is_offensive=True, time_limit=0.1)

        return ReflexCaptureAgent.choose_action(self, game_state)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # --- NEW LOGIC: Separating Ghosts ---
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        
        # 1. Active Ghosts (Danger) - Timer < 5
        active_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() and a.scared_timer <= 5]
        
        # 2. Scared Ghosts (Food!) - Timer > 5 (Buffer to avoid dying as timer ends)
        scared_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() and a.scared_timer > 5]

        dangerous_spots = set()
        
        # Handle Danger
        if active_ghosts:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
            min_dist = min(dists)
            if min_dist < 5:
                features['ghost_proximity'] = 5 / (min_dist * 1.5) 
                
            for g in active_ghosts:
                g_pos = g.get_position()
                ix, iy = int(g_pos[0]), int(g_pos[1])
                dangerous_spots.add((ix, iy))
 
        # Handle Scared Targets
        if scared_ghosts:
            # We want to minimize distance to the closest scared ghost
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in scared_ghosts]
            features['distance_to_scared_ghost'] = min(dists)

        # Food Handling
        if not self.retreating:
            food_list = self.get_food(successor).as_list()
            features['successor_score'] = -len(food_list)
            if len(food_list) > 0:
                features['distance_to_food'] = min([self.get_maze_distance(my_pos, food) for food in food_list])
            
            capsules = self.get_capsules(successor)
            if capsules:
                features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, c) for c in capsules])

        # Dead Ends (Only dangerous if ACTIVE ghosts are near)
        if my_pos in self.dead_ends and active_ghosts:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
            if min(dists) < 6:
                features['in_dead_end'] = 1

        # Return Home
        # Note: We calculate this even if chasing ghosts, so we don't accidentally get trapped
        if self.retreating or my_state.num_carrying > 0:
            features['distance_to_home'] = self.get_safe_distance_to_home(successor, my_pos, dangerous_spots)

        return features

    def get_weights(self, game_state, action):
        weights = {
            'successor_score': 100,
            'distance_to_food': -1,
            'distance_to_capsule': -2,
            'ghost_proximity': -1000, 
            'in_dead_end': -500,
            'stop': -100,
            'reverse': -2,
            'distance_to_home': 0,
            'distance_to_scared_ghost': 0 # Default 0
        }

        # Check context
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        scared_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() and a.scared_timer > 5]
        
        if self.retreating:
            # RETREAT MODE
            weights['successor_score'] = 0
            weights['distance_to_food'] = 0
            weights['distance_to_capsule'] = 0
            weights['distance_to_home'] = -5 
        
        else:
            # SCAVANGE MODE
            my_state = game_state.get_agent_state(self.index)
            if my_state.num_carrying > 0:
                weights['distance_to_home'] = -1 

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


