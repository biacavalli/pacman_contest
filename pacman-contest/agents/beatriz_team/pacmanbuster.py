# use the ghostbuster project idea. 
# instead of chasing ghosts, the ghosts will chase the pacman

import random, util
from util import Counter

class HybridInference(object):
    """
    Hybrid tracking of a single enemy agent using a combination of:
      - Exact belief tables when uncertain
      - Particle filtering when enough evidence is available
    """

    def __init__(self, index, gameState):
        self.index = index              # enemy agent index
        self.numParticles = 300
        self.particles = []
        self.useParticles = False

        self.legalPositions = [p for p in gameState.getWalls().asList(False)]
        self.__initializeBeliefs()

    # ------------ SETUP ---------------- #

    def __initializeBeliefs(self):
        """Uniform exact belief initially (no detection yet)."""
        self.beliefs = Counter()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def __initializeParticles(self):
        """Initialize particles based on current beliefs."""
        self.particles = util.nSample(self.legalPositions, self.beliefs, self.numParticles)

    # ------------ OBSERVATION UPDATE ------------ #

    def observe(self, agent, gameState):
        """
        Update beliefs with noisy distance and food disappearance.
        """
        myPos = gameState.getAgentPosition(agent.index)
        noisyDist = gameState.getAgentDistances()[self.index]

        # If visible, collapse belief and disable particles
        visiblePos = gameState.getAgentPosition(self.index)
        if visiblePos is not None:
            self.useParticles = False
            self.beliefs = Counter({visiblePos: 1.0})
            return

        # -------- SENSOR UPDATE -------- #
        if not self.useParticles:
            # Use exact update
            for p in self.legalPositions:
                trueDist = util.manhattanDistance(myPos, p)
                prob = gameState.getDistanceProb(trueDist, noisyDist)
                self.beliefs[p] *= prob
            self.beliefs.normalize()
        else:
            # Use particle update
            newParticles = []
            for p in self.particles:
                trueDist = util.manhattanDistance(myPos, p)
                prob = gameState.getDistanceProb(trueDist, noisyDist)
                if random.random() < prob:
                    newParticles.append(p)

            # If too few survived, resample
            if len(newParticles) < self.numParticles // 4:
                self.__initializeParticles()
            else:
                self.particles = newParticles

        # -------- FOOD DISAPPEARANCE BOOST -------- #
        self.boostFromFood(agent, gameState)

        # SWITCH MODE IF ENOUGH EVIDENCE
        if max(self.beliefs.values()) > 0.10 and not self.useParticles:
            self.useParticles = True
            self.__initializeParticles()

    # -------- HELPER: Food Evidence -------- #

    def boostFromFood(self, agent, gameState):
        """Increase belief near recently eaten food."""
        eatenFoods = self.getEatenFood(agent, gameState)
        for food in eatenFoods:
            self.beliefs[food] *= 4.0
        self.beliefs.normalize()

    def getEatenFood(self, agent, gameState):
        """Return positions of newly missing food."""
        if not hasattr(self, "prevFood"):
            self.prevFood = gameState.getFood().asList()
            return []

        currentFood = gameState.getFood().asList()
        eaten = [f for f in self.prevFood if f not in currentFood]
        self.prevFood = currentFood
        return eaten

    # ------------ TIME ELAPSE ------------- #

    def elapseTime(self, gameState):
        """Diffuse belief or particles over possible moves."""
        successors = util.Counter()

        if not self.useParticles:
            # Exact diffusion
            for p, prob in self.beliefs.items():
                for np in self.getLegalSuccessors(p, gameState):
                    successors[np] += prob / len(self.getLegalSuccessors(p, gameState))
            successors.normalize()
            self.beliefs = successors
        else:
            # Particle propagation
            newParticles = []
            for p in self.particles:
                moves = self.getLegalSuccessors(p, gameState)
                if moves:
                    newParticles.append(random.choice(moves))
            self.particles = newParticles

    # ------------ UTILS ------------- #

    def getLegalSuccessors(self, pos, gameState):
        """Legal movement actions in the maze."""
        x, y = pos
        actions = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        return [p for p in actions if p in self.legalPositions]

    def getBeliefDistribution(self):
        """Return combined belief distribution."""
        if not self.useParticles:
            return self.beliefs
        dist = Counter()
        for p in self.particles:
            dist[p] += 1.0
        dist.normalize()
        return dist

    def getMostLikelyPosition(self):
        """Return argmax of current belief."""
        return self.getBeliefDistribution().argMax()
