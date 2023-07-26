# valueIterationAgents.py
# -----------------------
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


import mdp, util
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        self.learningRate = 0.5
        actions = dict()
        for state in mdp.getStates():
            self.values[state] = 0
            actions[state] = self.mdp.getPossibleActions(state)
        
        counter = 0
        while counter < iterations:            
            for state in self.mdp.getStates():
                if (not self.mdp.isTerminal):
                    statesAndProbs = []
                    action1StatesAndProbs = [], action2StatesAndProbs = [], action3StatesAndProbs = [], action4StatesAndProbs = []
                    # for action in actions[state]:
                    #     statesAndProbs.append(mdp.getTransitionStatesAndProbs(state, action))          
                    actionStatesAndProbs = []
                    for stateAndProb in self.mdp.getTransitionStatesAndProbs(state, actions[0]): 
                        action1StatesAndProbs.append(stateAndProb) 
                    
                    for stateAndProb in self.mdp.getTransitionStatesAndProbs(state, actions[1]): 
                        action2StatesAndProbs.append(stateAndProb) 
                    
                    for stateAndProb in self.mdp.getTransitionStatesAndProbs(state, actions[2]): 
                        action3StatesAndProbs.append(stateAndProb) 
                    
                    for stateAndProb in self.mdp.getTransitionStatesAndProbs(state, actions[3]): 
                        action4StatesAndProbs.append(stateAndProb) 
                    
                    for elem in self.mdp.getTransitionStatesAndProbs(state, actions[1]): statesAndProbs.append(elem) 

                    self.values[state] = -0.9 + self.learningRate*max(
                        sum(prob*val for prob, val in action1StatesAndProbs),
                        sum(prob*val for prob, val in action2StatesAndProbs),
                        sum(prob*val for prob, val in action3StatesAndProbs),
                        sum(prob*val for prob, val in action4StatesAndProbs)
                    )

        
    # def somaacoes(self, mdp, state, listProbs(nextState, prob)):
    #     for elem in listProbs:
    #         listProbs[1] * mdp.getReward(state, )


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        prob = self.mdp.getTransitionStatesAndProbs(state, action)[0][1]
        value = self.values[state]
        return sum()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if(self.mdp.isTerminal()):
            return None
        return max(self.mdp.getPossibleActions(state))

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
