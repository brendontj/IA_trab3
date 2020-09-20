import random
import numpy as np

class QLearningAgent():
    def __init__(self, env, epsilon=0.01, alpha=0.8, gamma=0.2):
        self.epsilon = epsilon # exploration
        self.alpha = alpha   # learning rate
        self.gamma = gamma   # discount
        self.env = env
        self.q_table = np.zeros((env.getStateSpaceSize(),
                env.getActionSpaceSize()))

    def getAction(self, state):
        actions = self.env.getLegalActions(state)
        # Exemplo: Agente de decisao aleatoria
        
        #return random.choice(actions)
        ##
        if random.uniform(0,1) <= self.epsilon:
            return random.choice(actions)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value
        
