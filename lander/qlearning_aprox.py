import gym
import numpy as np
import pickle
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from base_agent import BaseAgent
from statistics import mode,mean
import random

warnings.filterwarnings("ignore", category = ConvergenceWarning)

class QLearningAgent(BaseAgent):
    def __init__(self, env, alpha=0.003, epsilon=0.05, gamma=0.999, possible_actions=4):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.legal_actions = list(range(possible_actions))

        # Utilize o dicionario networks de forma que cada indice eh representado
        # por um valor entre 0 e possible_actions-1
        # O dicionario devera guardar as instancias da classe MLPRegressor
        self.networks = {}
        # Se preferir, descomente o codigo abaixo para instanciar as redes, lembre-se
        # de experimentar arquiteturas diferentes (o parametro hidden_layer_sizes)
        for action in self.legal_actions:
            self.networks[action] = MLPRegressor(hidden_layer_sizes=(100,),
                                                 max_iter=200, random_state=0,
                                                 learning_rate_init=self.alpha)


    ## NAO ALTERE OS METODOS save_snapshot e getLegalActions ##
    def save_snapshot(self, name):
        self.snapshots[name] = {x:pickle.dumps(self.networks[x]) for x in self.networks}

    def getLegalActions(self, state):
        return self.legal_actions


    def getAction(self, state):
        '''
        state: vetor de numeros reais

        retorna um valor presente na lista self.getLegalActions(state)
        '''
        # Lembre-se que o Q-value do par (estado,acao) eh obtido pela
        # predicao da rede referente a cada acao e passando como entrada o estado:
        # self.networks[action].predict([state])[0]
        # OBS: o metodo predict sempre retorna um lista, mesmo que tenha apenas um elemento
        # por isso o [0] para pegar o primeiro elemento
        # IMPLEMENTE AQUI O METODO PARA ESCOLHER A ACAO
        actions = self.getLegalActions(state)
        # i = np.random.randint(len(actions))
        # return actions[i]
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(len(actions))
        else:
            list_max = []
            for i in range(len(actions)):
                list_max.append(self.getValue(actions[i], state))
            
        return np.argmax(list_max)

    def update(self, state, action, nextState, reward):
        '''
        Atualiza a rede na posicao self.networks[action].
        '''
        # Voce devera treinar a rede parcialmente pelo metodo partial_fit
        # da seguinte forma:
        # self.networks[action].partial_fit([state], [<target>])
        # onde <target> eh o Q-value esperado
        # OBS: nao esqueca dos colchetes pois os parametros devem ser listas
        # IMPLEMENTE AQUI O METODO PARA ATUALIZAR O AGENTE
        
        q_value_next_state = self.getValue(action,nextState)
        target = reward + (self.gamma * q_value_next_state)
        self.networks[action].partial_fit([state], [target])

    def getValue(self, action, state):
        try:
            return self.networks[action].predict([state])[0] 
        except:
            return 0