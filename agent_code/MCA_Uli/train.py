from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np
from datetime import datetime
import os

# This is only an example!
#Transition = namedtuple('Transition',
#                        ('state', 'action', 'next_state', 'reward'))

#Transitions = [[]]


TRANSITION_HISTORY_SIZE=300

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # [can move in direction, closest coin direction]
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.gamma = 0.8
    self.Transitions = []

    self.V = np.zeros([2,2,2,2,2,2,2,2])
    self.V = {(i,j,k,l,m,n,o,p,q):float() for i in range(2) for j in range(2) for k in range(2) for l in range(2) for m in range(2) for n in range(2) for o in range(2) for p in range(2) for q in range(30)}
    self.returns = {(i,j,k,l,m,n,o,p,q):list() for i in range(2) for j in range(2) for k in range(2) for l in range(2) for m in range(2) for n in range(2) for o in range(2) for p in range(2) for q in range(30)}

    if os.path.isfile("my-saved-model.pt"):
        print("model found")
        with open("my-saved-model.pt", "rb") as file:
            (self.V, self.returns) = pickle.load(file)

        #print(len(self.V.keys()))
        nz = 0
        for state in self.V.keys():
            if self.V[state] != 0:
                #print(state, self.V[state])
                nz += 1
        print("Found %s nonzero entries" % nz)





def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    features = state_to_features(new_game_state)
    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    self.Transitions += [[state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)]]


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    start = datetime.now()

    G = 0
    for i, step in enumerate(self.Transitions[::-1]):
        G = self.gamma*G + step[3]
        #print(i, step[0])
        #print(self.Transitions[::-1][len(self.Transitions)-i - 1:][0])
        if tuple(step[0]) not in [tuple(x[0]) for x in self.Transitions[::-1][len(self.Transitions) - i - 1:]]:
            self.returns[tuple(step[0].astype(int))].append(G)
            self.V[tuple(step[0].astype(int))] = np.average(self.returns[tuple(step[0])])

        #print(self.V[tuple(idx[0].astype(int))])               
        pass
    

    # Store the model
    saving_start = datetime.now()
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump((self.V, self.returns), file)
    print(datetime.now()-start, datetime.now() - saving_start)
    self.Transitions = []



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.INVALID_ACTION: -1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
