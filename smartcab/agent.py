from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from random import choice, random

OPTIONS = (None, 'forward', 'left', 'right')

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.q_matrix = {}
        self.alpha = 0.7
        self.gamma = 0.3
        self.explore = 0.15

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        deadline = self.env.get_deadline(self)

        state = self.observe_state()
        
        knowledge = self.knowledge_from_state(state)

        # TODO: Select action according to your policy
        if random() < self.explore:
            action = choice(OPTIONS)
        else:
            action = sorted(knowledge.items(), key=lambda x: x[1], reverse=True)[0][0]

        old_value = knowledge[action]

        # Execute action and get reward
        reward = self.env.act(self, action)
        new_state = self.observe_state()
        new_state_knowledge = self.knowledge_from_state(new_state)

        self.q_matrix[state][action] = (1-self.alpha) * old_value +\
                                       self.alpha * (reward + self.gamma * max(new_state_knowledge.values()))
        

        #print "LearningAgent.update(): deadline = {}, state = {}, action = {}, reward = {}".format(deadline, state, action, reward)  # [debug]

    def knowledge_from_state(self, state):
        """
        Gets what we know about the given state
        """
        if not state in self.q_matrix:
            self.q_matrix[state] = {option: 0 for option in OPTIONS}
        return self.q_matrix[state]

    def observe_state(self):
        """
        Observe the current state
        """
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        return (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
