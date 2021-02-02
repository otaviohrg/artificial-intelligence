import random
import math
from constants import *


class FiniteStateMachine(object):
    """
    A finite state machine.
    """
    def __init__(self, state):
        self.state = state

    def change_state(self, new_state):
        self.state = new_state

    def update(self, agent):
        self.state.check_transition(agent, self)
        self.state.execute(agent)


class State(object):
    """
    Abstract state class.
    """
    def __init__(self, state_name):
        """
        Creates a state.

        :param state_name: the name of the state.
        :type state_name: str
        """
        self.state_name = state_name

    def check_transition(self, agent, fsm):
        """
        Checks conditions and execute a state transition if needed.

        :param agent: the agent where this state is being executed on.
        :param fsm: finite state machine associated to this state.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")

    def execute(self, agent):
        """
        Executes the state logic.

        :param agent: the agent where this state is being executed on.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")


class MoveForwardState(State):
    def __init__(self):
        super().__init__("MoveForward")
        self.n = 0

    def check_transition(self, agent, state_machine):
        if agent.get_bumper_state():
            agent.behavior.change_state(GoBackState())
        elif self.n*SAMPLE_TIME >= MOVE_FORWARD_TIME:
            agent.behavior.change_state(MoveInSpiralState())
        pass

    def execute(self, agent):
        agent.set_velocity(FORWARD_SPEED, 0)
        self.n += 1
        pass


class MoveInSpiralState(State):
    def __init__(self):
        super().__init__("MoveInSpiral")
        self.n = 0
    
    def check_transition(self, agent, state_machine):
        if agent.get_bumper_state():
            agent.behavior.change_state(GoBackState())
        elif self.n * SAMPLE_TIME >= MOVE_IN_SPIRAL_TIME:
            agent.behavior.change_state(MoveForwardState())
        pass

    def execute(self, agent):
        time = self.n*SAMPLE_TIME
        agent.set_velocity(ANGULAR_SPEED * (INITIAL_RADIUS_SPIRAL + SPIRAL_FACTOR * time), ANGULAR_SPEED)
        self.n += 1
        pass


class GoBackState(State):
    def __init__(self):
        super().__init__("GoBack")
        self.n = 0

    def check_transition(self, agent, state_machine):
        if self.n * SAMPLE_TIME >= GO_BACK_TIME:
            agent.behavior.change_state(RotateState())
        pass

    def execute(self, agent):
        agent.set_velocity(BACKWARD_SPEED, 0)
        self.n += 1
        pass


class RotateState(State):
    def __init__(self):
        super().__init__("Rotate")
        self.n = 0
        self.angle = random.uniform(0, 2*math.pi)

    def check_transition(self, agent, state_machine):
        if ANGULAR_SPEED*self.n*SAMPLE_TIME >= self.angle:
            agent.behavior.change_state(MoveForwardState())
        pass
    
    def execute(self, agent):
        agent.set_velocity(0, ANGULAR_SPEED)
        self.n += 1
        pass
