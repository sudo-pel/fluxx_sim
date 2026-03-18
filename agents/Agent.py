from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def act(self, state):
        raise NotImplementedError

    def reset(self):
        pass

    def observe(self, obs, reward, terminated, truncated, info):
        pass
