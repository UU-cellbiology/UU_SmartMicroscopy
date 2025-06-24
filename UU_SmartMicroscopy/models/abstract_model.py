from abc import ABC, abstractmethod

class AbstractModel(ABC):

    def __init__(self, functionalities=None):
        self.processed_image = None
        self.mod_image = None

    @abstractmethod
    def process_step(self):
        pass

    @abstractmethod
    def controller_step(self):
        pass

    @abstractmethod
    def write_data(self):
        pass

    @abstractmethod
    def set_parameters(self, parameters):
        pass
