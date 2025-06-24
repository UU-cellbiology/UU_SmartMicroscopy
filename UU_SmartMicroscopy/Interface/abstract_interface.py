from abc import ABC, abstractmethod

class AbstractInterface(ABC):
    """
    Abstract class defining the structure of an interface module for the repository.

    This class serves as a blueprint for GUI or CLI-based interface implementations 
    that interact with the Smart Microscope Control System.
    """

    def __init__(self, config):
        super().__init__()

    @abstractmethod
    def refresh(self):
        pass

    @abstractmethod
    def acquire(self):
        pass

    @abstractmethod
    def abort(self):
        pass