"""Abstract classes for Library bridge between Python and microscope software."""

from abc import ABC, abstractmethod


class abstract_Bridge(ABC):
     
    def __init__(self):
        self.im_height = None
        self.im_width = None
        self.dmd_height = None
        self.dmd_width = None
        self.acq_path = None
        self.acq_name = None
        self.running = None
        self.image_ready = None
        self.Aq_image = None
        self.modulator_dict = None
        self.modulator_ready = None
        
    @abstractmethod
    def live_image(self): 
        pass

    @abstractmethod
    def set_modulation(self): 
        pass

    @abstractmethod
    def run_acquisition(self): 
        pass
    
    @abstractmethod
    def shutdown(self): 
        pass


    
