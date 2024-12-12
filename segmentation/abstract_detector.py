"""Abstract classes for cell segmentation and object detection."""


from abc import ABC, abstractmethod


class Abstract_Detector(ABC):
     
    def __init__(self):
        self.image_in = None
        self.image_out = None
        
    @abstractmethod
    def run(self): 
        pass

    @abstractmethod
    def setImage(self): 
        pass

    @abstractmethod
    def setup(self): 
        pass
    
    @abstractmethod
    def getResult(self): 
        pass


    
