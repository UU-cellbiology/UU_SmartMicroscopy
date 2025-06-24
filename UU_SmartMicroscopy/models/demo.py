#region Libraries
import numpy as np
from ..segmentation import Threshold
from .abstract_model import AbstractModel
from skimage import measure
#endregion 

class demo(AbstractModel):
    def __init__(self,functionalities=None):
        #initialize variables
        self.processed_image = None
        self.mod_image = None
        self.areaSelected = False
        self.lClick = None
        self.rClick = None

        #initialize segmentation
        self.modelCell = Threshold()

        # controlled parameters - parameters related to the control system
        self.controlled_parameters = {
            "Rid": np.array([0, 0]),
            "Rid_set": False,
            "Lid": np.array([0, 0]),
            "Lid_set": False,
        }
        #parameters for control - commands relating the GUI to the control model!
        self.parameter_controls = {
            "double_left": "Lid",
            "double_left_set": "Lid_set",
            "textbox": None,
            "right": "Rid",
            "right_set": "Rid_set"
        }

        #drop menu GUI list
        self.menu_list = [
            "Unassigned",
            "1- select top-left",
            "Unassigned",
            "2- select bottom-right"
        ]

    def process_step(self, cam_image):

        #select top-left corner
        if self.controlled_parameters["Lid_set"]:
                    print('Left top corner selected...')
                    self.controlled_parameters["Lid_set"] = False
                    self.lClick = self.controlled_parameters["Lid"]
                    self.rClick = None
                    self.areaSelected = False
        #select bottom-right corner and set area
        if self.controlled_parameters["Rid_set"] and self.lClick is not None:
                print('Bottom right corner selected...')
                self.controlled_parameters["Rid_set"] = False
                self.rClick = self.controlled_parameters["Rid"]
                self.modelCell.setImage(cam_image[np.newaxis])
                self.modelCell.setup(np.array([[self.lClick[1], self.lClick[0], self.rClick[1], self.rClick[0]]]))
                self.areaSelected = True
        #segment area
        if self.areaSelected:
            self.modelCell.run()  
            self.processed_image = self.modelCell.getResult()[0]
        else:
             self.processed_image = np.zeros(cam_image.shape)
    
    def controller_step(self):
        
        #get centroid of segmentation
        if self.areaSelected:
            m = measure.moments(self.processed_image)

            if m[0,0] !=0:
                centroid = np.array([int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])])
            else:
                centroid = np.array([0,0])

            #get width of box for face
            width = 2*np.sqrt(sum(self.processed_image.flatten()))


            #create googly eyes
            self.mod_image = self.create_googly_face_matrix(centroid,width)
        else:
             self.mod_image = np.zeros(self.processed_image.shape)

    def set_parameters(self, queue_dict):

        # iterate queue and try to change the parameter.
        for control in queue_dict:
            try:

                # get parameter to change from parameter_controls. if it exists, change the controlled_parameters'
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
            except:
                pass

    def write_data(self):
        pass

    def create_googly_face_matrix(self, center, prop):
        """
        Creates a boolean matrix with googly eyes and a smile.
        """
        #proportional smily to segmentation
        # prop = self.rClick[0]-self.lClick[0]
        prop /=400
        eye_radius=40*prop
        pupil_radius=10*prop
        eye_separation=80*prop
        pupil_offset_max=10*prop
        smile_radius=40*prop
        smile_thickness=200*prop
        smile_offset_y=10*prop

        #define from self
        rows, cols = self.processed_image.shape
        center_y, center_x = center

        # Create coordinate grids
        y, x = np.ogrid[:rows, :cols]
        
        # Initialize empty matrix
        result = np.zeros((rows, cols), dtype=bool)
        
        # Calculate eye centers
        left_eye_center_x = center_x - eye_separation // 2
        right_eye_center_x = center_x + eye_separation // 2
        eye_center_y = center_y
        
        # Generate random pupil offsets (googly effect)
        left_pupil_offset_x = np.random.randint(-pupil_offset_max, pupil_offset_max+1)
        left_pupil_offset_y = np.random.randint(-pupil_offset_max, pupil_offset_max+1)
        right_pupil_offset_x = np.random.randint(-pupil_offset_max, pupil_offset_max+1)
        right_pupil_offset_y = np.random.randint(-pupil_offset_max, pupil_offset_max+1)
        
        # Create eyes (white part)
        left_eye = (x - left_eye_center_x)**2 + (y - eye_center_y)**2 <= eye_radius**2
        right_eye = (x - right_eye_center_x)**2 + (y - eye_center_y)**2 <= eye_radius**2
        
        # Create pupils (black part)
        left_pupil = (x - (left_eye_center_x + left_pupil_offset_x))**2 + \
                    (y - (eye_center_y + left_pupil_offset_y))**2 <= pupil_radius**2
        right_pupil = (x - (right_eye_center_x + right_pupil_offset_x))**2 + \
                    (y - (eye_center_y + right_pupil_offset_y))**2 <= pupil_radius**2
        
        # Create smile (semi-circle)
        smile_center_y = eye_center_y + eye_radius + smile_offset_y
        in_smile_arc = ((x - center_x)**2 + (y - smile_center_y)**2 <= smile_radius**2)
        above_smile_line = (y >= smile_center_y - smile_thickness//2)
        below_smile_line = (y <= smile_center_y + smile_thickness//2)
        smile = in_smile_arc & above_smile_line & below_smile_line & (y >= smile_center_y)
        
        # Combine all elements (eyes but not pupils, plus smile)
        result = (left_eye | right_eye | smile) & ~(left_pupil | right_pupil)
        
        return result
