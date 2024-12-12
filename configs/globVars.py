#region libraries
import queue
import threading
import numpy as np
import json
from skimage import filters, feature, transform 
import configs.functions as functions
import scipy
#endregion

#region queues
'''
Create events and queues for parallel processes. 
These queues are connected to the user interface class.

Events:
    main_event          ... GUI and interface
    acq_event           ... acquisition process
    cal_event           ... calibration of modulator
Queues:
    image_control_queue ... image from camera
    acq_queue           ... acquisition parameters and modulator image
    mod_control_queue   ... modulator image
    parameter_queue     ... input parameters for feedback_model from GUI
    mod_ui_queue        ... modulator image - specifically to show in GUI
    image_ui_queue      ... image from camera - specifically to show in GUI
    processed_ui_queue  ... image of segmented cell to show in GUI
'''

#events - main, acquisition, calibration
main_event = threading.Event()
acq_event = threading.Event()
cal_event = threading.Event()

#queues - image, modulator image, parameter, id, etc.
image_control_queue = queue.LifoQueue()
mod_control_queue = queue.Queue()
parameter_queue = queue.Queue()
mod_ui_queue = queue.Queue()
image_ui_queue = queue.LifoQueue()
processed_ui_queue = queue.LifoQueue()
id_queue = queue.Queue()
#endregion

#region path from migration model
'''
The GUI has the possibility to show the path used for migration. 
This path is drawn by the model class. Here, we define two variables 
to achieve this. 
    hasPath          ... boolean to decide whether the model has or not path for migration
    modelPath        ... a list with the coordinates of each point in the path
'''
hasPath = False
modelPath = []
#endregion

#region extra shared variables
'''
....
'''
microscopeInfo = []
inputs = []
menuList = []
#endregion

#region calibration
'''
Class including all the calibration related to camera, modulator (DMD, SLM), and their images. 
The calibration consists on activating different points on the modulator, measuring an image, and relating the XY position of each pixel in the modulator with the ones in the camera.
The position of the points is given by the cal_file_60.txt file.
'''
class Calibration():
    def __init__(self):
        global microscopeInfo
        #get calibration point for modulator and camera
        self.slm_points, self.cam_points = self.read_points()

        #estimate the image transformation needed to relate from modulator to camera and back, based on calibration file
        self.t_slmtocam = np.linalg.inv(transform.estimate_transform("affine", self.slm_points,self.cam_points).params)
        self.t_camtoslm = np.linalg.inv(transform.estimate_transform("affine", self.cam_points,self.slm_points).params)

        #dimensions
        self.iw = microscopeInfo[0]
        self.ih = microscopeInfo[1]
        self.mh = microscopeInfo[2]
        self.mw = microscopeInfo[3]

    def read_points(self):
        '''
            Read calibration file to obtain magnification calibration for Modulator and camera. 
            This file is a .json file working as a dictionary. 'modulator' gives the coordinates for the modulator, and 'camera' for the camera.
            Saves every line as a list nested in a list, then convert to numpy array

            Return
            ----------
            :return: (2 numpy arrays) calibration points for modulator and camera, respectively.
        '''

        #try reading the file, if exists
        try:
            #open file and read json
            f = open(inputs['folder_name']+'/cal_file.json')
            cali_data = json.load(f)
            f.close()
            
            #get camera coordinates.
            cam_points_r = np.array(cali_data['camera'])

            #get modulator coordinates.
            if inputs['cali_coordinates'] is None:
                #if no input, get from file
                slm_points_r = np.array(cali_data['modulator'])
            else:
                #if input, don't get from file
                slm_points_r = np.array(inputs['cali_coordinates'])
                #if coordinate is different from cali_file, print a warning.
                if np.array(slm_points_r).tolist() != cali_data['modulator']:
                    print('Warning: input coordinate different from cali_file. Re-do calibration.')
        
        #if file doesn't exist use input coordiates - or zero.
        except:
            print("No calibration file found - calibration needed")
            if inputs['cali_coordinates'] is not None:
                slm_points_r = np.array(inputs['cali_coordinates'])
            else:
                slm_points_r = np.ones((3,2))    
            cam_points_r = np.array(inputs['cali_coordinates'])
        print(slm_points_r,cam_points_r)
        return slm_points_r, cam_points_r

    def write_points(self):
        '''
            Write calibration file to obtain magnification calibration for Modulator and camera. 
            This file is a .json file working as a dictionary. 'modulator' gives the coordinates for the modulator, and 'camera' for the camera.
            Note: this overwrites the file
        '''

        #get global inputs
        global inputs

        #create dictonary
        cali_dic = {
            'modulator' : self.slm_points.tolist(),
            'camera' : self.cam_points.tolist()
        }

        #open file and save data
        with open(inputs['folder_name']+'/cal_file.json', 'w') as f:
            json.dump(cali_dic, f)

        #print data
        print('Modulator coordinates: {}'.format(self.cam_points))
        print('Camera coordinates: {}'.format(self.slm_points))

    def set_calibration_image(self):
        '''
            Create a calibration image based on 2D gaussian distributions centered at the calibration points. 
            After creating the image based on the calibration file, start and activate modulator device.
        '''

        #gv global variables
        global cal_event
        global mod_control_queue

        #vectorize a 2D gaussian function
        f = np.vectorize(functions.gaussian_2d,signature='(),(),(k),()->(m,n)',excluded=['0','1','sigma'])
        
        #get calibration image by summing a 2D gaussian in the third dimension
        self.slm_calibration_image = np.sum(f(self.mw,self.mh,np.flip(self.slm_points,axis=1),1),axis=0)
        print(np.flip(self.slm_points,axis=1))

        #convert calibration image to uint8 and convert it to a vector
        self.slm_calibration_image = functions.image_to_uint8(self.slm_calibration_image)
        # self.slm_calibration_image = self.slm_calibration_image.flatten()

        #set parallel threading event announcing the calibration is starting, activating the modulator with the calibration image
        cal_event.set()
        mod_control_queue.put({"image":self.slm_calibration_image, "power": 1000, "set": False})
    
    def get_calibration_points(self):
        '''
            Get new calibration for camera and modulator based on new image. 
            To do this, filter image with a gaussian and find 3 peaks in the image.
        '''
        global inputs
        #gv global variables
        global image_control_queue
        global cal_event

        if inputs['microscope'] != 'demo':

            #get value from image_control_queue (wait for it) and clear thread's flag
            self.cam_calibration_image = image_control_queue.get(block=True)
            cal_event.clear()

            #filter (convolution) the image using a gaussian distribution
            self.cam_calibration_image = filters.gaussian(self.cam_calibration_image,sigma=3)

            #get position of 3 (intensity) peaks within the image
            temp_points = feature.peak_local_max(self.cam_calibration_image, num_peaks=3, exclude_border=False)

            #assign and re-arrange calibration for camera and modulator
            self.cam_points = temp_points
            self.cam_points = self.cam_points[np.argsort(self.cam_points[:,0])]
            self.slm_points = self.slm_points[np.argsort(self.slm_points[:,0])]

            #estimate the image transformation needed to relate from modulator to camera and back
            self.t_slmtocam = np.linalg.inv(transform.estimate_transform("affine", self.slm_points,self.cam_points).params)
            self.t_camtoslm = np.linalg.inv(transform.estimate_transform("affine", self.cam_points,self.slm_points).params)

            #write new calibration to file
            self.write_points()
        else:
            print('no calibration for demo microscope bridge.')

    def transform_slmtocam(self, slm_pixels):
        '''
            Transform modulator image to camera image using expected transformation. Data exported to uint8

            Parameters
            ----------
            :param slm_pixels: (uint8 2D matrix) image of modulator

            Return
            ----------
            :return: (uint8 2D matrix) image of modulator in camera dimension.
        '''

        #create image with camera dimensions and "print" modulator image on it
        cam_pixels = np.zeros((self.ih,self.iw))
        cam_pixels[:slm_pixels.shape[0],:slm_pixels.shape[1]] += slm_pixels

        #transform image to modulator dimensions
        cam_pixels = scipy.ndimage.affine_transform(cam_pixels, self.t_slmtocam)

        #normalize image with 255 and convert to uint8
        if np.max(cam_pixels)>0: 
            cam_pixels *= 255/np.max(cam_pixels)
        cam_pixels = cam_pixels.astype("uint8")

        #return image
        return cam_pixels
        
    def transform_camtoslm(self, cam_pixels):
        '''
            Transform camera image to modulator image using expected transformation. Data exported to uint8

            Parameters
            ----------
            :param cam_pixels: (uint8 2D matrix) image of camera

            Return
            ----------
            :return: (uint8 2D matrix) image of camera in modulator dimension.
        '''
        #create modulator image
        slm_pixels = np.zeros((self.mh,self.mw))

        #transform camera image to modulator dimensions and "print" it on the modulator image
        temp_pixels = scipy.ndimage.affine_transform(cam_pixels, self.t_camtoslm)
        slm_pixels += temp_pixels[:self.mh,:self.mw]

        #normalize image with 255 and convert to uint8
        if np.max(slm_pixels)>0: 
            slm_pixels *= 255/np.max(slm_pixels)
        slm_pixels = slm_pixels.astype("uint8")

        #return image
        return slm_pixels
#endregion