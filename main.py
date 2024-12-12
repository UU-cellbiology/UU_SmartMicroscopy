#region Prelude
#region Description
''' DESCRIPTION
Main program with user interface to acquire images with PEX microscope using feedback loop control. 
In this main program, we use user interfaces to connect to the microscope via micromanager. 
Three threads are used to simultaneusly have a window, the control loop, and the model loop.

Language:       Python
Author(s):      Jakob Schröder, Josiah Passmore, Alfredo Rates and Ihor Smal
Creation date:  2023-06-19
Contact:        a.ratessoriano@uu.nl
Git:            https://github.com/UU-cellbiology/FeedbackMicroscopy
Version:        4.0

'''
#endregion
#region Libraries
import tkinter                                                                          #GUI library
import time                                                                             # Time measurements
import datetime                                                                         # Time (and date) measurements
import threading                                                                        # Paralelize process
from skimage import filters, feature, transform                                         # Image analysis library
import configs.functions as functions                                                                        # Tailored-made functions
import os                                                                               # System handling
from pathlib import Path                                                                # Get and manage directories object-oriented
import importlib                                                                        # Necessary to initialize classes andmodels from strings 
import json                                                                             # Nice formatting for files like metadata and calibration
import models                                                                           # Package/folde that keeps all the Models as separate classes in separate files 
import configs.globVars as gv                                                           #shared variables with other modules
import tifffile
from Interface.GUI_tkinter import mainWindow
import yaml
#endregion
#endregion

#region Setting up
#region basic user input
'''
Basic inputs from user read from a yaml file. 
'''

with open("inputs.yaml", "r") as file:
    yaml_config = yaml.safe_load(file)

#merge dictonary
inputs = yaml_config["general_inputs"] | yaml_config["advanced_inputs"]
inp_k = list(yaml_config.keys())
inp_k = [k for k in inp_k if 'functionalities' in k]
for key in inp_k:
    inputs = inputs | yaml_config[key]

#region other user input
'''
Extra inputs are present in a second dictonary. Both dictonaries will merge. 
Additionally, create metadata file.
'''

#add time to PID coeff and extra variables
inputs["functionalities"] = inputs.copy()
inputs['functionalities']['PID_coef'].append(inputs['time_interval_s'])
inputs['functionalities']['file_name'] = inputs['file_name']
inputs['functionalities']['fluorescence_excitation_intensity'] = inputs['coolLED']
inputs['functionalities']['exposure_time'] = inputs['channels'][0][-1]
inputs['functionalities']['init_power'] = 1
inputs['led_set'] = False
inputs['led_power'] = 1
inputs['dt'] = 1700

#get demo file streaming if needed
if inputs['microscope']=='demo' and inputs['demo_path']=='':
    if inputs['model']=='PID_LEXY_SAM':
        inputs['demo_path'] = './_info/demo_data/LEXY_U2OS_mCherry_100ms.tif'
    else:
        inputs['demo_path'] = './_info/demo_data/Migration_HT1080_mCherry_100ms.tif'

#create metadata file name NOTE: the metadata will only be saved in a succesful run!
nameFile_metadata = '_{0}_{1}-{2}'.format(str(datetime.date.today()),time.ctime()[11:13],time.ctime()[14:16])
#endregion

#region Hardware setup
'''
Initialize microscope object. Change the name of the microscope to the one you are using. 
Available: PEX, AiryScan2, demo
'''

#define Microscope object
microscope_module = importlib.import_module('microscopeBridge.{}'.format(inputs['microscope']))
class_ = getattr(microscope_module, inputs['microscope'])
Microscope = class_(inputs)

#get model object from text extracted from pop-up window
#The third element called functionalities is a dictionary with possible additional functions. Mandatory for modularity. 
models_module = importlib.import_module('models.{}'.format(inputs['model']))
class_ = getattr(models_module, inputs['model'])
model = class_(Microscope.im_height, Microscope.im_width, functionalities = inputs["functionalities"])

#get variables for calibration
gv.microscopeInfo = [Microscope.im_width,Microscope.im_height,Microscope.mod_height,Microscope.mod_width]
gv.inputs = inputs
gv.menuList = model.menu_list
#endregion

#region Folder and Hash
'''
Make sure folder exists and store Git hash
'''

#make sure saving path exists
if not os.path.exists(inputs['folder_name']):
    os.makedirs(inputs['folder_name'])

#get Git hash and store it in inputs dictonary
#NOTE: this will only work if the code is used from the git folder!
git_folder = os.getcwd()+'\.git\\'
head_name = Path(git_folder, 'HEAD').read_text().split('\n')[0].split(' ')[-1]
head_ref = Path(git_folder,head_name)
inputs['git'] = head_ref.read_text().replace('\n','')
#endregion
#endregion

#region Threads
'''
Define the two classes for parallel processing. 
One is for modeling (i.e., process images and get modulator image) and the other is for controlling (i.e., control microscope and start acquisition). 
'''
class ModelThread(threading.Thread):
    def __init__(self):

        #initialize a threading object
        threading.Thread.__init__(self)

        #set experiment model from feedback_model
        self.model = model

        #set path, if possible
        if hasattr(self.model, 'create_path'):
            gv.hasPath = True
            gv.modelPath = self.model.create_path()
        else:
            gv.hasPath = False

        self.mask_addition = 0
        self.mask_addition = 0

        #start timer
        self.timer = time.time()

    def run(self):
        '''
        Run the adquisition and control loop. 
        Acquire image, process the image, compute the new modulator patern.
        The procedure is based on the functions from feedback_model
        '''

        #run as long as the main event is off - Window still open
        while not gv.main_event.isSet():

            #image acquisition - if image is available
            if gv.acq_event.isSet() and Microscope.image_ready:

                #get image
                self.cam_image = Microscope.Acq_image
                Microscope.image_ready = False

                #process image using processing function from feedback_model to segment cell
                self.model.process_step(self.cam_image)

                #add image to GUI queue as uint8
                gv.image_ui_queue.put(functions.image_to_uint8(self.cam_image))
                    
                #get new modulator image based on the control model
                self.model.controller_step()

                #add segmented cell to queue as uint8
                gv.processed_ui_queue.put(functions.image_to_uint8(self.model.processed_image))

                #export segmented cell
                #save mask in separated file
                try:
                    fname = inputs['folder_name']+'/'+inputs['file_name']+nameFile_metadata+'_mask_'+str(self.mask_addition)+'.tiff'
                    tifffile.imwrite(fname, self.model.processed_image.astype("uint8"), append=True)
                    if os.path.getsize(fname)>=4294967296:
                        self.mask_addition+=1
                except:
                    print('Error while saving mask file.')

                #set modulator image to the queue. If the model has control over intensity, assign this as well
                if hasattr(self.model, 'led_power'):
                    self.queue_mod_image(self.model.mod_image, self.model.led_power, set=True)
                else:
                    self.queue_mod_image(self.model.mod_image, set=True)

                #set flag on
                Microscope.modulator_ready = True

                #save data using feedback_model function
                self.model.write_data(time.ctime(), datetime.date.today(),inputs['folder_name'])

            elif not gv.acq_event.isSet() and not gv.image_control_queue.empty():

                #get image
                self.cam_image = gv.image_control_queue.get()

                #process image using processing function from feedback_model to segment cell
                self.model.process_step(self.cam_image)

                #add image to GUI queue as uint8
                gv.image_ui_queue.put(functions.image_to_uint8(self.cam_image))

                #add segmented cell to queue as uint8
                gv.processed_ui_queue.put(functions.image_to_uint8(self.model.processed_image))

                #control loop - if no modulator image is available
                if gv.mod_control_queue.empty():
                    
                    #get new modulator image based on the control model
                    self.model.controller_step()

                    #set modulator image to the queue. If the model has control over intensity, assign this as well
                    if hasattr(self.model, 'led_power'):
                        self.queue_mod_image(self.model.mod_image, self.model.led_power)
                    else:
                        self.queue_mod_image(self.model.mod_image)
                
                    #if not acquiring, try closing saving file
                    try:
                        if self.model.file is not None:
                            self.model.file.close()
                            self.model.file = None
                    except: pass 

            #add parameters from settings to feedback_model
            while not gv.parameter_queue.empty():
                self.model.set_parameters(gv.parameter_queue.get())

            #get new elapsed time
            elapsed = time.time() - self.timer
            self.timer = time.time()

    def queue_mod_image(self,mod_image, power=None, set=None):
        '''
        Convert Modulator image to uint8 and assign to corresponding queues

        Parameters
        ----------
        :param mod_image: (2D matrix uint8) original image from modulator
        :param power: (positive int) power related to the image to save in queue
        :param set: (boolean) set flag related to the image to save in queue
        '''
        
        #get global inputs
        global inputs

        #if no power or set available, set from global
        if power is None:
            power = inputs['led_power']
        if set is None:
            set = inputs['led_set']

        #convert image to uint8 and save in queue
        mod_image = functions.image_to_uint8(mod_image)
        gv.mod_ui_queue.put(mod_image)

        #calibrate image
        mod_image_cal = calibration.transform_camtoslm(mod_image) # quick fix

        #if calibration is not running, save parameters (image, power, set value) to queue
        if not gv.cal_event.isSet(): 

            if gv.acq_event.isSet(): 
                Microscope.modulator_dict = {"image":mod_image_cal, "power": power, "set": set, "image_ui": mod_image}

            gv.mod_control_queue.put({"image":mod_image_cal, "power": power, "set": set})
class ControlThread(threading.Thread):
    def __init__(self):
        
        #initialize a threading object
        threading.Thread.__init__(self)

        #start timer
        self.timer = time.time()

        #assign flag and channel name
        self.acq_check = True
        self.acq_channel1 = "asdf"
    
    def run(self):
        '''
        Get modulator image and assign it to the modulator and to queue. 
        '''
        
        #run as long as the main event is off, i.e., window still open
        while not gv.main_event.isSet():
            
            #run as long as the main event AND acquisition event is off
            while not gv.main_event.isSet() and not gv.acq_event.isSet():
                
                #get live image and add to queue
                gv.image_control_queue.put(self.snap_from_core())

                #if modulator queue is not empty, get and assign modulator image. 
                if not gv.mod_control_queue.empty():

                    #get and set image
                    modulator_dict = gv.mod_control_queue.get()
                    Microscope.set_modulation(modulator_dict["image"])

                #calculate and assign elapsed time
                time.sleep(0.001*(inputs['dt'] - int(1000*time.time()) % inputs['dt']))
                elapsed = time.time() - self.timer
                self.timer = time.time()

            #if we go out of the loop because the acquisition event is on, run the run_acquisition function
            if not gv.main_event.isSet():
                
                self.run_acquisition()
    
    def snap_from_core(self):
        '''
        Get image from core and reshape it. This communicates with pycromanager.

        Return
        ----------
        :return: (2D matrix uint8) reshaped image from core
        '''
        
        #acquire live image
        tagged_image = Microscope.live_image()
        
        #return reshaped image
        return tagged_image

    def run_acquisition(self):
        '''
        XXX
        '''

        Microscope.run_acquisition()

        while Microscope.running:
            time.sleep(0.5)

        # clear event object
        gv.acq_event.clear()

#endregion 
    
#region Actual execution
'''
Run program using the functions and GUI previously defined. 
We base our code in threads. there are three threads, one for calibration, other for control, and other for model. 
'''
#Start threads
calibration    = gv.Calibration()
control_thread = ControlThread()
model_thread   = ModelThread()

control_thread.start()
model_thread.start()

#start GUI
root   = tkinter.Tk()
window = mainWindow(root,inputs,Microscope.im_height, Microscope.im_width)
root.resizable(width=0, height=0)
root.mainloop()

#Close threads
gv.main_event.set()
model_thread.join()
control_thread.join()
threading._shutdown()

#Turn LED off
Microscope.shutdown()

#Save metadata as json file
with open(inputs['folder_name']+'/'+inputs['file_name']+nameFile_metadata+'_metadata.json', 'w') as f:
        json.dump(inputs, f)
#endregion