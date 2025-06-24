#region Libraries
import numpy as np
import scipy
import time
import csv
import importlib
import copy
from skimage.measure import regionprops,moments
import configs.globVars as gv
from ..configs import functions
from ..Controllers.PID_controller import PIDController
from .abstract_model import AbstractModel
#endregion

class LEXYCell():
    def __init__(self,
                 setpoint_array,
                 LEXY_parameters,
                 PIDdetails,
                 controlled_variable,
                 normalization_parameters,
                 init_power,
                 shape
                 ):
        '''
        XXX
        '''

        use_antiwindup,use_gainscheduling = PIDdetails
        self.maskReady = False

        if controlled_variable == "nucleus_intensity" or controlled_variable == "ratio" or controlled_variable == "normalized_nucleus_intensity":
            parity = -1.0
        else:
            parity = 1.0

        if controlled_variable == "ratio":
            lowerBound = -50/100
        else:
            lowerBound = -0.1

        # loop control: variables to change parameters (e.g. LED intensity) after certain loops
        self.smart_loop_parameters = {
            "setpoint_array": setpoint_array,  # array of tuples to change setpoint at a certain time (in seconds)
            "counter": time.time(),  # time counter
        }

        # controlled parameters - parameters related to the control of the cell
        self.controlled_parameters = {
            "id": np.array([0, 0]),  # position to get new id of segmented cell
            "id_set": False,  # flag to set new id of segmented cell
            "led_set": False,  # flag to set LED on
            "setpoint": 0.1,  # concentration setpoint
            "controller": PIDController(K=LEXY_parameters[0:-1], T=LEXY_parameters[-1], antiwindup=use_antiwindup, parity=parity, lb=lowerBound),  # PID Controller object
            "use_gainscheduling": use_gainscheduling,  # Use gain scheduling in the PID controller
            "use_antiwindup": use_antiwindup,  # Use anti-windup for the PID controller
            # "controller": PIDController(K=LEXY_parameters[0:-1],T=LEXY_parameters[-1])  # PID Controller object
        }

        self.controlled_variable = controlled_variable
        if 'normalized_' in controlled_variable:
            self.controlled_parameters["controlled_variable"]= controlled_variable[11:]
        else:
            self.controlled_parameters["controlled_variable"]= controlled_variable

        # internal parameters for the model
        self.internal_parameters = {
            "segment_id": [1,1],  # label/id of the segment to consider
            "nucleus": np.zeros(shape),  # mask of segmented nucleus of cell
            "cytosol": np.zeros(shape),  # mask of segmented cytosol of cell
            "minimum_int_measurement": normalization_parameters[0], # Minimum measured intensity in nucleus intensity
            "maximum_int_measurement": normalization_parameters[1], # Maximum measured intensity in nucleus intensity
            "normalized_nucleus_intensity": 1.0, # Normalized mean nucleus intensity
            "initial_measurement": 1.0, # Initial measurement of controlled variable, used for defining setpoints
            "bg": 1.0,  # background value
            "ratio": 1,  # concentration ratio between nucleus and cytosol
            "nucleus_intensity": 1.0,  # Mean intensity measured in nucleus
            "cytosol_intensity": 1.0,  # Mean intensity measured in cytosol
            "led_power": 1,  # Computed LED power sent to the LED module
            "max_led_power": 1000,  # Maximum LED power
            "min_led_power": 0, # Minimum LED power
            "LEXY_parameters": LEXY_parameters,  # PID controller (kP, kI, kD, imaging interval[s])
        }

        # Setup Gain scheduling if enabled
        if self.controlled_parameters["use_gainscheduling"]:
            self.controlled_parameters["controller"].setup_gain_scheduling()

        self.led_power = init_power
        self.mask_power = 0
        self.mask_nuc_filter = [0,0]

    def get_setpoint(self):
        self.time_passed = time.time() - self.smart_loop_parameters['counter'] #XXX
        if 'normalized' in self.controlled_variable:
            self.internal_parameters[self.controlled_parameters["controlled_variable"]] = ((self.internal_parameters[self.controlled_parameters["controlled_variable"]] - self.internal_parameters["minimum_int_measurement"]) /
                                                                     (self.internal_parameters["maximum_int_measurement"] - self.internal_parameters["minimum_int_measurement"]))
            relative_factor = 1.0
        else:
            relative_factor = 1.0
        setpoint = [item[1] * relative_factor for item in self.smart_loop_parameters['setpoint_array'] if self.time_passed > item[0]]
        if len(setpoint)<1:
            if 'nucleus' in self.controlled_variable:
                self.controlled_parameters["setpoint"] = 1
            else:
                self.controlled_parameters["setpoint"] = 0
        else:
            self.controlled_parameters["setpoint"] = setpoint[-1]

class PID_LEXY_SAM(AbstractModel):
    def __init__(self, im_height, im_width, functionalities=None):
        """
            Model that automatically illuminates the cell edge in a set direction and changes direction after a set
            number of frames.

            Parameters
            ----------
            :param im_height: (positive int) height of the processed image
            :param im_width: (positive int) width of the processed image
            :param functionalities: (dictionary) extra functions for every model. For now: change setpoint after
            some time
        """

        # create variables for processed image (empty array image & modulator image, flag for area_check)
        self.processed_image = np.zeros((im_height, im_width))

        self.mod_image = np.zeros((im_height, im_width))
        self.area_check = False

        #get segmentation class
        models_module = importlib.import_module('segmentation.{}'.format(functionalities['segmentation']))
        class_ = getattr(models_module, functionalities['segmentation'])
        self.modelCell = class_()

        #segmentation model
        self.maskReady = False
        self.initMeas = False
        self.id_x,self.id_y = 0,0

        self.file_name = functionalities['file_name']

        # create variables for LED control (frame counts, LED power, file, error array)
        self.frame_count = 0
        self.led_power = 0
        self.init_power = 0
        self.file = None
        self.e_array = []

        #tuple for segmentation of nucleus (previous,current)
        self.mask_nuc_filter = [0,0]
        self.mask_cyt_filter = [0,0]
        self.area_nuc = np.nan
        self.area_cyt = np.nan

        # Assign values from functionalities. Assign default of no input
        if functionalities is not None:
            self.led_power = functionalities['init_power']
            self.init_power = functionalities['init_power']
            setpoint_array = functionalities['setpoint_array']
            LEXY_parameters = functionalities['PID_coef']
            self.illuminate_nucleus = functionalities['illuminate_nucleus']
            att = functionalities['attenuation']
            use_gainscheduling = functionalities['use_gainscheduling']
            use_antiwindup = functionalities['use_antiwindup']
            controlled_variable = functionalities['LEXY_control_parameter']
            self.cellSize = functionalities['cell_size']
            self.nucleusSize = functionalities['nucleus_size']
            self.hh_c = self.cellSize[0]//2
            self.wh_c = self.cellSize[1]//2
            self.hh_n = self.nucleusSize[0]//2
            self.wh_n = self.nucleusSize[1]//2
            normalization_parameters = functionalities['LEXY_normalization_parameters']
            exposure_time = functionalities['exposure_time']
            excitation_intensity = functionalities["fluorescence_excitation_intensity"]
            self.microscope=functionalities['scope_type']

            # assign default values if some parameters are missing
            if not setpoint_array:
                setpoint_array = [(0, 0.1)]
            if not att:
                att = 0.0
            if not use_gainscheduling:
                use_gainscheduling = False
            if not use_antiwindup:
                use_antiwindup = False
            if not self.microscope:
                self.microscope = None

        else:
            # assign values if whole dictionary missing
            setpoint_array = [(0, 0.1)]
            LEXY_parameters = [0.1, 0.1, 0.1, 15]
            self.illuminate_nucleus = True
            use_gainscheduling = False
            use_antiwindup = False
            att = 0.0
            controlled_variable = "nucleus_intensity"
            self.hh_c, self.wh_c = 250,250
            self.hh_n, self.wh_n = 100,100
            normalization_parameters = [200.0, 3000.0]
            exposure_time = 100
            excitation_intensity = 5
            self.microscope = None

        #attenuation correction
        self.ODfit = att

        # loop control: variables to change parameters (e.g. LED intensity) after certain loops
        self.smart_loop_parameters = {
            "setpoint_array": setpoint_array,  # array of tuples to change setpoint at a certain time (in seconds)
            "counter": time.time(),  # time counter
        }

        # controlled parameters - parameters related to the control of the cell
        self.controlled_parameters = {
            "id": np.array([0, 0]),  # position to get new id of segmented cell
            "id_set": False,  # flag to set new id of segmented cell
            "led_set": False,  # flag to set LED on
        }

        # parameters for control - commands relating the GUI to the control model!
        self.parameter_controls = {
            "double_left": "id",  # position of double left click - new id position
            "double_left_set": "id_set",  # double left click boolean - change id of segmented area
            "textbox": "target_nucleus_intensity",  # ill-shape text input: control setpoint
            "right": None,  # position of right click - None
            "right_set": None  # right click boolean - None
        }

        # internal parameters for the model
        self.internal_parameters = {
            "segment_id": [1,1],  # label/id of the segment to consider
            "minimum_int_measurement": normalization_parameters[0], # Minimum measured intensity in nucleus intensity
            "maximum_int_measurement": normalization_parameters[0], # Maximum measured intensity in nucleus intensity
            "normalized_nucleus_intensity": 1.0, # Normalized mean nucleus intensity
            "initial_measurement": 1.0, # Initial measurement of controlled variable, used for defining setpoints
            "bg": 1.0,  # background value
            "ratio": 1,  # concentration ratio between nucleus and cytosol
            "nucleus_intensity": 1.0,  # Mean intensity measured in nucleus
            "cytosol_intensity": 1.0,  # Mean intensity measured in cytosol
            "led_power": self.led_power,  # Computed LED power sent to the LED module
            "max_led_power": 255,  # Maximum BIT power
            "min_led_power": 0, # Minimum BIT power
            "attenuation": 10**(-1*att),  # Configured attenuation - calculated from OD of the filter
            "LEXY_parameters": LEXY_parameters,  # PID controller (kP, kI, kD, imaging interval[s])
            "exposure_time": exposure_time,                 # Exposure time used in imaging
            "excitation_intensity": excitation_intensity,   # LED power sent to the fluorescence excitation light
        }

        #drop menu GUI list
        self.menu_list = [
            "Unassigned",
            "Select cell",
            "Unassigned",
            "Unassigned"
        ]
            
        #define cell to be selected!
        self.num_cells = len(setpoint_array)
        self.current_cell = 0

        #array of boxes for SAM
        self.setup_array = []

        #create cell objects same number as setpoint arrays
        self.cells = []

        for i in range(len(setpoint_array)):
            cell = LEXYCell(setpoint_array[i],
                            LEXY_parameters,    #XXX note, this could be for every cell but whatever
                            [use_antiwindup,use_gainscheduling], #XXX note, this could be for every cell but whatever
                            controlled_variable[i],
                            normalization_parameters[i],
                            self.led_power,
                            (im_height, im_width))
            self.cells.append(cell)

        #print click order:
        print('---LEXY EXPERIMENTS---')
        print('TO SEGMENT: double left click the cell.')

    def process_step(self, cam_image): 
        """
            Image processing of the camera image.
            Here we segment the cell and separate two areas: one for the nucleus (modulator image) and other for the
            cytosol (processed image). Furthermore, we use the areas of the cytosol, the nucleus, and the background to
            calculate concentrations. No return value, just changes the internal variable processed_image. Can be
            considered as extracting the measurement values from the image.

            Parameters
            ----------
            :param cam_image:  (2D array uint8) image collected from camera
        """

        #if the id_set flag is on, update segment_id based on the processed image, taking the coordinates from the controlled_parameters id, and set flag id_set off
        if self.controlled_parameters["id_set"]:

            #increase index of current cell
            if not (self.current_cell==0 and len(self.setup_array)==0) and not self.cells[self.current_cell].maskReady:
                self.current_cell = (self.current_cell+1)%self.num_cells

            #get click position
            self.controlled_parameters["id_set"] = False
            self.initMeas = True
            id_x, id_y= self.controlled_parameters["id"]

            #add position to array of regions
            if len(self.setup_array)>=2*self.num_cells:
                self.setup_array[self.current_cell] = [[id_y-self.hh_n, id_x-self.wh_n, id_y+self.hh_n, id_x+self.wh_n],
                                                       [id_y-self.hh_c, id_x-self.wh_c, id_y+self.hh_c, id_x+self.wh_c]]
            else:
                self.setup_array.append([id_y-self.hh_n, id_x-self.wh_n, id_y+self.hh_n, id_x+self.wh_n])
                self.setup_array.append([id_y-self.hh_c, id_x-self.wh_c, id_y+self.hh_c, id_x+self.wh_c])

            #set regions array to SAM
            # self.model.setImage(np.int32(self.processed_image_list[0])[np.newaxis]) #XXX
            self.modelCell.setup(np.array(self.setup_array))             # box for initial frame
            print('[CELL {0}] Setting SAM segmentation...'.format(self.current_cell))

        #get segmentation of cell and nucleus - considering time filtering as well
        self.mask_cell, self.mask_nucleus = self.get_segmentation(cam_image)

        #should we consider all or only until current cell?
        if len(self.setup_array)//2==self.num_cells:
            self.cells_ready = self.num_cells
        else:
            self.cells_ready = self.current_cell+1

        #get background mask and erode for every cell
        bkg_mask = self.mask_cell[0]
        
        for nCell in range(self.cells_ready):
            bkg_mask += scipy.ndimage.morphology.binary_fill_holes(self.mask_cell[nCell]==self.internal_parameters["segment_id"][1])
        bkg_mask = scipy.ndimage.binary_erosion(bkg_mask==0,iterations=5)

        #get background value bg for every cell
        if np.median(cam_image[bkg_mask]) is not None:
            self.internal_parameters["bg"] = np.median(cam_image[bkg_mask])
        if self.file is None:
            print('Background value: '+str(self.internal_parameters["bg"]))

        # if self.cells[self.current_cell].maskReady:
        cell_images = [[],[]]
        for nCell in range(self.cells_ready):
            
            #get initial mask for nucleus and cytosol
            self.cells[nCell].internal_parameters["nucleus"] = (self.mask_nucleus[nCell]==self.internal_parameters["segment_id"][0])
            self.cells[nCell].internal_parameters["cytosol"] = (self.mask_cell[nCell]==self.internal_parameters["segment_id"][1])

            #temp mask of dilated nucleus
            dilatedNucleus = 1*scipy.ndimage.binary_dilation(self.cells[nCell].internal_parameters["nucleus"],iterations=25)

            #erode masks to avoid borders
            self.cells[nCell].internal_parameters["nucleus"] = 1*scipy.ndimage.binary_erosion(self.cells[nCell].internal_parameters["nucleus"],iterations=5) #XXX 20x
            self.cells[nCell].internal_parameters["cytosol"] = 1*scipy.ndimage.binary_erosion(self.cells[nCell].internal_parameters["cytosol"],iterations=5)

            #substract dilated nucleus to cytosol
            self.cells[nCell].internal_parameters["cytosol"] = 1*((self.cells[nCell].internal_parameters["cytosol"]-dilatedNucleus)==1)

            cell_images[0].append([self.cells[nCell].internal_parameters["cytosol"]])
            cell_images[1].append([self.cells[nCell].internal_parameters["nucleus"]])

            # get average pixel value of the illumination area, i.e., in the nucleus (concentration in)
            c_i = np.mean(cam_image[self.cells[nCell].internal_parameters["nucleus"] != 0])

            # get average pixel value of the area of the cytosol, i.e., outside the nucleus (concentration out). We
            c_o = np.mean(cam_image[self.cells[nCell].internal_parameters["cytosol"] != 0])

            # Store mean measured intensity in nucleus and cytosol
            self.cells[nCell].internal_parameters["nucleus_intensity"] = c_i - self.internal_parameters["bg"]
            self.cells[nCell].internal_parameters["cytosol_intensity"] = c_o - self.internal_parameters["bg"]
            if self.file is None:
                print('[CELL {}: ]'.format(nCell)+'Nucleus concentration: '+str(c_i)+'| Cytosol concentration: '+str(c_o))

            self.cells[nCell].c_o = c_o
            self.cells[nCell].c_i = c_i

            # get concentration ratio as nucleus intensity / cytosol intensity. Divisor is clipped to prevent
            # division by 0.
            self.cells[nCell].internal_parameters["ratio"] = (self.cells[nCell].internal_parameters["nucleus_intensity"] /
                                                    np.maximum(self.cells[nCell].internal_parameters["cytosol_intensity"], 0.01))

            # On self.maskReady, update initial measurement for control
            if self.initMeas:
                self.initMeas = False
                self.internal_parameters["initial_measurement"] = copy.deepcopy(self.internal_parameters)
        
        self.processed_image = bkg_mask*1 + (np.sum(cell_images[0],0)>0)[0]*3 + (np.sum(cell_images[1],0)>0)[0]*7

    def controller_step(self): 
        """
            This function wraps all the control processing and in the end sets the new mod_image.
            Here we take the concentration values and the controller parameters to obtain the correct LED intensity.
        """

        mod_image = np.zeros(self.mask_nucleus[0].shape)

        for nCell in range(self.cells_ready):

            self.cells[nCell].get_setpoint()

            #scheduling variable
            if self.cells[nCell].controlled_variable == "normalized_cytosol_intensity":
                sch_var = 1-self.cells[nCell].internal_parameters[self.cells[nCell].controlled_parameters["controlled_variable"]]
            else:
                sch_var = self.cells[nCell].internal_parameters[self.cells[nCell].controlled_parameters["controlled_variable"]]          

            conversion_factor = 1
            # Compute control output
            u = self.cells[nCell].controlled_parameters["controller"].step(self.cells[nCell].internal_parameters[self.cells[nCell].controlled_parameters["controlled_variable"]],
                                                                           self.cells[nCell].controlled_parameters["setpoint"],
                                                                           prev_control_applied=self.cells[nCell].mask_power/conversion_factor,
                                                                           scheduling_variable=sch_var)
            u /= conversion_factor

            #calibrate based on OD filter
            if u>0:
                uu = self.odCalibration(u)
                # uu /= 4 #XXX correction???

                # Scale control value to attenuation in setup and clip to LED power bounds
                u = np.clip(uu,
                            self.internal_parameters["min_led_power"],
                            self.internal_parameters["max_led_power"])
            else:
                u = 0

            # set dmd either as area in nucleus or full cell iterate every cell
            if self.cells_ready==1:
                self.led_power = u
                self.cells[nCell].mask_power = u
                u = 255
                self.internal_parameters['max_led_power'] = 100 #XXX OPTOSCOPE
            else:
                self.led_power = self.init_power
                self.internal_parameters['max_led_power'] = 255
                self.cells[nCell].mask_power = u

            if self.illuminate_nucleus:
                if True:    #XXX testing making a circle instead of mask
                    m = moments(np.int32(self.cells[nCell].internal_parameters["nucleus"])==1)
                    cc = [0,0]
                    if m[0,0] !=0:
                        cc = np.array([int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])])
                    xx, yy = np.mgrid[:self.processed_image.shape[0],:self.processed_image.shape[1]]
                    circle = (xx-cc[0])**2 + (yy-cc[1])**2
                    mod_image += (circle < (self.hh_n*0.4)**2)*u
                else:
                    mod_image += (self.cells[nCell].internal_parameters["nucleus"])*u #XXX
                # mod_image += (self.mask_nucleus[nCell])*u
            else:
                # mod_image += (self.mask_cell[nCell]==self.internal_parameters["segment_id"][0])*u
                # mod_image += (self.mask_cell[nCell]!=-1)*u
                # print(self.mask_cell[nCell].dtype)
                mod_image = np.ones(self.mask_cell[nCell].shape)*255
                mod_image[0,0]=0

        self.mod_image = mod_image.astype('uint8')

    def write_data(self, timeStep, date, fpath=''): 
        """
            Function to export data to file.
            The dat exported is the relative concentration, power, setpoint, and (if needed) controller parameters.
            Here the led_Set is set to ON, as this functions is called when the acquisition starts. Not elegant but
            maintains the modularity of the system.

            Parameters
            ----------
            :param timeStep: (ctime format) time of the measurement
            :param date: (datetime.date format) day of the measurement
            :param fpath: (path, default empty) folder to save data
        """

        # save file name (using legacy self.file)
        if self.file is None:
            file = []
            for i in range(self.cells_ready):
                f = r'\{0}_data_{1}_{2}-{3}_cell{4}.csv'.format(self.file_name,str(date), timeStep[11:13], timeStep[14:16],i)
                file.append(f)
                #Create header
                with open(fpath+f, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
                    if 'normalized_' not in self.cells[i].controlled_variable:
                        csvwriter.writerow(['Cytosol','Nucleus', 'Power', 'Setpoint'])
                    else:
                        csvwriter.writerow(['Cytosol','Nucleus', 'Power', 'Setpoint','Controlled_variable'])
            self.file = file

            # set LED ON. This statement is here because this function is called when the acquisition starts. This way
            # the LED and controller don't start until the acquisition starts.
            if not self.controlled_parameters["led_set"]:
                for cell in self.cells:
                    cell.smart_loop_parameters["counter"] = time.time()
                    cell.controlled_parameters["controller"].integral["value"] = 0.0
                self.controlled_parameters["led_set"] = True

            #initial area and segmentation - when we press acquire
            self.area_nuc = []
            for i in range(self.cells_ready):
                props_nuc = regionprops(np.array(self.modelCell.getResult()[2*i].squeeze(), dtype=np.uint8))[0]
                self.area_nuc.append(props_nuc.area) # area at first timepoint
                self.cells[i].mask_nuc_filter[0] = self.modelCell.getResult()[2*i]

        # get data in one list
        for i in range(self.cells_ready):
            cell = self.cells[i]
            if 'normalized_' in cell.controlled_variable: 
                data = [cell.c_o - self.internal_parameters["bg"],cell.c_i - self.internal_parameters["bg"], cell.mask_power,
                        cell.controlled_parameters["setpoint"],cell.internal_parameters[cell.controlled_parameters["controlled_variable"]]]
            else:
                data = [cell.internal_parameters["cytosol_intensity"],cell.internal_parameters["nucleus_intensity"], cell.mask_power,
                        cell.controlled_parameters["setpoint"]]

            # save in csv
            with open(fpath + self.file[i], 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(data)

            # print information
            print('cell: ', i,
                ' power: ', cell.mask_power,
                ' setpoint: ', cell.controlled_parameters["setpoint"],
                ' control: ', cell.internal_parameters[cell.controlled_parameters["controlled_variable"]])
            
            # if i==0:
            #     # gv.plot_manager.add_data_point(time.time(), cell.controlled_parameters["setpoint"], 'primary')
            #     gv.plot_manager.add_data_point(time.time(), cell.internal_parameters[cell.controlled_parameters["controlled_variable"]], 'primary')
            #     gv.plot_manager.add_data_point(time.time(), cell.mask_power, 'secondary')
    def set_parameters(self, queue_dict):
        """
            This function takes a dictionary containing the changed parameter values.
            It assigns those to the entries in controlled parameters with the same key.

            Parameters
            ----------
            :param queue_dict: (dictionary) parameetr values to change
        """

        # iterate queue and try to change the parameter.
        for control in queue_dict:
            try:

                # get parameter to change from parameter_controls. if it exists, change the controlled_parameters'
                # parameter and print
                # note: parameter_controls has as value the keys of the controlled_parameters dictionary!
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control, "event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass

    def get_segmentation(self,cam_image):
        '''
        Get segmentation from SAM and apply time filter
        '''
        #segment cells in image
        if self.modelCell.getRegions().shape[0] > 0:
            self.maskReady = True

            x = self.modelCell.getRegions()
            self.modelCell.setImage(image=cam_image[np.newaxis])
            self.modelCell.run()

            #get all segmentations
            allRes = self.modelCell.getResult()
            masks_cell = []
            masks_nucleus = []

            #iterate for every cell
            for i in range(len(allRes)//2):

                #get cytosol segmentation
                masks_cell.append(self.modelCell.getResult()[2*i+1])

                if not np.isnan(self.area_nuc).all():

                    #get new nucleus segmentation to compare with previous frame
                    self.cells[i].mask_nuc_filter[1] = self.modelCell.getResult()[2*i]
                    props_nuc = regionprops(np.array(self.cells[i].mask_nuc_filter[1].squeeze(), dtype=np.uint8))[0]
                    intersection_nuc = np.sum(self.cells[i].mask_nuc_filter[0] * self.cells[i].mask_nuc_filter[1], axis=None) / props_nuc.area

                    #conditions to use last frame
                    cond1 = np.abs(np.log(props_nuc.area/self.area_nuc[i])/np.log(1.2))<1
                    cond2 = intersection_nuc > 0.85
                    cond3 = functions.circularity(props_nuc.area, props_nuc.perimeter) > 0.85
                    
                    #get nucleus segmentation
                    if cond1 and cond2 and cond3:
                        self.cells[i].mask_nuc_filter[0] = self.cells[i].mask_nuc_filter[1]
                        masks_nucleus.append(np.int32(self.cells[i].mask_nuc_filter[1]))
                    else:
                        masks_nucleus.append(np.int32(self.cells[i].mask_nuc_filter[0]))
                        # print('Using previous frame at cell {}'.format(i))
                else:
                    masks_nucleus.append(self.modelCell.getResult()[2*i])
        else:
            masks_nucleus = [np.zeros_like(cam_image)]
            masks_cell = [np.zeros_like(cam_image)]

        return masks_cell,masks_nucleus

    def odCalibration(self,irradiance):
        
        if self.ODfit in [0.0,0.5,0.9,1.4,1.5,2.0]:

            try:
                if self.microscope=='OPTO': 
                    if self.ODfit==0.0:
                        p1 = np.poly1d([ 3.76398739e-04, -6.24804269e-02,  1.39235446e+01,  0.0])
                        total = p1(self.led_power)*1e3  
                        if self.cells_ready==1:
                            inverse = (p1 - irradiance).roots
                            res = inverse[inverse.imag==0]
                            uu = res[res>0][0].real              
                        else:
                            uu = int(255*irradiance/total)
                    elif self.ODfit==0.5:
                        p1 = np.poly1d([ 8.74529579e-05, -1.49971967e-02,  3.93786138e+00,  0.00000000e+00])
                        total = p1(self.led_power)*1e3  
                        if self.cells_ready==1:
                            inverse = (p1 - irradiance).roots
                            res = inverse[inverse.imag==0]
                            uu = res[res>0][0].real              
                        else:
                            uu = int(255*irradiance/total)
                    elif self.ODfit==0.9:
                        p1 = np.poly1d([ 4.23176634e-05, -7.16502492e-03,  1.51825109e+00,  0.00000000e+00])
                        total = p1(self.led_power)*1e3  
                        if self.cells_ready==1:
                            inverse = (p1 - irradiance).roots
                            res = inverse[inverse.imag==0]
                            uu = res[res>0][0].real              
                        else:
                            uu = int(255*irradiance/total)
                    elif self.ODfit==1.5:
                        p1 = np.poly1d([ 8.32678156e-06, -1.43096457e-03,  3.06984879e-01,  0.00000000e+00])
                        total = p1(self.led_power)*1e3  
                        if self.cells_ready==1:
                            inverse = (p1 - irradiance).roots
                            res = inverse[inverse.imag==0]
                            uu = res[res>0][0].real              
                        else:
                            uu = int(255*irradiance/total)
                    elif self.ODfit==2.0:
                        p1 = np.poly1d([ 1.90953468e-06, -3.11938512e-04,  6.70000706e-02,  0.00000000e+00])
                        total = p1(self.led_power)*1e3  
                        if self.cells_ready==1:
                            inverse = (p1 - irradiance).roots
                            res = inverse[inverse.imag==0]
                            uu = res[res>0][0].real              
                        else:
                            uu = int(255*irradiance/total)
                    else:
                        inverse = (p0 - irradiance/self.internal_parameters["attenuation"]).roots
                        res = inverse[inverse.imag==0]
                        uu = res[res>=0][0].real
                else:
                    if self.ODfit==0.5:
                        p1 = np.poly1d([ 1.45540422e-25, -7.22998680e-22,  1.53730246e-18, -1.82709303e-15, 1.32886943e-12, -6.07825856e-10,  1.72518330e-07, -2.86652496e-05, 2.35478005e-03,  1.33829724e-01,  0.00000000e+00])
                        inverse = (p1 - irradiance).roots
                        res = inverse[inverse.imag==0]
                        uu = res[res>0][0].real
                    elif self.ODfit==0.9:
                        p1 = np.poly1d([ 6.27292492e-26, -3.11345623e-22,  6.61399806e-19, -7.85320229e-16, 5.70605024e-13, -2.60733403e-10,  7.39341928e-08, -1.22765249e-05, 1.00873168e-03,  5.64286799e-02,  0.00000000e+00])
                        inverse = (p1 - irradiance).roots
                        res = inverse[inverse.imag==0]
                        uu = res[res>0][0].real
                    elif self.ODfit==1.4:
                        p1 = np.poly1d([ 2.02891422e-26, -1.00739179e-22,  2.14093695e-19, -2.54331305e-16, 1.84900983e-13, -8.45470172e-11,  2.39943052e-08, -3.98844667e-06, 3.28216385e-04,  1.82445370e-02,  0.00000000e+00])
                        inverse = (p1 - irradiance).roots
                        res = inverse[inverse.imag==0]
                        uu = res[res>0][0].real
                    else:
                        p0 = np.poly1d([4.41480269e-25,-2.19459790e-21, 4.67006203e-18,-5.55564172e-15,4.04514753e-12,-1.85255380e-09, 5.26500384e-07,-8.75911068e-05,7.20427821e-03, 4.08846259e-01, 0.0])
                        inverse = (p0 - irradiance/self.internal_parameters["attenuation"]).roots
                        res = inverse[inverse.imag==0]
                        uu = res[res>=0][0].real

            except:
                # print('conversion curve failed')
                uu = irradiance / self.internal_parameters["attenuation"]
        
        else:
            uu = irradiance / self.internal_parameters["attenuation"]
            # print('OD attenuation not recognized!')

        return uu
