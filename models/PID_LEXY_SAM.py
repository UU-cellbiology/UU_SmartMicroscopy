#region Libraries
import configs.functions as functions
import numpy as np
import scipy
import time
import csv
from Controllers.PID_controller import PIDController
from models.abstract_model import AbstractModel
import importlib
import copy
from skimage.measure import regionprops
#endregion

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

            # assign default values if some parameters are missing
            if not setpoint_array:
                setpoint_array = [(0, 0.1)]
            elif setpoint_array[0][0] != 0:
                setpoint_array.insert((0, 0.1))
            if not att:
                att = 0.0
            if not use_gainscheduling:
                use_gainscheduling = False
            if not use_antiwindup:
                use_antiwindup = False

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

        if controlled_variable == "nucleus_intensity" or controlled_variable == "ratio" or controlled_variable == "normalized_nucleus_intensity":
            parity = -1.0
        else:
            parity = 1.0

        if controlled_variable == "ratio": #XXX
            lowerBound = -50/100
        else:
            lowerBound = -0.1

        self.controlled_variable = controlled_variable
        if 'normalized_' in controlled_variable:
            controlled_variable = controlled_variable[11:]
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
            "setpoint": 0.1,  # concentration setpoint
            "controller": PIDController(K=LEXY_parameters[0:-1], T=LEXY_parameters[-1], antiwindup=use_antiwindup, parity=parity, lb=lowerBound),  # PID Controller object
            "use_gainscheduling": use_gainscheduling,  # Use gain scheduling in the PID controller
            "use_antiwindup": use_antiwindup,  # Use anti-windup for the PID controller
            "controlled_variable": controlled_variable, # Either nucleus_intensity, cytosol_intensity or ratio, relating to the respective parameters in the internal_parameters dictionary
            # "controller": PIDController(K=LEXY_parameters[0:-1],T=LEXY_parameters[-1])  # PID Controller object
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
            "nucleus": np.zeros(self.processed_image.shape),  # mask of segmented nucleus of cell
            "cytosol": np.zeros(self.processed_image.shape),  # mask of segmented cytosol of cell
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

        # Setup Gain scheduling if enabled
        if self.controlled_parameters["use_gainscheduling"]:
            self.controlled_parameters["controller"].setup_gain_scheduling()
            
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
            self.controlled_parameters["id_set"] = False
            self.initMeas = True
            self.id_x, self.id_y= self.controlled_parameters["id"]
            self.modelCell.setImage(self.processed_image[np.newaxis])
            self.modelCell.setup(np.array([[self.id_y-self.hh_n, self.id_x-self.wh_n, self.id_y+self.hh_n, self.id_x+self.wh_n],
                                           [self.id_y-self.hh_c, self.id_x-self.wh_c, self.id_y+self.hh_c, self.id_x+self.wh_c],]))             # box for initial frame

        #get segmentation of cell and nucleus - considering time filtering as well
        mask_cell, mask_nucleus = self.get_segmentation(cam_image)

        if self.maskReady:
            
            #get initial mask for nucleus and cytosol
            self.internal_parameters["nucleus"] = (mask_nucleus==self.internal_parameters["segment_id"][0])
            self.internal_parameters["cytosol"] = (mask_cell==self.internal_parameters["segment_id"][1])

            #temp mask of dilated nucleus
            dilatedNucleus = 1*scipy.ndimage.binary_dilation(self.internal_parameters["nucleus"],iterations=10)

            #erode masks to avoid borders
            self.internal_parameters["nucleus"] = 1*scipy.ndimage.binary_erosion(self.internal_parameters["nucleus"],iterations=10)
            self.internal_parameters["cytosol"] = 1*scipy.ndimage.binary_erosion(self.internal_parameters["cytosol"],iterations=20)

            #substract dilated nucleus to cytosol XXX maybe eroded is enough?
            self.internal_parameters["cytosol"] = 1*((self.internal_parameters["cytosol"]-dilatedNucleus)==1)

            #assign nucleus area to processed_image for GUI visualization
            self.processed_image = self.internal_parameters["nucleus"]

            #get background mask and erode
            bkg_mask = scipy.ndimage.morphology.binary_fill_holes(mask_cell==self.internal_parameters["segment_id"][1])
            bkg_mask = scipy.ndimage.binary_erosion(bkg_mask==False,iterations=50)

            #get background value 
            if np.median(cam_image[bkg_mask]) is not None:
                self.internal_parameters["bg"] = np.median(cam_image[bkg_mask])

            # set dmd either as area in nucleus or full cell XXX may needa 1* or 255*?
            if self.illuminate_nucleus:
                self.mod_image = (mask_nucleus==self.internal_parameters["segment_id"][0])
            else:
                # self.mod_image = (mask_cell==self.internal_parameters["segment_id"][1])
                self.mod_image = np.ones(cam_image.shape)*255
                self.mod_image[0,0]=0
            self.processed_image = bkg_mask*1 + self.internal_parameters["cytosol"]*2 + self.internal_parameters["nucleus"]*3

            # get average pixel value of the illumination area, i.e., in the nucleus (concentration in)
            c_i = np.mean(cam_image[self.internal_parameters["nucleus"] != 0])

            # get average pixel value of the area of the cytosol, i.e., outside the nucleus (concentration out). We
            c_o = np.mean(cam_image[self.internal_parameters["cytosol"] != 0])

            # Store mean measured intensity in nucleus and cytosol
            self.internal_parameters["nucleus_intensity"] = c_i - self.internal_parameters["bg"]
            self.internal_parameters["cytosol_intensity"] = c_o - self.internal_parameters["bg"]
            if self.file is None:
                print('Nucleus concentration: '+str(c_i))
                print('Cytosol concentration: '+str(c_o))
                print('Background value: '+str(self.internal_parameters["bg"]))

            self.c_o = c_o
            self.c_i = c_i

            if self.internal_parameters["maximum_int_measurement"] == 3000.0 or self.internal_parameters["minimum_int_measurement"] == 200.0:
                print("Warning: Normalization parameters not configured")
        else:
            c_i=0
            c_o=0

        # get concentration ratio as nucleus intensity / cytosol intensity. Divisor is clipped to prevent
        # division by 0.
        self.internal_parameters["ratio"] = (self.internal_parameters["nucleus_intensity"] /
                                                np.maximum(self.internal_parameters["cytosol_intensity"], 0.01))

        # On self.maskReady, update initial measurement for control
        if self.maskReady and self.initMeas:
            self.initMeas = False
            self.internal_parameters["initial_measurement"] = copy.deepcopy(self.internal_parameters)

    def controller_step(self):
        """
            This function wraps all the control processing and in the end sets the new mod_image.
            Here we take the concentration values and the controller parameters to obtain the correct LED intensity.
        """

        # get setpoint based on the time that has passed
        setpoint = self.get_setpoint()

        # assign setpoint. This will overwrite any input from user
        self.controlled_parameters["setpoint"] = setpoint[-1]

        #scheduling variable
        if self.controlled_variable == "normalized_cytosol_intensity":
            sch_var = 1-self.internal_parameters[self.controlled_parameters["controlled_variable"]]
            design_expression_factor = self.internal_parameters["minimum_int_measurement"] / 360.0    # Cytosol intensity
        else:
            sch_var = self.internal_parameters[self.controlled_parameters["controlled_variable"]]          
            design_expression_factor = self.internal_parameters["maximum_int_measurement"] / 570.0    # Nucleus intensity

        conversion_factor = 1
        # Compute control output
        u = self.controlled_parameters["controller"].step(self.internal_parameters[self.controlled_parameters["controlled_variable"]],
                                                            self.controlled_parameters["setpoint"],
                                                            prev_control_applied=self.led_power/conversion_factor,
                                                            scheduling_variable=sch_var)
        u /= conversion_factor

        if self.ODfit in [0.5,0.9,1.4]:
            #transform output to irradiance
            p0 = np.poly1d([2.4462704767148365e-25, -1.216040771448087e-21, 2.5877113159722006e-18, -3.0784167004911703e-15, 2.241442186872485e-12, -1.0265119378942568e-09, 2.9173723888269126e-07, -4.8534793924076166e-05, 0.0039919367541054185, 0.22654433369393628, 0.0])
            irradiance = p0(u)

            #get LED value back
            try:
                if self.ODfit==0.5:
                    p1 = np.poly1d([8.064488092650016e-26, -4.0061820460761246e-22, 8.518291488117697e-19, -1.0124039647572173e-15, 7.363350707168426e-13, -3.368002043837635e-10, 9.559351312628607e-08, -1.5883598594692223e-05, 0.0013047987261475599, 0.07415590831534427, 0.0])
                    inverse = (p1 - irradiance).roots
                    res = inverse[inverse.imag==0]
                    uu = res[res>0][0].real
                elif self.ODfit==0.9:
                    p1 = np.poly1d([3.475867919799597e-26, -1.7251860586951488e-22, 3.66485873475562e-19, -4.351509748184323e-16, 3.161759027063436e-13, -1.44474050575047e-10, 4.096741031830757e-08, -6.80250115282812e-06, 0.0005589446925517952, 0.031267493383880865, 0.0])
                    inverse = (p1 - irradiance).roots
                    res = inverse[inverse.imag==0]
                    uu = res[res>0][0].real
                elif self.ODfit==1.4:
                    p1 = np.poly1d([1.124234376429241e-26, -5.582022497440792e-23, 1.186306891491435e-19, -1.409266071391446e-16, 1.024548203564747e-13, -4.6848044387807016e-11, 1.329539836209176e-08, -2.210023872730046e-06, 0.00018186680360326763, 0.010109414922018953, 0.0])
                    inverse = (p1 - irradiance).roots
                    res = inverse[inverse.imag==0]
                    uu = res[res>0][0].real
                else:
                        inverse = (p0 - irradiance/self.internal_parameters["attenuation"]).roots
                        res = inverse[inverse.imag==0]
                        uu = res[res>=0][0].real

            except:
                print('conversion curve failed')
                uu = u / self.internal_parameters["attenuation"]
        else:
            uu = u / self.internal_parameters["attenuation"]
            print('OD attenuation not recognized!')

        # Scale control value to attenuation in setup and clip to LED power bounds
        u = np.clip(uu,
                    self.internal_parameters["min_led_power"],
                    self.internal_parameters["max_led_power"])

        # store LED power from the controller value in different formats
        self.internal_parameters["led_power"] = u

        # get LED power as integer output
        self.led_power = self.internal_parameters["led_power"]

    def get_setpoint(self):
        self.time_passed = time.time() - self.smart_loop_parameters['counter']
        if self.controlled_variable == "normalized_nucleus_intensity" or self.controlled_variable == "normalized_cytosol_intensity":
            self.internal_parameters[self.controlled_parameters["controlled_variable"]] = ((self.internal_parameters[self.controlled_parameters["controlled_variable"]] - self.internal_parameters["minimum_int_measurement"]) /
                                                                     (self.internal_parameters["maximum_int_measurement"] - self.internal_parameters["minimum_int_measurement"]))
            relative_factor = 1.0
        else:
            relative_factor = 1.0

        setpoint = [item[1] * relative_factor for item in self.smart_loop_parameters['setpoint_array'] if self.time_passed > item[0]]
        return setpoint

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

        # get data in one list (ratio, power, setpoint, and 3 coeff)
        if 'normalized_' in self.controlled_variable: 
            data = [self.c_o - self.internal_parameters["bg"],self.c_i - self.internal_parameters["bg"], self.internal_parameters["led_power"],
                    self.controlled_parameters["setpoint"],self.internal_parameters[self.controlled_parameters["controlled_variable"]]]
        else:
            data = [self.internal_parameters["cytosol_intensity"],self.internal_parameters["nucleus_intensity"], self.internal_parameters["led_power"],
                    self.controlled_parameters["setpoint"]]

        # save file name (using legacy self.file)
        if self.file is None:
            self.file = '\{0}_data_{1}_{2}-{3}.csv'.format(self.file_name,str(date), timeStep[11:13], timeStep[14:16])

            # Create header
            with open(fpath + self.file, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
                if 'normalized_' not in self.controlled_variable:
                    csvwriter.writerow(['Cytosol','Nucleus', 'Power', 'Setpoint'])
                else:
                    csvwriter.writerow(['Cytosol','Nucleus', 'Power', 'Setpoint','Controlled_variable'])

            # set LED ON. This statement is here because this function is called when the acquisition starts. This way
            # the LED and controller don't start until the acquisition starts.
            if not self.controlled_parameters["led_set"]:
                self.smart_loop_parameters["counter"] = time.time()
                self.controlled_parameters["controller"].integral["value"] = 0.0
                self.controlled_parameters["led_set"] = True

            #initial area and segmentation - when we press acquire
            props_nuc = regionprops(np.array(self.modelCell.getResult()[0].squeeze(), dtype=np.uint8))[0]
            props_cyt = regionprops(np.array(self.modelCell.getResult()[1].squeeze(), dtype=np.uint8))[0]
            self.area_nuc = props_nuc.area # area at first timepoint
            self.mask_nuc_filter[0] = self.modelCell.getResult()[0]
            self.mask_cyt_filter[0] = self.modelCell.getResult()[1]

            self.area_cyt = props_cyt.area

        # save in csv
        with open(fpath + self.file, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(data)

        # print information
        print('power: ', self.internal_parameters["led_power"],
              'nucleus value: ', self.c_i - self.internal_parameters["bg"],
              'cytosol value: ', self.c_o - self.internal_parameters["bg"],
              'ratio value:   ', self.internal_parameters["ratio"],
              'setpoint:      ', self.controlled_parameters["setpoint"],
              'current controlled value: ', self.internal_parameters[self.controlled_parameters["controlled_variable"]])

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

            if not np.isnan(self.area_nuc):
                self.mask_nuc_filter[1] = self.modelCell.getResult()[0]
                props_nuc = regionprops(np.array(self.mask_nuc_filter[1].squeeze(), dtype=np.uint8))[0]
                intersection_nuc = np.sum(self.mask_nuc_filter[0] * self.mask_nuc_filter[1], axis=None) / props_nuc.area
                if np.abs(np.log(props_nuc.area/self.area_nuc)/np.log(1.2))<1 and intersection_nuc > 0.85 and functions.circularity(props_nuc.area, props_nuc.perimeter) > 0.85:
                    self.mask_nuc_filter[0] = self.mask_nuc_filter[1]
                    mask_nucleus = np.int32(self.mask_nuc_filter[1])
                else:
                    mask_nucleus = np.int32(self.mask_nuc_filter[0])
                    print('Nucleus segmentation incorrect')
                    #previous cytosol
            else:
                mask_nucleus = self.modelCell.getResult()[0]

            mask_cell = self.modelCell.getResult()[1]
        else:
            mask_nucleus = np.zeros_like(cam_image)
            mask_cell = np.zeros_like(cam_image)

        return mask_cell,mask_nucleus
