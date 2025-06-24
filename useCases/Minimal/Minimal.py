###############################################################
'''                 IMPORTS                                 '''
###############################################################

#Import global variables to share between modules
from UU_SmartMicroscopy.configs import globVars as gv, functions

#Import interface
from UU_SmartMicroscopy.Interface import mainWindow

#Import bridge to microscope
from UU_SmartMicroscopy.microscopeBridge import demo as demoMicroscope

#Import model to use, i.e., smart miroscopy experiment. 
#NOTE: you should define the MODEL that you want to use, and from there you import the respective CONTROLLER and SEGMENTATION
from UU_SmartMicroscopy.models import demo as demoModel

#Import other libraries
import os
import threading
import datetime
import time
import tkinter

###############################################################
'''                 THREADINGS                              '''
###############################################################

#create class for parallel analysis
class ModelThread(threading.Thread):
    def __init__(self,inputs,model,Microscope):

        #initialize a threading object
        threading.Thread.__init__(self)

        #set experiment model and microscope
        self.model = model
        self.Microscope = Microscope

        #set variables
        self.interval = inputs['time_interval_s']
        gv.hasPath = False
        gv.needsCali = False

    def run(self):
        while not gv.main_event.isSet():
            
            #add parameters from settings to model
            while not gv.parameter_queue.empty():
                self.model.set_parameters(gv.parameter_queue.get())

            if not self.Microscope.running:
                time.sleep(self.interval)

            #process image using processing function
            self.cam_image = self.Microscope.Acq_image
            self.model.process_step(self.cam_image)

            #take control decision using control function
            self.model.controller_step()
            mod = self.model.mod_image
            
            self.Microscope.modulator_dict = {"image":mod, "power": 1, "set": True}
            self.Microscope.modulator_ready = True

            #assign image, segmentation, control to GUI queue
            gv.image_ui_queue.put(functions.image_to_uint8(self.cam_image))
            gv.processed_ui_queue.put(functions.image_to_uint8(self.model.processed_image))
            gv.mod_ui_queue.put(functions.image_to_uint8(mod))


class ControlThread(threading.Thread):
    def __init__(self,inputs,Microscope):

        #initialize a threading object
        threading.Thread.__init__(self)

        #set experiment model and microscope
        self.Microscope = Microscope

        #set variables
        self.interval = inputs['time_interval_s']

    def run(self):
            while not gv.main_event.isSet():

                #get snap of data if acquisition not running
                snap = self.Microscope.live_image()
                self.Microscope.Acq_image = snap
                time.sleep(self.interval)

                #If acquisition button is pressed, run acquisition
                if gv.acq_event.isSet():
                    self.Microscope.run_acquisition()
                    gv.acq_event.clear()

if __name__ == "__main__":
    ###############################################################
    '''                 DEMO INPUTS                             '''
    ###############################################################

    #Example data from repository, try also LEXY_U2OS_mCherry_100ms.tif
    example_data    = 'Migration_HT1080_mCherry_100ms.tif'

    #Number of points for the whole experiment
    n_points        = 10

    #Time between points in seconds
    interval_s      = 1

    ###############################################################
    '''                 CONNECT TO MICROSCOPE                   '''
    ###############################################################

    #Get path to demo data
    demo_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir,os.pardir, 'info/demo_data'))
    demo_path = demo_path + '/' + example_data

    #Create directory with the inputs needed for the demo microscope
    microscope_inputs = {
        'demo_path'         : demo_path,
        'n_time_points'     : n_points,
        'time_interval_s'   : interval_s
    }

    #Define microscope object
    Microscope = demoMicroscope(microscope_inputs)

    ###############################################################
    '''                 DEFINE GLOBAL VARIABLES                 '''
    ###############################################################

    print('Demo experiment initiated...')

    #start timer
    timer = datetime.datetime.now()

    #start demo model
    demo = demoModel()

    #get variables for calibration
    gv.microscopeInfo = [Microscope.im_width,Microscope.im_height,Microscope.mod_height,Microscope.mod_width]
    gv.inputs = microscope_inputs
    gv.menuList = demo.menu_list

    ###############################################################
    '''                 RUN EXPERIMENT                          '''
    ###############################################################

    #Start thread
    model_thread    = ModelThread(microscope_inputs,demo,Microscope)
    control_thread  = ControlThread(microscope_inputs,Microscope)
    model_thread.start()
    control_thread.start()

    #start GUI
    root   = tkinter.Tk()
    window = mainWindow(root,
                        microscope_inputs,
                        Microscope.im_height,
                        Microscope.im_width)
    root.resizable(width=0, height=0)
    root.mainloop()

    #Close threads
    gv.main_event.set()
    model_thread.join()
    control_thread.join()
    threading._shutdown()

    print('Demo experiment finished.')

    #print elapsed time
    print('Elapsed time: {}'.format(datetime.datetime.now() - timer))
