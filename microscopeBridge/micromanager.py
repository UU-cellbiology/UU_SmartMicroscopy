#region Prelude

#region Description
'''
XXX

Language:       Python
Author(s):      Alfredo Rates and Josiah Passmore
Creation date:  2024-01-08
Contact:        a.ratessoriano@uu.nl
Git:            https://github.com/passm003/PEXscope
Version:        1.0
'''
#endregion

#region libraries
from microscopeBridge.abstract_bridge import abstract_Bridge
from pycromanager import Core
import numpy as np
import copy
import numpy as np
import time
from pycromanager import Acquisition, multi_d_acquisition_events
import datetime
import tifffile
import os
#endregion

#endregion 

#region PEX 
class micromanager(abstract_Bridge):
    def __init__(self,inputs):
        '''
            XXX

            Structural parameters:
                self.im_height
                self.im_width
                self.mod_height
                self.mod_width
                self.acq_path
                self.acq_name
                self.running
                self.image_ready
                self.Acq_image
                self.modulator_dict
                self.modulator_ready

            Structural methods:
                self.live_image
                self.shutdown
                self.run_acquisition
                self.set_modulation

            Parameters
            ----------
            :param inputs: dictonary with all user inputs.
        '''

        #initialize Modulator
        self.core = Core()
        self.Modulator = self.core.get_slm_device()
        self.mod_height, self.mod_width = self.core.get_slm_height(self.Modulator), self.core.get_slm_width(self.Modulator)
        self.im_height, self.im_width = self.core.get_image_height(), self.core.get_image_width()

        #print Modulator type
        print('Modulator initialized, model: {}'.format(self.core.get_device_type(self.Modulator)))

        #iterate to get channels info
        channel_lab_text = ''
        channel_exp_text = ''
        for channels in inputs['channels']:
            channel_exp_text += '{},'.format(channels[1])
            channel_lab_text += '"{}",'.format(channels[0])

        #create acquisition settings string
        acq_settings = r'"num_time_points":{0},"time_interval_s":{1},"channel_group": "Channel","channels":[{2}],"channel_exposures_ms":[{3}]'.format(inputs['n_time_points'],
                                                                                                                                                    inputs['time_interval_s'],
                                                                                                                                                    channel_lab_text[:-1],
                                                                                                                                                    channel_exp_text[:-1])
        self.acq_args = '{'+acq_settings+'}'
        self.acq_args = eval(self.acq_args)

        #get path and name to save data
        self.acq_path = inputs['folder_name']
        self.acq_name = inputs['file_name']
        self.channels = inputs['channels']
        self.waiting_multichannel = inputs['waiting_multichannel']

        #counter in case of multi-channel
        self.channelIdx=0

        #acquisition variables
        self.led_power = inputs['led_power']
        self.led_set = inputs['led_set']
        self.running = False
        self.image_ready = False
        self.modulator_ready = False
        self.hookWait = False
        self.acq_check = True

        #internal queue variables
        self.modulator_dict = {"image":np.zeros((self.im_height,self.im_width)), "power": self.led_power, "set": self.led_set}
        self.Acq_image = np.zeros((self.im_height,self.im_width))

        #filter wheel parameters
        self.prev_channel = self.mm_get_channel()

    def live_image(self):
        self.core.snap_image()
        tagged_image = self.core.get_tagged_image()
        return np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])

    def shutdown(self):
        self.mm_led(False)
        self.core.set_slm_image(self.Modulator,np.zeros((self.mod_height,self.mod_width)).astype("uint8").flatten())

    def set_modulation(self,mask=''):
        if type(mask)==type(''):
            self.core.set_slm_image(self.Modulator,self.modulator_dict["image"])
        else:
            mask = mask.flatten()
            self.core.set_slm_image(self.Modulator,mask)

    def run_acquisition(self):
        '''
        Run the acquisition procedure. This function takes all the acquisiton processes defined and communicate to micromanager. 
        This function depends on acq_process and mod_hook as linked functiosn to multi_d_acquisition_events().
        '''

        #set flag as true
        self.running = True

        #start acquisition event with defined arguments. the image process function is acq_process and the hardware hook is mod_hook
        with Acquisition(directory=str(self.acq_path), 
                         name=str(datetime.date.today())+self.acq_name, 
                         show_display=True, 
                         image_process_fn=self.acq_process, 
                         pre_hardware_hook_fn=self.mod_hook,
                         post_hardware_hook_fn=self.post_hardware_hook) as acq:
           events = multi_d_acquisition_events(**(self.acq_args))
           acq.acquire(events)

        #set flag as False
        self.running = False
    
    def acq_process(self,image,metadata):
        '''
        This function is called every time we take a new picture with Acquisition. 
        Here, the image is collected along with the modulator image and saved together, along with their metadata.
        The output of this function is saved automatically. 

        Parameters
        ----------
        :param image: return file from the Acquisition function from pycromanager, i.e., acquire image.
        :param metadata: return file from the Acquisition function from pycromanager, i.e., metadata of the acquire image.

        Return
        ----------
        :return: (list with 2 tuples) Each tuple consist of image+metadata. they correspond to camera image and modulator image, respectively
        '''

        #if the flag acq_check is on (only when an object of the class is created), get acquisition channel from metadata and set flag off 
        self.hookWait=True
        if self.acq_check:
            self.acq_channel1 = metadata["Channel"]
            self.acq_check = False
        
        #if the acquire image comes from the same initial channel
        if metadata["Channel"] == self.acq_channel1:

            #assign image to queue from real data
            self.Acq_image = image
            self.image_ready = True

            #get and set image for Modulator
            while not self.modulator_ready:
                time.sleep(0.1)
            self.set_modulation(self.modulator_dict["image"])
            self.modulator_ready = False

            #set LED power - even if it is off. 
            self.mm_power(self.modulator_dict['power'])
            
            #change global variable for post_hook of Modulator
            self.led_set = self.modulator_dict['set']
            self.led_power = self.modulator_dict['power']

        #counter to know what channel we are measuring
        self.channelIdx+=1

        #turn LED ON if assigned. if this is not the last channel, keep led on
        if self.led_set and self.channelIdx==len(self.channels):
            self.channelIdx=0
            self.mm_led(True)
        self.hookWait=False

        return [(image, metadata)]
    
    def mod_hook(self, event):
        '''
        Set modulator image and LED while acquiring, if needed.
        This function is called before the hardware updates. 
        XX edit after changes!

        Parameters
        ----------
        :param event: output from Aquisiton function from pycomanager.

        Return
        ----------
        :return: same input, event.
        '''

        #turn LED off just before the acquisition using preset of micromanager
        while self.hookWait:
            time.sleep(0.1)
        self.mm_led(False)
            
        #return input
        return event
    
    def post_hardware_hook(self,event):
        '''
        Wait for microscope to change channel
        '''

        if len(self.channels)>1:
            time.sleep(self.waiting_multichannel)
        return event

    def set_LED(self,power=None,set=None):
        '''
        Set power to LED 
        XXX
        '''

        #set LED status
        if set is not None:
            if set:
                print("change set ON")
                time.sleep(0.1)
                self.mm_led(True)
                time.sleep(0.5)
            else:
                print("change set OFF")
                time.sleep(0.1)
                self.mm_led(False)
                time.sleep(0.5)

        #set LED power
        if power is not None:
            print("change power")
            time.sleep(0.1)
            self.mm_power(power)
            time.sleep(0.5)

    def set_channel(self,channel='1-Transmission',previous=False):
        '''
        Change filter wheel - for now only for calibration
        XXX
        '''

        if previous:
            
            try:
                time.sleep(0.1)
                tmp = self.mm_get_channel()
                time.sleep(0.5)
                self.mm_channel(self.prev_channel)
                time.sleep(1)
                self.prev_channel = tmp
            except:
                print('Change of channel failed')
        else:
            try:
                time.sleep(0.1)
                self.prev_channel = self.mm_get_channel()
                time.sleep(0.5)
                self.mm_channel(channel)
                time.sleep(1)
            except:
                print('Change of channel failed')

    def mm_channel(self,channel=''):
        '''
        The functions mm_ depend on the configuration file and hardware used in the micromanager software. 
        '''
        self.core.set_config("Channel", channel)

    def mm_power(self,val):
        '''
        The functions mm_ depend on the configuration file and hardware used in the micromanager software. 
        '''
        self.core.set_property("Mightex_BLS(USB)", "normal_CurrentSet", val)

    def mm_led(self,set):
        '''
        The functions mm_ depend on the configuration file and hardware used in the micromanager software. 
        '''
        if set:
            self.core.set_config("LED", "Blue ON")
        else:
            self.core.set_config("LED", "Blue OFF")

    def mm_get_channel(self):
        '''
        The functions mm_ depend on the configuration file and hardware used in the micromanager software. 
        '''
        self.core.get_property("TIFilterBlock1","Label")
#endregion