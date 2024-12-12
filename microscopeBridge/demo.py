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
import numpy as np
import time
from skimage import io
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tkinter
#endregion

#endregion 

#region DEMO
class demo(abstract_Bridge):
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

        #initial variables
        self.stack = io.imread(inputs['demo_path'])#[:,0]
        self.frame = 0
        self.im_height, self.im_width = self.stack.shape[1], self.stack.shape[2]
        self.mod_height, self.mod_width = self.stack.shape[1], self.stack.shape[2]
        print('DEMO initialized')

        #time series variables
        self.n_time_points = inputs['n_time_points']
        self.time_interval_s = inputs['time_interval_s']

        #acquisition variables
        self.led_power = inputs['led_power']
        self.led_set = inputs['led_set']
        self.running = False
        self.image_ready = False
        self.modulator_ready = False

        #internal queue variables
        self.modulator_dict = {"image":np.zeros((self.im_height,self.im_width)), "power": self.led_power, "set": self.led_set}
        self.Acq_image = np.zeros((self.im_height,self.im_width))

    def live_image(self):
        #get next image from stack
        image = self.stack[self.frame]
        #increase index of frame. If we reach the limit, we start again.
        # self.frame += 1
        self.frame %= self.stack.shape[0]
        #return image from the stack
        return image 

    def shutdown(self):
        pass

    def set_modulation(self,mask=''):
        pass

    def run_acquisition(self,rf=0.5):

        #fixed size of the image
        self.reduce_factor = rf

        #set flag as true
        self.running = True

        #start GUI
        self.window = tkinter.Tk()
        self.canvas = tkinter.Canvas(self.window, width = 2*int(self.im_width*self.reduce_factor), height = int(self.im_height*self.reduce_factor))
        self.canvas.pack()
        self.window.title("DEMO")

        #create empty images and references
        image = np.zeros((int(self.im_width*self.reduce_factor),int(self.im_height*self.reduce_factor)))
        im=Image.fromarray(image)
        self.canvasPhoto = ImageTk.PhotoImage(image=im,master=self.window)
        self.canvasImage = self.canvas.create_image(0, 0, image=self.canvasPhoto, anchor=tkinter.NW)
        self.canvasPhotoMask = ImageTk.PhotoImage(image=im,master=self.window)
        self.canvasMask = self.canvas.create_image(int(self.im_width*self.reduce_factor)+1, 0, image=self.canvasPhoto, anchor=tkinter.NW)

        #run DEMO GUI
        self.picNum = 0
        self.window.after(0, self.AcqLoop())
        self.window.mainloop()

        #set flag as False
        self.running = False

    def AcqLoop(self):

        #increase acquisition image
        self.picNum+=1

        #get image
        self.Acq_image = self.live_image()
        self.image_ready = True

        #get and set image for modulator
        while not self.modulator_ready:
            time.sleep(0.1)
        self.modulator_ready = False
        mod_image = self.modulator_dict["image"]

        #set LED power
        self.led_set = self.modulator_dict['set']
        self.led_power = self.modulator_dict['power']

        #get image to show in GUI
        image = 255*(self.Acq_image/np.max(self.Acq_image))
        im=Image.fromarray(image)
        im=im.resize((int(self.im_width*self.reduce_factor),int(self.im_height*self.reduce_factor)),Image.Resampling.LANCZOS)
        
        #get modulator mask to show in GUI
        mod_image = 255*(mod_image/np.max(mod_image))
        mod_image = Image.fromarray(mod_image)
        mod_image=mod_image.resize((int(self.im_width*self.reduce_factor),int(self.im_height*self.reduce_factor)),Image.Resampling.LANCZOS)
        
        #update image in GUI
        self.canvasPhoto = ImageTk.PhotoImage(image=im,master=self.window)
        self.canvasPhotoMask = ImageTk.PhotoImage(image=mod_image,master=self.window)
        self.canvas.itemconfig(self.canvasImage, image = self.canvasPhoto)
        self.canvas.itemconfig(self.canvasMask, image = self.canvasPhotoMask)

        #wait for timestep
        if self.picNum==self.n_time_points:
            pass
        else:
            self.window.after(self.time_interval_s*1000, self.AcqLoop)
#endregion