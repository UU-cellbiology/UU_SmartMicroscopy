import tkinter 
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile
import numpy as np
from PIL import Image, ImageTk
import configs.functions as functions
import configs.globVars as gv
from Interface.abstract_interface import AbstractInterface

class mainWindow(AbstractInterface):
    def __init__(self, parent,inputs,ih,iw):

        #calibration from gv
        calibration = gv.Calibration()

        #microscope sizes
        self.iw = iw
        self.ih = ih

        #define inputs variable
        self.inputs = inputs

        #create GUI frame with a canvas
        self.frame = tkinter.Frame(parent, height=self.inputs['max_height'], width=self.inputs['max_height'])
        self.frame.grid(row = 0, column = 0)#.pack()
        self.canvas = tkinter.Canvas(self.frame, height=self.inputs['max_height'], width=self.inputs['max_height'])
        self.canvas.place(x=-1,y=-1)
        self.frame_controls = tkinter.Frame(parent, width=120)#, bg="yellow")  
        self.frame_controls.grid(row = 0, column = 1)#.pack()
        self.frame_info = tkinter.Frame(parent, width=self.inputs['max_height'], height=120)#, bg="yellow")  
        self.frame_info.grid(row = 1, column = 0)#pack()

        #create empty variables for camera and modulator image in various formats
        self.pixels = np.zeros((self.ih,self.iw))
        self.processed_image = np.zeros((self.ih,self.iw))
        self.mod_pattern = np.zeros((self.ih,self.iw))
        self.display_it = 0

        #create and resize starting image
        #NOTE: this may deform the image. 
        self.im=Image.fromarray(self.pixels)
        self.reduce_factor = self.inputs['max_height']/max(self.ih,self.iw)
        self.im=self.im.resize((int(self.iw*self.reduce_factor),int(self.ih*self.reduce_factor)),Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=self.im)

        #show image at GUI
        self.display_image = self.canvas.create_image(0, 0, image = self.photo, anchor=tkinter.NW)

        self.button_width = 18

        #create buttons, i.e., set & get calibrate button

        self.button_show_path = tkinter.Button(self.frame_controls, width = self.button_width, text="Show Path", command=self.show_path)

        self.button_show_camera = tkinter.Button(self.frame_controls, bg='yellow', text="Camera", width = self.button_width, command = self.display_switch_camera)
        self.button_show_illumination = tkinter.Button(self.frame_controls, bg='cyan', text="Illumination", width = self.button_width, command = self.display_switch_illumination)        
        self.button_show_segmentation = tkinter.Button(self.frame_controls, bg='blue', fg='white',text="Segmentation", width = self.button_width, command = self.display_switch_segmentation)
        self.button_show_RGB = tkinter.Button(self.frame_controls, bg='orange', fg='white',text="RGB", width = self.button_width, command = self.display_switch_RGB)
        self.showAsRGB = False

        self.set_calibrate_button = tkinter.Button(self.frame_controls, text="Set Calibration Image",width = self.button_width, command=calibration.set_calibration_image)
        self.get_calibrate_button = tkinter.Button(self.frame_controls,  text="Get Calibration",width = self.button_width, command=calibration.get_calibration_points)

        #create button to acquire
        self.acq_button = tkinter.Button(self.frame_controls, text="Acquire", bg='green', fg='white',  width = self.button_width, command=self.acquire)
        self.abort_button = tkinter.Button(self.frame_controls, text="Abort", bg='red', fg='white', width = self.button_width, command=self.abort)

        # labels 
        temp_lab_rois = tkinter.Label(self.frame_controls, text="Selections:", justify="left", fg="blue")
        temp_lab_calib = tkinter.Label(self.frame_controls, text="Calibration:", justify="left",  fg="blue")
        temp_lab_view = tkinter.Label(self.frame_controls, text="Views:", justify="left",  fg="blue")
        temp_lab_runs = tkinter.Label(self.frame_controls, text="Functions:", justify="left",  fg="blue")

        temp_label_status_text = tkinter.Label(self.frame_info, text="Status:", justify="left",  fg="blue")
        temp_label_folder_name = tkinter.Label(self.frame_info, text="Folder name:", justify="left",  fg="blue")
        temp_label_file_name = tkinter.Label(self.frame_info, text="File name:", justify="left",  fg="blue")

        self.label_status = tkinter.Label(self.frame_info, text="Ready!", justify="left",  fg="black")

        # positioning
        temp_lab_rois.grid                    (row = 0, column = 0, pady = 1, padx = 1)
        self.button_show_path.grid            (row = 2, column = 0, pady = 1, padx = 1)
        
        temp_lab_calib.grid                   (row = 4, column = 0, pady = 5, padx = 1)
        self.set_calibrate_button.grid        (row = 5, column = 0, pady = 1, padx = 1)
        self.get_calibrate_button.grid        (row = 6, column = 0, pady = 1, padx = 1)
        
        temp_lab_view.grid                    (row = 8, column = 0, pady = 5, padx = 1)
        self.button_show_camera.grid          (row = 9, column = 0, pady = 1, padx = 1)
        self.button_show_illumination.grid    (row = 10, column = 0, pady = 1, padx = 1)
        self.button_show_segmentation.grid    (row = 11, column = 0, pady = 1, padx = 1)
        self.button_show_RGB.grid             (row = 12, column = 0, pady = 1, padx = 1)
        
        temp_lab_runs.grid                    (row = 13, column = 0, pady = 5, padx = 1)
        self.acq_button.grid                  (row = 14, column = 0, pady = 1, padx = 1)#place(x=self.inputs['max_height']-200, y= canvas_height+30)
        self.abort_button.grid                (row = 15, column = 0, pady = 1, padx = 1)

        self.acq_path_box = tkinter.Text(self.frame_info, height=1, width=60)
        self.acq_path_box.insert(tkinter.END, self.inputs['folder_name'])
        
        self.acq_name_box = tkinter.Text(self.frame_info, height=1, width=60)
        self.acq_name_box.insert(tkinter.END, self.inputs['file_name'])
        
        temp_label_status_text.grid(sticky="E", row = 0, column = 0, pady = 1, padx = 2)
        self.label_status.grid(sticky="W", row = 0, column = 1, pady = 1, padx = 1)
        temp_label_folder_name.grid(sticky="E", row = 1, column = 0, pady = 1, padx = 2)
        self.acq_path_box.grid(row = 1, column = 1, pady = 1, padx = 2)#place(x= 200, y =canvas_height+30)
        temp_label_file_name.grid(sticky="E", row = 2, column = 0, pady = 1, padx = 2)
        self.acq_name_box.grid(row = 2, column = 1, pady = 1, padx = 2)#place(x= 200, y =canvas_height+60)

        # popup menu 
        self.popup = tkinter.Menu(self.canvas, tearoff = 0)

        #Adding Menu Items
        self.popup.add_command(label=gv.menuList[0], command = self.mouse_left)
        self.popup.add_command(label=gv.menuList[1], command = self.mouse_double_left)
        self.popup.add_command(label=gv.menuList[2], command = self.mouse_left_release)
        self.popup.add_command(label=gv.menuList[3], command = self.mouse_right)


        #create test circular image 
        self.test = Image.fromarray((255*functions.circle(self.iw,self.inputs['max_height'],[int(self.iw/2),int(self.inputs['max_height']/2)],100)).astype("uint8"))

        self.tag = self.canvas.create_text(10, 10, text="", anchor="nw", font=("Courier", 12), fill='yellow') 

        #relate canvas to functions depending of type of click
        self.canvas.bind("<Button 3>", self.right)

        self.canvas.bind("<Button 2>", self.display_switch)

        # roi setup
        self.start_roi_x = None
        self.start_roi_y = None

        self.roi_rect = None
        self.poly_path = None
        self.poly_path_array = None

        #refresh canvas
        self.refresh()

    def display_switch_RGB(self):
        '''
            Show all 3 images (camera, illumination and segmentation) as one RGB 
        '''
        self.showAsRGB = True

    def abort(self):
        '''
            Abort execution of Control and Model Threads
        '''

        gv.main_event.set()

    def show_path(self):
        '''
            Shows and hides the path which is loaded/used by the model
        '''

        if gv.hasPath:
            if self.poly_path is None:
                path = gv.modelPath
                self.poly_path_array = []
                for i in range(len(path)):
                    self.poly_path_array.append(path[i][1] * self.reduce_factor)
                    self.poly_path_array.append(path[i][0] * self.reduce_factor)

                self.poly_path = self.canvas.create_line(self.poly_path_array, width=2, fill='lightgreen')
                self.button_show_path.config(text ='Hide Path')  
            else:
                self.canvas.delete(self.poly_path)
                self.poly_path = None
                self.poly_path_array = []
                self.button_show_path.config(text ='Show Path')  

    def refresh(self):
        '''
            refresh canvas from GUI based on selected image.
        '''

        #if queue not empty, get raw and processed image
        if not gv.image_ui_queue.empty():
            self.pixels = gv.image_ui_queue.get()
            self.processed_image = gv.processed_ui_queue.get()

        #if queue not empty, get modulator pattern
        if not gv.mod_ui_queue.empty():
            self.mod_pattern = gv.mod_ui_queue.get()

        #select which image to display (image, modulator pattern, processed image) depending on the value of display_it
        if self.showAsRGB:
            if len(self.processed_image.shape) == 3:
                # flatten the multichannel image
                self.processed_image[:,:,0] *= 64
                self.processed_image[:,:,1] *= 128
                self.processed_image[:,:,2] *= 255
                self.processed_image = self.processed_image.max(axis = 2)

            self.im = Image.fromarray(np.concatenate((self.processed_image[:,:,np.newaxis], self.pixels[:,:,np.newaxis], self.mod_pattern[:,:,np.newaxis]), axis = 2, dtype = np.uint8)).convert('RGB')
        else:
            if self.display_it==0: 
                self.im=Image.fromarray(self.pixels)
            elif self.display_it==1: 
                self.im=Image.fromarray(self.mod_pattern)
            else: 
                self.im=Image.fromarray(self.processed_image)

        #resize image to window. This reduces the image such that the largest dimension fix in the canvas window. 
        self.im=self.im.resize((int(self.iw*self.reduce_factor),int(self.ih*self.reduce_factor)),Image.Resampling.LANCZOS)

        #refresh canva with new image
        self.photo = ImageTk.PhotoImage(self.im)
        self.canvas.itemconfig(self.display_image, image=self.photo)
        self.canvas.after(int(self.inputs['dt']/2), self.refresh)

    def double_left(self, event):
        '''
            Save position of double click with left click into parameter queue 
            note: position corrected with resize factor of window
        '''

        gv.parameter_queue.put({"double_left": np.array([int(event.y/self.reduce_factor),int(event.x/self.reduce_factor)]), "double_left_set": True})
    
    def mouse_double_left(self):

        gv.parameter_queue.put({"double_left": np.array([int(self.start_roi_y/self.reduce_factor),int(self.start_roi_x/self.reduce_factor)]), "double_left_set": True})

    def right(self, event):        
        # display the popup menu
        self.start_roi_x = self.canvas.canvasx(event.x)
        self.start_roi_y = self.canvas.canvasx(event.y)

        try:
            self.popup.tk_popup(event.x_root, event.y_root, 0)
        finally:
            #Release the grab
            self.popup.grab_release()

    def left(self, event):
        '''
            Save position of left click into parameter queue 
            note: position corrected with resize factor of window
        '''
        gv.parameter_queue.put({"left":[int(event.y/self.reduce_factor),int(event.x/self.reduce_factor)], "left_set": True})
    
    def mouse_left(self):

        gv.parameter_queue.put({"left":[int(self.start_roi_y / self.reduce_factor), int(self.start_roi_x / self.reduce_factor)], "left_set": True})
        
    def mouse_right(self):

        gv.parameter_queue.put({"right":[int(self.start_roi_y / self.reduce_factor), int(self.start_roi_x / self.reduce_factor)], "right_set": True})

    def left_release(self, event):
        '''
            Save position of release left click into parameter queue 
            note: position corrected with resize factor of window
        '''

        gv.parameter_queue.put({"left_release":[int(event.y/self.reduce_factor),int(event.x/self.reduce_factor)], "left_release_set": True})
    
    def mouse_left_release(self):

        gv.parameter_queue.put({"left_release":[int(self.start_roi_y / self.reduce_factor),int(self.start_roi_x  /self.reduce_factor)], "left_release_set": True})

    def display_switch(self, event):
        '''
            Wheel click to change what to display [camera image, illumination pattern, segmented cell] 
        '''
        self.display_it = (self.display_it + 1) % 3

    def display_switch_camera(self):
        '''
           switch to the previous view [camera image, illumination pattern, segmented cell] 
        '''
        self.display_it = 0
        self.showAsRGB = False

    def display_switch_segmentation(self):
        '''
           switch to the previous view [camera image, illumination pattern, segmented cell] 
        '''
        self.display_it = 2
        self.showAsRGB = False
    
    def display_switch_illumination(self):
        '''
           switch to the previous view [camera image, illumination pattern, segmented cell] 
        '''
        self.display_it = 1
        self.showAsRGB = False

    def acquire(self):
        '''
            get acquisition info (parameters, path, name) and add it to the queue. 
            Get flag of acq_event true.
        '''

        gv.acq_event.set()
#endregion