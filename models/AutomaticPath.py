#region Libraries
import configs.functions as functions                                                                        # Tailored-made functions
import numpy as np                                                                      # Pytho's backbone - math
from skimage import measure, filters                                                    # Image analysis library                                                                      # Time measurement
import csv                                                                              # save data in .csv format
from Controllers.Direction_controller import Direction_controller
from models.abstract_model import AbstractModel
import importlib
#endregion 

class MigrationCell():
    def __init__(self,path_type,path_param):
        '''
            XXX
        '''

        self.path_param = path_param

        self.maskReady = False

        self.mask_area = 0

        #loop control: variables to change parameters (e.g. LED intensity) after certain loops
        self.smart_loop_parameters = {
            "loop_number" : -1,                  #loop counter, starting from -1 so we increase it for the first step and start from 0.
            "start_id" : 0                      #starting id of the path
        }

        #controlled parameters - parameters related to the control of the cell
        self.controlled_parameters = {
            "direction": np.array([0,0]),       #vector pointing in the direction from centroid of cell to closest point of path
            "shape": 1.5,                       #angular width for the illuminated area
            "id": np.array([0,0]),              #position to take as a reference to select segmented cell
            "id_set": False,                    #flag to set new id of the cell 
            "path_set": False                   #flag to set new mod_image
        }

        #internal parameters for the model
        self.internal_parameters = {
            "centroid": np.array([0,0]),        #position of cell's centroid
            "segment_id": 1,                    #label/id of the segment to consider
            "path": self.create_path(path_type),         #defined path to follow 
            "pp_id": None,                      #path point ID - point of the path closest to the cell
            "L2": 0                             #distance to next point to decide if new step is needed
        }

    def create_path(self,path_type='circle'):
        '''
            Create the path to follow. This is called when creating an object from the class.
            There are 3 types of paths: circle, square, and line.
            To be done? path from file, center/radius/number of line.

            Parameters
            ----------
            :param path_type: (string, default 'square') choose between 'circle' for a circle, 'square' for a square, and 'line' for a line.

            Return
            ----------
            :return: (numpy array) array with indexes (both in x and y) of the points in the path.
        '''

        #check if we have input from user
        if self.path_param is not None:
            center = np.array(self.path_param[0])
            radius = self.path_param[1] 
            n_points = self.path_param[2]
            
        #draw a circular path
        if path_type=='circle' or (path_type is None):

            #define parameters for circle (center, radius, number of points, angles of circle)
            angles = np.arange(0,2*np.pi,2*np.pi/n_points)

            #debug to check if number of points match
            assert angles.shape[0] == n_points

            #create an array with 2 arrays corresponding to the two indexes for the circle, and calculate positions
            path = np.ones((n_points,2))
            path[:,0] = center[0] + radius * np.cos(angles)
            path[:,1] = center[1] + radius * np.sin(angles)

            #return array of indexes for the circle
            return path
        
        #draw a square path
        elif path_type=='square':

            #create an array with 2 arrays corresponding to the two indexes for the square
            path = np.ones((n_points,2))

            #get an array with 4 elements, each of them containing an array of 2 elements corresponding to the indexes of the vertices of the square
            vertices = np.tile(center,(4,1)) + np.array([[radius,radius],[-radius,radius],[-radius,-radius],[radius,-radius]])

            #iterate the vertices and the side of the square
            for i in range(4):
                for j in range(int(n_points/4)):

                    #get index of the path to update
                    index = i*int(n_points/4) + j

                    #update value of the path based on which side it is iterating (from vertice to vertice)
                    path[index] = vertices[i] + 4 * j * (vertices[((i+1) % 4)] - vertices[i]) / n_points

            #return array of indexes for the square
            return path 
        
        #draw a triangle path
        elif path_type=='triangle':
            angles = [np.pi/3,np.pi,5*np.pi/3]
            x0 = center[0]
            y0 = center[1] + radius
            n = n_points//3
            L = 2*radius/np.sin(angles[0])
            path = np.ones((n*3,2))
            ii = 0
            for i in range(3):
                for j in range(n):
                    print(ii)
                    path[ii,0] = x0+(j*L/n)*np.cos(angles[i])
                    path[ii,1] = y0-(j*L/n)*np.sin(angles[i])
                    ii+=1
                x0 = x0+(L)*np.cos(angles[i])
                y0 = y0-(L)*np.sin(angles[i])

        #draw a line path
        elif path_type=='line':

            #define parameters for line (start, edges, number of points, position in Y-axis)
            startx = 650
            leftx = 250
            rightx = 1050
            numPoints = 5
            yVal = 515
            
            #create the line coordinates
            intervalx = (startx-leftx)/(numPoints-1)
            x1 = np.arange(startx,leftx-intervalx,-intervalx)
            x2 = np.arange(startx, rightx+intervalx,intervalx)
            x = np.concatenate((x1,x2), axis=0)
            y = np.repeat(yVal,numPoints*2)

            #restructure to format for interface
            path = np.vstack((x,y)).T  

            #return array of indexes for the line
            return path
        
        #draw a line path
        elif path_type=='hline':

            #define parameters for line (start, edges, number of points, position in Y-axis)
            x0 = center[0]
            y0 = center[1] - radius

            #get arrays
            y = np.linspace(y0,y0+2*radius,n_points//2)
            x = np.ones(2*(n_points//2))*x0
            y = np.append(y,np.flipud(y))

            #restructure to format for interface
            path = np.vstack((x,y)).T  

            #return array of indexes for the line
            return path
        
        #draw a line path
        elif path_type=='vline':

            #define parameters for line (start, edges, number of points, position in Y-axis)
            x0 = center[0] - radius
            y0 = center[1]

            #get arrays
            x = np.linspace(x0,x0+2*radius,n_points//2)
            y = np.ones(2*(n_points//2))*y0
            x = np.append(x,np.flipud(x))

            #restructure to format for interface
            path = np.vstack((x,y)).T  

            #return array of indexes for the line
            return path

        #if the path_type is not available, try reading from file or restart with circle
        else:

            #try reading file
            try: 
                path = []
                with open(path_type, 'r', newline='') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                    for row in csvreader:
                        tmp = [int(float(item)) for item in row]
                        path.append([tmp[1],tmp[0]])
                path = np.array(path)

                #center
                x = min(path[:,0])
                y = min(path[:,1])
                w = max(path[:,0])-min(path[:,0])
                h = max(path[:,1])-min(path[:,1])

                #scale by width
                scale = 2*radius/(w)
                x0 = center[0]-radius
                y0 = center[1]-(h*scale/2)

                #path scaled
                for i in range(len(path)):
                    path[i,0] = x0+(path[i,0]-x)*scale
                    path[i,1] = y0+(path[i,1]-y)*scale
                return path
                
            #continue with circle
            except:
                print('Path type not recognize, changing to circle instead. Select between \'circle\' and \'square\' or type file name.')
                return self.create_path('circle')

class AutomaticPath(AbstractModel):
    def __init__(self,im_height,im_width,functionalities=None):
        '''
            Model that automatically illuminates the cell edge in a set direction. 
            This control is based solely on the closest path from center of cell to follow the defined path. 

            Parameters
            ----------
            :param im_height: (positive int) height of the processed image
            :param im_width: (positive int) width of the processed image
            :param functionalities: (dictonary) extra functions for every model. For now: list of loops and their LED power and initial LED power
        '''

        #create variables for processed image (empty array image & modulator image, flag for area_check)
        self.processed_image = np.zeros((im_height,im_width))
        self.processed_image_list = [np.zeros((im_height,im_width))]
        self.mod_image = np.zeros((684,608))
        self.file = None

        #get segmentation class
        models_module = importlib.import_module('segmentation.{}'.format(functionalities['segmentation']))
        class_ = getattr(models_module, functionalities['segmentation'])
        self.model = class_()

        #Assign valaues from functionalities. Assign default of no input
        if functionalities is not None:
            self.led_power = functionalities['init_power']
            power_array = functionalities['power_array']
            path_type = functionalities['path_type']
            self.path_param = functionalities['path_pos']
            self.cellSize = functionalities['cell_size']
            if not power_array:
                power_array = [[np.inf,0,0]]
            if not self.cellSize:
                self.cellSize = [250,250]
            self.wh = self.cellSize[0]//2
            self.hh = self.cellSize[1]//2
        else:
            self.led_power = 1
            power_array = [[np.inf,0,0]]
            self.path_param = None
            path_type = []
            self.wh, self.hh = 250,250

        #if type is a single word and not a list, make it a list
        if type(path_type)==type(''):
            path_type = [path_type]

        #parameters for control - commands relating the GUI to the control model!
        self.parameter_controls = {
            "double_left": None,                #position of double left click - position of new selected cell
            "double_left_set": "path_set",      #double left click boolean - set a new dmd image
            "textbox": "shape",                 #ill-shape text input: change width of illuminated area
            "right": "id",                      #position of right click - position of initial box
            "right_set": "id_set"               #riht click boolean - flag to change selected cell
        }

        
        #controlled parameters - parameters related to the control of the cell
        self.controlled_parameters = {
            "shape": 1.5,                       #angular width for the illuminated area
            "id": np.array([0,0]),              #position to take as a reference to select segmented cell
            "id_set": False,                    #flag to set new id of the cell 
            "path_set": False                   #flag to set new mod_image
        }

        #create cell objects same number as paths
        self.cells = []
        for i in range(len(path_type)):
            cell = MigrationCell(path_type[i],self.path_param[i])
            self.cells.append(cell)

        #loop control: variables to change parameters (e.g. LED intensity) after certain loops
        self.smart_loop_parameters = {
            "loopsNpower" : power_array,        #array of tuples to change LED power after certain loops. First element of tuple is loops, second is power. Must be in increasing order of loops
        }

        #define cell to be selected!
        self.num_cells = len(path_type)
        self.current_cell = 0

        #array of boxes for SAM
        self.setup_array = []

        self.controller = Direction_controller()

        
        #drop menu GUI list
        self.menu_list = [
            "Unassigned",
            "Begin controller",
            "Unassigned",
            "Select region"
        ]
        
        #print click order:
        print('---MIGRATION EXPERIMENTS---')
        print('TO SEGMENT: first right click the cell. once the segmentation is correct, double left click the cell.')

    def process_step(self, cam_image):
        '''
            Image processing of the camera image.
            No return value, just changes the internal variable processed_image. 
            This segments the image into the selected cell and saves the centroid's position and label.

            Parameters
            ----------
            :param cam_image: (2D array uint8) image collected from camera
        '''

        #segment cells in image
        if self.model.getRegions().shape[0] > 0:
            x = self.model.getRegions()[0]
            self.model.setImage(image=cam_image[np.newaxis])
            self.model.run()
            if not self.cells[self.current_cell].maskReady: #XXX sure?
                self.model.updateRegions(square=False)
            new_segmentation = self.model.getResult() # returns an N x h x w binary segmentation. XXX remember np.int32()
        else:
            new_segmentation = [np.zeros_like(cam_image)]

        #update processed_image
        self.processed_image_list = new_segmentation
    
    def controller_step(self):
        '''
            This function wraps all the controll processing and in the end sets the new mod_image. 
            It gets the point of the path closest to the centroid and calculate the new modulator image based on this and the shape.
            XX controller_step and create_directed_pattern are too mixed. For next version they have to be properly separated.
        '''

        #for current segmenting cell
        #if flag of path_set is on, get the (new) point in the path closest to the centroid of the cell
        if self.controlled_parameters["path_set"]:
            
            #If right click already pressed, get coordinate of cell, get centroid, and continue
            if self.cells[self.current_cell].maskReady:
                self.cells[self.current_cell].internal_parameters["centroid"] = self.controlled_parameters["id"]
                self.cells[self.current_cell].maskReady = False
                dump = self.create_directed_pattern(self.controlled_parameters["shape"],self.current_cell)
                self.cells[self.current_cell].internal_parameters["pp_id"] = None
                print('[CELL {0}] Setting segmented cell...'.format(self.current_cell))
            
        #should we consider all or only until current cell?
        if len(self.setup_array)==self.num_cells:
            cells_ready = self.num_cells
        else:
            cells_ready = self.current_cell+1

        #iterating for every cell
        cell_images = []
        for n_cell in range(cells_ready):
            if self.controlled_parameters["path_set"]:
            
                #get number of points of the set path
                n_points = self.cells[n_cell].internal_parameters["path"].shape[0]

                #if there is no id for the path, create one
                if self.cells[n_cell].internal_parameters["pp_id"] is None:

                    #set initial "infinite" distance
                    distance = 999999
                    self.cells[n_cell].internal_parameters["pp_id"] = 1

                    #iterate every point of the path
                    for i in range(n_points):

                        #calculate (eucledian) distance from centroid of cell to the path
                        temp = self.cells[n_cell].internal_parameters["path"][i] - self.cells[n_cell].internal_parameters["centroid"] 
                        temp = np.dot(temp, temp)

                        #get the index of the path point that has the minimum distance from centroid to path and set as pp_id
                        if temp < distance:
                            distance = temp
                            self.cells[n_cell].internal_parameters["pp_id"] = i

                    #print current point path id and save 
                    print("[CELL {0}] start id:".format(self.current_cell), self.cells[n_cell].internal_parameters["pp_id"])
                    self.cells[n_cell].smart_loop_parameters["start_id"] = self.cells[n_cell].internal_parameters["pp_id"]
                
                    #check number of loops to change LED power
                    self.smart_loop(n_cell)

                #get next point in path (circular) and calculate distance to centroid
                check_id = (self.cells[n_cell].internal_parameters["pp_id"] + 1) % n_points
                temp = self.cells[n_cell].internal_parameters["path"][check_id] - self.cells[n_cell].internal_parameters["centroid"] 
                temp = np.dot(temp, temp)

                #if next point in path has a smaller distance than L2, assign that point instead and print information
                if temp < self.cells[n_cell].internal_parameters["L2"]: 

                    #assign point
                    self.cells[n_cell].internal_parameters["pp_id"] = check_id 

                    #check number of loops to change LED power
                    self.smart_loop(n_cell)

                    #print information
                    print("[CELL {0}] new direction id:".format(self.current_cell), check_id, "L2=", self.cells[n_cell].internal_parameters["L2"])
                    print("[CELL {0}] centroid:".format(self.current_cell), self.cells[n_cell].internal_parameters["centroid"], "direction:", self.cells[n_cell].internal_parameters["path"][self.cells[n_cell].internal_parameters["pp_id"]])

                #assign the value of the path at index pp_id to the direction
                self.cells[n_cell].controlled_parameters["direction"] = self.cells[n_cell].internal_parameters["path"][self.cells[n_cell].internal_parameters["pp_id"]]

            #get new modulator image based on the path and the shape. if path_set is off, the image is empty
            cell_image = self.controlled_parameters["path_set"]*self.create_directed_pattern(self.controlled_parameters["shape"],n_cell)

            #sum mask area in pixels #XXX this is wrong, it should be processed_image_list!
            self.cells[n_cell].mask_area = self.processed_image_list[n_cell].flatten().sum()

            #assign to list
            cell_images.append(cell_image)

        #get mask in layers XXX
        processed_images = np.int32(self.processed_image_list[0])!=0
        for image in self.processed_image_list[1:]:
            processed_images += np.int32(image)!=0
        self.processed_image = processed_images*3

        #assign total mod_image
        self.mod_image = 0
        for cell in range(len(cell_images)):

            #check traffic light!
            if len(cell_images)>1:
                greenLight = self.traffic_light(cell,self.wh*1.5,0.8)
                if greenLight:
                    self.mod_image += cell_images[cell]
                else:
                    projection = self.cells[cell].internal_parameters["centroid"] - self.cells[cell].controlled_parameters["direction"]
                    stopCell = self.controlled_parameters["path_set"]*functions.cell_edge(np.int32(self.processed_image_list[cell]),25)*functions.sector_mask(np.int32(self.processed_image_list[cell]).shape,self.cells[cell].internal_parameters["centroid"], projection, 1)*(np.int32(self.processed_image_list[cell])==1)
                    self.mod_image += stopCell
            else:
                self.mod_image += cell_images[cell]

        ##add to processed_image
        self.processed_image += self.mod_image*7
    
    def create_directed_pattern(self, width, cell_num):
        '''
            Create a modulator image/pattern based on the position of the center of the cell and the closest point of the path.
            The image is a mask overlapping the area of the cell, the edge of the cell, and the sector of a circle pointing to the defined direction centered at the cell.

            Parameters
            ----------
            :param width: (positive float) angular width of the sector of the cirle

            Return
            ----------
            :return: (2D uint8 matrix) modulator image based on the direction
        '''

        #for the current segmenting cell
        #if the id_set flag is on, update segment_id based on the processed image, taking the coordinates from the controlled_parameters id, and set flag id_set off
        if self.controlled_parameters["id_set"]:

            #move to next cell!
            if not (self.current_cell==0 and len(self.setup_array)==0) and not self.cells[self.current_cell].maskReady:
                self.current_cell = (self.current_cell+1)%self.num_cells

            self.cells[self.current_cell].maskReady = True
            self.controlled_parameters["id_set"] = False
            self.controlled_parameters["path_set"] = False
            id_x, id_y= self.controlled_parameters["id"]

            if len(self.setup_array)>=self.num_cells:
                self.setup_array[self.current_cell] = [id_y-self.hh, id_x-self.wh, id_y+self.hh, id_x+self.wh]
            else:
                self.setup_array.append([id_y-self.hh, id_x-self.wh, id_y+self.hh, id_x+self.wh])

            self.model.setImage(np.int32(self.processed_image_list[0])[np.newaxis]) #XXX process image will be a list?
            self.model.setup(np.array(self.setup_array))             # box for initial frame
            self.cells[self.current_cell].internal_parameters["centroid"] = np.array(self.controlled_parameters["id"])
            print('[CELL {0}] Setting box for SAM... wait for segmentation and then left click.'.format(self.current_cell))
        
        #controller step
        self.controller.step(self.processed_image_list[cell_num],
                             (self.controlled_parameters["id_set"] and  not self.cells[cell_num].maskReady),
                             self.cells[cell_num],
                             width)
        self.cells[cell_num].internal_parameters["L2"] = self.controller.L2
        self.cells[cell_num].internal_parameters["centroid"] = self.controller.centroid

        #return new mask from controller
        return self.controller.area

    def set_parameters(self, queue_dict):
        '''
            This function takes a dictionary containing the changed parameter values.
            It assigns those to the entries in controlled parameters with the same key.
            This function is called from the main code to change parameters of the model.

            Parameters
            ----------
            :param queue_dict: (dictionary) parameetr values to change
        '''

        #iterate queue and try to change the parameter.
        for control in queue_dict:
            try:

                #get parameter to change from parameter_controls. if it exists, change the controlled_parameters' parameter and print
                #parameter_controls has as value the keys of the controlled_parameters dictionary!
                parameter = self.parameter_controls[control]
                if parameter is not None:
                    self.controlled_parameters[parameter] = queue_dict[control]
                    print(control,"event set", parameter, "to", self.controlled_parameters[parameter])
            except:
                pass

    def write_data(self, time='', date='',fpath=''):
        '''
            Function to export data to file. 
            In this model, we do not export data every frame, so this is an empty function.
        '''

        #save file name (using legacy self.file)
        if self.file is None:
            file = []
            for i in range(len(self.cells)):
                f = '\Acq_{0}_{1}-{2}_cell{3}.csv'.format(str(date),time[11:13],time[14:16],i)
                file.append(f)
                #Create header
                with open(fpath+f, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow(['Step','Loop','LED', 'Centroid_X', 'Centroid_Y','mask_area'])
            self.file = file
        
        for i,cell in enumerate(self.cells):
            #get data in one list (ratio, power, setpoint, and 3 coeff)
            data = [cell.internal_parameters["pp_id"], 
                    cell.smart_loop_parameters["loop_number"],
                    self.led_power,
                    cell.internal_parameters["centroid"][1], #inverted
                    cell.internal_parameters["centroid"][0],
                    cell.mask_area
                    ]

            #save in csv
            with open(fpath+self.file[i], 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(data)

    def create_path(self):
        path = []
        for cell in self.cells:
            for coo in cell.internal_parameters["path"]:
                path.append(coo)
        return path
        # return cell.internal_parameters["path"]

    def smart_loop(self,n_cell):
        '''
            Function to check control LED power for every loop. 
            This can be extended to any functionality for every loop (e.g., start new circuit, go backwards)
        '''

        #increase loop number - we start from 0!
        if self.cells[n_cell].internal_parameters["pp_id"] == self.cells[n_cell].smart_loop_parameters["start_id"]:
            self.cells[n_cell].smart_loop_parameters["loop_number"] +=1
            print('[CELL {0}] Starting loop {1}'.format(self.current_cell,self.cells[n_cell].smart_loop_parameters["loop_number"]))

        #temps for loopsNpower iteration (power, loop, index)
        ln = self.cells[n_cell].smart_loop_parameters["loop_number"]
        li = self.cells[n_cell].internal_parameters["pp_id"] - self.cells[n_cell].smart_loop_parameters["start_id"]
        li %=len(self.cells[n_cell].internal_parameters["path"])
        print('[CELL {0}] loop {1} step {2}'.format(self.current_cell,ln,li))

        #get LED power from input
        power = []
        for item in self.smart_loop_parameters['loopsNpower']:

            #if the loop number ln and loop index li exist, append
            if (item[0]==ln) and (item[1]==li):
                power.append(item[2])
        
        #assign power - only if it is in the list
        if power and self.num_cells==1:
            self.led_power = power[0]
            print('LED power for loop {0} set to {1}'.format(self.cells[n_cell].smart_loop_parameters["loop_number"],power[0]))

    def traffic_light(self,cell_num=1,dist=100, width=1):

        if cell_num<len(self.processed_image_list):
            #get all masks except of current cell
            others = 0
            for i,image in enumerate(self.processed_image_list):
                if i!=cell_num:
                    others += np.int32(image)!=0

            #calculate cone from center of current cell XXX pointing 5 point ahead!
            if self.cells[cell_num].internal_parameters["pp_id"] is not None:
                direction = self.cells[cell_num].internal_parameters["path"][(self.cells[cell_num].internal_parameters["pp_id"]+5)%self.cells[cell_num].internal_parameters["path"].shape[0]]
            else:
                direction = self.cells[cell_num].internal_parameters["path"][0]
            projection = direction - self.cells[cell_num].internal_parameters["centroid"]
            view = functions.view_mask(np.int32(others).shape,self.cells[cell_num].internal_parameters["centroid"], projection, width, dist)

            #add view to processed_image
            self.processed_image+=view*11

            #return greenlight
            if sum((view*others).flatten())>1:
                # print('collision')
                return False
            else:
                # print('Free')
                return True
        else:
            # print('Lonely cell')
            return True
