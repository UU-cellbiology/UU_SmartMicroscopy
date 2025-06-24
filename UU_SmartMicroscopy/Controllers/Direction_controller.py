import numpy as np
from skimage import measure
from ..configs import functions

class Direction_controller:
    def __init__(self):
        pass

    def step(self,image,getCentr,cell,width,reverse=0):

        #for this specific cell from input
        if getCentr:
            #calculate L2 based on the area of the segmented cell
            self.L2 = (0.75/np.pi)*np.sum((image==1))

            #get centroid of cell based on the coordinates of its (1st) moment
            m = measure.moments(image)

            if m[0,0] !=0:
                self.centroid = np.array([int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])])
            else:
                self.centroid = np.array([0,0])
        else:
            self.L2 = cell.internal_parameters["L2"]
            self.centroid = cell.internal_parameters["centroid"]

        #get direction from centroid to the closest point of the path
        if reverse:
            projection = self.centroid - cell.controlled_parameters["direction"]
        else:
            projection = cell.controlled_parameters["direction"] - self.centroid

        #return modulator image as an overlap between the edge of the cell, a sector of a circle from the centroid to the defined direction, and the area of the segmented cell
        self.area =  functions.cell_edge(image)*functions.sector_mask(image.shape,self.centroid, projection, width)*(np.int32(image)==1)

    def get_dir(self):
        return self.area