""" This module implements threshold segmentation and naive tracking."""
import warnings
import numpy as np
from segmentation.abstract_detector import Abstract_Detector
import scipy 
from skimage import filters, feature, measure, segmentation

class Threshold(Abstract_Detector):
    """
    XXX
    """

    def __init__(self):
        """
        XXX
        """
            
        # initialize attributes
        self.regions = np.zeros([0, 0])                  # regions used for image segmentation
        self.image   = np.zeros([0,0,0])               # image array is stored in detector
        self.result  = np.zeros([0,0,0])                 # segmentation mask b x h x w
        self.imageCrop = np.zeros([0,0,0])
        self.isList = False
        
    def setImage(self, image: np.ndarray):
        """
        XXX
        """
        
        self.image = image[0] # class stores original input image

    def setup(self, regions: np.ndarray):
        """Specify regions for segmentation."""

        if len(self.regions.shape)==1:
            self.regions = [regions[1],regions[0],regions[3],regions[2]]
            self.regions = [self.regions]
        else:
            self.regions = []
            for reg in regions:
                self.regions.append([reg[1],reg[0],reg[3],reg[2]])
        self.regions = np.array(self.regions)
    def run(self):
        """
        XXX 
        """
        # check if image is specified
        if self.image.shape[0] == 0:
            warnings.warn('No image specified. Returning empty mask.')
            self.result = np.zeros([0,0])
            return

        self.imageCrop = []
        for reg in self.regions:
            self.imageCrop.append(self.image[reg[0]:reg[2],reg[1]:reg[3]])

        self.result = []
        for image in self.imageCrop:
            #add gaussian filter to data
            image = filters.gaussian(image, sigma=4)
            #substract background and use cuadratic scale to highlight differences
            image -= image.mean()

            #separate image in 2 classes using the Multi-Otsu threshold method
            thresholds = filters.threshold_multiotsu(image, classes=2)

            #isolate cells as the pixels with higher intensity than the threshold
            cells = image > thresholds[0]

            #get distance from each non-zero pixel (i.e., cell pixels) to the closest zero pixel (i.e., background)
            distance = scipy.ndimage.distance_transform_edt(cells)

            #get coordinates of local maxima OF DISTANCE MATRIX with a minimum separation of 0.4x the size of the image. 
            #this gives the center of each cell
            local_max_coords = feature.peak_local_max(distance, min_distance=int(0.4*image.shape[0]), exclude_border=False)

            #create new boolean matrix with only the center of each cell as True
            local_max_mask = np.zeros(distance.shape, dtype=bool)
            local_max_mask[tuple(local_max_coords.T)] = True

            #join local maximas in areas or "labels" for watershed
            markers = measure.label(local_max_mask)

            #use the watershed algorithm to separate the cells in areas. 
            #the previusly calculated markers are used to assign labels and the mask ennsure that only the True pixels (higher than threshold) are assigned to areas.
            segmented_cells = segmentation.watershed(-distance, markers, mask=cells)

            #return segmented cells as mask
            self.result.append(segmented_cells)
        self.result = np.array(self.result)
    
    def getImage(self) -> np.ndarray:
        """Return the (reshaped) image as numpy array.
        
        :return: The image as numpy array.
        """
        if self.image.shape[0] == 0:
            warnings.warn('No image found. Returning empty image.')
        return self.image
    
    def getResult(self) -> np.ndarray:
        """Retrieves the segmentation mask as numpy array.

        :return: The segmentation mask as numpy array.
        """
        if self.result.shape[0] == 0:
            warnings.warn('No masks found. Returning empty mask.')

        res = []
        for i in range(len(self.result)):
            empty = np.zeros(self.image.shape)
            empty[self.regions[i][0]:self.regions[i][2],self.regions[i][1]:self.regions[i][3]] = self.result[i]
            res.append(empty)
        return res
    
    def getRegions(self) -> np.ndarray:
        """Retrieves the regions used as input to SAM.
        
        :return: Regions as numpy array.
        """
        if self.regions.shape[0] == 0:
            warnings.warn('No regions found. Returning empty mask.')
        
        return self.regions
      
    def updateRegions(self, square: bool = True):
        """
        XXX
        """

        regions = []
        for i in range(len(self.imageCrop)):
            xArr = np.sum(self.imageCrop[i],0)
            yArr = np.sum(self.imageCrop[i],1)

            xc = np.argmax(xArr)
            yc = np.argmax(yArr)

            regions.append([self.regions[i][1]+xc-self.imageCrop[i].shape[0]//2,
                           self.regions[i][0]+yc-self.imageCrop[i].shape[1]//2,
                           self.regions[i][3]+xc-self.imageCrop[i].shape[0]//2,
                           self.regions[i][2]+yc-self.imageCrop[i].shape[1]//2])
        self.setup(np.array(regions))

if __name__ == "__main__":
    segm  = Threshold()
    test_input = np.float32(np.random.rand(3, 612, 512))
    segm.setImage(test_input)
    
    segm.setup(np.array([[100, 200, 300, 400]]))
    segm.run()
    mask = segm.getResult()

    