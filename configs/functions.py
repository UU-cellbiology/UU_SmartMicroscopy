#region Prelude
#region Description
'''
Functions file for feedback loop control in PEX microscope. 
Utility functions, mostly for image processing.

Language:       Python
Author(s):      Jakob Schröder, Josiah Passmore, and Alfredo Rates
Creation date:  2023-06-19
Contact:        a.ratessoriano@uu.nl
Git:            https://github.com/passm003/PEXscope
Version:        2.0
'''
#endregion

#region Libraries
import numpy as np                                                                      # Pytho's backbone - math
from skimage import filters, feature, measure, segmentation, io                         # Image analysis library 
import scipy                                                                            # Scientific library for python
#endregion
#endregion

#region functions
def gaussian_2d(width, height, mu, sigma = 1.0):
    '''
        Creates a 2D gaussian pattern in a meshgrid, normalized to 255 with format uint8.

        Parameters
        ----------
        :param width: (positive int) number of pixels in width dimension
        :param height: (positive int) number of pixels in height dimension
        :param mu: (list of floats) expected value in width [0] and height [1]
        :param sigma: (float, default 1.0) standard deviation of gaussian

        Return
        ----------
        :return: (numpy meshgrid of uint8 type) 2D gaussian pattern
    '''
    #crete mesh
    x, y = np.meshgrid(np.linspace(0,1,width)*width, np.linspace(0,1,height)*height)
    #calculate gaussian
    g = np.exp(-( ((x-mu[0])**2 + (y-mu[1])**2) / ( 2.0 * sigma**2 ) ) )
    #normalize gaussian
    g *= 255/np.max(g)
    #change type to uint8
    g = g.astype("uint8")
    #return
    return g

def circle(width, height, center, radius):
    '''
        Create a circle mask in a 2D array. 0 out of the circle, 1 inside the circle. 
        Method based on https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle

        Parameters
        ----------
        :param width: (positive int) width of the mask
        :param height: (positive int) height of the mask
        :param center: (positive int) center of the circle in the mask
        :param radius: (positive int) radius of the circle

        Return
        ----------
        :return: (2D matrix floats) mask of a circle
    '''
    #create axis vectors and empty mask
    x = np.arange(0, width)
    y = np.arange(0, height)
    arr = np.zeros((y.size, x.size))
    #make mask
    pixels = (x[np.newaxis,:]-center[0])**2 + (y[:,np.newaxis]-center[1])**2 < radius**2
    #return the mask with a circle with float values
    return pixels*1.0

def image_to_uint8(image, max_i=None):
    '''
        Convert an image to uint8 by converting it to float64, substract minimum value, and normalize with maximum.

        Parameters
        ----------
        :param image: (2D matrix) image
        :param max_i: (defaulte None, int) maximum value of image matrix

        Return
        ----------
        :return: (uint8 type 2D matrix) converted image
    '''
    #make sure values are within range
    if max_i is not None: 
        assert max_i > -1 and max_i < 256
    else: 
        max_i = 255
    #change image format to float64 and substract minimum. If needed, normalized with max_i
    image = image.astype("float64")
    image -= np.min(image)
    if np.max(image)>0:
        image *= max_i/np.max(image)
    #change image format to uint8
    image = image.astype("uint8")
    #return
    return image

def centroid(array):
    '''
        calculate and return (1st) moment of the image as the centroid of the object.

        Parameters
        ----------
        :param array: (2D uint8 matrix) image of the segmented cell

        Return
        ----------
        :return:tuple with the (2) moments of image
    '''
    m = measure.moments(array)
    return int(m[1,0]/m[0,0]), int(m[0,1]/m[0,0])

def segment_cellsN(image, n=1, cellSize=None, cuadratic = False):
    '''
        Image processing to get area of pixels related to cells for cell segmentation.
        This function uses watershed for segmentation. 
        This is a combination of legacy functions segment_cells and segment_cells2.
        The only difference when changing n is some tunning parameters.

        :param image: (uint8 2D matrix) image to segment
        :param n: (positive int, default 1) option from the two legacy functions, 1 for segment_cells, 2 for segment_cells2
        :param cellSize: (float, default None) minimum distance of centroinds of segments, related to the cell size
        :param cuadratic: (bool, default True) check if you want to change the intensity scale to cuadratic, good for low-contrast images
        :return: (int32 2D matrix) segmented cells as mask
    '''

    #filter for possible cases of n
    if n not in [1,2]:
        print('selected segment_cells option not possible. Changing to original function instead')
        n=1

    #default values for each case
    class_threshold = [2, 4]
    threshold_num = [0, 2]

    #change min distance if there is an input
    if cellSize is not None:
        min_distance = cellSize
    else:
        min_distance = [0.4, 0.3]
        min_distance = min_distance[n-1]

    #add gaussian filter to data
    image = filters.gaussian(image, sigma=4)

    #substract background and use cuadratic scale to highlight differences
    image -= image.mean()
    if cuadratic:
        image =  image**2

    #separate image in 2 classes using the Multi-Otsu threshold method
    thresholds = filters.threshold_multiotsu(image, classes=class_threshold[n-1])

    #isolate cells as the pixels with higher intensity than the threshold
    cells = image > thresholds[threshold_num[n-1]]

    #get distance from each non-zero pixel (i.e., cell pixels) to the closest zero pixel (i.e., background)
    distance = scipy.ndimage.distance_transform_edt(cells)

    #get coordinates of local maxima OF DISTANCE MATRIX with a minimum separation of 0.4x the size of the image. 
    #this gives the center of each cell
    local_max_coords = feature.peak_local_max(distance, min_distance=int(min_distance*image.shape[0]), exclude_border=False)

    #create new boolean matrix with only the center of each cell as True
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True

    #join local maximas in areas or "labels" for watershed
    markers = measure.label(local_max_mask)

    #use the watershed algorithm to separate the cells in areas. 
    #the previusly calculated markers are used to assign labels and the mask ennsure that only the True pixels (higher than threshold) are assigned to areas.
    segmented_cells = segmentation.watershed(-distance, markers, mask=cells)

    #return segmented cells as mask
    return segmented_cells

def segment_cells_cytosol(image,mask_size=0.3):
    '''
        Image processing to get area of pixels related to cells. 
        This function is used to separate the cytosol from the nucleus of the cell. 
        This function first uses segment_cellsN with n=2, then makes a preliminar circular filter around the cells, and repeats segment_cellsn with n=2.

        Parameters
        ----------
        :param image: (uint8 2D matrix) image to segment
        :param mask_size: (float, default 0.3) size of the qmask relative to the image

        Return
        ----------
        :return: (uint8 2D matrix) segmented cells as mask
    '''

    #add gaussian filter to data (large sigma)
    filtered_image = filters.gaussian(image, sigma=10)

    #get segmented cell using 2nd segmentation function to find nuclei
    nuclei = segment_cellsN(image,n=2)

    #create grid, empty image, empty list of centroids, and size of mask
    x,y = np.ogrid[:image.shape[0],:image.shape[1]]
    size = int(mask_size*image.shape[0])
    stitched_image = np.zeros(filtered_image.shape)
    centroids = []

    #iterate labels of the segmented cell - or areas of the cell
    for i in range(np.max(nuclei)+1):

        #skip first idex 
        if i==0: 
            continue

        #get position of centroid of area with current id using moments of the image and append to list
        point = centroid(nuclei*(nuclei==i))   
        centroids.append(point) 

        #get mask (circle) around the position of the centroid
        qmask = ((x-point[0])**2 < size**2) * ((y-point[1])**2 < size**2)

        #sum up every mask of each segmented area by multiplying the filtered image with the mask. Avoid overlappinng with last parenthesis
        stitched_image += (filtered_image * qmask) * (stitched_image == 0)
    
    #avoid 0 values, merge every 0 with the minimum value of the image
    stitched_image[stitched_image==0] = np.min(stitched_image[stitched_image.nonzero()])

    #make a last segmentation for the masked image to get cytosol
    segmented_image = segment_cellsN(stitched_image,n=1)

    #return final segmented image
    return segmented_image

def cell_edge(segmented_cells,size=25):
    '''
        Get the edge of the segmented area of the cell using smoothing filters (difference of gaussians, gaussian).

        Parameters
        ----------
        :param segmented_cells: (2D matrix uint8) mask of segmented cells (output from, e.g., segment_cell)
        :param size: (positive float, default 25) standard deviation of (second) gaussian filter

        Return
        ----------
        :return: (2D matrix uint8) mask of the edges of the cell
    '''

    #filter the image to blur (smooth) it using the Difference of Gaussians method [NOTE: used to enhance edges?]
    border = abs(filters.difference_of_gaussians(segmented_cells,1))

    #filter once more with a single gaussian
    border = filters.gaussian(border,size)

    #get edges (?) of cell as all the pixels with a value above average, after filtering
    border = border > np.mean(border)

    #return image with edges of the cell (?)
    return border

def sector_mask(shape, centroid_pos, projection, width):
    """
        Return a boolean mask for a circular sector from centroid of cell in direction of projection with a certain width

        Parameters
        ----------
        :param shape: (int tuple) dimensions of mask image
        :param centroid_pos: (int tuple) position of centroid of cell
        :param projection: (list of 2 int) direction to take, i.e., from cell centroid to the closest point of the path
        :param width: (float) width of the arch for the illumination area

        Return
        ----------
        :return: return mask for modulator
    """

    #create mesh grid
    x,y = np.ogrid[:shape[0],:shape[1]]

    #get posotion of centroid
    cx,cy = centroid_pos

    #get angle of the path to take and the angle range of  the arch
    direction = projection / np.sqrt(np.sum(projection[:]**2))
    angle = np.arctan2(direction[0], direction[1])
    tmin,tmax = angle - width/2, angle + width/2

    #ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    #convert cartesian to polar coordinates of the mesh and wrap angles between 0 and 2pi
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin
    theta %= (2*np.pi)

    #create circular and angular masks
    circmask = r2 <= shape[0]*shape[1]
    anglemask = theta <= (tmax-tmin)

    #return overlap between circular and angular masks
    return circmask*anglemask

def view_mask(shape, centroid_pos, projection, width,radius):
    """
        XXX Return a boolean mask for a circular sector from centroid of cell in direction of projection with a certain width

        Parameters
        ----------
        :param shape: (int tuple) dimensions of mask image
        :param centroid_pos: (int tuple) position of centroid of cell
        :param projection: (list of 2 int) direction to take, i.e., from cell centroid to the closest point of the path
        :param width: (float) width of the arch for the illumination area

        Return
        ----------
        :return: return mask for modulator
    """

    #create mesh grid
    x,y = np.ogrid[:shape[0],:shape[1]]

    #get posotion of centroid
    cx,cy = centroid_pos

    #get angle of the path to take and the angle range of  the arch
    direction = projection / np.sqrt(np.sum(projection[:]**2))
    angle = np.arctan2(direction[0], direction[1])
    tmin,tmax = angle - width/2, angle + width/2

    #ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    #convert cartesian to polar coordinates of the mesh and wrap angles between 0 and 2pi
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin
    theta %= (2*np.pi)

    #create circular and angular masks
    circmask = r2 <= radius**2
    anglemask = theta <= (tmax-tmin)

    #return overlap between circular and angular masks
    return circmask*anglemask

def roundness(area, major_axis):
    """roundness as defined in FIJI"""
    return 4 * area / (np.pi * major_axis**2)

def circularity(area, perimeter):
    """circularity as defined in FIJI"""
    return 4 * np.pi * area / perimeter**2
#endregion