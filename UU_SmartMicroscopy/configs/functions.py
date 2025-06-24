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

def cell_edge(segmented_cells,size=50):
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

    # #filter the image to blur (smooth) it using the Difference of Gaussians method [NOTE: used to enhance edges?]
    # border = abs(filters.difference_of_gaussians(segmented_cells,1))

    # #filter once more with a single gaussian
    # border = filters.gaussian(border,size)

    # #get edges (?) of cell as all the pixels with a value above average, after filtering
    # border = border > np.mean(border)

    eroded = 1*scipy.ndimage.binary_erosion(segmented_cells,iterations=size)
    border = segmented_cells-eroded

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

def circularity(area, perimeter):
    """circularity as defined in FIJI"""
    return 4 * np.pi * area / perimeter**2
#endregion