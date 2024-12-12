""" This module implements the SAM segmentation and naive tracking."""


import warnings
import numpy as np
import torch
from torch import Tensor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segmentation.abstract_detector import Abstract_Detector
import urllib
import os
from PIL import Image
from torch.nn import functional as F

    
class SAM(Abstract_Detector):
    """Implementation of SAM segmentation.
    Runs SAM segmentation for specified regions of interest.
    If none are specified, the whole image is segmented (slow).
    Class atrributes used during inference are Tensors on the same device for speed,
    which can be ascribed or obtained as np.ndarray using set or get methods.
    """

    def __init__(self,
                 model_type: str = "vit_h",
                 sam_checkpoint: str = "sam_vit_h_4b8939.pth"):
        """
        Initialize the SAM segmentation.
        If the model is not available locally, it will be downloaded from Facebook AI Research.

        :param model_type: The type of the model to use.
        :param sam_checkpoint: The path to the checkpoint of the model.
        """
        # check if GPU is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            warnings.warn('No GPU available. Using CPU (slow).')
            self.device = torch.device("cpu")
            
        # initialize attributes
        self.regions = torch.empty(0, 0).to(device=self.device)                  # regions used for image segmentation
        self.image   = np.empty((0,0,0))                                         # image array is stored in detector
        self.result  = torch.empty(0,0,0).to(device=self.device)                 # segmentation mask b x h x w
        
        # code to load a model from the Facebook AI research if not locally available
        if not os.path.exists("segmentation/" + sam_checkpoint):
            try:
                print('Could not find model in segmentation directory. Downloading from Facebook AI Research.')
                url = "https://dl.fbaipublicfiles.com/segment_anything/"
                urllib.request.urlretrieve(url+sam_checkpoint, "segmentation/" + sam_checkpoint)
            except (urllib.error.URLError, IOError, FileNotFoundError) as e:
                warnings.warn('Could not find model in segmentation directory and downloading failed.')
                raise e        
        
        # initialize the model
        self.model = sam_model_registry[model_type](checkpoint="segmentation/" + sam_checkpoint)
        self.model.to(device=self.device)
        for param in self.model.parameters():
            param.grad = None
        self.predictor = SamPredictor(self.model)                               # initialize the predictor
        self.mask_generator = SamAutomaticMaskGenerator(self.model)             # initialize the mask generator
        
    def setImage(self, image: np.ndarray):
        """Set the image to segment. Will be stored as [0 255] float Tensor [1 x 3 x h x w]

        :param image: The image to segment (3[rbg] x h x w).
        """
        
        # check if image is valid
        if image.shape[0] not in [1, 3] or len(image.shape) != 3:
            warnings.warn('Image must be [3 x h x w] array.\n' +
                          'Single channel images are automatically converted but must be [1 x h x w].\n' +
                          'Setting image to empty tensor.')
            self.image = np.empty((0,0,0))
            self.predictor.reset_image()
            return
        
        self.image = image # class stores original input image
        
        ### Process image to tensor and store in predictor for fast inference ###
        # normalize the image with floating point accuracy to [0 255]
        image = np.float32(image)
        image = (image - np.min(image, axis=(1,2), keepdims=True))
        image = image/(np.max(image, axis=(1,2), keepdims=True)+10**-5) * 255
        
        # convert single channel to three channels ('rgb') image
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        
        # store image as torch tensor with appropriate shape and device for inference
        image_torch = torch.Tensor(image[None]).to(device=self.device)          # 1 x 3 x h x w
        target_size = self.predictor.transform.get_preprocess_shape(image_torch.shape[2],
                                                                    image_torch.shape[3],
                                                                    self.predictor.transform.target_length)
        image_torch = F.interpolate(image_torch, target_size, mode="bilinear", align_corners=False, antialias=True)
        self.predictor.set_torch_image(image_torch, image.shape[1:])            # set the image for the predictor attribute
        
    def setup(self, regions: np.ndarray):
        """Specify regions for segmentation."""
        self.regions = Tensor(regions).to(device=self.device)
        
    def run(self):
        """Segment the image using SAM.

        :param image: The image to segment (1 x 3[rbg] x h x w).
        :return: The segmentation mask.
        """
        # check if image is specified
        if self.image.shape[0] == 0:
            warnings.warn('No image specified. Returning empty mask.')
            self.result = torch.empty(0,0,0).to(device=self.device)
            return
                
        # if no regions are specified, segment the whole image
        if self.regions.shape[0] == 0:
            warnings.warn('No regions specified. Segmenting the whole image (slow).')
            
            # convert to three channel image
            if self.image.shape[0] == 1:
                image = np.repeat(self.image, 3, axis=0)
            else:
                image = self.image
            
            # normalize and convert to 8 bit image
            image = np.float32(image)
            image = (image - np.min(image, axis=(1,2), keepdims=True))
            image = image/(np.max(image, axis=(1,2), keepdims=True)+10**-5) * 255
            image = np.uint8(image.transpose(1,2,0))                            # convert to 8 bit rgb image (h x w x 3)

            predictions = self.mask_generator.generate(image)                   # generate the predictions
            masks = [p['segmentation'] for p in predictions]                    # extract the masks
            masks = [np.array(Image.fromarray(m).resize((image.shape[1], image.shape[0]))) for m in masks]
            self.result = Tensor(np.array(masks)).to(device=self.device)
        # else segment the specified regions
        else:
            transformed = self.predictor.transform.apply_boxes_torch(self.regions,
                                                                     self.predictor.original_size)
            predictions = self.predictor.predict_torch(point_coords=None,
                                                       point_labels=None,
                                                       boxes=transformed)[0]   # segment using the specified regions
            predictions = predictions[:,0].to(device=self.device)
            self.result = predictions
    
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
        if self.device == torch.device("cuda"):
            return self.result.detach().cpu().numpy()
        else:
            return self.result.numpy()
    
    def getRegions(self) -> np.ndarray:
        """Retrieves the regions used as input to SAM.
        
        :return: Regions as numpy array.
        """
        if self.regions.shape[0] == 0:
            warnings.warn('No regions found. Returning empty mask.')
        if self.device == torch.device("cuda"):
            return self.regions.detach().cpu().numpy()
        else:
            return self.regions.numpy()
      
    def resetRegions(self):
        """Reset the regions attribute to empty tensor."""
        self.regions = torch.empty(0,0).to(device=self.device)
    
    def updateRegions(self, square: bool = True):
        """Compute bounding boxes for binary masks and update regions. Boxes are square if "square" is True, else rectangular.
        
        :param square: Whether to use square bounding boxes.
        """
        # compute bounding boxes
        sum_masks_x = torch.sum(self.result, dim=2)>0                           # rows with at least one pixel in the mask
        sum_masks_y = torch.sum(self.result, dim=1)>0                           # columns with at least one pixel in the mask
        idx = torch.arange(self.result.shape[1], 0, -1, device=self.device)
        idy = torch.arange(self.result.shape[2], 0, -1, device=self.device)

        xmin = torch.argmax(sum_masks_x*idx, dim=1)                             # first row with a pixel in the mask
        xmax = torch.argmax(sum_masks_x*idx.flip(dims=[0]), dim=1)              # last row with a pixel in the mask
        ymin = torch.argmax(sum_masks_y*idy, dim=1)                             # first column with a pixel in the mask
        ymax = torch.argmax(sum_masks_y*idy.flip(dims=[0]), dim=1)              # last column with a pixel in the mask
        w, h = xmax-xmin, ymax-ymin                                             # width and height of the bounding box
        
        # add some additional space around the bounding box, unless out of FOV
        W, H = self.result.shape[1:]                                            # height and width of the image
        if square:
            l = torch.max(torch.cat([w[None], h[None]]), dim=0).values          # longest sidelength of the bounding box
            xmin = torch.clamp(xmin//2+xmax//2 - l//2 - l//10, 0, W-1)
            xmax = torch.clamp(xmin//2+xmax//2 - l//2 + l//10, 0, W-1)
            ymin = torch.clamp(ymin//2+ymax//2 - l//2 - l//10, 0, H-1)
            ymax = torch.clamp(ymin//2+ymax//2 - l//2 + l//10, 0, H-1)
        else:
            xmin = torch.clamp(xmin - w//10, 0, W-1)
            xmax = torch.clamp(xmax + w//10, 0, W-1)
            ymin = torch.clamp(ymin - h//10, 0, H-1)
            ymax = torch.clamp(ymax + h//10, 0, H-1)

        self.regions = torch.stack([ymin, xmin, ymax, xmax]).T.to(device=self.device)


if __name__ == "__main__":
    sam  = segmenter_SAM()
    test_input = np.float32(np.random.rand(3, 612, 512))
    sam.setImage(test_input)
    
    sam.setup(np.array([[100, 200, 300, 400]]))
    sam.run()
    mask = sam.getResult()

    