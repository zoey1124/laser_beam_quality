from psana import *
from model import *
import numpy as np
import matplotlib.pyplot as plt
import torch

class BeamQualityDetector(object):
    
    def __init___(self, img, ref_img, threshold, model_type='CNN_28_1'):
        """
        valid mode type: CNN_128, CNN_28_1, CNN_28_2
        """
        self.img = img
        self.ref_img = ref_img
        self.threshold = threshold
        self.model = load_model(model_type)

    def is_referenced(self):
        """
        Returns True if img is within threshold range of ref_img
        """
        assert self.img.shape == self.ref_img.shape
        return (abs(self.img - self.ref_img) < threshold).all()


    def show_numpy_img(self, image):
        plt.imshow(image)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("image visualization")
        plt.show()


    def predict(self, test_img):
        """
        test_img - numpy array format test image 
        """
        # first change numpy to tensor, check size of image, resize
        outputs = self.model(test_img)
        _, predicted = outputs.max(1)
        return predicted
        

        
        
        



    



