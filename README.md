# Beam Quality Detector
## _For Automatic Detection of Laser Beam Quality_


Beam Quality Detector is a Python API that make it easy to automate laser beam image aberration detection for all cameras that captures laser beam images.

Two main parts of design: threshold reference and CNN model. Threshold Reference is used to identify the aberration laser images, and if the image is identified as abnormal, the CNN model will output which category it is (Hot Spot, Clipped Edge, or Airy Ring)




## Features

- Detect Laser Beam Image Aberration
- Classify Aberration Category
- Compatible to any image size/shape
- Data Processing from xtc format file
- Image Transformation



## Tech & Dependency

Beam Quality Detector requires following packages & dependencies to run properly:

- [psana](https://confluence.slac.stanford.edu/display/PSDMInternal/psana+-+Reference+Manual) - used for LCLS internal data analysis/processing tool
- [h5py](https://docs.h5py.org/en/stable/quick.html) - a container for datasets and groups.
- [torch](https://pytorch.org/docs/stable/index.html) - for CNN model building and running to classify image


## Usage

The running environment uses [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Install the dependencies and devDependencies and start jupyter-notebook, more usage examples can be viewed in use_case_example.ipynb.
```sh
conda env create -f environment.yml
conda activate laser_beam
```

## License

**Free Software, Hell Yeah!**
