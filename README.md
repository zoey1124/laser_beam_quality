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
cd dillinger
npm i
node app
```

For production environments...

```sh
npm install --production
NODE_ENV=production node app
```

## Plugins

Dillinger is currently extended with the following plugins.
Instructions on how to use them in your own application are linked below.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Development

Want to contribute? Great!

Dillinger uses Gulp + Webpack for fast developing.
Make a change in your file and instantaneously see your updates!

Open your favorite Terminal and run these commands.

First Tab:

```sh
node app
```

Second Tab:

```sh
gulp watch
```

(optional) Third:

```sh
karma test
```

#### Building for source

For production release:

```sh
gulp build --prod
```

Generating pre-built zip archives for distribution:

```sh
gulp build dist --prod
```

## Docker

Dillinger is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the
Dockerfile if necessary. When ready, simply use the Dockerfile to
build the image.

```sh
cd dillinger
docker build -t <youruser>/dillinger:${package.json.version} .
```

This will create the dillinger image and pull in the necessary dependencies.
Be sure to swap out `${package.json.version}` with the actual
version of Dillinger.

Once done, run the Docker image and map the port to whatever you wish on
your host. In this example, we simply map port 8000 of the host to
port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart=always --cap-add=SYS_ADMIN --name=dillinger <youruser>/dillinger:${package.json.version}
```

> Note: `--capt-add=SYS-ADMIN` is required for PDF rendering.

Verify the deployment by navigating to your server address in
your preferred browser.

```sh
127.0.0.1:8000
```

## License

**Free Software, Hell Yeah!**


