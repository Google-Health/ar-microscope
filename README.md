# ARM

The Augmented Reality Microscope (ARM), brings real-time machine learning into an optical microscope for applications in cancer diagnostics and other areas that rely on brightfield microscopes for visual specimen inspection. The Augmented Reality Microscope (ARM) offers novel optics for parallax-free digital overlay projection in an optical microscope, real-time machine learning inference and
state-of-the-art CNNs for accurate cancer detection and classification.


This Github repository contains all driver code and documentation
for ARM. You can find the ARM research paper [here](https://www.nature.com/articles/s41591-019-0539-7).

## Requirements

### Microscope Hardware
* Olympus Microscope BX 43 Series Microscope
* Jenoptik ARM Modules including the ARM Camera & Microdisplay

### Desktop Hardware
The machine the AR Microscope software is built on is referred to as the
"Build Machine" and the machine the software runs on is referred to as the
"Host Machine". We have validated different configurations for these machines.

#### Build Machine
Our build process has been tested on the following configurations:

* Architecture: x86/64
* CPU Platform: Intel Cascade Lake, AMD EPYC Milan processor or AMD EPYC Rome
* OS: Ubuntu 22.04, Ubuntu 20.04, Debian 11 or Debian 12

#### Host Machine
Internally, we use an 6-core Intel Xeon Workstation with an NVIDIA Titan RTX
with 64 GB of RAM and a 512 GB SSD as our ARM host machine.

More broadly, ARM supports:

* NVIDIA GPU of compute architecture 3.5 - 8.6, including:
  * Titan [Xp](https://www.nvidia.com/en-us/titan/titan-xp/),
[V](https://www.nvidia.com/en-us/titan/titan-v/),
or [RTX](https://www.nvidia.com/en-us/deep-learning-ai/products/titan-rtx/)
  * [NVIDIA GTX 10 Series](https://www.nvidia.com/en-us/geforce/10-series/)
  * [NVIDIA RTX 20 Series](https://www.nvidia.com/en-us/geforce/20-series/)
  * [NVIDIA RTX 30 Series](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/)
* x86_64

Because of the necessity for backwards compatibility, our constraining factors
for hardware stem from the Tensorflow version we use: version 2.7. Tensorflow
constraints can be found [here](https://www.tensorflow.org/install/source#gpu).
Currently, Tensorflow 2.7.0 supports CUDA 11.2. CUDA 11.2 supports NVIDIA GPU
compute architectures 3.5 - 8.6.

### Software
* Jenoptik Camera SDK (included with the Jenoptik ARM Modules)
* Ubuntu 18.0.4
* CUDA 11.2
* All dependencies listed in `ar_microscope/build/initial_setup.md`.

## Installing ARM

### Initial Setup
Before building ARM you will need to install its dependencies. You can do this
by running `ar_microscope/build/initial_setup.md`.

### Building and Installing ARM

See the [`README.md`](https://github.com/Google-Health/ar-microscope/blob/main/ar_microscope/build/README.md)
in `ar_microscope/build`.

## Using ARM

Once you have installed ARM, you can run it via Terminal or by double
clicking the application. If you run it via Terminal, you can view the logs in
real time. You can run it by executing the following:

```shell
foo@bar$ bash usr/local/bin/ar_microscope
```

The ARM software will start two windows, one window with a control panel that
allows you to select the model, objective, and brightness, and another that
shows a live feed of the input from the microscope. You should drag the live
feed to the ARM microdisplay. If you can't view windows you may find a
combination of `alt + tab` and `option + space` then `down down enter` useful.


### Calibrating ARM

The first time you use the ARM, you may need to calibrate its microdisplay. In
order to calibrate the ARM Microdisplay, you will need to run ARM via
Terminal in calibration mode. To do this run the following:

```shell
foo@bar$ bash usr/local/bin/ar_microscope --calibration_mode
```

The ARM software will start in calibration mode, allowing you to adjust the left
and right margin for the microdisplay on the main ARM window. You will then need
to update the ARM Config. See the `Configuring ARM` for more details.

### Configuring ARM

See the [`README.md`](https://github.com/Google-Health/ar-microscope/blob/main/ar_microscope/deb_package/README.md)
in `ar_microscope/deb_package`.

### Adding custom models to ARM

Adding custom models is not yet supported. ARM's models are available but you
must request access first. A form to request the models is coming soon!

## Acknowledgments

If you make any publication related to your use of ARM, please cite our
[paper](https://www.nature.com/articles/s41591-019-0539-7) on the research
behind ARM. However, you may not use Google’s name to endorse or promote your
work without our permission.  To request our permission, contact us at
arm-oss-team@google.com.

## Contributing

See [`CONTRIBUTING.md`](https://github.com/Google-Health/ar-microscope/blob/main/CONTRIBUTING.md)
for details.

## License

See [`LICENSE`](https://github.com/Google-Health/ar-microscope/blob/main/LICENSE)
for details.