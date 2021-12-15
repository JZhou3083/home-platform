<<<<<<< HEAD
# HoME Platform [![Build Status](https://travis-ci.org/HoME-Platform/home-platform.svg?branch=master)](https://travis-ci.org/HoME-Platform/home-platform)

HoME is a platform for artificial agents to learn from vision, audio, semantics, physics, and interaction with objects and
other agents, all within a realistic context.

Check out the paper on Arxiv for more details: [HoME: a Household Multimodal Environment](https://arxiv.org/abs/1711.11017)

![alt tag](https://github.com/HoME-Platform/home-platform/raw/master/doc/images/overview.png)

## Dependencies

Main requirements:
- Python 2.7+ with Numpy, Scipy and Matplotlib
- (or Python 3.5+ but currently WIP, might have bugs)
- [Panda3d](https://www.panda3d.org/) game engine for 3D rendering
- [EVERT](https://github.com/sbrodeur/evert) engine for 3D acoustic ray-tracing
- [PySoundFile](https://github.com/bastibe/PySoundFile) for Ogg Vorbis decoding

To install dependencies on Ubuntu operating systems:
```
sudo apt-get install python-pip python-tk python-dev build-essential libsndfile1 portaudio19-dev
sudo pip2 install --upgrade pip numpy scipy matplotlib gym panda3d pysoundfile pyaudio resampy nose coverage Pillow imageio
```

or, for Python 3:
```
sudo apt-get install python3-pip python3-tk python3-dev build-essential libsndfile1 portaudio19-dev
sudo pip3 install --upgrade pip numpy scipy matplotlib gym panda3d pysoundfile pyaudio resampy nose coverage Pillow imageio
```

(Packages `nose` and `coverage` are for tests only and can be omitted)

Finally you have to install EVERT. In order to do so, please follow the instructions over at 
https://github.com/sbrodeur/evert

## SUNCG Dataset

The Home environment is based on the [SUNCG](http://suncg.cs.princeton.edu/) dataset. 

**Important!** Before you can use this library, you need to obtain the SUNCG dataset.
In order to do so, please follow the instructions on their website.

For the test suit we included a single small house as sample in this repository.

## Installing the library

Download the source code from the git repository:
```
mkdir -p $HOME/work
cd $HOME/work
git clone https://github.com/HoME-Platform/home-platform.git
```

Note that the library must be in the PYTHONPATH environment variable for Python to be able to find it:
```
export PYTHONPATH=$HOME/work/home-platform:$PYTHONPATH 
```
This can also be added at the end of the configuration file $HOME/.bashrc

## Running unit tests

To ensure all libraries where correctly installed, it is advised to run the test suite:
```
cd $HOME/work/home-platform/tests
./run_tests.sh
```
Note that this can take some time.


## Usage

**Before you start please read steps 0 and 1 so that you know what to expect. 
The data installation and preparation can take some time (because it's a big dataset).**

#### 0 Obtain and install the SUNCG dataset (only once)

The dataset can be acquired by contacting the authors at their website: http://suncg.cs.princeton.edu/

Currently the dataset downloaded consists of a single zip file, which in turn contains multiple zip files
(one for each directory in [house, object, object_vox, room, texture]).

Please unzip all files so that as a result you have the folloging directory structure somewhere on your PC:
```
/some/path/doesnt/matter/SUNCG/
/some/path/doesnt/matter/SUNCG/house/
/some/path/doesnt/matter/SUNCG/house/0004d52d1aeeb8ae6de39d6bd993e992/
/some/path/doesnt/matter/SUNCG/house/0004dd3cb11e50530676f77b55262d38/
...
/some/path/doesnt/matter/SUNCG/object/100/
/some/path/doesnt/matter/SUNCG/object/101/
...
``` 

The unzipped files take up approx 28.1GB disk space on an NTFS-formatted drive.

**Warning** Unzipping may take considerable time and should not be done on a network drive
due to the overhead in network communication. We advise to extract everything on a local machine 
and if necessary copy it to networked machines via `rsync` like so:

    rsync -avh --info=progress2 --remove-source-files /some/path/doesnt/matter/SUNCG /cluster/datasets/
    
Once everything is unzipped, you can remove the original zip files.

As final step please "install" the dataset by (a) symlinking it into your home directory

    ln -s /some/path/doesnt/matter/SUNCG/ ~/.suncg 

**or** by (b) setting the environmental variable to the right path:

    export SUNCG_DATA_DIR="/cluster/datasets/SUNCG"
    
If you do both of these things, the environmental variable takes precedent.

#### 1 Convert the houses into a usable format (only once)

In order for Panda3D to be able to read the 3D files of the houses, you need
to first convert them to a different format (from OBJ/Wavefront to EGG and BAM).

To do so, just cd into the scripts folder in this repository and run the conversion script:

    # assuming you currently are in the directory of this readme file:
    cd scripts
    
    # if you installed the dataset via symlink:
    ./convert_suncg.sh ~/.suncg/ # the trailing slash is important
    
    # if you installed the dataset via environmental vairable:
    ./convert_suncg.sh $SUNCG_DATA_DIR/ # the trailing slash might be important
    
=======
# jz1n20_home

>>>>>>> a2f9e510447913cf25afd5480abb5fe760e78396
