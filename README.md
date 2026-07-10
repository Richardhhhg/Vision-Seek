<h1 align="center">Vision Seek</h1>

---

> [!IMPORTANT]
> Vision seek is still in development and may not be reliable in production. Please use with caution.

Vision Seek detects objects in drone footage and maps it onto a GIS map showing an annotated version of the location.

## What is This Project?

While there are services for mapping drone flights into the real world, few support object detection and mapping the location of the object into the real world as well. Vision Seek solves this problem by supporting a all in one platform for detection objects from done footage, and mapping the location of the objects into the real world.
  
The detection module is built to support various types of object detection algorithms utilizing abstractions.
Currently supported detection algorithms:

- YOLO
  
The mapping module takes the results of the detection module and maps it into the real world.
>[!NOTE]
> The mapping module has yet to be implemented

## Installation

>[!NOTE]
> This project has only so far been tested with Python 3.11 and Arch Linux operating system. Python versions more recent than 3.9 should be supported.

To install simply ensure you have all libraries needed in requirements.txt installed and run main.py.

## Contributing

This project welcomes new contributions!
  
Make sure you are working on a branch that isn't main. Once you have made your changes simply make a new pull request and wait for a contributor to review it.

To help speed up the review, please record the result of pytest in your PR. This is just to make sure nothing broke and we are still meeting development standards.

## TODO

- [x] Detecting Objects
- [ ] Extracting GPS data about the location of the detected objects
- [ ] Displaying Mapping
