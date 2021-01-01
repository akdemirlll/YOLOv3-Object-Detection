# YOLOv3

## USAGE

To download weights and setup model:
  - Windows users: run `setup.bat`
  - MacOS or UNIX/LINUX users: run `setup.sh`
  - Windows users with git bash can also run `setup.sh`

This step will download the weights, build the model and test setup.

To feed images from web camera:
  - Windows users: run `main.bat`
  - MacOS or UNIX/LINUX users: run `main.sh`
  - Windows users with git bash can also run `main.sh`

`main` script allows some arguments for debugging and configuration purposes:
- `--no-nms` : Disable non-max suppression.
- `--profile`: Print function execution times.
- `--shallow`: Use predictions from the first output of Yolo.
  - This will speed up prediction time and decrease number of boxes.
