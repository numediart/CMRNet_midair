# CMRNet On Midair

## Prerequisites 
1. Install all the tools needed to run CMRNet as explained on the official [repository](https://github.com/cattaneod/CMRNet);
2. Download the Midair dataset [here](https://midair.ulg.ac.be/download.html). You will need left and right RGB as well as depth for both the sunny kite environment and the spring fall environment.

## How to use
First, you need to generate the local maps needed to train CMRNet. In a terminal, type: 

`python preprocess/midair_maps.py --sequence trajectory_5014 --start nbr --end nbr --midair_folder path_to_root_dir_of_trajectory`

This will generate the local maps for the trajectory_5014 between the start and end index. You can repeat for all the frames of the trajectory and for all the trajectories. The repository has script called `midair_maps.sh` to automate the process. It only misses the `midair_folder` flag, but you can change the default value of the parameter in `preprocess\midair_maps.py`.

Once you have generated the maps, you can train the model on the dataset using the `main_visibility_CALIB_midair.py` script. The syntax is as follows:

`python main_visibility_CALIB_midair.py with batch_size=16 data_folder=path_to_root_dir_of_trajectory epochs=100 max_r=10 max_t=2 BASE_LEARNING_RATE=0.0001 savemodel=./checkpoints/ test_sequence=trajectory_5001`
