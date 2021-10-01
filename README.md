# Keys and Doors

This package provides a simple RL gridworld environment in which navigating the maps requires the agent to pickup keys in order to open doors separating each room. 
Reward locations, door locations, key locations as well as the overall structure of the map can be fully customised. 
The environment is aimed specifically at continual learning research; this is achieved primarily via a "curriculum" interface that 
allows for modification of the environment at certain timepoints during training. Modifications include the positions of the keys and rewards, as well as the reward statistics. 

## Getting Started

There are minimal requirements for basic use of this package. The python requirements are listed in ```requirements.txt``` and can be installed via ```pip install -r requirements.txt```.
To ensure correct installation, you can run tests via ```python -m tests/test.py```.

## Example Usage

Maps are specified by a combination of an ASCII text file (with the map layout) and YAML configuration file (with key, door, reward specifications).

For instance, the ASCII and YAML snippets below (A, B) produce the map shown below (C). These example files can also be found under the ```tests/test_map_files```.

#### A. ASCII file
```
#################################
#         K    #       R#       #
#              #        D       #
#              #        #       #
#              #        #       #
#S             D     K  #   R   #
################################# 
```
#### B. YAML File
```
start_position: [1, 1]

key_positions:
    - [23, 5] # key for door 1
    - [2, 5] # key for door 2

door_positions:
    - [15, 1] # door 1 
    - [24, 4] # door 2 

reward_positions:
    - [17, 4]
    - [31, 5]

# provide either one set of statistics (used for all rewards)
# or one set of statistics for each reward
reward_statistics:
    gaussian:
        mean: 1
        std: 0
```
#### C. Example Map
![Sample Map](./tests/test_map_files/test_map.png "Title")

TODO:

- Document
- Tests
- Sample maps
- Plotting functions
- Video generation functions
- README

Thoughts:

- init with either path to yaml or set of python args?
- Start position options, random, fixed, set etc.
- Environment curriculum
- Constants??
- Throw error if key.door/rewrD etc in same position?
- Throw error if key.door/rewrD etc in wall?