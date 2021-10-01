# Keys and Doors

This package provides a simple RL gridworld environment in which navigating the maps requires the agent to pickup keys in order to open doors separating each room. 
Reward locations, door locations, key locations as well as the overall structure of the map can be fully customised. 
The environment is aimed specifically at continual learning research; this is achieved primarily via a "curriculum" interface that 
allows for modification of the environment at certain timepoints during training. Modifications include the positions of the keys and rewards, as well as the reward statistics. 

# Getting Started

There are minimal requirements for basic use of this package. The python requirements are listed in ```requirements.txt``` and can be installed via ```pip install -r requirements.txt```.

# Example Usage

Maps are specified by a combination of an ASCII text file (with the map layout) and YAML configuration file (with key, door, reward specifications).

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