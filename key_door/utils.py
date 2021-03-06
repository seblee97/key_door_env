import itertools
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from key_door import constants


def parse_map_outline(map_file_path: str, mapping: Dict[str, int]) -> np.ndarray:
    """Method to parse ascii map and map settings from yaml file.

    Args:
        map_file_path: path to file containing map schematic.
        map_yaml_path: path to yaml file containing map config.

    Returns:
        multi_room_grid: numpy array of map state.
    """
    map_rows = []

    with open(map_file_path) as f:
        map_lines = f.read().splitlines()

        # flip indices for x, y referencing
        for i, line in enumerate(map_lines[::-1]):
            map_row = [mapping[char] for char in line]
            map_rows.append(map_row)

    assert all(
        len(i) == len(map_rows[0]) for i in map_rows
    ), "ASCII map must specify rectangular grid."

    multi_room_grid = np.array(map_rows, dtype=float)

    return multi_room_grid


def parse_map_positions(map_yaml_path: str) -> Tuple[List, List, List, List]:
    """Method to parse map settings from yaml file.

    Args:
        map_yaml_path: path to yaml file containing map config.

    Returns:
        initial_start_position: x,y coordinates for
            agent at start of each episode.
        key_positions: list of x, y coordinates of keys.
        door_positions: list of x, y coordinates of doors.
        reward_positions: list of x, y coordinates of rewards.
    """
    with open(map_yaml_path) as yaml_file:
        map_data = yaml.load(yaml_file, yaml.SafeLoader)

    start_positions = [tuple(map_data[constants.START_POSITION])]
    reward_positions = [tuple(rp) for rp in map_data[constants.REWARD_POSITIONS]]
    key_positions = [tuple(kp) for kp in map_data[constants.KEY_POSITIONS]]
    door_positions = [tuple(dp) for dp in map_data[constants.DOOR_POSITIONS]]

    reward_statistics = map_data[constants.REWARD_STATISTICS]

    assert (
        len(start_positions) == 1
    ), "maximally one start position 'S' should be specified in ASCII map."

    assert len(door_positions) == len(
        key_positions
    ), "number of key positions must equal number of door positions."

    return (
        start_positions[0],
        key_positions,
        door_positions,
        reward_positions,
        reward_statistics,
    )


def setup_reward_statistics(
    reward_positions, reward_specifications
) -> Dict[Tuple, Callable]:

    per_reward_specification = len(reward_positions) == len(reward_specifications)
    single_reward_specification = len(reward_specifications) == 1

    assert (
        per_reward_specification or single_reward_specification
    ), "number of rewards statistics must either be 1 or match number of reward positions."

    def _get_reward_function(reward_type: str, reward_parameters: Dict) -> Callable:

        if reward_type == constants.GAUSSIAN:

            def _sample_gaussian():
                return np.random.normal(
                    loc=reward_parameters[constants.MEAN],
                    scale=reward_parameters[constants.VARIANCE],
                )

            return _sample_gaussian

    reward_types = list(reward_specifications.keys())
    reward_parameters = list(reward_specifications.values())

    if single_reward_specification:
        reward_function = _get_reward_function(reward_types[0], reward_parameters[0])
        rewards = {
            reward_position: reward_function for reward_position in reward_positions
        }
    else:
        rewards = {
            reward_position: _get_reward_function(reward_type, reward_parameter)
            for reward_position, reward_type, reward_parameter in zip(
                reward_positions, reward_types, reward_parameters
            )
        }

    return rewards


def configure_state_space(map_outline, key_positions, reward_positions):
    """Get state space for the environment from the parsed map.
    Further split state space into walls, valid positions, key possessions etc.
    """
    state_indices = np.where(map_outline == 0)
    wall_indices = np.where(map_outline == 1)

    positional_state_space = list(zip(state_indices[1], state_indices[0]))
    key_possession_state_space = list(
        itertools.product([0, 1], repeat=len(key_positions))
    )
    rewards_received_state_space = list(
        itertools.product([0, 1], repeat=len(reward_positions))
    )
    state_space = [
        i[0] + i[1] + i[2]
        for i in itertools.product(
            positional_state_space,
            key_possession_state_space,
            rewards_received_state_space,
        )
    ]

    wall_state_space = list(zip(wall_indices[1], wall_indices[0]))

    return (
        positional_state_space,
        key_possession_state_space,
        rewards_received_state_space,
        state_space,
        wall_state_space,
    )


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    # rgb channel last
    grayscale = np.dot(rgb[..., :3], [[0.299], [0.587], [0.114]])
    return grayscale
