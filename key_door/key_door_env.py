import copy
import itertools
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from key_door import base_environment, constants, utils


class KeyDoorGridworld(base_environment.BaseEnvironment):
    """Grid world environment with multiple rooms.
    Between each room is a door, that requires a key to unlock.
    """

    def __init__(
        self,
        map_ascii_path: str,
        map_yaml_path: str,
        representation: str,
        episode_timeout: Optional[Union[int, None]] = None,
    ) -> None:
        """Class constructor.

        Args:
            map_ascii_path: path to txt or other ascii file with map specifications.
            map_yaml_path: path to yaml file with map settings (reward locations etc.)
            representation: agent_position (for tabular) or pixel (for function approximation)
            episode_timeout: number of steps before episode automatically terminates.
        """

        self._active: bool = False

        self._training: bool
        self._episode_step_count: int
        self._train_episode_position_history: List[List[int]]
        self._test_episode_position_history: List[List[int]]
        self._train_episode_history: List[np.ndarray]
        self._test_episode_history: List[np.ndarray]
        self._agent_position: np.ndarray
        self._rewards_state: np.ndarray
        self._keys_state: np.ndarray

        self._representation = representation
        self._episode_timeout = episode_timeout or np.inf

        self._map = utils.parse_map_outline(
            map_file_path=map_ascii_path, mapping=self.MAPPING
        )

        (
            self._starting_xy,
            self._key_positions,
            self._door_positions,
            reward_positions,
            reward_statistics,
        ) = utils.parse_map_positions(map_yaml_path)

        self._rewards = utils.setup_reward_statistics(
            reward_positions, reward_statistics
        )
        self._total_rewards = len(self._rewards)

        (
            self._positional_state_space,
            self._key_possession_state_space,
            self._rewards_received_state_space,
            self._state_space,
            self._wall_state_space,
        ) = utils.configure_state_space(
            map_outline=self._map,
            key_positions=self._key_positions,
            reward_positions=reward_positions,
        )

        # states are zero, -1 removes walls from counts.
        self._visitation_counts = -1 * copy.deepcopy(self._map)

    def _env_skeleton(
        self,
        rewards: Union[None, str, Tuple[int]] = "state",
        keys: Union[None, str, Tuple[int]] = "state",
        doors: Union[None, str] = "state",
        agent: Union[None, str, np.ndarray] = "state",
    ) -> np.ndarray:
        """Get a 'skeleton' of map e.g. for visualisation purposes.

        Args:
            rewards: # TODO whether or not to mark out rewards (ignores magnitudes).
            show_doors: whether or not to mark out doors.
            show_keys: whether or not to mark out keys.
            show_agent: whether or not to mark out agent position

        Returns:
            skeleton: np array of map.
        """
        # flip size so indexing is consistent with axis dimensions
        skeleton = np.ones(self._map.shape + (3,))

        # make walls black
        skeleton[self._map == 1] = np.zeros(3)

        if rewards is not None:
            if isinstance(rewards, str):
                if rewards == constants.STATIONARY:
                    reward_iterate = list(self._rewards.keys())
                elif rewards == constants.STATE:
                    reward_positions = list(self._rewards.keys())
                    reward_iterate = [
                        reward_positions[i]
                        for i, r in enumerate(self._rewards_state)
                        if not r
                    ]
                else:
                    raise ValueError(f"Rewards keyword {rewards} not identified.")
            elif isinstance(rewards, list):
                reward_positions = list(self._rewards.keys())
                reward_iterate = [
                    reward_positions[i] for i, r in enumerate(rewards) if not r
                ]
            else:
                raise ValueError(
                    "Rewards must be string with relevant keyword or keystate list"
                )
            # show reward in red
            for reward in reward_iterate:
                skeleton[reward[::-1]] = [1.0, 0.0, 0.0]

        if keys is not None:
            if isinstance(keys, str):
                if keys == constants.STATIONARY:
                    keys_iterate = self._key_positions
                elif keys == constants.STATE:
                    keys_iterate = [
                        self._key_positions[i]
                        for i, k in enumerate(self._keys_state)
                        if not k
                    ]
                else:
                    raise ValueError(f"Keys keyword {keys} not identified.")
            elif isinstance(keys, list):
                keys_iterate = [
                    self._key_positions[i] for i, k in enumerate(keys) if not k
                ]
            else:
                raise ValueError(
                    "Keys must be string with relevant keyword or keystate list"
                )
            # show key in yellow
            for key_position in keys_iterate:
                skeleton[tuple(key_position[::-1])] = [1.0, 1.0, 0.0]

        if doors is not None:
            if isinstance(doors, str):
                if doors == constants.STATE:
                    doors_iterate = self._door_positions
                elif doors == constants.STATIONARY:
                    doors_iterate = self._door_positions
                else:
                    raise ValueError(f"Doors keyword {doors} not identified.")
            # show door in maroon
            for door in doors_iterate:
                skeleton[tuple(door[::-1])] = [0.5, 0.0, 0.0]

        if agent is not None:
            if isinstance(agent, str):
                if agent == constants.STATE:
                    agent_position = self._agent_position
                elif agent == constants.STATIONARY:
                    agent_position = self._starting_xy
            else:
                agent_position = agent
            # show agent
            skeleton[tuple(agent_position[::-1])] = 0.5 * np.ones(3)

        return skeleton

    def _get_state_representation(
        self,
        tuple_state: Optional[Tuple] = None,
    ) -> Union[tuple, np.ndarray]:
        """From current state, produce a representation of it.
        This can either be a tuple of the agent and key positions,
        or a top-down pixel view of the environment (for DL)."""
        if self._representation == constants.AGENT_POSITION:
            return (
                tuple(self._agent_position)
                + tuple(self._keys_state)
                + tuple(self._rewards_state)
            )
        elif self._representation == constants.PIXEL:
            if tuple_state is None:
                env_skeleton = self._env_skeleton()  # H x W x C
            else:
                agent_position = tuple_state[:2]
                keys = tuple_state[2 : 2 + len(self._key_positions)]
                rewards = tuple_state[2 + len(self._key_positions) :]
                env_skeleton = self._env_skeleton(
                    rewards=rewards, keys=keys, agent=agent_position
                )
            grayscale_env_skeleton = utils.rgb_to_grayscale(env_skeleton)
            transposed_env_skeleton = np.transpose(
                grayscale_env_skeleton, axes=(2, 0, 1)
            )  # C x H x W
            # add batch dimension
            state = np.expand_dims(transposed_env_skeleton, 0)
            return state

    def _move_agent(self, delta: np.ndarray) -> float:
        """Move agent. If provisional new position is a wall, no-op."""
        provisional_new_position = self._agent_position + delta

        moving_into_wall = tuple(provisional_new_position) in self._wall_state_space
        locked_door = tuple(provisional_new_position) in self._door_positions

        if locked_door:
            door_index = self._door_positions.index(tuple(provisional_new_position))
            if self._keys_state[door_index]:
                locked_door = False

        if not moving_into_wall and not locked_door:
            self._agent_position = provisional_new_position

        if tuple(self._agent_position) in self._key_positions:
            key_index = self._key_positions.index(tuple(self._agent_position))
            if not self._keys_state[key_index]:
                self._keys_state[key_index] = 1

        return self._compute_reward()

    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent.

        Args:
            action: 0: left, 1: up, 2: right, 3: down

        Returns:
            reward: float indicating reward, 1 for target reached, 0 otherwise.
            next_state: new coordinates of agent.
        """
        assert (
            self._active
        ), "Environment not active. call reset_environment() to reset environment and make it active."
        assert (
            action in self.ACTION_SPACE
        ), f"Action given as {action}; must be 0: left, 1: up, 2: right or 3: down."

        reward = self._move_agent(delta=self.DELTAS[action])

        if self._training:
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1
            self._train_episode_position_history.append(tuple(self._agent_position))
            self._train_episode_history.append(self._env_skeleton())
        else:
            self._test_episode_position_history.append(tuple(self._agent_position))
            self._test_episode_history.append(self._env_skeleton())

        self._active = self._remain_active(reward=reward)

        new_state = self._get_state_representation()

        self._episode_step_count += 1

        return reward, new_state

    def _compute_reward(self) -> float:
        """Check for reward, i.e. whether agent position is equal to a reward position.
        If reward is found, add to rewards received log.
        """
        if (
            tuple(self._agent_position) in self._rewards
            and tuple(self._agent_position) not in self._rewards_received
        ):
            reward = self._rewards.get(tuple(self._agent_position))()
            reward_index = list(self._rewards.keys()).index(tuple(self._agent_position))
            self._rewards_state[reward_index] = 1
            self._rewards_received.append(tuple(self._agent_position))
        else:
            reward = 0.0

        return reward

    def _remain_active(self, reward: float) -> bool:
        """Check on reward / timeout conditions whether episode should be terminated.

        Args:
            reward: total reward accumulated so far.

        Returns:
            remain_active: whether to keep episode active.
        """
        conditions = [
            self._episode_step_count == self._episode_timeout,
            len(self._rewards_received) == self._total_rewards,
        ]
        return not any(conditions)

    def reset_environment(self, train: bool = True) -> Tuple[int, int, int]:
        """Reset environment.

        Bring agent back to starting position.

        Args:
            train: whether episode is for train or test (affects logging).
        """
        self._active = True
        self._episode_step_count = 0
        self._training = train
        self._agent_position = np.array(self._starting_xy)
        self._rewards_received = []
        self._keys_state = np.zeros(len(self._key_positions), dtype=int)
        self._rewards_state = np.zeros(len(self._rewards), dtype=int)

        if train:
            self._train_episode_position_history = [tuple(self._agent_position)]
            self._train_episode_history = [self._env_skeleton()]
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1
        else:
            self._test_episode_position_history = [tuple(self._agent_position)]
            self._test_episode_history = [self._env_skeleton()]

        initial_state = self._get_state_representation()

        return initial_state

    # def _average_values_over_key_states(
    #     self, values: Dict[Tuple[int], float]
    # ) -> Dict[Tuple[int], float]:
    #     """For certain analyses (e.g. plotting value functions) we want to
    #     average the values for each position over all non-positional state information--
    #     in this case the key posessions.

    #     Args:
    #         values: full state-action value information

    #     Returns:
    #         averaged_values: (positional-state)-action values.
    #     """
    #     averaged_values = {}
    #     for state in self._positional_state_space:
    #         non_positional_set = [
    #             values[state + i[0] + i[1]]
    #             for i in itertools.product(
    #                 self._key_possession_state_space, self._rewards_received_state_space
    #             )
    #         ]
    #         non_positional_mean = np.mean(non_positional_set, axis=0)
    #         averaged_values[state] = non_positional_mean
    #     return averaged_values

    # def _get_value_combinations(
    #     self, values: Dict[Tuple[int], float]
    # ) -> Dict[Tuple[int], Dict[Tuple[int], float]]:
    #     """Get each possible combination of positional state-values
    #     over non-positional states.

    #     Args:
    #         values: values over overall state-space

    #     Returns:
    #         value_combinations: values over positional state space
    #             for each combination of non-positional state.
    #     """
    #     value_combinations = {}
    #     for key_state in self._key_possession_state_space:
    #         value_combination = {}
    #         for state in self._positional_state_space:
    #             value_combination[state] = values[state + key_state]
    #         value_combinations[key_state] = value_combination

    #     return value_combinations

    # def animate_value_function(
    #     self, all_values: List[np.ndarray], save_path: str, over_actions: str
    # ):
    #     """Create an animation of value function(s) saved throughout training.

    #     Args:
    #         all_values: list of saved numpy arrays corresponding to
    #         values at different times in training.
    #         save_path: path to save location for animation.
    #         over_actions: method to perform over actions e.g. mean.
    #     """
    #     caxes = []

    #     if 0 < len(self._key_positions) <= 2:
    #         fig, axes = plt.subplots(nrows=1 + 2 * len(self._key_positions), ncols=1)
    #         for ax in axes:
    #             divider = make_axes_locatable(ax)
    #             cax = divider.append_axes("right", "5%", "5%")
    #             caxes.append(cax)
    #     else:
    #         fig, axes = plt.subplots(nrows=1, ncols=1)
    #         divider = make_axes_locatable(axes)
    #         cax = divider.append_axes("right", "5%", "5%")
    #         caxes.append(cax)

    #     def _update(values):
    #         self.plot_value_function(
    #             values=values,
    #             plot_max_values=True,
    #             quiver=False,
    #             over_actions=over_actions,
    #             save_path=None,
    #             fig=fig,
    #             axes=axes,
    #             caxes=caxes,
    #         )
    #         # for i, ax in enumerate(updated_axes):
    #         #     axes[i] = ax

    #     anim = FuncAnimation(fig, _update, frames=all_values, interval=200)
    #     anim.save(save_path, dpi=200, writer="imagemagick")
    #     plt.close()

    # def plot_value_function(
    #     self,
    #     values: Dict,
    #     plot_max_values: bool,
    #     quiver: bool,
    #     over_actions: str,
    #     save_path: Union[str, None],
    #     fig=None,
    #     axes=None,
    #     caxes=None,
    # ) -> None:
    #     """
    #     Plot value function over environment.

    #     The multiroom states include non-positional state information (key posession),
    #     so plots can be constructed with this information averaged over. Alternatively
    #     multiple plots for each combination can be made. For cases with up to 2 keys,
    #     we construct the combinations. Beyond this only the average is plotted.

    #     Args:
    #         values: state-action values.
    #         plot_max_values: whether or not to plot colormap of values.
    #         quiver: whether or not to plot arrows with implied best action.
    #         over_actions: 'mean' or 'max'; how to flatten action dimension.
    #         save_path: path to save graphs.
    #     """
    #     if len(self._key_positions) <= 0:
    #         value_combinations = {}
    #         if fig is None and axes is None:
    #             fig, axes = plt.subplots(nrows=1, ncols=1)
    #         averaged_values_axis = axes
    #     # elif len(self._key_positions) <= 2:
    #     #     value_combinations = self._get_value_combinations(values=values)
    #     #     if fig is None and axes is None:
    #     #         fig, axes = plt.subplots(
    #     #             nrows=1 + 2 * len(self._key_positions), ncols=1
    #     #         )
    #     #     averaged_values_axis = axes[0]
    #     else:
    #         value_combinations = {}
    #         if fig is None and axes is None:
    #             fig, axes = plt.subplots(nrows=1, ncols=1)
    #         averaged_values_axis = axes

    #     caxes = caxes or [None for _ in range(1 + len(value_combinations))]

    #     fig.subplots_adjust(hspace=0.5)

    #     averaged_values = self._average_values_over_key_states(values=values)

    #     self._value_plot(
    #         fig=fig,
    #         ax=averaged_values_axis,
    #         values=averaged_values,
    #         plot_max_values=plot_max_values,
    #         quiver=quiver,
    #         over_actions=over_actions,
    #         subtitle="Positional Average",
    #         cax=caxes[0],
    #     )

    #     for i, (key_state, value_combination) in enumerate(value_combinations.items()):
    #         self._value_plot(
    #             fig=fig,
    #             ax=axes[i + 1],
    #             values=value_combination,
    #             plot_max_values=plot_max_values,
    #             quiver=quiver,
    #             over_actions=over_actions,
    #             subtitle=f"Key State: {key_state}",
    #             cax=caxes[i + 1],
    #         )

    #     if save_path is not None:
    #         fig.savefig(save_path, dpi=100)
    #         plt.close()

    # def _value_plot(
    #     self,
    #     fig,
    #     ax,
    #     values: Dict,
    #     plot_max_values: bool,
    #     quiver: bool,
    #     over_actions: str,
    #     subtitle: str,
    #     cax=None,
    # ):

    #     if plot_max_values:
    #         image, min_val, max_val = self._get_value_heatmap(
    #             values=values, over_actions=over_actions
    #         )
    #         im = ax.imshow(
    #             image, origin="lower", cmap=self._colormap, vmin=min_val, vmax=max_val
    #         )
    #         if cax is not None:
    #             cax.cla()
    #             fig.colorbar(im, ax=ax, cax=cax)
    #         else:
    #             fig.colorbar(im, ax=ax)

    #     if quiver:
    #         map_shape = self._env_skeleton().shape
    #         X, Y, arrow_x, arrow_y = self._get_quiver_data(
    #             map_shape=map_shape, values=values
    #         )
    #         ax.quiver(
    #             X,
    #             Y,
    #             arrow_x,
    #             arrow_y,
    #             color="red",
    #         )
    #     ax.title.set_text(subtitle)
    #     return ax

    # def _get_value_heatmap(
    #     self, values: Dict, over_actions: str
    # ) -> Tuple[np.ndarray, float, float]:
    #     """Heatmap of values over states."""
    #     environment_map = self._env_skeleton(rewards=False, doors=False, keys=False)

    #     if over_actions == constants.MAX:
    #         values = {k: max(v) for k, v in values.items()}
    #     elif over_actions == constants.MEAN:
    #         values = {k: np.mean(v) for k, v in values.items()}
    #     elif over_actions == constants.STD:
    #         values = {k: np.std(v) for k, v in values.items()}
    #     elif over_actions == constants.SELECT:
    #         values = values

    #     all_values = list(values.values())
    #     current_max_value = np.max(all_values)
    #     current_min_value = np.min(all_values)

    #     for state, value in values.items():
    #         # remove alpha from rgba in colormap return
    #         # normalise value for color mapping
    #         environment_map[state[::-1]] = self._colormap(
    #             (value - current_min_value) / (current_max_value - current_min_value)
    #         )[:-1]

    #     return environment_map, current_min_value, current_max_value

    # def _get_quiver_data(self, map_shape: Tuple[int], values: Dict) -> Tuple:
    #     """Get data for arrow quiver plot.

    #     Args:
    #         map_shape: map skeleton.
    #         values: state-action values.

    #     Returns:
    #         X: x part of meshgrid
    #         Y: y part of meshgrid
    #         arrow_x_directions: x component of arrow over grid
    #         arrow_y_directions: y component of arrow over grid
    #     """
    #     action_arrow_mapping = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}
    #     X, Y = np.meshgrid(
    #         np.arange(map_shape[1]),
    #         np.arange(map_shape[0]),
    #         indexing="ij",
    #     )
    #     # x, y size of map (discard rgb from environment_map)
    #     arrow_x_directions = np.zeros(map_shape[:-1][::-1])
    #     arrow_y_directions = np.zeros(map_shape[:-1][::-1])

    #     for state, action_values in values.items():
    #         action_index = np.argmax(action_values)
    #         action = action_arrow_mapping[action_index]
    #         arrow_x_directions[state] = action[0]
    #         arrow_y_directions[state] = action[1]

    #     return X, Y, arrow_x_directions, arrow_y_directions

    # def show_grid(self) -> np.ndarray:
    #     """Generate 2d array of current state of environment."""
    #     grid_state = copy.deepcopy(self._map)
    #     for reward in self._rewards.keys():
    #         grid_state[reward] = np.inf
    #     grid_state[tuple(self._agent_position)] = -1
    #     return grid_state
