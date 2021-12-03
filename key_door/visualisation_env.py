from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from key_door import constants, wrapper

try:
    import cv2
    import matplotlib
    from matplotlib import cm
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except:
    raise AssertionError(
        "To use visualisation wrapper, further package requirements need to be satisfied. Please consult README."
    )


class VisualisationEnv(wrapper.Wrapper):

    COLORMAP = cm.get_cmap("plasma")

    def __init__(self, env):
        super().__init__(env=env)

    def render(
        self,
        save_path: Optional[str] = None,
        dpi: Optional[int] = 60,
        format: str = "state",
    ) -> None:
        if format == constants.STATE:
            assert (
                self._env.active
            ), "To render map with state, environment must be active."
            "call reset_environment() to reset environment and make it active."
            "Else render stationary environment skeleton using format='stationary'"
        if save_path:
            fig = plt.figure()
            plt.imshow(
                self._env._env_skeleton(
                    rewards=format, keys=format, doors=format, agent=format
                ),
                origin="lower",
            )
            fig.savefig(save_path, dpi=dpi)
        else:
            plt.imshow(
                self._env.env_skeleton(
                    rewards=format, keys=format, doors=format, agent=format
                ),
                origin="lower",
            )

    def visualise_episode_history(
        self, save_path: str, history: Union[str, List[np.ndarray]] = "train"
    ) -> None:
        """Produce video of episode history.

        Args:
            save_path: name of file to be saved.
            history: "train", "test" to plot train or test history, else provide an independent history.
        """
        if isinstance(history, str):
            if history == constants.TRAIN:
                history = self._env.train_episode_history
            elif history == constants.TEST:
                history = self._env.test_episode_history

        SCALING = 20
        FPS = 30

        map_shape = history[0].shape
        frameSize = (SCALING * map_shape[1], SCALING * map_shape[0])

        out = cv2.VideoWriter(
            filename=save_path,
            fourcc=cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            fps=FPS,
            frameSize=frameSize,
        )

        for frame in history:
            bgr_frame = frame[..., ::-1].copy()
            flipped_frame = np.flip(bgr_frame, 0)
            scaled_up_frame = np.kron(flipped_frame, np.ones((SCALING, SCALING, 1)))
            out.write((scaled_up_frame * 255).astype(np.uint8))

        out.release()

    def plot_heatmap_over_env(
        self,
        heatmap: Dict[Tuple[int, int], float],
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        save_name: Optional[str] = None,
    ) -> None:
        assert (
            ax is not None and fig is not None
        ) or save_name is not None, "Either must provide axis to plot heatmap over,"
        "r file name to save separate figure."
        environment_map = self._env.env_skeleton(
            rewards=None, keys=None, doors=None, agent=None
        )

        all_values = list(heatmap.values())
        current_max_value = np.max(all_values)
        current_min_value = np.min(all_values)

        for position, value in heatmap.items():
            # remove alpha from rgba in colormap return
            # normalise value for color mapping
            environment_map[position[::-1]] = self.COLORMAP(
                (value - current_min_value) / (current_max_value - current_min_value)
            )[:-1]

        if save_name is not None:
            fig = plt.figure()
            plt.imshow(environment_map, origin="lower", cmap=self.COLORMAP)
            plt.colorbar()
            fig.savefig(save_name, dpi=60)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(environment_map, origin="lower", cmap=self.COLORMAP)
            fig.colorbar(im, ax=ax, cax=cax, orientation="vertical")

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
