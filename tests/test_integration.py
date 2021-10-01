import argparse
import copy
import os
import random
import sys
import tempfile
import unittest

import numpy as np
import yaml
from key_door import key_door_env

try:
    import matplotlib.pyplot as plt
    from key_door import visualisation_env
except:
    pass


FILE_PATH = os.path.dirname(os.path.abspath(__file__))

TEST_MAP_PATH = os.path.join(FILE_PATH, "test_map_files", "sample_map.txt")
TEST_MAP_YAML_PATH = os.path.join(FILE_PATH, "test_map_files", "sample_map.yaml")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--full",
    action="store_true",
    help="if flagged, test visualisations as well as basic functionality.",
)


class TestIntegration(unittest.TestCase):
    """Test class for integration of functionality in key_door_env package."""

    def setUp(self):
        self._tabular_env = key_door_env.KeyDoorGridworld(
            map_ascii_path=TEST_MAP_PATH,
            map_yaml_path=TEST_MAP_YAML_PATH,
            representation="agent_position",
        )

        self._pixel_env = key_door_env.KeyDoorGridworld(
            map_ascii_path=TEST_MAP_PATH,
            map_yaml_path=TEST_MAP_YAML_PATH,
            representation="pixel",
        )

        self._envs = [self._tabular_env, self._pixel_env]

    def test_basic_setup(self):
        for env in self._envs:
            self.assertFalse(env.active)
            with self.assertRaises(AssertionError):
                env.step(random.choice(env.action_space))

    def test_random_rollout(self):
        for env in self._envs:
            env.reset_environment()
            for i in range(100):
                env.step(random.choice(env.action_space))

            non_wall_counts = copy.deepcopy(env.visitation_counts)
            non_wall_counts[non_wall_counts < 0] = 0

            # Note: 100 steps = 101 states
            self.assertEqual(len(env.train_episode_history), 101)
            self.assertEqual(len(env.train_episode_position_history), 101)
            self.assertEqual(np.sum(non_wall_counts), 101)
            self.assertEqual(env.episode_step_count, 100)

    def test_render(self):
        for env in self._envs:
            env = visualisation_env.VisualisationEnv(env=env)
            with tempfile.NamedTemporaryFile() as tmp:
                env.render(save_path=tmp, format="stationary")

    def test_episode_visualisation(self):
        for env in self._envs:

            env = visualisation_env.VisualisationEnv(env=env)

            env.reset_environment()

            # random rollout
            for i in range(100):
                env.step(random.choice(env.action_space))

            with tempfile.TemporaryDirectory() as tmpdir:
                video_file_name = os.path.join(tmpdir, "t.mp4")
                env.visualise_episode_history(
                    save_path=video_file_name, history="train"
                )

    def test_axis_heatmap_visualisation(self):
        for env in self._envs:

            env = visualisation_env.VisualisationEnv(env=env)

            fig, axs = plt.subplots(3, 2)

            for i in range(3):
                for j in range(2):
                    random_value_function = {
                        p: np.random.random() for p in env.positional_state_space
                    }
                    env.plot_heatmap_over_env(
                        random_value_function, fig=fig, ax=axs[i, j]
                    )
                    axs[i, j].set_title(f"{i}_{j}")

            fig.tight_layout()

            with tempfile.NamedTemporaryFile() as tmp:
                fig.savefig(tmp)


def get_suite(full: bool):
    model_tests = ["test_basic_setup", "test_basic_setup", "test_random_rollout"]
    if full:
        model_tests.extend(
            [
                "test_render",
                "test_episode_visualisation",
                "test_axis_heatmap_visualisation",
            ]
        )
    return unittest.TestSuite(map(TestIntegration, model_tests))


args = parser.parse_args()

if args.full:
    assert (
        "matplotlib" in sys.modules
    ), "To run full test suite, additional requirements need to be met. Please consult README."

runner = unittest.TextTestRunner(buffer=True, verbosity=1)
runner.run(get_suite(full=args.full))
