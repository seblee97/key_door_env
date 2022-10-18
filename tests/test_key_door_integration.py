import argparse
import copy
import os
import random
import sys
import tempfile
import unittest
import warnings

import numpy as np
from key_door import curriculum_env, key_door_env

try:
    import matplotlib.pyplot as plt
    from key_door import visualisation_env
except:
    pass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--full",
    action="store_true",
    help="if flagged, test visualisations as well as basic functionality.",
)
parser.add_argument(
    "--save",
    action="store_true",
    help="if flagged, output files from tests are save.",
)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

TEST_MAP_PATH = os.path.join(
    FILE_PATH, "key_door_test_map_files", "key_door_sample_map.txt"
)
TEST_MAP_YAML_PATH = os.path.join(
    FILE_PATH, "key_door_test_map_files", "key_door_sample_map.yaml"
)
TEST_MAP_YAML2_PATH = os.path.join(
    FILE_PATH, "key_door_test_map_files", "key_door_sample_map2.yaml"
)


class TestIntegration(unittest.TestCase):
    """Test class for integration of functionality in key_door_env package."""

    def __init__(self, test_name: str, save_path: str):
        super().__init__(test_name)
        self._save_path = save_path

    def setUp(self):
        self._tabular_env = key_door_env.KeyDoorEnv(
            map_ascii_path=TEST_MAP_PATH,
            map_yaml_path=TEST_MAP_YAML_PATH,
            representation="agent_position",
            episode_timeout=1000,
        )

        self._pixel_env = key_door_env.KeyDoorEnv(
            map_ascii_path=TEST_MAP_PATH,
            map_yaml_path=TEST_MAP_YAML_PATH,
            representation="pixel",
            episode_timeout=1000,
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
            env.render(
                save_path=os.path.join(self._save_path, "test_render.pdf"),
                format="stationary",
            )

    def test_episode_visualisation(self):
        for env in self._envs:

            env = visualisation_env.VisualisationEnv(env=env)

            env.reset_environment()

            # random rollout
            while env.active:
                env.step(random.choice(env.action_space))

            video_file_name = os.path.join(self._save_path, "visualisation.mp4")
            env.visualise_episode_history(save_path=video_file_name, history="train")

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

            fig.savefig(os.path.join(self._save_path, "heatmap_visualisation.pdf"))

    def test_curriculum_without_visualisation(self):
        for env in self._envs:

            env = visualisation_env.VisualisationEnv(env=env)
            env = curriculum_env.CurriculumEnv(
                env=env, transitions=[TEST_MAP_YAML_PATH, TEST_MAP_YAML2_PATH]
            )

            env.reset_environment()

            # follow random policy
            while env.active:
                action = random.choice(env.action_space)
                env.step(action)

            next(env)

            # follow random policy
            while env.active:
                action = random.choice(env.action_space)
                env.step(action)

    def test_curriculum_with_visualisation(self):
        for env in self._envs:

            env = visualisation_env.VisualisationEnv(env=env)
            env = curriculum_env.CurriculumEnv(
                env=env, transitions=[TEST_MAP_YAML_PATH, TEST_MAP_YAML2_PATH]
            )

            env.reset_environment()

            # follow random policy
            while env.active:
                action = random.choice(env.action_space)
                env.step(action)

            video_file_name = os.path.join(self._save_path, "episode_curr_1.mp4")
            env.visualise_episode_history(save_path=video_file_name, history="train")

            next(env)

            # follow random policy
            while env.active:
                action = random.choice(env.action_space)
                env.step(action)

            video_file_name = os.path.join(self._save_path, "episode_curr_2.mp4")
            env.visualise_episode_history(save_path=video_file_name, history="train")


def get_suite(full: bool, save_path: str):
    model_tests = [
        "test_basic_setup",
        "test_basic_setup",
        "test_random_rollout",
        "test_curriculum_without_visualisation",
    ]
    if full:
        model_tests.extend(
            [
                "test_render",
                "test_episode_visualisation",
                "test_axis_heatmap_visualisation",
                "test_curriculum_with_visualisation",
            ]
        )
    suite = unittest.TestSuite()
    for model_test in model_tests:
        suite.addTest(TestIntegration(model_test, save_path))
    return suite


args = parser.parse_args()

if args.full:
    assert (
        "matplotlib" in sys.modules
    ), "To run full test suite, additional requirements need to be met. Please consult README."
else:
    if args.save:
        warnings.warn(
            "Saving is only defined for visualisation, so will have no effect unless"
            " --full flag is used and visualisation tests are run."
        )

runner = unittest.TextTestRunner(buffer=True, verbosity=1)

if args.save:
    save_path = os.path.join(FILE_PATH, "test_outputs")
    os.makedirs(save_path, exist_ok=True)
    runner.run(get_suite(full=args.full, save_path=save_path))
else:
    with tempfile.TemporaryDirectory() as tmpdir:
        runner.run(get_suite(full=args.full, save_path=tmpdir))
