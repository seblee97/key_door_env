import random

import matplotlib.pyplot as plt
import numpy as np
from key_door import key_door_env

env = key_door_env.KeyDoorGridworld(
    map_ascii_path="test_map_files/sample_map.txt",
    map_yaml_path="test_map_files/sample_map.yaml",
    representation="pixel",
)

env.render("test_map.pdf", format="stationary")

env.reset_environment()

# follow random policy
for i in range(1000):
    action = random.choice(env.action_space)
    env.step(action)

# env.visualise_episode_history("test_vid.mp4")

random_value_function = {p: np.random.random() for p in env.positional_state_space}
env.plot_heatmap_over_env(random_value_function, save_name="text_vals.pdf")

fig, axs = plt.subplots(3, 2)

for i in range(3):
    for j in range(2):
        random_value_function = {
            p: np.random.random() for p in env.positional_state_space
        }
        env.plot_heatmap_over_env(random_value_function, fig=fig, ax=axs[i, j])
        axs[i, j].set_title(f"{i}_{j}")

fig.tight_layout()
fig.savefig("test_sub.pdf")
