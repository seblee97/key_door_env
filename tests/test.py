from key_door import key_door_env

env = key_door_env.KeyDoorGridworld(
    map_ascii_path="sample_map.txt",
    map_yaml_path="sample_map.yaml",
    representation="agent_position",
)

env.render("test_map.pdf")
