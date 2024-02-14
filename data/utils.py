import numpy as np

# greater_gtreward_adjs = ["better", "more successfully"]
# greater_gtreward_adjs_val = ["more effectively"]
# less_gtreward_adjs = ["worse", "not as well"]
# less_gtreward_adjs_val = ["less successfully"]
# greater_speed_adjs = ["faster", "quicker", "swifter", "at a higher speed"]
# greater_speed_adjs_val = ["more quickly"]
# less_speed_adjs = ["slower", "more moderate", "more sluggish", "at a lower speed"]
# less_speed_adjs_val = ["more slowly"]
# greater_height_adjs = ["higher", "taller", "at a greater height"]
# greater_height_adjs_val = ["to a greater height"]
# less_height_adjs = ["lower", "shorter", "at a lesser height"]
# less_height_adjs_val = ["to a lower height"]
# greater_distance_adjs = ["further", "farther", "more distant"]
# greater_distance_adjs_val = ["less nearby"]
# less_distance_adjs = ["closer", "nearer", "more nearby"]
# less_distance_adjs_val = ["less distant"]
greater_gtreward_adjs = ["better", "more successfully"]
less_gtreward_adjs = ["worse", "not as well"]
greater_speed_adjs = ["faster", "quicker", "swifter", "at a higher speed"]
less_speed_adjs = ["slower", "more moderate", "more sluggish", "at a lower speed"]
greater_height_adjs = ["higher", "taller", "at a greater height"]
less_height_adjs = ["lower", "more down", "at a lesser height"]
greater_distance_adjs = ["further", "farther", "more distant"]
less_distance_adjs = ["closer", "nearer", "more nearby"]

height_nouns = ["stature", "height"]
speed_nouns = ["speed", "pace", "hasten your movement", "advance"]
distance_nouns = ["distance"]

GT_REWARD_MEAN = None
GT_REWARD_STD = None
SPEED_MEAN = None
SPEED_STD = None
HEIGHT_MEAN = None
HEIGHT_STD = None
DISTANCE_TO_BOTTLE_MEAN = None
DISTANCE_TO_BOTTLE_STD = None
DISTANCE_TO_CUBE_MEAN = None
DISTANCE_TO_CUBE_STD = None


def calc_and_set_global_vars(trajs):
    horizon = len(trajs[0])
    avg_gt_rewards = []
    avg_speeds = []
    avg_heights = []
    avg_distance_to_bottles = []
    avg_distance_to_cubes = []

    for traj in trajs:
        avg_gt_rewards.append(np.mean([gt_reward(traj[t]) for t in range(horizon)]))
        avg_speeds.append(np.mean([speed(traj[t]) for t in range(horizon)]))
        avg_heights.append(np.mean([height(traj[t]) for t in range(horizon)]))
        avg_distance_to_bottles.append(np.mean([distance_to_bottle(traj[t]) for t in range(horizon)]))
        avg_distance_to_cubes.append(np.mean([distance_to_cube(traj[t]) for t in range(horizon)]))

    global GT_REWARD_MEAN
    global GT_REWARD_STD
    global SPEED_MEAN
    global SPEED_STD
    global HEIGHT_MEAN
    global HEIGHT_STD
    global DISTANCE_TO_BOTTLE_MEAN
    global DISTANCE_TO_BOTTLE_STD
    global DISTANCE_TO_CUBE_MEAN
    global DISTANCE_TO_CUBE_STD
    global DISTANCE_DICT

    GT_REWARD_MEAN = np.mean(avg_gt_rewards)
    GT_REWARD_STD = np.std(avg_gt_rewards)
    SPEED_MEAN = np.mean(avg_speeds)
    SPEED_STD = np.std(avg_speeds)
    HEIGHT_MEAN = np.mean(avg_heights)
    HEIGHT_STD = np.std(avg_speeds)
    DISTANCE_TO_BOTTLE_MEAN = np.mean(avg_distance_to_bottles)
    DISTANCE_TO_BOTTLE_STD = np.std(avg_speeds)
    DISTANCE_TO_CUBE_MEAN = np.mean(avg_distance_to_cubes)
    DISTANCE_TO_CUBE_STD = np.std(avg_speeds)
    DISTANCE_DICT = {
        "gt_reward": [GT_REWARD_MEAN, GT_REWARD_STD],
        "speed": [SPEED_MEAN, SPEED_STD],
        "height": [HEIGHT_MEAN, HEIGHT_STD],
        "distance_to_bottle": [DISTANCE_TO_BOTTLE_MEAN, DISTANCE_TO_BOTTLE_STD],
        "distance_to_cube": [DISTANCE_TO_CUBE_MEAN, DISTANCE_TO_CUBE_STD]
    }


# NOTE: For this function, we produce commands that would change traj1 to traj2.
def generate_synthetic_comparisons_commands(traj1, traj2, feature_name=None, augmented_comps=None, validation=False,
                                            split='train'):
    horizon = len(traj1)

    value_func = {
        "gt_reward": gt_reward,
        "speed": speed,
        "height": height,
        "distance_to_bottle": distance_to_bottle,
        "distance_to_cube": distance_to_cube
    }
    ori_commands = {
        "gt_reward":
            [["Lift the cube " + w + "." for w in comps] for comps in
             [greater_gtreward_adjs, less_gtreward_adjs]],
        "speed":
            [["Move " + w + "." for w in comps] for comps in
             [greater_speed_adjs, less_speed_adjs]],
        "height":
            [["Move " + w + "." for w in comps] for comps in
             [greater_height_adjs, less_height_adjs]],
        "distance_to_bottle":
            # [["Move " + w + " from the bottle." for w in comps] for comps in
            #  [greater_distance_adjs, less_distance_adjs, greater_distance_adjs_val, less_distance_adjs_val]],
            [["Move " + w + " from the bottle." for w in greater_distance_adjs],
             ["Move " + w + " to the bottle." for w in less_distance_adjs]],
             # ["Move " + w + " from the bottle." for w in greater_distance_adjs_val],
             # ["Move " + w + " to the bottle." for w in less_distance_adjs_val]],
        "distance_to_cube":
            # [["Move " + w + " from the cube." for w in comps] for comps in
            #  [greater_distance_adjs, less_distance_adjs, greater_distance_adjs_val, less_distance_adjs_val]]
            [["Move " + w + " from the cube." for w in greater_distance_adjs],
             ["Move " + w + " to the cube." for w in less_distance_adjs]],
             # ["Move " + w + " from the cube." for w in greater_distance_adjs_val],
             # ["Move " + w + " to the cube." for w in less_distance_adjs_val]]
    }

    if feature_name is None:
        feature_names = ["gt_reward", "speed", "height", "distance_to_bottle", "distance_to_cube"]
    else:
        feature_names = [feature_name]

    commands = []
    for feature_name in feature_names:
        traj1_feature_values = [value_func[feature_name](traj1[t]) for t in range(horizon)]
        traj2_feature_values = [value_func[feature_name](traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) < np.mean(traj2_feature_values):
            comps = ori_commands[feature_name][0]
        else:
            comps = ori_commands[feature_name][1]
        if augmented_comps is not None:
            for comp in comps:
                if split == 'train':
                    num_comps_train = int(np.floor(len(augmented_comps[comp]) * 0.8))
                    new_comps = augmented_comps[comp][: num_comps_train]
                elif split == 'val':
                    num_comps_train = int(np.floor(len(augmented_comps[comp]) * 0.8))
                    num_comps = int(np.floor(len(augmented_comps[comp]) * 0.1))
                    new_comps = augmented_comps[comp][num_comps_train: num_comps_train + num_comps]
                elif split == 'test':
                    num_comps_train = int(np.floor(len(augmented_comps[comp]) * 0.8))
                    num_comps = int(np.floor(len(augmented_comps[comp]) * 0.1))
                    new_comps = augmented_comps[comp][num_comps_train + num_comps:]
                else:
                    raise NotImplemented("Split not implemented")
                commands.extend(new_comps)

    return commands
    # if feature_name == "gt_reward":
    #     traj1_feature_values = [gt_reward(traj1[t]) for t in range(horizon)]
    #     traj2_feature_values = [gt_reward(traj2[t]) for t in range(horizon)]
    #
    #     if np.mean(traj1_feature_values) < np.mean(traj2_feature_values):  # Here, we take the MEAN gt_reward
    #         ori_comps = ["Lift the cube " + w + "." for w in greater_gtreward_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #         return commands
    #     else:
    #         ori_comps = ["Lift the cube " + w + "." for w in less_gtreward_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #         return commands
    #
    # elif feature_name == "speed":
    #     traj1_feature_values = [speed(traj1[t]) for t in range(horizon)]
    #     traj2_feature_values = [speed(traj2[t]) for t in range(horizon)]
    #
    #     if np.mean(traj1_feature_values) < np.mean(traj2_feature_values):  # Here, we take the MEAN speed
    #         ori_comps = ["Move " + w + "." for w in greater_speed_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #         return commands
    #     else:
    #         ori_comps = ["Move " + w + "." for w in less_speed_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #         return commands
    #
    # elif feature_name == "height":
    #     traj1_feature_values = [height(traj1[t]) for t in range(horizon)]
    #     traj2_feature_values = [height(traj2[t]) for t in range(horizon)]
    #
    #     if np.mean(traj1_feature_values) < np.mean(traj2_feature_values):  # Here, we take the MEAN height
    #         ori_comps = ["Move " + w + "." for w in greater_height_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #         return commands
    #     else:
    #         ori_comps = ["Move " + w + "." for w in less_height_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #         return commands
    #
    # elif feature_name == "distance_to_bottle":
    #     traj1_feature_values = [distance_to_bottle(traj1[t]) for t in range(horizon)]
    #     traj2_feature_values = [distance_to_bottle(traj2[t]) for t in range(horizon)]
    #
    #     # TODO: Later, we can make this non-Markovian (e.g., the MINIMUM distance)
    #     if np.mean(traj1_feature_values) < np.mean(traj2_feature_values):  # Here, we take the MEAN distance
    #         ori_comps = ["Move " + w + " from the bottle." for w in greater_distance_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #         return commands
    #     else:
    #         # Move further from the bottle.
    #         ori_comps = ["Move " + w + " to the bottle." for w in less_distance_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #         return commands
    #
    # elif feature_name == "distance_to_cube":
    #     traj1_feature_values = [distance_to_cube(traj1[t]) for t in range(horizon)]
    #     traj2_feature_values = [distance_to_cube(traj2[t]) for t in range(horizon)]
    #
    #     # TODO: Later, we can make this non-Markovian (e.g., the FINAL distance)
    #     if np.mean(traj1_feature_values) < np.mean(traj2_feature_values):  # Here, we take the MEAN distance
    #         ori_comps = ["Move " + w + " from the cube." for w in greater_distance_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #         return commands
    #     else:
    #         ori_comps = ["Move " + w + " to the cube." for w in less_distance_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #         return commands


# NOTE: For this function, we produce commands that would change traj1 to traj2.
# We generate n comparisons per trajectory pair, where the proportion that are greaterly labeled
# is equal to the sigmoid of the difference in the feature values.
# IMPORTANT: User needs to have run calc_and_set_global_vars() at some point before this function.
def generate_noisyaugmented_synthetic_comparisons_commands(traj1, traj2, n_duplicates, feature_name=None,
                                                           augmented_comps=None, validation=False, split='train'):
    horizon = len(traj1)
    value_func = {
        "gt_reward": gt_reward,
        "speed": speed,
        "height": height,
        "distance_to_bottle": distance_to_bottle,
        "distance_to_cube": distance_to_cube
    }
    ori_commands = {
        "gt_reward":
            [["Lift the cube " + w + "." for w in comps] for comps in
             [greater_gtreward_adjs, less_gtreward_adjs]],
        "speed":
            [["Move " + w + "." for w in comps] for comps in
             [greater_speed_adjs, less_speed_adjs]],
        "height":
            [["Move " + w + "." for w in comps] for comps in
             [greater_height_adjs, less_height_adjs]],
        "distance_to_bottle":
            [["Move " + w + " from the bottle." for w in greater_distance_adjs],
             ["Move " + w + " to the bottle." for w in less_distance_adjs]],
        "distance_to_cube":
            [["Move " + w + " from the cube." for w in greater_distance_adjs],
             ["Move " + w + " to the cube." for w in less_distance_adjs]],
    }

    if feature_name is None:
        feature_names = ["gt_reward", "speed", "height", "distance_to_bottle", "distance_to_cube"]
    else:
        feature_names = [feature_name]

    total_commands = []
    for feature_name in feature_names:
        traj1_feature_values = [value_func[feature_name](traj1[t]) for t in range(horizon)]
        traj2_feature_values = [value_func[feature_name](traj2[t]) for t in range(horizon)]

        feature_diff = np.mean(traj2_feature_values) - np.mean(traj1_feature_values)
        greater_prob = 1 / (1 + np.exp(-feature_diff / DISTANCE_DICT[feature_name][1]))
        num_greater = int(np.around(n_duplicates * greater_prob))
        num_lesser = n_duplicates - num_greater

        commands = []
        for i in range(num_greater):
            comps = ori_commands[feature_name][0]
            if augmented_comps is not None:
                for comp in ori_commands[feature_name][0]:
                    if split == 'train':
                        num_comps_train = int(np.floor(len(augmented_comps[comp]) * 0.8))
                        new_comps = augmented_comps[comp][: num_comps_train]
                    elif split == 'val':
                        num_comps_train = int(np.floor(len(augmented_comps[comp]) * 0.8))
                        num_comps = int(np.floor(len(augmented_comps[comp]) * 0.1))
                        new_comps = augmented_comps[comp][num_comps_train: num_comps_train + num_comps]
                    elif split == 'test':
                        num_comps_train = int(np.floor(len(augmented_comps[comp]) * 0.8))
                        num_comps = int(np.floor(len(augmented_comps[comp]) * 0.1))
                        new_comps = augmented_comps[comp][num_comps_train + num_comps:]
                    else:
                        raise NotImplemented("Split not implemented")
                    commands.extend(new_comps)
            else:
                commands.extend(comps)
        for i in range(num_lesser):
            comps = ori_commands[feature_name][1]
            if augmented_comps is not None:
                for comp in ori_commands[feature_name][1]:
                    if split == 'train':
                        num_comps_train = int(np.floor(len(augmented_comps[comp]) * 0.8))
                        new_comps = augmented_comps[comp][: num_comps_train]
                    elif split == 'val':
                        num_comps_train = int(np.floor(len(augmented_comps[comp]) * 0.8))
                        num_comps = int(np.floor(len(augmented_comps[comp]) * 0.1))
                        new_comps = augmented_comps[comp][num_comps_train: num_comps_train + num_comps]
                    elif split == 'test':
                        num_comps_train = int(np.floor(len(augmented_comps[comp]) * 0.8))
                        num_comps = int(np.floor(len(augmented_comps[comp]) * 0.1))
                        new_comps = augmented_comps[comp][num_comps_train + num_comps:]
                    else:
                        raise NotImplemented("Split not implemented")
                    commands.extend(new_comps)
            else:
                commands.extend(comps)
        total_commands.extend(commands)

    return total_commands

    # if feature_name == "gt_reward":
    #     traj1_feature_values = [gt_reward(traj1[t]) for t in range(horizon)]
    #     traj2_feature_values = [gt_reward(traj2[t]) for t in range(horizon)]
    #
    #     feature_diff = np.mean(traj2_feature_values) - np.mean(traj1_feature_values)
    #     greater_prob = 1 / (1 + np.exp(-feature_diff / GT_REWARD_STD))
    #     num_greater = int(np.around(n_duplicates * greater_prob))
    #     num_lesser = n_duplicates - num_greater
    #
    #     commands = []
    #     for i in range(num_greater):
    #         ori_comps = ["Lift the cube " + w + "." for w in greater_gtreward_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #
    #     for i in range(num_lesser):
    #         ori_comps = ["Lift the cube " + w + "." for w in less_gtreward_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #
    #     return commands
    #
    # elif feature_name == "speed":
    #     traj1_feature_values = [speed(traj1[t]) for t in range(horizon)]
    #     traj2_feature_values = [speed(traj2[t]) for t in range(horizon)]
    #
    #     feature_diff = np.mean(traj2_feature_values) - np.mean(traj1_feature_values)
    #     greater_prob = 1 / (1 + np.exp(-feature_diff / SPEED_STD))
    #     num_greater = int(np.around(n_duplicates * greater_prob))
    #     num_lesser = n_duplicates - num_greater
    #
    #     commands = []
    #     for i in range(num_greater):
    #         ori_comps = ["Move " + w + "." for w in greater_speed_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #     for i in range(num_lesser):
    #         ori_comps = ["Move " + w + "." for w in less_speed_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #
    #     return commands
    #
    # elif feature_name == "height":
    #     traj1_feature_values = [height(traj1[t]) for t in range(horizon)]
    #     traj2_feature_values = [height(traj2[t]) for t in range(horizon)]
    #
    #     feature_diff = np.mean(traj2_feature_values) - np.mean(traj1_feature_values)
    #     greater_prob = 1 / (1 + np.exp(-feature_diff / HEIGHT_STD))
    #     num_greater = int(np.around(n_duplicates * greater_prob))
    #     num_lesser = n_duplicates - num_greater
    #
    #     commands = []
    #     for i in range(num_greater):
    #         ori_comps = ["Move " + w + "." for w in greater_height_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #     for i in range(num_lesser):
    #         ori_comps = ["Move " + w + "." for w in less_height_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #
    #     return commands
    #
    # elif feature_name == "distance_to_bottle":
    #     traj1_feature_values = [distance_to_bottle(traj1[t]) for t in range(horizon)]
    #     traj2_feature_values = [distance_to_bottle(traj2[t]) for t in range(horizon)]
    #
    #     feature_diff = np.mean(traj2_feature_values) - np.mean(traj1_feature_values)
    #     greater_prob = 1 / (1 + np.exp(-feature_diff / DISTANCE_TO_BOTTLE_STD))
    #     num_greater = int(np.around(n_duplicates * greater_prob))
    #     num_lesser = n_duplicates - num_greater
    #
    #     commands = []
    #     for i in range(num_greater):
    #         ori_comps = ["Move " + w + " from the bottle." for w in greater_distance_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #     for i in range(num_lesser):
    #         ori_comps = ["Move " + w + " to the bottle." for w in less_distance_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #
    #     return commands
    #
    # elif feature_name == "distance_to_cube":
    #     traj1_feature_values = [distance_to_cube(traj1[t]) for t in range(horizon)]
    #     traj2_feature_values = [distance_to_cube(traj2[t]) for t in range(horizon)]
    #
    #     feature_diff = np.mean(traj2_feature_values) - np.mean(traj1_feature_values)
    #     greater_prob = 1 / (1 + np.exp(-feature_diff / DISTANCE_TO_CUBE_STD))
    #     num_greater = int(np.around(n_duplicates * greater_prob))
    #     num_lesser = n_duplicates - num_greater
    #
    #     commands = []
    #     for i in range(num_greater):
    #         ori_comps = ["Move " + w + " from the cube." for w in greater_distance_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #     for i in range(num_lesser):
    #         ori_comps = ["Move " + w + " to the cube." for w in less_distance_adjs]
    #         commands.extend(ori_comps)
    #         if augmented_comps is not None:
    #             for comp in ori_comps:
    #                 commands.extend(augmented_comps[comp])
    #
    #     return commands


OBSERVATION_DIM = 65
ACTION_DIM = 4  # OSC_POSITION controller
OBJECT_STATE_DIM = 25
PROPRIO_STATE_DIM = 40


def gt_reward(gym_obs):
    assert len(gym_obs) == OBSERVATION_DIM or len(
        gym_obs) == OBSERVATION_DIM + ACTION_DIM  # Ensure that we are using the right observation (OBSERVATION_DIM) or observation+action (OBSERVATION_DIM+ACTION_DIM) space.
    object_state = gym_obs[0:OBJECT_STATE_DIM]
    proprio_state = gym_obs[OBJECT_STATE_DIM:OBSERVATION_DIM]

    reward = 0.0

    # Check success (sparse completion reward)
    cube_pos = object_state[0:3]
    cube_height = cube_pos[2]
    # print("cube_height:", cube_height)
    if cube_height > 0.9:  # refer to line 428 in lift.py. based on printing out the observation, the cube starts at a height of 0.81857747, and the bottle starts at a height of 0.88023976.
        reward = 2.25

    else:
        # reaching reward
        dist = distance_to_cube(gym_obs)
        reaching_reward = 1 - np.tanh(10.0 * dist)
        reward += reaching_reward
        # print("reaching_reward:", reaching_reward)

        # grasping reward
        is_grasping_cube = object_state[-1]
        if is_grasping_cube:
            reward += 0.25
    # print("total reward returned:", reward)
    return reward


def speed(gym_obs):
    assert len(gym_obs) == OBSERVATION_DIM or len(
        gym_obs) == OBSERVATION_DIM + ACTION_DIM  # Ensure that we are using the right observation (OBSERVATION_DIM) or observation+action (OBSERVATION_DIM+ACTION_DIM) space.
    object_state = gym_obs[0:OBJECT_STATE_DIM]
    proprio_state = gym_obs[OBJECT_STATE_DIM:OBSERVATION_DIM]

    # joint_vel = proprio_state[14:21]
    # gripper_qvel = proprio_state[34:40]
    # efc_vel = object_state[20]
    hand_vel = object_state[21:24]

    # return np.linalg.norm(gripper_qvel, 2)  # Returning L2 norm of the gripper q-velocities as the speed.
    # return efc_vel
    return np.linalg.norm(hand_vel, 2)  # Returning L2 norm of the hand velocities as the speed.


prev_eef_pos = np.zeros(3)


def finite_diff_speed(gym_obs):
    global prev_eef_pos
    object_state = gym_obs[0:OBJECT_STATE_DIM]
    proprio_state = gym_obs[OBJECT_STATE_DIM:OBSERVATION_DIM]
    eef_pos = proprio_state[21:24]
    vel = eef_pos - prev_eef_pos
    prev_eef_pos = eef_pos
    return np.linalg.norm(vel, 2)


def height(gym_obs):
    assert len(gym_obs) == OBSERVATION_DIM or len(
        gym_obs) == OBSERVATION_DIM + ACTION_DIM  # Ensure that we are using the right observation (OBSERVATION_DIM) or observation+action (OBSERVATION_DIM+ACTION_DIM) space.
    object_state = gym_obs[0:OBJECT_STATE_DIM]
    proprio_state = gym_obs[OBJECT_STATE_DIM:OBSERVATION_DIM]

    eef_pos = proprio_state[21:24]
    return eef_pos[2]  # Returning z-component of end-effector position as height.


def distance_to_bottle(gym_obs):
    assert len(gym_obs) == OBSERVATION_DIM or len(
        gym_obs) == OBSERVATION_DIM + ACTION_DIM  # Ensure that we are using the right observation (OBSERVATION_DIM) or observation+action (OBSERVATION_DIM+ACTION_DIM) space.
    object_state = gym_obs[0:OBJECT_STATE_DIM]
    proprio_state = gym_obs[OBJECT_STATE_DIM:OBSERVATION_DIM]

    gripper_to_bottle_pos = object_state[17:20]
    return np.linalg.norm(gripper_to_bottle_pos, 2)


def distance_to_cube(gym_obs):
    assert len(gym_obs) == OBSERVATION_DIM or len(
        gym_obs) == OBSERVATION_DIM + ACTION_DIM  # Ensure that we are using the right observation (OBSERVATION_DIM) or observation+action (OBSERVATION_DIM+ACTION_DIM) space.
    object_state = gym_obs[0:OBJECT_STATE_DIM]
    proprio_state = gym_obs[OBJECT_STATE_DIM:OBSERVATION_DIM]

    gripper_to_cube_pos = object_state[7:10]
    return np.linalg.norm(gripper_to_cube_pos, 2)
