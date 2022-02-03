import sys

sys.path.append("../")

from Grasping_Agent_multidiscrete import Grasp_Agent
from termcolor import colored
import torch
from collections import defaultdict

GAMMA = 0
SAVE_WEIGHTS = True


def main():

    for rand_seed in [122]:
        for lr in [0.001]:
            LOAD_PATH = "../DQN_RESNET_LR_0.001_OPTIM_ADAM_H_200_W_200_STEPS_35000_BUFFER_SIZE_2000_BATCH_SIZE_12_SEED_81_9_7_2020_9_52_weights.pt"

            FILE_SIZE = 12

            agent = Grasp_Agent(
                seed=rand_seed,
                load_path=LOAD_PATH,
                learning_rate=lr,
                depth_only=False,
                mem_size=100,
            )
            agent.optimizer.zero_grad()
            states = []
            actions = []
            rewards = []
            save_dic = defaultdict(list)
            states_full = False
            actions_full = False
            rewards_full = False
            number_saved = 0
            reward_counter = defaultdict(int)
            for episode in range(1, 101):
                state = agent.env.reset()
                if not states_full:
                    states.append(state)
                if len(states) == FILE_SIZE:
                    save_dic["states"] = states
                    states_full = True
                state = agent.transform_observation(state)
                print(
                    colored(
                        "CURRENT EPSILON: {}".format(agent.eps_threshold),
                        color="blue",
                        attrs=["bold"],
                    )
                )
                for step in range(1, 51):
                    print("#################################################################")
                    print(
                        colored(
                            "EPISODE {} STEP {}".format(episode, step),
                            color="white",
                            attrs=["bold"],
                        )
                    )
                    print("#################################################################")
                    action = agent.epsilon_greedy(state)
                    if not actions_full:
                        actions.append(action.item())
                    if len(actions) == FILE_SIZE:
                        save_dic["actions"] = actions
                        actions_full = True
                    env_action = agent.transform_action(action)
                    next_state, reward, done, _ = agent.env.step(
                        env_action, action_info=agent.last_action
                    )
                    if not rewards_full:
                        rewards.append(reward)
                    reward_counter[str(reward)] += 1
                    if len(rewards) == FILE_SIZE:
                        save_dic["rewards"] = rewards
                        rewards_full = True

                    if states_full and rewards_full and actions_full:
                        number_saved += 1
                        file_name = f"Data/grasping_data_{number_saved}.pt"
                        torch.save(save_dic, file_name)
                        print(f"Saved data to {file_name}.")
                        states = []
                        actions = []
                        rewards = []
                        states_full = False
                        actions_full = False
                        rewards_full = False
                        print("Current reward counter:")
                        for k, v in reward_counter.items():
                            print(f"{k}: {v}")
                    if not states_full:
                        states.append(next_state)
                    if len(states) == FILE_SIZE:
                        save_dic["states"] = states
                        states_full = True

                    agent.update_tensorboard(reward, env_action)
                    reward = torch.tensor([[reward]])

                    next_state = agent.transform_observation(next_state)
                    if GAMMA == 0.0:
                        agent.memory.push(state, action, reward)
                    else:
                        agent.memory.push(state, action, next_state, reward)

                    state = next_state

                    agent.learn()

            if SAVE_WEIGHTS:
                torch.save(
                    {
                        "step": agent.steps_done,
                        "model_state_dict": agent.policy_net.state_dict(),
                        "optimizer_state_dict": agent.optimizer.state_dict(),
                        "epsilon": agent.eps_threshold,
                        "greedy_rotations": agent.greedy_rotations,
                        "greedy_rotations_successes": agent.greedy_rotations_successes,
                        "random_rotations_successes": agent.random_rotations_successes,
                    },
                    agent.WEIGHT_PATH,
                )

                print("Saved checkpoint to {}.".format(agent.WEIGHT_PATH))

            print(f"Finished training (rand_seed = {rand_seed}).")
            agent.writer.close()
            agent.env.close()


if __name__ == "__main__":
    main()
