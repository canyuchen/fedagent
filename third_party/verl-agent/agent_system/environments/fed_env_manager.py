from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import json
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if "extra.gamefile" in info:
            gamefile.append(info["extra.gamefile"])
        else:
            gamefile.append(None)
    return gamefile


def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if "extra.gamefile" in infos[i]:
            infos[i]["extra.gamefile"] = gamefile[i]
        else:
            infos[i]["extra.gamefile"] = None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, client_id=None, client_num=None):
        self.memory = SimpleMemory()
        self.client_id = client_id
        self.client_num = client_num
        super().__init__(envs, projection_f, config)

    def reset(self):
        # import pdb;pdb.set_trace()
        # breakpoint()
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size=len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(
            text_obs, self.envs.get_admissible_commands, init=True
        )
        return {"text": full_text_obs, "image": image_obs, "anchor": text_obs}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(
            text_actions, self.envs.get_admissible_commands
        )
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        next_observations = {
            "text": full_text_obs,
            "image": image_obs,
            "anchor": text_obs,
        }
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find("Your task is to: ")

            if task_start != -1:
                self.tasks.append(obs[task_start + len("Your task is to: ") :].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def build_text_obs(
        self,
        text_obs: List[str],
        admissible_actions: List[List[str]],
        init: bool = False,
    ) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(
                f"'{s}'" for s in admissible_actions[i] if s != "help"
            )

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                )
            else:
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length :]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                success["success_rate"].append(won_value)

                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]

        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }

    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == "rgb_array"
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode="tiny_rgb_array")
            observations = {
                "text": self.build_text_obs(infos, init=True),
                "image": obs,
                "anchor": obs,
            }
        else:
            self.pre_text_obs = obs
            observations = {
                "text": self.build_text_obs(infos, obs, init=True),
                "image": None,
                "anchor": obs,
            }
        self.memory.reset(batch_size=len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        self.memory.store(
            {
                "text_obs": self.pre_text_obs,
                "action": [self.ACTION_LOOKUP[act] for act in actions],
            }
        )
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode="tiny_rgb_array")
            next_observations = {
                "text": self.build_text_obs(infos),
                "image": next_obs,
                "anchor": next_obs,
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                "text": self.build_text_obs(infos, next_obs),
                "image": None,
                "anchor": next_obs,
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self, infos, text_obs: List[str] = None, init: bool = False
    ) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = (
                    SOKOBAN_VISUAL_TEMPLATE
                    if self.is_multi_modal
                    else SOKOBAN_TEMPLATE_NO_HIS.format(
                        current_observation=text_obs[i],
                    )
                )
            else:
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length :]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    if self.is_multi_modal:
                        action_history += (
                            f"\n[Action {step_number}: '{record['action']}']"
                        )
                    else:
                        action_history += f"\n[Text Observation {step_number}: \n{record['text_obs']}\nAction {step_number}: '{record['action']}']"

                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)

    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {
            "text": self.build_text_obs(infos),
            "image": obs,
            "anchor": obs.copy(),
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)

        # add text observation to next_observations
        next_observations["text"] = self.build_text_obs(infos)
        next_observations["anchor"] = next_observations["image"].copy()

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos: Tuple[Dict] = None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if "ezpoints" in self.config.env.env_name.lower():
                text_formula = (
                    "".join(str(element) for element in infos[i]["Formula"])
                    if infos[i] is not None
                    else ""
                )
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif "points24" in self.config.env.env_name.lower():
                text_formula = (
                    "".join(str(element) for element in infos[i]["Formula"])
                    if infos[i] is not None
                    else ""
                )
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif "numberline" in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, client_id=None, client_num=None):
        self.client_id = client_id
        self.client_num = client_num
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {
            "text": self.build_text_obs(obs, infos, init=True),
            "image": None,
            "anchor": obs.copy(),
        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size=len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        self.pre_text_obs = next_obs

        next_observations = {
            "text": self.build_text_obs(next_obs, infos),
            "image": None,
            "anchor": next_obs.copy(),
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1] == "Instruction:"
            tasks.append(parts[2])
        return tasks

    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index + 1 :])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs

    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions

    def build_text_obs(
        self, text_obs: List[str], infos: List[List[str]], init: bool = False
    ) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):

            available_actions = self.format_avail_actions(infos[i]["available_actions"])
            reformatted_available_actions = "\n".join(
                f"'{s}'," for s in available_actions
            )

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions,
                )
            else:
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length :]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions,
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions,
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                score_value = float(info["task_score"])
                success["success_rate"].append(won_value)
                success["webshop_task_score (not success_rate)"].append(score_value)
                return


class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self):
        text_obs, infos = self.envs.reset()

        self.supervisors = [info["supervisor"] for info in infos]
        self.memory.reset(batch_size=len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {"text": full_text_obs, "image": None, "anchor": text_obs}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({"text_obs": text_obs, "action": actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        next_observations = {"text": full_text_obs, "image": None, "anchor": text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                    supervisor_first_name=self.supervisors[i]["first_name"],
                    supervisor_last_name=self.supervisors[i]["last_name"],
                    supervisor_email=self.supervisors[i]["email"],
                    supervisor_phone_number=self.supervisors[i]["phone_number"],
                    task_description=self.tasks[i],
                )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length :]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"

                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                    supervisor_first_name=self.supervisors[i]["first_name"],
                    supervisor_last_name=self.supervisors[i]["last_name"],
                    supervisor_email=self.supervisors[i]["email"],
                    supervisor_phone_number=self.supervisors[i]["phone_number"],
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs


def fed_make_envs(config, client_id=None, client_num=None):
    """
    Create enviroments
    """
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import (
            build_gymcards_envs,
            gym_projection,
        )

        _envs = build_gymcards_envs(
            env_name=config.env.env_name,
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            is_train=True,
        )
        _val_envs = build_gymcards_envs(
            env_name=config.env.env_name,
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            is_train=False,
        )

        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import (
            build_alfworld_envs,
            alfworld_projection,
        )

        # breakpoint()
        if config.env.env_name == "alfworld/AlfredThorEnv":
            alf_config_path = os.path.join(
                os.path.dirname(__file__), "env_package/alfworld/configs/config_tw.yaml"
            )
        elif config.env.env_name == "alfworld/AlfredTWEnv":
            alf_config_path = os.path.join(
                os.path.dirname(__file__), "env_package/alfworld/configs/config_tw.yaml"
            )
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            "eval_dataset": "eval_in_distribution",  # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        
        # Read federated learning parameters from the config
        min_goals_per_client = config.data.get('min_goals_per_client', 100)
        val_batch_size = config.data.get('val_batch_size', 500)

        # Read partition_strategy and partition_kwargs from environment variables or the config
        partition_strategy = os.environ.get('PARTITION_STRATEGY', 'uniform')
        partition_kwargs = {}

        # First try reading the parameters from environment variables
        if os.environ.get('SIZE_STD'):
            partition_kwargs['size_std'] = float(os.environ.get('SIZE_STD'))
        # Dirichlet PreferencePartition uses OMEGA; legacy used TAU.
        # Read both; OMEGA takes precedence when both present.
        if os.environ.get('OMEGA'):
            partition_kwargs['omega'] = float(os.environ.get('OMEGA'))
        elif os.environ.get('TAU'):
            partition_kwargs['tau'] = float(os.environ.get('TAU'))
        if os.environ.get('SUCCESS_STD'):
            partition_kwargs['success_std'] = float(os.environ.get('SUCCESS_STD'))
        if os.environ.get('SHUFFLE_SEED'):
            partition_kwargs['shuffle_seed'] = int(os.environ.get('SHUFFLE_SEED'))
        # Env-level heterogeneity (env_disjoint) kwargs from env vars.
        # See docs/heterogeneity.md
        if os.environ.get('ENV_DIV'):
            partition_kwargs['env_div'] = float(os.environ.get('ENV_DIV'))
        if os.environ.get('FALLBACK'):
            partition_kwargs['fallback'] = os.environ.get('FALLBACK')
        if os.environ.get('HOLDOUT_FILE'):
            partition_kwargs['holdout_file'] = os.environ.get('HOLDOUT_FILE')
        
        # If the config has a `federated` section, prefer the parameters defined there
        if hasattr(config, 'federated') and hasattr(config.federated, 'data_sharding'):
            # Try reading the new `partition` format
            if hasattr(config.federated.data_sharding, 'partition'):
                partition_strategy = config.federated.data_sharding.partition.strategy
                if hasattr(config.federated.data_sharding.partition, 'kwargs'):
                    partition_kwargs = dict(config.federated.data_sharding.partition.kwargs)

            # Read the shuffle_seed parameter
            if hasattr(config.federated.data_sharding, 'shuffle_seed'):
                partition_kwargs['shuffle_seed'] = config.federated.data_sharding.shuffle_seed

        # ============================================================
        # Env-level heterogeneity: resolve holdout_file → holdout_scenes
        # See docs/heterogeneity.md
        # ============================================================
        if partition_strategy == 'env_disjoint':
            holdout_file = partition_kwargs.pop('holdout_file', None)
            if holdout_file:
                # Resolve relative path against project_root (subprocess cwd is verl-agent dir)
                if not os.path.isabs(holdout_file):
                    paths_yaml = './config/paths.yaml'
                    if os.path.exists(paths_yaml):
                        from omegaconf import OmegaConf
                        path_cfg = OmegaConf.to_container(OmegaConf.load(paths_yaml), resolve=True)
                        holdout_file = os.path.join(path_cfg['project_root'], holdout_file)
                if not os.path.exists(holdout_file):
                    raise FileNotFoundError(
                        f"[ENV-AlfWorld] holdout_file not found: {holdout_file}\n"
                        f"  cwd: {os.getcwd()}\n"
                        f"  Run `python tools/env_heterogeneity/gen_holdout_alfworld.py` to generate."
                    )
                with open(holdout_file) as f:
                    holdout_data = json.load(f)
                partition_kwargs['holdout_scenes'] = holdout_data.get('scenes', [])
                print(f"[ENV-AlfWorld] loaded {len(partition_kwargs['holdout_scenes'])} "
                      f"holdout scenes from {holdout_file}: {partition_kwargs['holdout_scenes']}")
            else:
                partition_kwargs.setdefault('holdout_scenes', [])
                print('[ENV-AlfWorld] WARNING: no holdout_file set; '
                      'OOD eval will not have unseen scenes')
            # Surface env_div / fallback values for visibility
            print(f"[ENV-AlfWorld] env_div={partition_kwargs.get('env_div', 0.7)} "
                  f"fallback={partition_kwargs.get('fallback', 'skip')}")

        _envs = build_alfworld_envs(
            alf_config_path,
            config.env.seed,
            config.data.train_batch_size,
            group_n,
            is_train=True,
            env_kwargs=env_kwargs,
            client_id=client_id,        # Federated parameter
            client_num=client_num,
            min_goals_per_client=min_goals_per_client,  # Minimum number of goals per client
            val_batch_size=val_batch_size,  # Validation set size
            partition_strategy=partition_strategy,  # Partition strategy
            **partition_kwargs  # Forward partition-strategy parameters
        )
        _val_envs = build_alfworld_envs(
            alf_config_path,
            config.env.seed + 1000,
            config.data.val_batch_size,
            1,
            is_train=False,
            env_kwargs=env_kwargs,
            client_id=client_id,        # Federated parameter
            client_num=client_num,
            min_goals_per_client=min_goals_per_client,  # Minimum number of goals per client
            val_batch_size=val_batch_size  # Validation set size
        )

        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config,client_id=client_id, client_num=client_num)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config,client_id=client_id, client_num=client_num)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import (
            build_sokoban_envs,
            sokoban_projection,
        )

        env_kwargs = {
            "dim_room": config.env.sokoban.dim_room,
            "num_boxes": config.env.sokoban.num_boxes,
            "max_steps": config.env.max_steps,
            "search_depth": config.env.sokoban.search_depth,
        }
        _envs = build_sokoban_envs(
            config.env.seed,
            config.data.train_batch_size,
            group_n,
            mode=config.env.sokoban.mode,
            is_train=True,
            env_kwargs=env_kwargs,
        )
        _val_envs = build_sokoban_envs(
            config.env.seed + 1000,
            config.data.val_batch_size,
            1,
            mode=config.env.sokoban.mode,
            is_train=False,
            env_kwargs=env_kwargs,
        )

        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import (
            build_webshop_envs,
            webshop_projection,
        )

        if config.env.webshop.use_small:
            file_path = os.path.join(
                os.path.dirname(__file__),
                "env_package/webshop/webshop/data/items_shuffle_1000.json",
            )
            attr_path = os.path.join(
                os.path.dirname(__file__),
                "env_package/webshop/webshop/data/items_ins_v2_1000.json",
            )
        else:
            file_path = os.path.join(
                os.path.dirname(__file__),
                "env_package/webshop/webshop/data/items_shuffle.json",
            )
            attr_path = os.path.join(
                os.path.dirname(__file__),
                "env_package/webshop/webshop/data/items_ins_v2.json",
            )
        
        # Read the shuffle_seed parameter
        shuffle_seed = 42  # Default value
        if os.environ.get('SHUFFLE_SEED'):
            shuffle_seed = int(os.environ.get('SHUFFLE_SEED'))
        elif hasattr(config, 'federated') and hasattr(config.federated, 'data_sharding'):
            if hasattr(config.federated.data_sharding, 'get'):
                shuffle_seed = config.federated.data_sharding.get('shuffle_seed', 42)
            else:
                shuffle_seed = getattr(config.federated.data_sharding, 'shuffle_seed', 42)
        elif hasattr(config, 'data') and hasattr(config.data, 'get'):
            shuffle_seed = config.data.get('shuffle_seed', 42)
        elif hasattr(config, 'data'):
            shuffle_seed = getattr(config.data, 'shuffle_seed', 42)

        env_kwargs = {
            "observation_mode": "text",
            "num_products": None,
            "human_goals": config.env.webshop.human_goals,
            "file_path": file_path,
            "attr_path": attr_path,
            "shuffle_seed": shuffle_seed,  # shuffle_seed parameter
        }

        # ============================================================
        # Env-level heterogeneity (distractor_disjoint partition)
        # See docs/heterogeneity.md
        # Triggered when PARTITION_STRATEGY env var is "distractor_disjoint",
        # set by core/fed/script_builder.py from yaml partition.strategy.
        # ============================================================
        partition_strategy_env = os.environ.get('PARTITION_STRATEGY', 'uniform')
        if partition_strategy_env == 'distractor_disjoint':
            from agent_system.environments.partition_strategy import (
                _distractor_disjoint_partition_webshop,
            )
            # Lower-bound check on SEARCH_RETURN_N for env-level experiments.
            # Default 50 (from engine.py) is too small once a per-client catalog
            # filter strips out ~30% of distractor ASINs from BM25 hits — the page
            # can drop below PRODUCT_WINDOW=10 in pathological cases.
            srn = int(os.environ.get('WEBSHOP_SEARCH_RETURN_N', 50))
            if srn < 100:
                raise ValueError(
                    f"[ENV-LEVEL] env-level experiments require WEBSHOP_SEARCH_RETURN_N >= 100; "
                    f"current = {srn}. Set search_return_n in yaml partition.kwargs (recommended: 200)."
                )
            env_div_v = float(os.environ.get('ENV_DIV', 0.7))
            keep_ratio_v = float(os.environ.get('KEEP_RATIO', 0.7))
            holdout_file = os.environ.get('HOLDOUT_FILE', '')
            holdout_distractor_asins = []
            if holdout_file:
                # fail-loud: an empty holdout ("not configured") is tolerable, but a "wrong path" must be surfaced
                # script_builder should already have resolved the relative path to an absolute one; this is defense in depth
                if not os.path.exists(holdout_file):
                    raise FileNotFoundError(
                        f"[ENV-LEVEL] HOLDOUT_FILE specified but not found: {holdout_file}\n"
                        f"  cwd: {os.getcwd()}\n"
                        f"  Either fix yaml partition.kwargs.holdout_file path "
                        f"or run `python tools/env_heterogeneity/gen_holdout_webshop.py`."
                    )
                with open(holdout_file) as f:
                    holdout_distractor_asins = json.load(f).get('asins', [])
                print(f'[ENV-LEVEL] loaded {len(holdout_distractor_asins)} holdout '
                      f'distractor ASINs from {holdout_file}')
            else:
                print('[ENV-LEVEL] WARNING: no HOLDOUT_FILE set; '
                      'OOD eval will not have unseen distractors')
            with open(file_path) as f:
                _products = json.load(f)
            with open(attr_path) as f:
                _ins = json.load(f)
            catalog_asins = _distractor_disjoint_partition_webshop(
                products=_products,
                ins=_ins,
                client_id=client_id,
                client_num=client_num,
                env_div=env_div_v,
                keep_ratio=keep_ratio_v,
                holdout_distractor_asins=holdout_distractor_asins,
                base_seed=42,
            )
            env_kwargs['catalog_filter_asins'] = catalog_asins
            print(f'[ENV-LEVEL] client {client_id}/{client_num}: '
                  f'env_div={env_div_v} keep_ratio={keep_ratio_v} '
                  f'|catalog|={len(catalog_asins)}')

        # ============================================================
        # Catalog-Split: per-client target floor distractor disjoint
        # See docs/heterogeneity.md
        # Differs from v4 (`distractor_disjoint`):
        #   - task partition: uniform_partition 100/client (matches main exp)
        #   - env partition: protects per-client target ASINs only (~50-80),
        #                    distractor_pool ≈ 920 (vs v4's 585 shared by all clients)
        #   - returns (catalog_asins, client_goal_idxs); webshop/envs.py uses
        #     client_goal_idxs to set self.goal_idxs (no longer hardcodes range(500, len(goals))).
        # ============================================================
        if partition_strategy_env == 'catalog_split':
            from agent_system.environments.partition_strategy import (
                _distractor_disjoint_partition_webshop_v5,
            )
            srn = int(os.environ.get('WEBSHOP_SEARCH_RETURN_N', 50))
            if srn < 100:
                raise ValueError(
                    f"[ENV-LEVEL Catalog-Split] env-level experiments require WEBSHOP_SEARCH_RETURN_N >= 100; "
                    f"current = {srn}. Set search_return_n in yaml partition.kwargs (recommended: 200)."
                )
            env_div_v = float(os.environ.get('ENV_DIV', 0.7))
            keep_ratio_v = float(os.environ.get('KEEP_RATIO', 0.7))
            min_goals_per_client = int(os.environ.get('MIN_GOALS_PER_CLIENT',
                                              config.data.get('min_goals_per_client', 100)))
            holdout_file = os.environ.get('HOLDOUT_FILE', '')
            holdout_distractor_asins = []
            if holdout_file:
                if not os.path.exists(holdout_file):
                    raise FileNotFoundError(
                        f"[ENV-LEVEL Catalog-Split] HOLDOUT_FILE specified but not found: {holdout_file}"
                    )
                with open(holdout_file) as f:
                    holdout_distractor_asins = json.load(f).get('asins', [])
                print(f'[ENV-LEVEL Catalog-Split] loaded {len(holdout_distractor_asins)} holdout distractor ASINs')
            with open(file_path) as f:
                _products = json.load(f)
            with open(attr_path) as f:
                _ins = json.load(f)
            catalog_asins, client_goal_idxs = _distractor_disjoint_partition_webshop_v5(
                products=_products,
                ins=_ins,
                client_id=client_id,
                client_num=client_num,
                min_goals_per_client=min_goals_per_client,
                env_div=env_div_v,
                keep_ratio=keep_ratio_v,
                holdout_distractor_asins=holdout_distractor_asins,
                base_seed=42,
            )
            env_kwargs['catalog_filter_asins'] = catalog_asins
            # webshop/envs.py reads this to set self.goal_idxs (replaces hardcoded list(range(500, len(goals))))
            env_kwargs['client_goal_idxs'] = client_goal_idxs
            print(f'[ENV-LEVEL Catalog-Split] client {client_id}/{client_num}: '
                  f'env_div={env_div_v} keep_ratio={keep_ratio_v} '
                  f'|catalog|={len(catalog_asins)} |goal_idxs|={len(client_goal_idxs)}')

        # ============================================================
        # Transition-level env heterogeneity: Lookalike Injection (lookalike adversarial)
        # See docs/heterogeneity.md
        #   - task partition: uniform
        #   - env partition: per-client extra_products (lookalike attack on
        #                    one attribute dimension; price/color/...)
        # ============================================================
        if partition_strategy_env == 'lookalike_injection':
            from agent_system.environments.partition_strategy import (
                _lookalike_injection_partition_webshop,
            )
            n_variants = int(os.environ.get('N_VARIANTS', 2))
            extra_products = _lookalike_injection_partition_webshop(
                client_id=client_id,
                client_num=client_num,
                N=n_variants,
                base_seed=42,
            )
            env_kwargs['extra_products'] = extra_products

        # ============================================================
        # Transition-level env heterogeneity: search-engine TYPE swap
        # 4 variants break different baseline-policy assumptions while
        # preserving reward gradient.
        # ============================================================
        if partition_strategy_env == 'rank_wrapper':
            from agent_system.environments.partition_strategy import (
                _rank_wrapper_partition_webshop,
            )
            n_variants = int(os.environ.get('N_VARIANTS', 4))
            search_cfg = _rank_wrapper_partition_webshop(
                client_id=client_id, client_num=client_num,
                N=n_variants, base_seed=42,
            )
            env_kwargs['search_engine_variant'] = search_cfg

        # ============================================================
        # Transition-level env heterogeneity: BM25 Reweighting (in-memory BM25 variants)
        # See docs/heterogeneity.md
        #   - task partition: uniform (no catalog filter, no goal_idxs override)
        #   - env partition: per-client InMemoryBM25Searcher with (fields, k1, b)
        # ============================================================
        if partition_strategy_env == 'bm25_variant':
            from agent_system.environments.partition_strategy import (
                _bm25_variant_partition_webshop,
            )
            n_variants = int(os.environ.get('N_VARIANTS', 4))
            bm25_cfg = _bm25_variant_partition_webshop(
                client_id=client_id,
                client_num=client_num,
                N=n_variants,
                base_seed=42,
            )
            env_kwargs['bm25_in_memory_config'] = bm25_cfg
        # Read the min_goals_per_client parameter from the config.
        # During client training this comes from the `data` config; on the server side it comes from the `federated` config.

        min_goals_per_client = config.data.get('min_goals_per_client', 100)

        # Read the val_batch_size parameter from the config
        val_batch_size = config.data.get('val_batch_size', 500)

        # partition_strategy was already read from the env var above, before the distractor_disjoint branch
        # (variable name partition_strategy_env); reuse it directly here to avoid re-reading and
        # conflicting with the yaml fallback chain.
        if partition_strategy_env != 'uniform':
            partition_strategy = partition_strategy_env
        elif hasattr(config, 'federated') and hasattr(config.federated, 'data_sharding'):
            partition_strategy = config.federated.data_sharding.get('partition_strategy',
                                                                  config.data.get('partition_strategy', 'uniform'))
        else:
            partition_strategy = config.data.get('partition_strategy', 'uniform')
        
        # Read the sharding-strategy parameters from the config.
        # Precedence: environment variables first, then federated.data_sharding, then data.
        tau = None
        size_std = None
        shuffle_seed = 42  # Default value

        # First try reading from environment variables
        if os.environ.get('TAU'):
            tau = float(os.environ.get('TAU'))
        elif hasattr(config, 'federated') and hasattr(config.federated, 'data_sharding'):
            tau = config.federated.data_sharding.get('tau',
                                                   config.data.get('tau', 0.3))
        else:
            tau = config.data.get('tau', 0.3)

        # Read the size_std parameter (used by the coverage strategy)
        if os.environ.get('SIZE_STD'):
            size_std = float(os.environ.get('SIZE_STD'))
        elif hasattr(config, 'federated') and hasattr(config.federated, 'data_sharding'):
            size_std = config.federated.data_sharding.get('size_std', 
                                                        config.data.get('size_std', 150))
        else:
            size_std = config.data.get('size_std', 150)
        if os.environ.get('SUCCESS_STD'):
            success_std = float(os.environ.get('SUCCESS_STD'))
        elif hasattr(config, 'federated') and hasattr(config.federated, 'data_sharding'):
            success_std = config.federated.data_sharding.get('success_std', 
                                                        config.data.get('success_std', 0.1))
        else:
            success_std = config.data.get('success_std', 0.1)

        # Read the shuffle_seed parameter
        if os.environ.get('SHUFFLE_SEED'):
            shuffle_seed = int(os.environ.get('SHUFFLE_SEED'))
        elif hasattr(config, 'federated') and hasattr(config.federated, 'data_sharding'):
            if hasattr(config.federated.data_sharding, 'get'):
                shuffle_seed = config.federated.data_sharding.get('shuffle_seed', 42)
            else:
                shuffle_seed = getattr(config.federated.data_sharding, 'shuffle_seed', 42)
        elif hasattr(config, 'data') and hasattr(config.data, 'get'):
            shuffle_seed = config.data.get('shuffle_seed', 42)
        elif hasattr(config, 'data'):
            shuffle_seed = getattr(config.data, 'shuffle_seed', 42)
        else:
            shuffle_seed = 42


        _envs = build_webshop_envs(
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            is_train=True,
            env_kwargs=env_kwargs,
            client_id=client_id,  # Pass through client_id
            client_num=client_num,
            min_goals_per_client=min_goals_per_client,  # Pass through min_goals_per_client
            val_batch_size=val_batch_size,  # Pass through val_batch_size
            partition_strategy=partition_strategy,  # Pass through partition_strategy
            tau=tau,  # tau parameter
            size_std=size_std,  # size_std parameter
            success_std=success_std,  # success_std parameter
        )
        # Env-level heterogeneity: val SimServer runs on FULL 1000 catalog
        # (catalog_filter_asins=None), decoupled from per-client train partition.
        # Without this override, val/success_rate at step 0 measures aggregated model
        # on each client's filtered catalog → cross-env_div curves not comparable.
        # See docs/heterogeneity.md
        # Force val to use the standard env (full Lucene + full 1000 catalog),
        # decoupled from per-client train heterogeneity. Aligns with the
        # Environment-level heterogeneity section of docs/heterogeneity.md.
        val_env_kwargs = {
            **env_kwargs,
            'catalog_filter_asins': None,
            'bm25_in_memory_config': None,  # force val -> default LuceneSearcher
            'extra_products': None,         # Lookalike Injection: drop lookalikes for val
            'search_engine_variant': None,  # search variant: val uses default Lucene
        }
        _val_envs = build_webshop_envs(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            is_train=False,
            env_kwargs=val_env_kwargs,
            client_id=client_id,  # Pass through client_id
            client_num=client_num,
            min_goals_per_client=min_goals_per_client,  # Pass through min_goals_per_client
            val_batch_size=val_batch_size,  # Pass through val_batch_size
            partition_strategy=partition_strategy,  # Pass through partition_strategy
            tau=tau,  # tau parameter
            size_std=size_std,  # size_std parameter
            success_std=success_std,  # success_std parameter
        )

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config, client_id=client_id, client_num=client_num)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config, client_id=client_id, client_num=client_num)
        import time

        time.sleep(
            (config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1
        )  # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import (
            build_appworld_envs,
            appworld_projection,
        )

        _envs = build_appworld_envs(
            dataset_name="train",
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            start_server_id=0,
        )
        _val_envs = build_appworld_envs(
            dataset_name="test_normal",
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            start_server_id=config.data.train_batch_size * group_n,
        )

        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
