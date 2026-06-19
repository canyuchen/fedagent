from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import re
import json
import time
from datetime import datetime
import ray
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
import hashlib


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    """
    AlfWorld Environment Manager with trajectory saving capabilities.
    
    This manager handles AlfWorld environments and can save trajectories in JSON format.
    Trajectories are saved with the following structure:
    
    {
        "task_info": {
            "task_id": "alfworld_timestamp_env_idx",
            "intent": "task description",
            "environment_name": "alfworld", 
            "actions_count": number_of_actions,
            "gamefile": "gamefile_path"
        },
        "traj_info": {
            "success": boolean,
            "duration": float,
            "timestamp": "ISO timestamp"
        },
        "trajectory": [
            {
                "role": "environment",
                "observation_text": "text observation",
                "observation_image": "image observation",
                "step_id": int,
                "reward": float,
                "done": boolean
            },
            {
                "role": "agent",
                "action": "original text action",
                "action_parsed": "parsed action",
                "step_id": int
            },
            ...
        ]
    }
    
    Configuration:
    - config.env.save_trajectories: Enable/disable trajectory saving (default: True)
    - config.env.trajectory_save_dir: Directory to save trajectories (default: /scratch/czr/agentscale/dataset/alfworld/online_trajs)
    
    Usage Example:
        # Enable trajectory saving in config
        config.env.save_trajectories = True
        config.env.trajectory_save_dir = "/path/to/save/trajectories"
        
        # Create environment manager
        env_manager = AlfWorldEnvironmentManager(envs, projection_f, config)
        
        # Trajectories will be automatically saved when episodes complete
        # or when the environment manager is destroyed
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
        
        # Trajectory saving setup
        self.save_trajectories = getattr(config.env, 'save_trajectories', True)
        if self.save_trajectories:
            self.trajectory_save_dir = getattr(config.env, 'trajectory_save_dir', None)
            if self.trajectory_save_dir is None:
                raise ValueError("trajectory_save_dir must be set when save_trajectories is True")
        
        if self.save_trajectories:
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
            
        # Initialize trajectory tracking
        self.trajectories = []
        self.trajectory_step_counts = []
        self.trajectory_start_times = []
    
    def reset(self):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        # Initialize trajectories for each environment
        if self.save_trajectories:
            self._init_trajectories(text_obs, image_obs, infos)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos

    def _init_trajectories(self, text_obs, image_obs, infos):
        """Initialize trajectory tracking for each environment"""
        num_envs = len(text_obs)
        self.trajectories = []
        self.trajectory_step_counts = []
        self.trajectory_start_times = []
        self.trajectory_saved = []  # Track which trajectories have been saved
        
        for i in range(num_envs):
            # Create unique task ID for AlfWorld using gamefile
            if i < len(self.gamefile) and self.gamefile[i] is not None:
                # Extract task identifier from gamefile path
                gamefile_path = self.gamefile[i]
                
                # Extract the last two path parts and join with underscore
                # e.g., "~/.cache/alfworld/json_2.1.1/valid_seen/look_at_obj_in_light-CD-None-DeskLamp-320/trial_T20190907_224451_655673_game.tw-pddl"
                # should become "look_at_obj_in_light-CD-None-DeskLamp-320_trial_T20190907_224451_655673_game"
                if '/' in gamefile_path:
                    path_parts = gamefile_path.split('/')
                    if len(path_parts) >= 2:
                        # Get the last two parts
                        task_identifier = '_'.join(path_parts[-3:])
                    else:
                        # If only one part, use it as is
                        task_identifier = path_parts[-1]
                else:
                    task_identifier = gamefile_path
                
                # Remove .tw-pddl suffix if present
                if task_identifier.endswith('.tw-pddl'):
                    task_identifier = task_identifier[:-8]  # Remove '.tw-pddl' (8 characters)
                
                task_id = f"alfworld_{task_identifier}"
            else:
                # Fallback to timestamp if no gamefile
                task_id = f"alfworld_{int(time.time())}_{i}"
            
            # Create trajectory for this environment
            trajectory = {
                "task_info": {
                    "task_id": task_id,
                    "intent": self.tasks[i] if i < len(self.tasks) else "Unknown task",
                    "environment_name": "alfworld",
                    "actions_count": 0,
                    "gamefile": self.gamefile[i] if i < len(self.gamefile) else None
                },
                "traj_info": {
                    "success": False
                },
                "trajectory": []
            }
            
            # Add initial environment step
            env_step = {
                "role": "environment",
                "observation_text": text_obs[i],
                "observation_image": image_obs[i] if image_obs is not None else None,
                "step_id": 0
            }
            
            trajectory["trajectory"].append(env_step)
            
            self.trajectories.append(trajectory)
            self.trajectory_step_counts.append(0)
            self.trajectory_start_times.append(time.time())
            self.trajectory_saved.append(False)  # Initialize as not saved
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        
        # Capture agent actions for trajectory saving
        if self.save_trajectories:
            self._save_agent_actions(text_actions, actions)
        
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        # Save environment observations and check for episode completion
        if self.save_trajectories:
            self._save_env_observations(text_obs, image_obs, infos, rewards, dones)

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def _save_agent_actions(self, text_actions, actions):
        """Save agent actions to trajectory"""
        for i, (text_action, action) in enumerate(zip(text_actions, actions)):
            if i < len(self.trajectories):
                self.trajectory_step_counts[i] += 1
                agent_step = {
                    "role": "agent",
                    "action": text_action,
                    "action_parsed": action,
                    "step_id": self.trajectory_step_counts[i]
                }
                self.trajectories[i]["trajectory"].append(agent_step)
                self.trajectories[i]["task_info"]["actions_count"] = self.trajectory_step_counts[i]
    
    def _save_env_observations(self, text_obs, image_obs, infos, rewards, dones):
        """Save environment observations to trajectory"""
        # Handle case where image_obs might be None
        if image_obs is None:
            image_obs = [None] * len(text_obs)
        
        for i, (obs, img_obs, info, reward, done) in enumerate(zip(text_obs, image_obs, infos, rewards, dones)):
            if i < len(self.trajectories):
                env_step = {
                    "role": "environment",
                    "observation_text": obs,
                    "observation_image": img_obs if img_obs is not None else None,
                    "step_id": self.trajectory_step_counts[i] + 1,
                    "reward": float(reward),
                    "done": bool(done)
                }
                
                self.trajectories[i]["trajectory"].append(env_step)
                
                # If episode is done, save the trajectory (only if not already saved)
                if done and not self.trajectory_saved[i]:
                    self.trajectories[i]["traj_info"]["success"] = info.get('won', False)
                    self._save_trajectory(i)
                    self.trajectory_saved[i] = True  # Mark as saved
                    # Don't break here - continue processing other environments
                elif i == len(self.trajectories) - 1 and not self.trajectory_saved[i]:
                    self.trajectories[i]["traj_info"]["success"] = False
                    self._save_trajectory(i)
                    self.trajectory_saved[i] = True  # Mark as saved
                    # Don't break here - continue processing other environments

    def _save_trajectory(self, env_idx):
        """Save a completed trajectory to JSON file"""
        try:
            trajectory = self.trajectories[env_idx]
            
            # Add timing information
            end_time = time.time()
            start_time = self.trajectory_start_times[env_idx]
            trajectory["traj_info"]["duration"] = end_time - start_time
            trajectory["traj_info"]["timestamp"] = datetime.now().isoformat()
            
            # Create filename using only task-related fields
            task_id = trajectory["task_info"]["task_id"]
            filename = f"trajectory_{task_id}.json"
            filepath = os.path.join(self.trajectory_save_dir, filename)
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False)
            
            print(f"Saved trajectory for environment {env_idx} to {filepath}")
            
        except Exception as e:
            print(f"Error saving trajectory for environment {env_idx}: {e}")

    def save_all_trajectories(self):
        """Save all remaining trajectories (useful when closing environment manager)"""
        if not self.save_trajectories:
            return
            
        for i, trajectory in enumerate(self.trajectories):
            if trajectory and len(trajectory["trajectory"]) > 1 and not self.trajectory_saved[i]:  # Only save if not already saved
                try:
                    # Mark as incomplete if not already marked as complete
                    if not trajectory.get("traj_info", {}).get("success", False):
                        trajectory["traj_info"]["success"] = False
                        trajectory["traj_info"]["incomplete"] = True
                    
                    self._save_trajectory(i)
                    self.trajectory_saved[i] = True  # Mark as saved
                except Exception as e:
                    print(f"Error saving trajectory {i}: {e}")
    
    def __del__(self):
        """Ensure trajectories are saved when environment manager is destroyed"""
        try:
            self.save_all_trajectories()
        except:
            pass  # Ignore errors during cleanup
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
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
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Save trajectory if trajectory saving is enabled
                if self.save_trajectories and batch_idx < len(self.trajectories) and not self.trajectory_saved[batch_idx]:
                    try:
                        self.trajectories[batch_idx]["traj_info"]["success"] = bool(won_value)
                        self._save_trajectory(batch_idx)
                        self.trajectory_saved[batch_idx] = True  # Mark as saved
                    except Exception as e:
                        print(f"Error saving trajectory in batch processing: {e}")
                
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
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    if self.is_multi_modal:
                        action_history += f"\n[Action {step_number}: '{record['action']}']"
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
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    """
    WebshopEnvironmentManager with trajectory saving capabilities.
    
    This manager handles both text observations for agent training and HTML observations 
    for trajectory saving. Trajectories are saved in JSON format with the following structure:
    
    {
        "task_info": {
            "task_id": "webshop_timestamp_env_idx",
            "intent": "task description",
            "environment_name": "webshop", 
            "actions_count": number_of_actions
        },
        "traj_info": {
            "success": boolean,
            "task_score": float,
            "duration": float,
            "timestamp": "ISO timestamp"
        },
        "trajectory": [
            {
                "role": "environment",
                "url": "current_url",
                "observation_text": "simplified text observation",
                "observation_html": "full HTML observation", 
                "step_id": int,
                "reward": float,
                "done": boolean
            },
            {
                "role": "agent",
                "action": "original text action",
                "action_parsed": "parsed action",
                "step_id": int
            },
            ...
        ]
    }
    
    Configuration:
    - config.env.save_trajectories: Enable/disable trajectory saving (default: True)
    - config.env.trajectory_save_dir: Directory to save trajectories (default: /scratch/czr/agentscale/dataset/webshop/online_trajs)
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
        
        # Trajectory saving setup
        self.save_trajectories = getattr(config.env, 'save_trajectories', True)
        if self.save_trajectories:
            self.trajectory_save_dir = getattr(config.env, 'trajectory_save_dir', None)
            if self.trajectory_save_dir is None:
                raise ValueError("trajectory_save_dir must be set when save_trajectories is True")
        
        if self.save_trajectories:
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
            
        # Initialize trajectory tracking
        self.trajectories = []
        self.trajectory_step_counts = []
        self.trajectory_start_times = []
        self.trajectory_saved = []  # Track which trajectories have been saved
        
        # Create HTML observation environments for trajectory saving
        if self.save_trajectories:
            self._init_html_envs()
    
    def _init_html_envs(self):
        """Initialize HTML observation environments for trajectory saving"""
        try:
            # Import here to avoid circular imports
            from agent_system.environments.env_package.webshop import build_webshop_envs
            
            # Create HTML observation environments
            if hasattr(self.config.env, 'webshop') and hasattr(self.config.env.webshop, 'use_small'):
                if self.config.env.webshop.use_small:
                    file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
                    attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
                else:
                    file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
                    attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
            else:
                file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
                attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
                
            html_env_kwargs = {
                'observation_mode': 'html',
                'num_products': None,
                'human_goals': getattr(self.config.env.webshop, 'human_goals', False),
                'file_path': file_path,
                'attr_path': attr_path
            }
            
            # Build HTML environments with same configuration as main envs
            group_n = self.config.env.rollout.n if self.config.env.rollout.n > 0 else 1
            self.html_envs = build_webshop_envs(
                seed=self.config.env.seed, 
                env_num=self.config.data.train_batch_size, 
                group_n=group_n, 
                is_train=True, 
                env_kwargs=html_env_kwargs
            )
            
            # Reset HTML environments to sync with main environments
            self.html_envs.reset()
            
        except Exception as e:
            print(f"Warning: Could not initialize HTML environments for trajectory saving: {e}")
            self.html_envs = None
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        
        # Initialize trajectories for each environment
        if self.save_trajectories:
            self._init_trajectories(obs, infos)
        
        return observations, infos

    def _init_trajectories(self, obs, infos):
        """Initialize trajectory tracking for each environment"""
        num_envs = len(obs)
        self.trajectories = []
        self.trajectory_step_counts = []
        self.trajectory_start_times = []
        self.trajectory_saved = []
        
        # Get goals to extract real task IDs
        goals = None
        try:
            # Get goals from the environment
            if hasattr(self.envs, '_workers') and len(self.envs._workers) > 0:
                goals_future = self.envs._workers[0].get_goals.remote()
                goals = ray.get(goals_future)
                print(f"DEBUG: Retrieved {len(goals)} goals from environment")
        except Exception as e:
            print(f"Warning: Could not get goals for task IDs: {e}")
        
        # Get HTML observations if available
        html_obs = None
        if self.html_envs is not None:
            try:
                html_obs, html_infos = self.html_envs.reset()
            except Exception as e:
                print(f"Warning: Could not get HTML observations: {e}")
                html_obs = None
        
        for i in range(num_envs):
            # Get real task ID from goals if available
            task_id = f"webshop_{int(time.time())}_{i}"  # fallback
            if goals is not None and hasattr(self.envs, 'goal_idxs') and i < len(self.envs.goal_idxs):
                goal_idx = self.envs.goal_idxs[i]
                print(f"DEBUG: Environment {i}, goal_idx={goal_idx}, goals_length={len(goals)}")
                if goal_idx < len(goals):
                    goal = goals[goal_idx]
                    # Create unique task ID for synthetic goals
                    if 'asin' in goal:
                        asin = goal['asin']
                        if 'goal_options' in goal and goal['goal_options']:
                            # For synthetic goals: use asin + goal_options hash
                            options_str = str(sorted(goal['goal_options'].items()))
                            options_hash = int(hashlib.md5(options_str.encode()).hexdigest(), 16)
                            task_id = f"{asin}_{abs(options_hash)}"
                            print(f"DEBUG: Successfully got synthetic task_id={task_id} for env {i} (asin={asin})")
                        else:
                            # Fallback to asin + instruction_text hash for human goals
                            if 'instruction_text' in goal:
                                instruction_hash = int(hashlib.md5(goal['instruction_text'].encode()).hexdigest(), 16)
                                task_id = f"{asin}_{abs(instruction_hash)}"
                                print(f"DEBUG: Successfully got human task_id={task_id} for env {i} (asin={asin})")
                            else:
                                task_id = asin
                                print(f"DEBUG: Using asin as task_id={task_id} for env {i}")
                    else:
                        print(f"DEBUG: No asin in goal for env {i}, using fallback")
                else:
                    print(f"DEBUG: Failed to get goal for env {i}, goal_idx={goal_idx} >= goals_length={len(goals)}")
            else:
                print(f"DEBUG: Using fallback task_id for env {i}, goals={goals is not None}, has_goal_idxs={hasattr(self.envs, 'goal_idxs')}, i={i} < goal_idxs_len={len(self.envs.goal_idxs) if hasattr(self.envs, 'goal_idxs') else 'N/A'}")
            
            # Create trajectory for this environment
            trajectory = {
                "task_info": {
                    "task_id": task_id,
                    "intent": self.tasks[i],
                    "environment_name": "webshop",
                    "actions_count": 0
                },
                "traj_info": {
                    "success": False
                },
                "trajectory": []
            }
            
            # Add initial environment step
            env_step = {
                "role": "environment",
                "url": infos[i].get('url', ''),
                "observation_text": obs[i],
                "step_id": 0
            }
            
            # Add HTML observation if available
            if html_obs is not None and i < len(html_obs):
                env_step["observation_html"] = html_obs[i]
            
            trajectory["trajectory"].append(env_step)
            
            self.trajectories.append(trajectory)
            self.trajectory_step_counts.append(0)
            self.trajectory_start_times.append(time.time())
            self.trajectory_saved.append(False)  # Initialize as not saved

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        
        # Capture agent actions for trajectory saving
        if self.save_trajectories:
            self._save_agent_actions(text_actions, actions)
        
        next_obs, rewards, dones, infos = self.envs.step(actions)
        
        # Get HTML observations if available
        html_obs = None
        if self.save_trajectories and self.html_envs is not None:
            try:
                html_obs, html_rewards, html_dones, html_infos = self.html_envs.step(actions)
            except Exception as e:
                print(f"Warning: Could not get HTML observations: {e}")
                html_obs = None

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        # Save environment observations and check for episode completion
        if self.save_trajectories:
            self._save_env_observations(next_obs, infos, html_obs, rewards, dones)

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def _save_agent_actions(self, text_actions, actions):
        """Save agent actions to trajectory"""
        for i, (text_action, action) in enumerate(zip(text_actions, actions)):
            if i < len(self.trajectories):
                self.trajectory_step_counts[i] += 1
                agent_step = {
                    "role": "agent",
                    "action": text_action,
                    "action_parsed": action,
                    "step_id": self.trajectory_step_counts[i]
                }
                self.trajectories[i]["trajectory"].append(agent_step)
                self.trajectories[i]["task_info"]["actions_count"] = self.trajectory_step_counts[i]
    
    def _save_env_observations(self, next_obs, infos, html_obs, rewards, dones):
        """Save environment observations to trajectory"""
        for i, (obs, info, reward, done) in enumerate(zip(next_obs, infos, rewards, dones)):
            if i < len(self.trajectories):
                env_step = {
                    "role": "environment",
                    "url": info.get('url', ''),
                    "observation_text": obs,
                    "step_id": self.trajectory_step_counts[i] + 1,
                    "reward": float(reward),
                    "done": bool(done)
                }
                
                # Add HTML observation if available
                if html_obs is not None and i < len(html_obs):
                    env_step["observation_html"] = html_obs[i]
                
                self.trajectories[i]["trajectory"].append(env_step)
                
                # If episode is done, save the trajectory
                if done and not self.trajectory_saved[i]:
                    self.trajectories[i]["traj_info"]["success"] = info.get('won', False)
                    self._save_trajectory(i)
                    self.trajectory_saved[i] = True  # Mark as saved

    def extract_task(self, text_obs: List[str]):
        tasks = []
        # Compile regex pattern for better performance if many obs
        pattern = re.compile(r"<instruction>(.*?)</instruction>", re.DOTALL)
        for obs in text_obs:
            match = pattern.search(obs)
            if match:
                tasks.append(match.group(1).strip())
            else:
                # Fallback: try to extract from info if available
                tasks.append("Browse and interact with the website")
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        pattern = re.compile(r"<instruction>.*?</instruction>", re.DOTALL)
        for obs in text_obs:
            # Remove the instruction part if it exists
            if '<instruction>' in obs and '</instruction>' in obs:
                reformatted_obs = pattern.sub('', obs).strip()
                # Collapse any extra whitespace caused by removal
                reformatted_obs = re.sub(r'\s+', ' ', reformatted_obs).strip()
            else:
                reformatted_obs = obs
            postprocess_text_obs.append(reformatted_obs)
        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        if avail.get("has_search_bar", False):
            actions.append("search[<your query>]")

        for txt in avail.get("clickables", []):
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
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
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                
                # Save trajectory if trajectory saving is enabled
                if self.save_trajectories and batch_idx < len(self.trajectories) and not self.trajectory_saved[batch_idx]:
                    try:
                        self.trajectories[batch_idx]["traj_info"]["success"] = bool(won_value)
                        self.trajectories[batch_idx]["traj_info"]["task_score"] = score_value
                        self._save_trajectory(batch_idx)
                        self.trajectory_saved[batch_idx] = True  # Mark as saved
                    except Exception as e:
                        print(f"Error saving trajectory in batch processing: {e}")
                
                return

    def _save_trajectory(self, env_idx):
        """Save a completed trajectory to JSON file"""
        try:
            trajectory = self.trajectories[env_idx]
            
            # Add timing information
            end_time = time.time()
            start_time = self.trajectory_start_times[env_idx]
            trajectory["traj_info"]["duration"] = end_time - start_time
            trajectory["traj_info"]["timestamp"] = datetime.now().isoformat()
            
            # Create filename using only task-related fields
            task_id = trajectory["task_info"]["task_id"]
            filename = f"trajectory_{task_id}.json"
            filepath = os.path.join(self.trajectory_save_dir, filename)
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False)
            
            print(f"Saved trajectory for environment {env_idx} to {filepath}")
            
        except Exception as e:
            print(f"Error saving trajectory for environment {env_idx}: {e}")

    def save_all_trajectories(self):
        """Save all remaining trajectories (useful when closing environment manager)"""
        if not self.save_trajectories:
            return
            
        for i, trajectory in enumerate(self.trajectories):
            if trajectory and len(trajectory["trajectory"]) > 1 and not self.trajectory_saved[i]:  # Only save if not already saved
                try:
                    # Mark as incomplete if not already marked as complete
                    if not trajectory.get("traj_info", {}).get("success", False):
                        trajectory["traj_info"]["success"] = False
                        trajectory["traj_info"]["incomplete"] = True
                    
                    self._save_trajectory(i)
                    self.trajectory_saved[i] = True  # Mark as saved
                except Exception as e:
                    print(f"Error saving trajectory {i}: {e}")
    
    def __del__(self):
        """Ensure trajectories are saved when environment manager is destroyed"""
        try:
            self.save_all_trajectories()
        except:
            pass  # Ignore errors during cleanup

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
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
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
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
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': 'eval_in_distribution', # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        
        # get AlfWorld's start_idx and end_idx parameters. Use .get on the intermediate
        # 'alfworld' key: the base trainer config has no env.alfworld block, and
        # dereferencing config.env.alfworld directly raises ConfigAttributeError before
        # getattr's default applies. That broke the standalone eval paths that pass no
        # +env.alfworld.* (evaluate.sh alfworld; batch_alfworld_eval.sh SPLIT=val); the
        # '+'-prefixed train/sweep paths add the key, so they were unaffected.
        _alf_cfg = config.env.get('alfworld', {})
        start_idx = _alf_cfg.get('start_idx', None)
        end_idx = _alf_cfg.get('end_idx', None)
        
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs)
        
        # if start_idx and end_idx are provided, the validation environment uses the training set but stays in validation mode to support index-based selection
        if start_idx is not None and end_idx is not None:
            _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, start_idx=start_idx, end_idx=end_idx)
        else:
            _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        # get the infer_special parameter, defaults to False
        infer_special = getattr(config.env.webshop, 'infer_special', False)
        start_idx = getattr(config.env.webshop, 'start_idx', None)
        end_idx = getattr(config.env.webshop, 'end_idx', None)
        
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, infer_special=infer_special, start_idx=start_idx, end_idx=end_idx)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)