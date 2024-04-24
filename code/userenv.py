from typing import Optional, List

import numpy as np
import gym
from gym import spaces

from user import UserProperties
from task import Instance, Task, TaskPropertiesDistribution, AntiCheatSettings
from util.exputil import Config


class UserModelEnv(gym.Env):
    """
    The crowdworking platform as a RL environment
    """

    # constants for addressing action indices
    ACTION_QUIT = 0
    ACTION_ANS_RND = 1  # answer negligently
    ACTION_ANS_INTENT = 2  # answer diligently
    SWITCH_TASK0 = 3

    # metadata for RL gym
    metadata = {'render.modes': ["text"]}

    def __init__(self, config: Config, user_properties: UserProperties,
                 tasks_properties_distributions: List[TaskPropertiesDistribution],
                 anti_cheat_settings: AntiCheatSettings):
        """
        :param config: experimental configuration
        :param user_properties: properties of the worker
        :param tasks_properties_distributions: list of task-givers, decide/generate the properties of the tasks,
                one for each task in each episode
        :param anti_cheat_settings: settings to deter cheating
        """
        super().__init__()

        self.config: Config = config

        self.num_tasks: int = len(tasks_properties_distributions)

        # seven possible actions (quit, answer randomly, answer intentionally, switch to task i)
        self.num_actions = 3 + self.num_tasks
        self.action_space = spaces.Discrete(self.num_actions)

        # observations of the user:
        # for each of the four tasks:
        #   payout,
        #   how many rounds of labeling already done (-1 is task is not active),
        #   expertise, effort, interestingness (-1 if task not yet started)
        # current task idx, user reputation [0,1], user time budget, overall time spent
        min_values = [0, -1, -1, -1, -1] * self.num_tasks
        min_values.extend([-1, 0, 0, 0])
        max_values = [ 1, float("inf"), float("inf"), float("inf"), float("inf")] * self.num_tasks
        max_values.extend([self.num_tasks, 1, float("inf"), float("inf")])
        self.observation_space = spaces.Box(low=np.float32(min_values),
                                            high=np.float32(max_values),
                                            shape=(len(min_values),))

        self.user_properties: UserProperties = user_properties
        self.user_reputation = -1

        # task properties distributions = task givers
        self.tasks_properties_distributions: List[TaskPropertiesDistribution] = tasks_properties_distributions
        self.anti_cheat_settings = anti_cheat_settings
        self.tasks: List[Task] = []
        # the order of the tasks is randomized in each episode so that the agent can not learn which task
        # maps to which task-giver. This mapping allows to reconstruct the task-giver/properties-distribution
        # to each task
        self.task_task_dist_map = {}

        self.last_action = None
        self.current_instance: Optional[Instance] = None
        self.current_task_idx: int = -1 # last task that was selected by agent
        self.overall_time_spent: float = 0

        self.random = None

    def step(self, action: int):
        """ environment reacts to the user's action in this step function
        """
        self.last_action = action

        info = {}

        # worker quits -> end of episode
        if action == UserModelEnv.ACTION_QUIT:
            done = True
            reward = (self.user_properties.time_budget - self.overall_time_spent) * self.user_properties.time_sensitivity # reward for using time for something else
            info["end_reason"] = "user_quit"

            obs = self.create_observation()
            return obs, reward, done, info

        # worker runs out of time -> end of episode
        if self.overall_time_spent > self.user_properties.time_budget:
            done = True
            reward = 0
            info["end_reason"] = "end_of_user_time_budget"

            obs = self.create_observation()
            return obs, reward, done, info

        done = False

        current_task = self.tasks[self.current_task_idx]

        if action == UserModelEnv.ACTION_ANS_RND or action == UserModelEnv.ACTION_ANS_INTENT:
            # no task selected or task is not active, so answering does not make sense
            if self.current_task_idx == -1 or not current_task.is_active(self.user_reputation):
                reward = -1
                self.overall_time_spent += self.user_properties.random_answer_time
                obs = self.create_observation()
                return obs, reward, done, info

            reward_payout = self.user_properties.payout_sensitivity * current_task.properties.payout

            # negligent answer
            if action == UserModelEnv.ACTION_ANS_RND:
                time_spent = self.user_properties.random_answer_time
                answer_to_task = self.random.integers(0, current_task.properties.num_classes)
                reward = reward_payout  # only monetary reward

            # diligent answer
            else:
                assert action == UserModelEnv.ACTION_ANS_INTENT
                time_spent = self.user_properties.intentional_answer_time * current_task.properties.effort
                if self.random.random() < current_task.properties.expertise:
                    answer_to_task = current_task.current_instance.true_label
                else:
                    answer_to_task = self.random.integers(0, current_task.properties.num_classes)
                reward_interestingness = self.user_properties.interestingness_sensitivity * current_task.properties.interestingness
                reward = reward_payout + reward_interestingness  # monetary + interestingness reward

            self.overall_time_spent += time_spent

            # Behavior of the task and task-giver
            reputation_change = current_task.receive_answer(answer_to_task)
            self.user_reputation += reputation_change
            self.user_reputation = max(0, min(1, self.user_reputation))

            # task-giver bans worker or has run out of questions, no longer supplies user with new questions
            if not current_task.is_active(self.user_reputation):
                self.current_task_idx = -1
            else:
                current_task.give_new_instance(self.random)

            obs = self.create_observation()
            return obs, reward, done, info

        # select new task
        if UserModelEnv.SWITCH_TASK0 <= action:
            previous_task_idx = self.current_task_idx
            self.current_task_idx = action - UserModelEnv.SWITCH_TASK0

            self.overall_time_spent += self.user_properties.switch_task_time

            # invalid action, can not select a task that is inactive
            if not self.tasks[self.current_task_idx].is_active(self.user_reputation):
                # task is not active
                reward = -1
                self.current_task_idx = -1
                obs = self.create_observation()
                return obs, reward, done, info

            # we are already in this task, switching does not make sense
            # might be used in the future by agent to get a new, different question
            if previous_task_idx == self.current_task_idx:
                reward = -1
            else:
                reward = 0

            self.tasks[self.current_task_idx].give_new_instance(self.random)

            obs = self.create_observation()
            return obs, reward, done, info

    def create_observation(self):
        """
        the part of the environment visible to the worker
        """
        obs = []

        for i, task in enumerate(self.tasks):
            # payout for the task
            obs.append(task.properties.payout)
            # how many rounds of labeling already done for the tasks or -1 if task is inactive
            if task.is_active(self.user_reputation):
                obs.append(task.instance_counter)
            else:
                obs.append(-1)
            # expertise, effort, interestingness (-1 if not at least one instance has been done, i.e. the worker
            # has tried out the task)
            if task.instance_counter > 0:
                obs.append(task.properties.expertise)
                obs.append(task.properties.effort)
                obs.append(task.properties.interestingness)
            else:
                obs.extend([-1, -1, -1])
        # current task idx
        obs.append(self.current_task_idx)
        # user reputation
        obs.append(self.user_reputation)
        # user time budget
        obs.append(self.user_properties.time_budget)
        obs.append(self.overall_time_spent)
        return np.array(obs)

    def reset(self):
        """
        Reset environment for a new episode
        """
        self.tasks = []
        self.task_task_dist_map = {}
        new_tasks = []
        for task_properties_distribution in self.tasks_properties_distributions:
            task_props = task_properties_distribution.create_properties(self.random)
            new_tasks.append(Task(task_props, self.anti_cheat_settings))
        # shuffle tasks, so that even if the task property distributions are different, it can not learn which is which
        # but store mapping so that we can recover the creating task distribution
        task_dist_indices = list(range(len(self.tasks_properties_distributions)))
        self.random.shuffle(task_dist_indices)
        for task_index, task_dist_index in enumerate(task_dist_indices):
            self.tasks.append(new_tasks[task_dist_index])
            self.task_task_dist_map[task_index] = task_dist_index

        self.current_task_idx = -1
        self.last_action = None
        self.current_instance= None
        self.overall_time_spent = 0
        self.user_reputation = self.user_properties.start_reputation

        return self.create_observation()

    def render(self, mode='text', close=False):
        if mode == "text":
            print(self.observation_to_string(self.create_observation()))

    def observation_to_string(self, obs):
        output = "Observation:\n"
        # observations of the user:
        # for each of the four tasks:
        #   payout,
        #   how many rounds of labeling already done (-1 is task is done),
        #   expertise, effort, interestingness (-1 if task not yet started)
        # current task idx
        # user time budget, overall time spent
        i = 0
        for i in range(self.num_tasks):
            skip_idx = i * 5
            output += f"  Task {i}:\n"
            output += f"      payout {obs[skip_idx]} | rounds {obs[skip_idx+1]}\n"
            output += f"      expert {obs[skip_idx+2]} | effort {obs[skip_idx+3]} | interest {obs[skip_idx+4]}\n"
        obs_idx = (i+1)*5
        output += f"  current task: {obs[obs_idx]}\n"
        output += f"  reputation: {obs[obs_idx+1]}\n"
        output += f"  time: {obs[obs_idx+3]}/{obs[obs_idx+2]}\n"
        return output

    def seed(self, seed=None):
        """
        Set the random seed for reproducibility of the environment
        """
        self.random = np.random.default_rng(seed)
        # not sure if needed, but probably doesn't hurt
        # https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
        self.action_space.seed(seed)

    @staticmethod
    def action_to_str(action_idx):
        """
        Map the action index (numeric int value) to the name representation,
        e.g. 0 to "QUIT" or 1 to "ANSWER NEGLIGENTLY"
        """
        if action_idx == UserModelEnv.ACTION_QUIT:
            return "QUIT"
        if action_idx == UserModelEnv.ACTION_ANS_RND:
            return "ANSWER NEGLIGENTLY"
        if action_idx == UserModelEnv.ACTION_ANS_INTENT:
            return "ANSWER DILIGENTLY"
        if action_idx >= UserModelEnv.SWITCH_TASK0:
            return f"SWITCH TO TASK {action_idx - UserModelEnv.SWITCH_TASK0}"
