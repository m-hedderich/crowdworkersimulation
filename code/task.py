import json
import pickle
from typing import Optional

import numpy as np


class Instance:
    """
    A question, e.g. an image instance in a machine learning labeling task
    """

    def __init__(self, true_label):
        self.true_label = true_label
        self.label = None # label assigned by worker


class TaskPropertiesDistribution:
    """
    TaskPropertiesDistribution represents a task-giver that provides the properties of a  task
    (with create_properties()) in each episode. The child methods define the actual properties.
    """

    def create_properties(self, random):
        """ Overwritten by child class """
        pass

    @staticmethod
    def save_list(list, config):
        """ Store a list of TaskPropertiesDistribution objects both in human-readable and
            machine-readable (pickle) form.
        """
        with open(config.path("task_properties_distributions.txt"), "w") as out_file:
            for item in list:
                out_file.write(str(item))
                out_file.write("\n")
        with open(config.path("task_properties_distributions.pickle"), "wb") as out_file:
            pickle.dump(list, out_file)

    @staticmethod
    def load_list(config):
        """ Load a list of TaskPropertiesDistribution objects from machine-readable (pickle) form
         """
        with open(config.path("task_properties_distributions.pickle"), "rb") as in_file:
            return pickle.load(in_file)

    def __str__(self):
        return self.__class__.__name__


class TaskPropertiesBetaDistribution(TaskPropertiesDistribution):

    def create_properties(self, random):
        return TaskProperties(payout=random.beta(10, 10), expertise=random.beta(40, 10), effort=random.beta(10, 10),
                              interestingness=random.beta(10, 10)-0.5, target_num_instances=random.beta(10, 10)*100)


class TaskPropertiesCustomDistribution(TaskPropertiesDistribution):
    def __str__(self):
        return self.__class__.__name__ + "(" + ";".join([f"{k}:{v}" for k, v in vars(self).items()]) + ")"


class TaskPropertiesCustomBetaDistribution(TaskPropertiesDistribution):
    """
    Properties following beta distributions, specified by two parameters per distribution (a and b)
    TODO: refactor to have TaskPropertiesCustomDistribution as parent class
    """

    def __init__(self, payout=(10, 10), expertise=(40, 10), effort=(10, 10), interestingness=(10, 10),
                 target_num_instances=(10, 10), target_num_instances_scale=100):
        """
        :param payout: distribution of the payout
        :param expertise: distribution of the expertise (also user dependent)
        :param effort: distribution of the time effort
        :param interestingness: distribution of the interestingness factor
        :param target_num_instances: distribution of the number of questions/instances to label. Like all other values
                is between [0,1], but will be scaled up by target_num_instances_scale
        :param target_num_instances_scale: scaling factor for the number of questions/instances
        """
        self.payout = payout
        self.expertise = expertise
        self.effort = effort
        self.interestingness = interestingness
        self.target_num_instances = target_num_instances
        self.target_num_instances_scale = target_num_instances_scale

    def create_properties(self, random):
        return TaskProperties(payout=random.beta(*self.payout), expertise=random.beta(*self.expertise),
                              effort=random.beta(*self.effort), interestingness=random.beta(*self.interestingness)-0.5,
                              target_num_instances=random.beta(*self.target_num_instances)*self.target_num_instances_scale)

    def __str__(self):
        return self.__class__.__name__ + "(" + ";".join([f"{k}:{v}" for k, v in vars(self).items()]) + ")"


class TaskPropertiesCustomFixedDistribution(TaskPropertiesCustomDistribution):
    """
    Fixed values (instead of beta distributions) for the task properties. For a more controlled environment.
    """

    def __init__(self, payout=0.5, expertise=0.8, effort=0.5, interestingness=0.5,
                 target_num_instances=50):
        """
        :param payout: fixed value, payout per question
        :param expertise: fixed expertise value
        :param effort: fixed time effort
        :param interestingness: fixed interestingness factor
        :param target_num_instances: fixed number of questions/instances to label.
        """
        self.payout = payout
        self.expertise = expertise
        self.effort = effort
        self.interestingness = interestingness
        self.target_num_instances = target_num_instances

    def create_properties(self, random):
        return TaskProperties(payout=self.payout, expertise=self.expertise,
                              effort=self.effort, interestingness=self.interestingness-0.5,
                              target_num_instances=self.target_num_instances)


class TaskProperties:
    """
    The properties of a task. The actual code for the task is all grouped in the Task class.
    """

    def __init__(self, payout: float, expertise: float, effort: float, interestingness: float, target_num_instances: int):
        self.payout: float = payout
        self.expertise: float = expertise
        self.effort: float  = effort
        self.interestingness: float = interestingness
        self.target_num_instances: int = target_num_instances
        self.num_classes: int = 10


class AntiCheatSettings:
    """
    Settings for the methods to deter cheating: hidden gold questions (quality assurance / qa questions) and reputation
    system.
    """

    def __init__(self, qa_false_max: int, qa_mode_prob: float, reputation_punishment: float, reputation_bonus: float, min_reputation: float):
        """
        :param qa_false_max: maximum number of hidden, known-answer gold questions the worker can answer incorrectly
                before being banned from a task (exclusive, i.e. <)
        :param qa_mode_prob: probability of introducing a hidden gold question
        :param reputation_punishment: decrease in reputation if a gold question is answered incorrectly (value of -0.05
                results in a decrease of 0.05)
        :param reputation_bonus: increase in reputation if a gold question is answered correctly (value of 0.05
                results in an increase of 0.05)
        :param min_reputation: minimum reputation before the worker is banned
        """
        self.qa_false_max: int = qa_false_max
        self.qa_mode_prob: float = qa_mode_prob
        self.reputation_punishment: float = reputation_punishment
        self.reputation_bonus: float = reputation_bonus
        self.min_reputation: float = min_reputation

    def save(self, config):
        with open(config.path("anti_cheat_settings.json"), "w") as out_file:
            json.dump(self.__dict__, out_file, indent=4, sort_keys=True)

    @staticmethod
    def load(config):
        path = config.path("anti_cheat_settings.json")
        props_json = json.load(open(path, "r"))
        props = AntiCheatSettings(-1, -1, -1, -1, -1)
        props.__dict__ = {}
        for key, value in props_json.items():
            props.__dict__[key] = value
        return props


class Task:
    """
    A task
    """

    QUALITY_CONTROL_MODE = 0 # hidden gold question mode
    LABEL_MODE = 1 # actual labeling (unknown answer) mode

    def __init__(self, properties: TaskProperties, anti_cheat_settings: AntiCheatSettings):
        self.properties: TaskProperties = properties
        self.anti_cheat_settings: AntiCheatSettings = anti_cheat_settings

        self.mode: Optional[int] = None # whether the last question was a hidden gold question or a normal question
        self.current_instance: Optional[Instance] = None

        self.instance_counter: int = 0 # how many rounds of labeling the user has done
        self.real_instance_counter: int = 0 # without the QA instances
        self.qa_false_counter: int = 0 # how many qa instances answered incorrectly

        self.last_response_type = "not-set" # logging what type the last response was (correct, failed qa, etc.)

    def receive_answer(self, answer):
        assert self.mode is not None

        self.instance_counter += 1
        reputation_change = 0

        if self.mode == Task.QUALITY_CONTROL_MODE:
            if answer != self.current_instance.true_label:
                self.qa_false_counter += 1
                reputation_change = self.anti_cheat_settings.reputation_punishment
                self.last_response_type = "qa_incorrect"
            else:
                reputation_change = self.anti_cheat_settings.reputation_bonus
                self.last_response_type = "qa_correct"

        elif self.mode == Task.LABEL_MODE:
            self.real_instance_counter += 1
            if answer != self.current_instance.true_label:
                self.last_response_type = "incorrect"
            else:
                self.last_response_type = "correct"

        return reputation_change

    def give_new_instance(self, random: np.random.Generator) -> (Optional[Instance], str):
        if random.random() < self.anti_cheat_settings.qa_mode_prob:
            self.mode = Task.QUALITY_CONTROL_MODE
            self.current_instance = self.get_next_known_answer_instance(random)
        else:
            self.mode = Task.LABEL_MODE
            self.current_instance = self.get_next_unlabeled_instance(random)

    def get_next_unlabeled_instance(self, random: np.random.Generator):
        return self.get_next_known_answer_instance(random)

    def get_next_known_answer_instance(self, random: np.random.Generator):
        return Instance(true_label=random.integers(0, self.properties.num_classes))

    def is_active(self, current_user_reputation):
        return self.real_instance_counter < self.properties.target_num_instances and \
               self.qa_false_counter < self.anti_cheat_settings.qa_false_max and \
               current_user_reputation >= self.anti_cheat_settings.min_reputation

    def __str__(self):
        output = "TaskProvider ("
        output += f", {self.instance_counter} instances labeled with {self.real_instance_counter} real instances and "
        output += f"{self.qa_false_counter} QA answers failed)"
        return output
