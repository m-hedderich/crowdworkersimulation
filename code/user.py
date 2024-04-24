import json


class UserProperties:
    """ The properties of the worker/user
    """

    def __init__(self, interestingness_sensitivity, payout_sensitivity, time_sensitivity, time_budget, start_reputation,
                 switch_task_time=1):
        """
        :param interestingness_sensitivity: multiplier of the interestingness factor for the reward function
        :param payout_sensitivity: multiplier of the payout for the reward function
        :param time_sensitivity: no longer used, TODO: refactor and remove
        :param time_budget: how much time the worker can spend on the crowdworking platform, is reduced by a task's
                            time effort
        :param start_reputation: the reputation the worker starts with
        :param switch_task_time: the time it takes the worker to switch between tasks (time effort for the switch task
                                 action)
        """
        # factors for reward
        self.interestingness_sensitivity = interestingness_sensitivity
        self.payout_sensitivity = payout_sensitivity
        self.time_sensitivity = time_sensitivity

        # other factors
        self.time_budget = time_budget
        self.start_reputation = start_reputation

        # factors for answer time
        self.random_answer_time = 0.1
        self.intentional_answer_time = 1
        self.switch_task_time = switch_task_time

    def save(self, config):
        with open(config.path("user_properties.json"), "w") as out_file:
            json.dump(self.__dict__, out_file, indent=4, sort_keys=True)

    @staticmethod
    def load(config):
        path = config.path("user_properties.json")
        props_json = json.load(open(path, "r"))
        props = UserProperties(-1, -1, -1, -1, -1)
        props.__dict__ = {}
        for key, value in props_json.items():
            props.__dict__[key] = value
        return props
