import json
import shutil
import argparse
import os
import time
import subprocess


class Config(argparse.Namespace):
    """ A configuration file that stores the settings of an experiment. Can be stored and loaded.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def create(name, exist_ok=False):
        config = Config()
        config._create_config(name, exist_ok)
        return config

    def _create_config(self, name, exist_ok=False):
        """
        Creates a config object via argparse.Namespace
        """
        self.name = name

        self.timestamp = time.strftime('%H:%M%p %Z on %b %d, %Y')
	# if you use git, you can store the current git hash of the repo with the model     
	#self.git_hash = str(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=".").strip())

        self.exp_dir_path = os.path.join("..", "exp", self.name)
        os.makedirs(self.exp_dir_path, exist_ok=exist_ok)

    def path(self, filename):
        return os.path.join("..", "exp", self.name, filename)

    def save(self):
        path = os.path.join(self.exp_dir_path, "config.json")
        config_dict = vars(self)
        print(config_dict)
        with open(path, "w") as out_file:
            json.dump(config_dict, out_file, indent=4, sort_keys=True)

    def copy_exp_dir(self, new_name):
        """ copy content of an exp dir to a new directory and returns a new config file.
        Tries to merge the content of the old config into the new one.
        Use carefully as content in the directory or config might still have dependencies
        to the old directory!
        """

        # create new config file and exp directory
        new_config = Config.create(new_name, exist_ok=False)

        # copy to new directory, but remove old config file
        shutil.copytree(self.exp_dir_path, new_config.exp_dir_path, dirs_exist_ok=True)
        os.remove(new_config.path("config.json"))

        # merge content of old config file into new one
        # just use entries that do not already exist in the new config (like name, path, git hash)
        for key, value in vars(self).items():
            if not key in vars(new_config):
                setattr(new_config, key, value)

        new_config.based_on = f"This exp-result directory is based on {self.name}."
        new_config.save()
        return new_config

    @staticmethod
    def load(name):
        path = os.path.join("..", "exp", name, "config.json")
        config_json = json.load(open(path, "r"))
        config = Config()
        config.__dict__ = {}
        for key, value in config_json.items():
            config.__dict__[key] = value
        return config
