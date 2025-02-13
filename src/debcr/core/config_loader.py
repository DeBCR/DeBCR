import yaml
#import argparse -> dict_to_namespace()

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
    
def load_yaml_config(config_path):
    """Load training configuration from a YAML file."""
    with open(config_path, 'r') as stream:
        config_dict = yaml.load(stream, Loader)
    config = DictConfig(config_dict)
    return config

class DictConfig(object):
    """Creates a Config object from a dict 
       such that object attributes correspond to dict keys.    
    """

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __str__(self):
        return '\n'.join(f"{key}: {val}" for key, val in self.__dict__.items())

    def __repr__(self):
        return self.__str__()

'''
def dict_to_namespace(config):
    """
    Converts a dictionary to a namespace object.
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively convert nested dictionaries
            new_value = dict_to_namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
'''