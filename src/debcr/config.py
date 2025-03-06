from ._core import config

def load(config_path: str = None):

    config_obj = config.Config()
    
    if config_path is not None:
        config_obj = config_obj.from_yaml(config_path)
    
    return config_obj.to_dict()

def save(config_dict: dict, config_path: str = 'config.yaml'):
    
    config_obj = config.Config().from_dict(config_dict)
    
    return config_obj.to_yaml(config_path)
