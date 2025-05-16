from ._core import config as _config

def load(config_path: str = None):

    config = _config.Config()
    
    if config_path is not None:
        config = config.from_yaml(config_path)
    
    return config.to_dict()

def save(config_dict: dict, config_path: str = 'config.yaml'):
    
    config = _config.Config().from_dict(config_dict)
    
    return config.to_yaml(config_path)
