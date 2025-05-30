from ._core import config as _config

def load(config_path: str = None):
    """
    Load configuration from a YAML file or return default configuration.

    If a path to a YAML configuration file is provided, the configuration is
    loaded from that file. Otherwise, a default configuration is used.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file. If None, uses default configuration.

    Returns
    -------
    dict
        Configuration data as a dictionary.
    """
    
    config = _config.Config()
    
    if config_path is not None:
        config = config.from_yaml(config_path)
    
    return config.to_dict()

def save(config_dict: dict, config_path: str = 'config.yaml'):
    """
    Save a configuration dictionary to a YAML file.

    Converts the provided configuration dictionary into a Config object,
    then writes it to a YAML file at the specified path.

    Parameters
    ----------
    config_dict : dict
        Configuration data to save.
    config_path : str, optional
        Path where the YAML file will be saved. Defaults to 'config.yaml'.

    Returns
    -------
    str
        Path to the saved YAML configuration file.
    """
    
    config = _config.Config().from_dict(config_dict)
    
    return config.to_yaml(config_path)
