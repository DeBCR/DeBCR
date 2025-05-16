import yaml

from dataclasses import dataclass, asdict

@dataclass
class Config:

    weights_path: str = './checkpoint/'
    batch_size: int = 32
    NUM_STEPS: int = 2000
    patience: int = 10
    learning_rate: float = 0.001
    save_freq: int = 50
    val_freq: int = 100

    def update(self, config_dict: dict):
        if isinstance(config_dict, dict):
            for param in self.__annotations__:
                if param in config_dict:
                    setattr(self, param, config_dict[param])
    
    def to_dict(self):
        return asdict(self)

    def to_yaml(self, config_path: str):        
        with open(config_path, 'w') as fout:
            yaml.dump(self.to_dict(), fout)
        return config_path
            
    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls()        
        config.update(config_dict)
        return config
    
    @classmethod
    def from_yaml(cls, config_path: str):  
        config = cls()
        with open(config_path, 'r') as fin:
            user_dict = yaml.safe_load(fin)
        config.update(user_dict)
        return config