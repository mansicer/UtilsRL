import os
import numpy as np
from operator import itemgetter

from typing import Dict as DictLike
from typing import Optional, Any, Union, Sequence, str

from UtilsRL.logger.base_logger import make_unique_name, LogLevel
from UtilsRL.logger.base_logger import BaseLogger
from UtilsRL.logger.tensorboard_logger import TensorboardLogger
from UtilsRL.logger.text_logger import ColoredLogger, FileLogger
from UtilsRL.logger.wandb_logger import WandbLogger


numpy_compatible = np.ndarray
try:
    import torch
    numpy_compatible = torch.Tensor
except ImportError:
    pass


class CompositeLogger(BaseLogger):
    logger_registry = {
        "ColoredLogger": ColoredLogger, 
        "FileLogger": FileLogger, 
        "TensorboardLogger": TensorboardLogger, 
        "WandbLogger": WandbLogger, 
    }
    logger_default_args = {
        "ColoredLogger": {"activate": True}, 
        "FileLogger": {"activate": False}, 
        "TensorboardLogger": {"activate": False}, 
        "WandbLogger": {"activate": False}
    }
    def __init__(self, 
                 log_path: str, 
                 name: str, 
                 unique_name: Optional[str]=None, 
                 activate: bool=True, 
                 level: int=LogLevel.WARNING, 
                 logger_configs: DictLike={}, 
                 *args, **kwargs):
        super().__init__(activate, level)
        
        if unique_name:
            self.unique_name = unique_name
        else:
            self.unique_name = make_unique_name(name)
        self.log_path = os.path.join(log_path, self.unique_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.logger_configs = logger_configs
            
        self.loggers = []
        for _logger_cls, _logger_config in logger_configs.items():
            self.loggers[_logger_cls] = self.logger_registry[_logger_cls](
                **self.logger_default_args[_logger_cls], 
                **_logger_config
            )
        
    def __getattr__(self, __name: str):
        # if the method does not exist for CompositeLogger
        for _logger_cls, _logger in self.loggers.items():
            if hasattr(_logger, __name):
                return _logger.__getattribute__(__name)

    def _call_by_group(self, func: str, group: list, *args, **kwargs):
        def on_exception(_logger_cls):
            raise NameError(f"Class {_logger_cls} does not have method: {func}.")
        return {
            _logger_cls: getattr(self.loggers[_logger_cls], func, on_exception)(*args, **kwargs)\
                for _logger_cls in group
        }
    
    def info(self, msg: str, level: int=LogLevel.WARNING):
        return self._call_by_group(
            func="info", 
            group=["ColoredLogger", "FileLogger"], 
            msg=msg, level=level
        )
        
    def debug(self, msg: str, level: int=LogLevel.WARNING):
        return self._call_by_group(
            func="debug", 
            group=["ColoredLogger", "FileLogger"], 
            msg=msg, level=level
        )
    
    def warning(self, msg: str, level: int=LogLevel.WARNING):
        return self._call_by_group(
            func="warning", 
            group=["ColoredLogger", "FileLogger"], 
            msg=msg, level=level
        )

    def error(self, msg: str, level: int=LogLevel.WARNING):
        return self._call_by_group(
            func="error", 
            group=["ColoredLogger", "FileLogger"], 
            msg=msg, level=level
        )
        
    def log_scalar(
        self, 
        tag: str, 
        value: Union[float, numpy_compatible], 
        step: Optional[int] = None):
        
        return self._call_by_group(
            func="log_scalar", 
            group=["TensorboardLogger", "WandbLogger"], 
            tag=tag, value=value, step=step
        )
        
    def log_scalars(
        self, 
        main_tag: str, 
        tag_scalar_dict: DictLike[str, Union[float, int, numpy_compatible]], 
        step: Optional[int]=None):
        
        return self._call_by_group(
            func="log_scalars", 
            group=["TensorboardLogger", "WandbLogger"], 
            main_tag=main_tag, tag_scalar_dict=tag_scalar_dict, step=step
        )
        
    def log_image(self, 
                  tag: str, 
                  img_tensor: numpy_compatible, 
                  step: Optional[int]=None, 
                  dataformat: str="CHW"):
        
        return self._call_by_group(
            func="log_image", 
            group=["TensorboardLogger"], 
            tag=tag, img_tensor=img_tensor, step=step, dataformat=dataformat
        )
        
    def log_video(
        self, 
        tag: str, 
        vid_tensor: numpy_compatible, 
        step: Optional[int] = None, 
        fps: Optional[Union[int, float]] = 4, 
        dataformat: Optional[str] = "NTCHW"):
        
        return self._call_by_group(
            func="log_video", 
            group=["TensorboardLogger"], 
            tag=tag, vid_tensor=vid_tensor, step=step, fps=fps, dataformat=dataformat
        )
        
    def log_object(self, 
                   name: str, 
                   object: Any, 
                   path: Optional[str]=None):
        
        return self._call_by_group(
            func="log_object",
            group=["TensorboardLogger"], 
            name=name, object=object, path=path
        )
        
    def load_object(self, 
                    name: str, 
                    path: Optional[str]=None):
        return self._call_by_group(
            func="load_object", 
            group=["TensorboardLogger"], 
            name=name, path=path
        )
