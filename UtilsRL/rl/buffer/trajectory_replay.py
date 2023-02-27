from typing import Optional, Union, Any, Sequence
from typing import Dict as DictLike

import numpy as np

from .base import SimpleReplay, FlexReplay


class TrajectorySimpleReplay(SimpleReplay):
    def __init__(self, max_size: int, max_traj_length: int, field_specs: Optional[DictLike] = None, *args, **kwargs):
        self._max_traj_length = max_traj_length
        super().__init__(max_size, field_specs, *args, **kwargs)

    def reset(self):
        self._pointer = self._size = 0
        self.fields = self.fields or {}
        for _key, _specs in self.field_specs.items():
            initializer = _specs.get("initialzier", np.zeros)
            self.fields[_key] = initializer(shape=[self._max_size, self._max_traj_length] + list(_specs["shape"]), dtype=_specs["dtype"])

    def add_fields(self, new_field_specs: Optional[DictLike] = None):
        new_field_specs = new_field_specs or {}
        self.fields = self.fields or {}
        for _key, _specs in new_field_specs.items():
            _old_specs = self.field_specs.get(_key, None)
            if _old_specs is None or _old_specs != _specs:
                self.field_specs[_key] = _specs
                initializer = _specs.get("initializer", np.zeros)
                self.fields[_key] = initializer(shape=[self._max_size, self._max_traj_length] + list(_specs["shape"]), dtype=_specs["dtype"])

    def add_sample(self, key_or_dict: Union[str, DictLike], data: Optional[Any] = None, timesteps: Union[int, slice] = slice(None)):
        if isinstance(key_or_dict, str):
            key_or_dict = {key_or_dict: data}

        is_single_timestep = isinstance(timesteps, int)
        if isinstance(timesteps, int):
            timesteps = slice(timesteps, timesteps + 1)

        unsqueeze = None
        data_len = None
        for _key, _data in key_or_dict.items():
            if unsqueeze is None:
                unsqueeze = len(_data.shape) == len(self.field_specs[_key]["shape"]) if is_single_timestep else len(_data.shape[1:]) == len(self.field_specs[_key]["shape"])
                data_len = 1 if unsqueeze else _data.shape[0]

            if unsqueeze:
                _data = np.expand_dims(_data, axis=0)
            if is_single_timestep:
                _data = np.expand_dims(_data, axis=1)

            assert _data.shape[1] == len(timesteps), "Trajectory length does not match given timesteps, key: {}, data shape: {}, given timesteps: {}".format(_key, _data.shape, timesteps)
            assert _data.shape[1] <= self._max_traj_length, "Trajectory length exceeds maximum length, key: {}, data shape: {}".format(_key, _data.shape)

            # padding to max_traj_length, a little tricky but simpler than using `np.pad`
            if timesteps == slice(None) and _data.shape[1] < self._max_traj_length:
                _data = np.concatenate([_data, np.zeros(shape=[data_len, self._max_traj_length - _data.shape[1]] + list(_data.shape[2:]), dtype=_data.dtype)], axis=1)

            index_to_go = np.arange(self._pointer, self._pointer + data_len) % self._max_size
            self.fields[_key][index_to_go, timesteps] = _data

        self._pointer = (self._pointer + data_len) % self._max_size
        self._size = min(self._size + data_len, self._max_size)

    def random_batch(self, batch_size: Optional[int] = None, fields: Optional[Sequence[str]] = None, return_idx: bool = False):
        if batch_size is None:
            batch_idx = np.arange(0, len(self))
            np.random.shuffle(batch_idx)
        else:
            buffer_len = len(self) if len(self) != 0 else self._max_size
            batch_idx = np.random.randint(0, buffer_len, batch_size)
        if fields is None:
            fields = self.field_specs.keys()
        batch_data = {_key: self.fields[_key][batch_idx] for _key in fields}
        return (batch_data, batch_idx) if return_idx else batch_data
