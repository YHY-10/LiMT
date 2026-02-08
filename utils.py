
import os
from datetime import datetime
def log_dir(file_name, log_directory = "./log"):
    # 获取当前日期和时间
    now = datetime.now()
    date_time_str = now.strftime("%y%m%d_%H%M")

    # 创建路径
    new_folder_path = os.path.join(log_directory, f"{date_time_str}_{file_name}")

    # 创建log文件夹（如果不存在的话）
    os.makedirs(log_directory, exist_ok=True)

    # 创建新文件夹
    os.makedirs(new_folder_path, exist_ok=True)
    return new_folder_path+'/'
    
import torch
import numpy as np
import random

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def model_size(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()  # nelement: number of elements, element_size: size of each element in bytes
        param_sum += param.nelement()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = (param_size + buffer_size) / 1024**2  # Convert to MB
    print(f"Model parameters: {param_sum}")
    print(f"Model size: {total_size:.2f} MB")

import shutil
# def copy_py_files(source_path,save_path):
#     if not os.path.exists(source_path):
#         print(f"Source path {source_path} does not exist.")
#         return
#     for root, dirs, files in os.walk(source_path):
#         for file in files:
#             if file.endswith('.py'):
#                 source_file = os.path.join(root, file)
#                 destination_file = os.path.join(save_path, file)
#                 try:
#                     shutil.copy2(source_file, destination_file)
#                 except Exception as e:
#                     print(f"Failed to copy {source_file} to {destination_file}: {e}")
                    
def copy_py_files(source_path, save_path):
    if not os.path.exists(source_path):
        print(f"Source path {source_path} does not exist.")
        return
    files = os.listdir(source_path)
    for file in files:
        if file.endswith('.py') or file.endswith('.sh'):
            source_file = os.path.join(source_path, file)
            destination_file = os.path.join(save_path, file)
            try:
                shutil.copy2(source_file, destination_file)
            except Exception as e:
                print(f"Failed to copy {source_file} to {destination_file}: {e}")


import pandas as pd
import numpy as np
import bisect
from copy import deepcopy
from typing import Callable, Union, List, Tuple, Dict, Text, Optional
from packaging import version
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch.multiprocessing as mp

def np_ffill(arr: np.array):
    """
    forward fill a 1D numpy array

    Parameters
    ----------
    arr : np.array
        Input numpy 1D array
    """
    mask = np.isnan(arr.astype(float))  # np.isnan only works on np.float
    # get fill index
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

def get_level_index(df: pd.DataFrame, level: Union[str, int]) -> int:
    """

    get the level index of `df` given `level`

    Parameters
    ----------
    df : pd.DataFrame
        data
    level : Union[str, int]
        index level

    Returns
    -------
    int:
        The level index in the multiple index
    """
    if isinstance(level, str):
        try:
            return df.index.names.index(level)
        except (AttributeError, ValueError):
            # NOTE: If level index is not given in the data, the default level index will be ('datetime', 'instrument')
            return ("datetime", "instrument").index(level)
    elif isinstance(level, int):
        return level
    else:
        raise NotImplementedError(f"This type of input is not supported")

def time_to_slc_point(t: Union[None, str, pd.Timestamp]) -> Union[None, pd.Timestamp]:
    """
    Time slicing in Qlib or Pandas is a frequently-used action.
    However, user often input all kinds of data format to represent time.
    This function will help user to convert these inputs into a uniform format which is friendly to time slicing.

    Parameters
    ----------
    t : Union[None, str, pd.Timestamp]
        original time

    Returns
    -------
    Union[None, pd.Timestamp]:
    """
    if t is None:
        # None represents unbounded in Qlib or Pandas(e.g. df.loc[slice(None, "20210303")]).
        return t
    else:
        return pd.Timestamp(t)
    
is_deprecated_lexsorted_pandas = version.parse(pd.__version__) > version.parse("1.3.0")

def lazy_sort_index(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    """
    make the df index sorted

    df.sort_index() will take a lot of time even when `df.is_lexsorted() == True`
    This function could avoid such case

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame:
        sorted dataframe
    """
    idx = df.index if axis == 0 else df.columns
    if (
        not idx.is_monotonic_increasing
        or not is_deprecated_lexsorted_pandas
        and isinstance(idx, pd.MultiIndex)
        and not idx.is_lexsorted()
    ):  # this case is for the old version
        return df.sort_index(axis=axis)
    else:
        return df

class TSDataSampler:
    """
    (T)ime-(S)eries DataSampler
    This is the result of TSDatasetH

    It works like `torch.data.utils.Dataset`, it provides a very convenient interface for constructing time-series
    dataset based on tabular data.
    - On time step dimension, the smaller index indicates the historical data and the larger index indicates the future
      data.

    If user have further requirements for processing data, user could process them based on `TSDataSampler` or create
    more powerful subclasses.

    Known Issues:
    - For performance issues, this Sampler will convert dataframe into arrays for better performance. This could result
      in a different data type

    """

    def __init__(
        self, data: pd.DataFrame, start, end, step_len: int, fillna_type: str = "none", dtype=None, flt_data=None
    ):
        """
        Build a dataset which looks like torch.data.utils.Dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The raw tabular data
        start :
            The indexable start time
        end :
            The indexable end time
        step_len : int
            The length of the time-series step
        fillna_type : int
            How will qlib handle the sample if there is on sample in a specific date.
            none:
                fill with np.nan
            ffill:
                ffill with previous sample
            ffill+bfill:
                ffill with previous samples first and fill with later samples second
        flt_data : pd.Series
            a column of data(True or False) to filter data.
            None:
                kepp all data

        """
        self.start = start
        self.end = end
        self.step_len = step_len
        self.fillna_type = fillna_type
        assert get_level_index(data, "datetime") == 0
        self.data = lazy_sort_index(data)

        kwargs = {"object": self.data}
        if dtype is not None:
            kwargs["dtype"] = dtype

        self.data_arr = np.array(**kwargs)  # Get index from numpy.array will much faster than DataFrame.values!
        # NOTE:
        # - append last line with full NaN for better performance in `__getitem__`
        # - Keep the same dtype will result in a better performance
        self.data_arr = np.append(
            self.data_arr, np.full((1, self.data_arr.shape[1]), np.nan, dtype=self.data_arr.dtype), axis=0
        )
        self.nan_idx = -1  # The last line is all NaN

        # the data type will be changed
        # The index of usable data is between start_idx and end_idx
        self.idx_df, self.idx_map = self.build_index(self.data)
        self.data_index = deepcopy(self.data.index)

        if flt_data is not None:
            if isinstance(flt_data, pd.DataFrame):
                assert len(flt_data.columns) == 1
                flt_data = flt_data.iloc[:, 0]
            # NOTE: bool(np.nan) is True !!!!!!!!
            # make sure reindex comes first. Otherwise extra NaN may appear.
            flt_data = flt_data.reindex(self.data_index).fillna(False).astype(np.bool)
            self.flt_data = flt_data.values
            self.idx_map = self.flt_idx_map(self.flt_data, self.idx_map)
            self.data_index = self.data_index[np.where(self.flt_data)[0]]
        self.idx_map = self.idx_map2arr(self.idx_map)

        self.start_idx, self.end_idx = self.data_index.slice_locs(
            start=time_to_slc_point(start), end=time_to_slc_point(end)
        )
        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)  # for better performance

        # del self.data  # save memory

    @staticmethod
    def idx_map2arr(idx_map):
        # pytorch data sampler will have better memory control without large dict or list
        # - https://github.com/pytorch/pytorch/issues/13243
        # - https://github.com/airctic/icevision/issues/613
        # So we convert the dict into int array.
        # The arr_map is expected to behave the same as idx_map

        dtype = np.int32
        # set a index out of bound to indicate the none existing
        no_existing_idx = (np.iinfo(dtype).max, np.iinfo(dtype).max)

        max_idx = max(idx_map.keys())
        arr_map = []
        for i in range(max_idx + 1):
            arr_map.append(idx_map.get(i, no_existing_idx))
        arr_map = np.array(arr_map, dtype=dtype)
        return arr_map

    @staticmethod
    def flt_idx_map(flt_data, idx_map):
        idx = 0
        new_idx_map = {}
        for i, exist in enumerate(flt_data):
            if exist:
                new_idx_map[idx] = idx_map[i]
                idx += 1
        return new_idx_map

    def get_index(self):
        """
        Get the pandas index of the data, it will be useful in following scenarios
        - Special sampler will be used (e.g. user want to sample day by day)
        """
        return self.data_index[self.start_idx : self.end_idx]

    def config(self, **kwargs):
        # Config the attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def build_index(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        The relation of the data

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe with <datetime, DataFrame>

        Returns
        -------
        Tuple[pd.DataFrame, dict]:
            1) the first element:  reshape the original index into a <datetime(row), instrument(column)> 2D dataframe
                instrument SH600000 SH600004 SH600006 SH600007 SH600008 SH600009  ...
                datetime
                2021-01-11        0        1        2        3        4        5  ...
                2021-01-12     4146     4147     4148     4149     4150     4151  ...
                2021-01-13     8293     8294     8295     8296     8297     8298  ...
                2021-01-14    12441    12442    12443    12444    12445    12446  ...
            2) the second element:  {<original index>: <row, col>}
        """
        # object incase of pandas converting int to float
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
        idx_df = lazy_sort_index(idx_df.unstack())
        # NOTE: the correctness of `__getitem__` depends on columns sorted here
        idx_df = lazy_sort_index(idx_df, axis=1)

        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    @property
    def empty(self):
        return len(self) == 0

    def _get_indices(self, row: int, col: int) -> np.array:
        """
        get series indices of self.data_arr from the row, col indices of self.idx_df

        Parameters
        ----------
        row : int
            the row in self.idx_df
        col : int
            the col in self.idx_df

        Returns
        -------
        np.array:
            The indices of data of the data
        """
        indices = self.idx_arr[max(row - self.step_len + 1, 0) : row + 1, col]

        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])

        if self.fillna_type == "ffill" or self.fillna_type == 'ffill+zero': # 前值填充+0填充

            # ffill是可以在索引阶段完成

            indices = np_ffill(indices)

        elif self.fillna_type == "ffill+bfill":

            indices = np_ffill(np_ffill(indices)[::-1])[::-1]

        else:

            # assertion that fillna_type must be 'none' or 'mean'

            assert self.fillna_type in ['none', 'mean', 'ffill+mean']

            # assert self.fillna_type == "none"
        return indices

    def _get_row_col(self, idx) -> Tuple[int]:
        """
        get the col index and row index of a given sample index in self.idx_df

        Parameters
        ----------
        idx :
            the input of  `__getitem__`

        Returns
        -------
        Tuple[int]:
            the row and col index
        """
        # The the right row number `i` and col number `j` in idx_df
        if isinstance(idx, (int, np.integer)):
            real_idx = self.start_idx + idx
            if self.start_idx <= real_idx < self.end_idx:
                i, j = self.idx_map[real_idx]  # TODO: The performance of this line is not good
            else:
                raise KeyError(f"{real_idx} is out of [{self.start_idx}, {self.end_idx})")
        elif isinstance(idx, tuple):
            # <TSDataSampler object>["datetime", "instruments"]
            date, inst = idx
            date = pd.Timestamp(date)
            i = bisect.bisect_right(self.idx_df.index, date) - 1
            # NOTE: This relies on the idx_df columns sorted in `__init__`
            j = bisect.bisect_left(self.idx_df.columns, inst)
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return i, j

    def __getitem__(self, idx: Union[int, Tuple[object, str], List[int]]):
        """
        # We have two method to get the time-series of a sample
        tsds is a instance of TSDataSampler

        # 1) sample by int index directly
        tsds[len(tsds) - 1]

        # 2) sample by <datetime,instrument> index
        tsds['2016-12-31', "SZ300315"]

        # The return value will be similar to the data retrieved by following code
        df.loc(axis=0)['2015-01-01':'2016-12-31', "SZ300315"].iloc[-30:]

        Parameters
        ----------
        idx : Union[int, Tuple[object, str]]
        """
        # Multi-index type
        mtit = (list, np.ndarray)
        if isinstance(idx, mtit):
            indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
            indices = np.concatenate(indices)
        else:
            indices = self._get_indices(*self._get_row_col(idx))

        # 1) for better performance, use the last nan line for padding the lost date
        # 2) In case of precision problems. We use np.float64. # TODO: I'm not sure if whether np.float64 will result in
        # precision problems. It will not cause any problems in my tests at least
        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)

        data = self.data_arr[indices]
        if isinstance(idx, mtit):
            # if we get multiple indexes, addition dimension should be added.
            # <sample_idx, step_idx, feature_idx>
            data = data.reshape(-1, self.step_len, *data.shape[1:])
        
        if 'zero' in self.fillna_type:
            data = data.copy()  # Make a copy to avoid modifying original data
            data = np.where(np.isinf(data), np.nan, data)
            data = np.where(np.isnan(data), 0, data)

            if np.isnan(data).any():
                raise Exception("Data contains NaN values")
        return data

    def __len__(self):
        return self.end_idx - self.start_idx



class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index(), dtype='float32').groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                # print('***************', np.arange(idx, idx + count))
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source.loc[self.start:self.end])