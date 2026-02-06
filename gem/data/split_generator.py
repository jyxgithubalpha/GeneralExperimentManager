from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass
from .data_dataclasses import SplitSpec, SplitGeneratorOutput


class SplitGenerator(ABC):
    @abstractmethod
    def generate(self) -> SplitGeneratorOutput:
        pass
    
    @staticmethod
    def _date_to_int(d: pd.Timestamp) -> int:
        return int(d.strftime("%Y%m%d"))
    
    @staticmethod
    def _int_to_date(d: int) -> pd.Timestamp:
        return pd.to_datetime(str(d), format="%Y%m%d")
    
    @staticmethod
    def _generate_date_range(start: int, end: int) -> np.ndarray:
        start_ts = pd.to_datetime(str(start), format="%Y%m%d")
        end_ts = pd.to_datetime(str(end), format="%Y%m%d")
        if start_ts > end_ts:
            raise ValueError("start date must be less than end date")
        dates = pd.date_range(start=start_ts, end=end_ts, freq="D")
        return np.array([int(d.strftime("%Y%m%d")) for d in dates], dtype=np.int32)


class RollingWindowSplitGenerator(SplitGenerator):
    def __init__(
        self,
        test_date_start: int = 20230101,
        test_date_end: int = 20241231,
        train_len: int = 90,
        val_len: int = 14,
        test_len: int = 14,
        gap: int = 0,
        step: int = 7,
        expanding: bool = False
    ):
        self.test_date_start = test_date_start
        self.test_date_end = test_date_end
        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len
        self.gap = gap
        self.step = step
        self.expanding = expanding
    
    def generate(self) -> SplitGeneratorOutput:
        test_start_ts = self._int_to_date(self.test_date_start)
        test_end_ts = self._int_to_date(self.test_date_end)
        
        offset_days = self.train_len + self.gap + self.val_len + self.gap
        first_train_start_ts = test_start_ts - pd.Timedelta(days=offset_days)
        
        split_dates = self._generate_date_range(
            self._date_to_int(first_train_start_ts),
            self._date_to_int(test_end_ts)
        )
        
        splitspec_list: List[SplitSpec] = []
        split_total = self.train_len + self.gap + self.val_len + self.gap + self.test_len
        
        start_idx = 0
        split_id = 0
        first_train_start_idx = 0  # 用于 expanding 模式
        
        while start_idx + split_total <= len(split_dates):
            if self.expanding:
                train_start_idx = first_train_start_idx
            else:
                train_start_idx = start_idx
            train_end_idx = start_idx + self.train_len - 1
            
            val_start_idx = train_end_idx + 1 + self.gap
            val_end_idx = val_start_idx + self.val_len - 1
            
            test_start_idx = val_end_idx + 1 + self.gap
            test_end_idx = test_start_idx + self.test_len - 1
            
            train_date_list = split_dates[train_start_idx:train_end_idx + 1].tolist()
            val_date_list = split_dates[val_start_idx:val_end_idx + 1].tolist()
            test_date_list = split_dates[test_start_idx:test_end_idx + 1].tolist()
            
            splitspec = SplitSpec(
                split_id=split_id,
                train_date_list=train_date_list,
                val_date_list=val_date_list,
                test_date_list=test_date_list,
                anchor_time=None,
            )
            splitspec_list.append(splitspec)
            
            split_id += 1
            start_idx += self.step
        
        date_start = int(split_dates[0])
        date_end = int(split_dates[-1])
        
        return SplitGeneratorOutput(
            splitspec_list=splitspec_list,
            date_start=date_start,
            date_end=date_end,
        )


class TimeSeriesKFoldSplitGenerator(SplitGenerator):
    def __init__(
        self,
        test_date_start: int = 20230101,
        test_date_end: int = 20241231,
        train_val_date_start: int = 20220101,
        pretrain_date_start: Optional[int] = None,
        n_folds: int = 5,
        gap: int = 0
    ):
        self.test_date_start = test_date_start
        self.test_date_end = test_date_end
        self.train_val_date_start = train_val_date_start
        self.pretrain_date_start = pretrain_date_start
        self.n_folds = n_folds
        self.gap = gap
    
    def generate(self) -> SplitGeneratorOutput:
        test_start_ts = self._int_to_date(self.test_date_start)
        test_end_ts = self._int_to_date(self.test_date_end)
        train_val_start_ts = self._int_to_date(self.train_val_date_start)
        train_val_end_ts = test_start_ts - pd.Timedelta(days=self.gap + 1)
        
        pretrain_start_ts = None
        if self.pretrain_date_start is not None:
            pretrain_start_ts = self._int_to_date(self.pretrain_date_start)
        
        if pretrain_start_ts is not None:
            full_start = pretrain_start_ts
        else:
            full_start = train_val_start_ts
        full_end = test_end_ts
        
        pretrain_dates: List[int] = []
        if pretrain_start_ts is not None:
            pretrain_range = self._generate_date_range(
                self._date_to_int(pretrain_start_ts),
                self._date_to_int(train_val_start_ts - pd.Timedelta(days=1))
            )
            pretrain_dates = list(pretrain_range)
        
        train_val_dates = list(self._generate_date_range(
            self._date_to_int(train_val_start_ts),
            self._date_to_int(train_val_end_ts)
        ))
        
        test_date_list = list(self._generate_date_range(
            self._date_to_int(test_start_ts),
            self._date_to_int(test_end_ts)
        ))
        
        splitspec_list: List[SplitSpec] = []
        fold_size = len(train_val_dates) // self.n_folds
        
        for fold_id in range(self.n_folds):
            val_start_idx = fold_id * fold_size
            val_end_idx = (fold_id + 1) * fold_size if fold_id < self.n_folds - 1 else len(train_val_dates)
            
            val_date_list = train_val_dates[val_start_idx:val_end_idx]
            
            train_date_list = pretrain_dates.copy()
            for i, d in enumerate(train_val_dates):
                if i < val_start_idx - self.gap or i >= val_end_idx + self.gap:
                    train_date_list.append(int(d))
            
            if not train_date_list:
                continue
            
            splitspec = SplitSpec(
                split_id=fold_id,
                train_date_list=[int(d) for d in train_date_list],
                val_date_list=[int(d) for d in val_date_list],
                test_date_list=[int(d) for d in test_date_list],
                anchor_time=None,
            )
            splitspec_list.append(splitspec)
        
        date_start = self._date_to_int(full_start)
        date_end = self._date_to_int(full_end)
        
        return SplitGeneratorOutput(
            splitspec_list=splitspec_list,
            date_start=date_start,
            date_end=date_end,
        )