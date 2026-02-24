from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

from .data_dataclasses import SplitGeneratorOutput, SplitSpec


class SplitGenerator(ABC):
    @abstractmethod
    def generate(self) -> SplitGeneratorOutput:
        pass

    @staticmethod
    def _date_to_int(d: datetime) -> int:
        return int(d.strftime("%Y%m%d"))

    @staticmethod
    def _int_to_date(d: int) -> datetime:
        return datetime.strptime(str(d), "%Y%m%d")

    @staticmethod
    def _generate_date_range(start: int, end: int) -> np.ndarray:
        start_ts = datetime.strptime(str(start), "%Y%m%d")
        end_ts = datetime.strptime(str(end), "%Y%m%d")
        if start_ts > end_ts:
            raise ValueError("start date must be <= end date")

        dates: List[int] = []
        current = start_ts
        while current <= end_ts:
            dates.append(int(current.strftime("%Y%m%d")))
            current += timedelta(days=1)
        return np.array(dates, dtype=np.int32)


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
        expanding: bool = False,
    ):
        if train_len <= 0 or val_len <= 0 or test_len <= 0:
            raise ValueError("train_len, val_len and test_len must all be > 0")
        if step <= 0:
            raise ValueError("step must be > 0")
        if gap < 0:
            raise ValueError("gap must be >= 0")

        self.test_date_start = test_date_start
        self.test_date_end = test_date_end
        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len
        self.gap = gap
        self.step = step
        self.expanding = expanding

    def _build_split_spec(self, split_id: int, split_dates: np.ndarray, start_idx: int, first_train_start_idx: int) -> SplitSpec:
        if self.expanding:
            train_start_idx = first_train_start_idx
        else:
            train_start_idx = start_idx

        train_end_idx = start_idx + self.train_len - 1
        val_start_idx = train_end_idx + 1 + self.gap
        val_end_idx = val_start_idx + self.val_len - 1
        test_start_idx = val_end_idx + 1 + self.gap
        test_end_idx = test_start_idx + self.test_len - 1

        train_date_list = split_dates[train_start_idx : train_end_idx + 1].tolist()
        val_date_list = split_dates[val_start_idx : val_end_idx + 1].tolist()
        test_date_list = split_dates[test_start_idx : test_end_idx + 1].tolist()

        return SplitSpec(
            split_id=split_id,
            train_date_list=train_date_list,
            val_date_list=val_date_list,
            test_date_list=test_date_list,
        )

    def generate(self) -> SplitGeneratorOutput:
        test_start_ts = self._int_to_date(self.test_date_start)
        test_end_ts = self._int_to_date(self.test_date_end)

        offset_days = self.train_len + self.gap + self.val_len + self.gap
        first_train_start_ts = test_start_ts - timedelta(days=offset_days)

        split_dates = self._generate_date_range(
            self._date_to_int(first_train_start_ts),
            self._date_to_int(test_end_ts),
        )

        split_total = self.train_len + self.gap + self.val_len + self.gap + self.test_len
        splitspec_list: List[SplitSpec] = []

        start_idx = 0
        split_id = 0
        first_train_start_idx = 0

        while start_idx + split_total <= len(split_dates):
            splitspec_list.append(
                self._build_split_spec(
                    split_id=split_id,
                    split_dates=split_dates,
                    start_idx=start_idx,
                    first_train_start_idx=first_train_start_idx,
                )
            )
            split_id += 1
            start_idx += self.step

        return SplitGeneratorOutput(
            splitspec_list=splitspec_list,
            date_start=int(split_dates[0]),
            date_end=int(split_dates[-1]),
        )


class TimeSeriesKFoldSplitGenerator(SplitGenerator):
    def __init__(
        self,
        test_date_start: int = 20230101,
        test_date_end: int = 20241231,
        train_val_date_start: int = 20220101,
        pretrain_date_start: Optional[int] = None,
        n_folds: int = 5,
        gap: int = 0,
    ):
        if n_folds <= 0:
            raise ValueError("n_folds must be > 0")
        if gap < 0:
            raise ValueError("gap must be >= 0")

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
        train_val_end_ts = test_start_ts - timedelta(days=self.gap + 1)

        pretrain_start_ts = None
        if self.pretrain_date_start is not None:
            pretrain_start_ts = self._int_to_date(self.pretrain_date_start)

        full_start = pretrain_start_ts if pretrain_start_ts is not None else train_val_start_ts
        full_end = test_end_ts

        pretrain_dates: List[int] = []
        if pretrain_start_ts is not None:
            pretrain_dates = list(
                self._generate_date_range(
                    self._date_to_int(pretrain_start_ts),
                    self._date_to_int(train_val_start_ts - timedelta(days=1)),
                )
            )

        train_val_dates = list(
            self._generate_date_range(
                self._date_to_int(train_val_start_ts),
                self._date_to_int(train_val_end_ts),
            )
        )
        test_date_list = list(
            self._generate_date_range(
                self._date_to_int(test_start_ts),
                self._date_to_int(test_end_ts),
            )
        )

        splitspec_list: List[SplitSpec] = []
        if not train_val_dates:
            return SplitGeneratorOutput(
                splitspec_list=splitspec_list,
                date_start=self._date_to_int(full_start),
                date_end=self._date_to_int(full_end),
            )

        fold_size = max(1, len(train_val_dates) // self.n_folds)

        for fold_id in range(self.n_folds):
            val_start_idx = fold_id * fold_size
            if val_start_idx >= len(train_val_dates):
                break

            if fold_id < self.n_folds - 1:
                val_end_idx = min((fold_id + 1) * fold_size, len(train_val_dates))
            else:
                val_end_idx = len(train_val_dates)

            val_date_list = train_val_dates[val_start_idx:val_end_idx]

            train_date_list = pretrain_dates.copy()
            for idx, day in enumerate(train_val_dates):
                if idx < val_start_idx - self.gap or idx >= val_end_idx + self.gap:
                    train_date_list.append(int(day))

            if not train_date_list or not val_date_list:
                continue

            splitspec_list.append(
                SplitSpec(
                    split_id=fold_id,
                    train_date_list=[int(d) for d in train_date_list],
                    val_date_list=[int(d) for d in val_date_list],
                    test_date_list=[int(d) for d in test_date_list],
                )
            )

        return SplitGeneratorOutput(
            splitspec_list=splitspec_list,
            date_start=self._date_to_int(full_start),
            date_end=self._date_to_int(full_end),
        )
