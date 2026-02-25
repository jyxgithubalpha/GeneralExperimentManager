"""
DataModule coordinates reading, preprocessing and assembly.
"""


from typing import Dict

from .data_assembler import GlobalDataAssembler
from .data_dataclasses import GlobalStore, SourceSpec
from .data_preprocessors import (
    MultiSourceDataPreprocessorPipeline,
    SingleSourceDataPreprocessorPipeline,
)
from .data_reader import DataReader


class DataModule:
    def __init__(
        self,
        sourcespec_dict: Dict[str, SourceSpec],
        data_reader: DataReader,
        single_source_data_preprocessor_pipeline_dict: Dict[str, SingleSourceDataPreprocessorPipeline],
        multi_source_data_preprocessor_pipeline: MultiSourceDataPreprocessorPipeline,
        global_data_assembler: GlobalDataAssembler,
    ):
        self.sourcespec_dict = sourcespec_dict
        self.data_reader = data_reader
        self.single_source_data_preprocessor_pipeline_dict = (
            single_source_data_preprocessor_pipeline_dict or {}
        )
        self.multi_source_data_preprocessor_pipeline = multi_source_data_preprocessor_pipeline
        self.global_data_assembler = global_data_assembler
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not self.sourcespec_dict:
            raise ValueError("sourcespec_dict must not be empty.")

        supported_formats = {"feather"}
        for source_key, source_spec in self.sourcespec_dict.items():
            if source_spec.name != source_key:
                raise ValueError(
                    "SourceSpec name mismatch: "
                    f"dict key='{source_key}' but SourceSpec.name='{source_spec.name}'."
                )
            if source_spec.format not in supported_formats:
                raise ValueError(
                    f"Unsupported source format '{source_spec.format}' for source "
                    f"'{source_key}'. Supported formats: {sorted(supported_formats)}."
                )

        unknown_pipeline_sources = sorted(
            set(self.single_source_data_preprocessor_pipeline_dict.keys())
            - set(self.sourcespec_dict.keys())
        )
        if unknown_pipeline_sources:
            raise ValueError(
                "single_source_data_preprocessor_pipeline_dict contains unknown "
                f"sources: {unknown_pipeline_sources}"
            )

    def _read_sources(self, date_start: int, date_end: int):
        return self.data_reader.read_sources(self.sourcespec_dict, date_start, date_end)

    def _apply_single_source_preprocessors(self, source_dict):
        for source_name, pipeline in self.single_source_data_preprocessor_pipeline_dict.items():
            if source_name not in source_dict:
                continue
            source_dict[source_name] = pipeline.fit_transform(source_dict[source_name])
        return source_dict

    def _apply_multi_source_preprocessor(self, source_dict):
        if self.multi_source_data_preprocessor_pipeline is None:
            return source_dict
        return self.multi_source_data_preprocessor_pipeline.fit_transform(source_dict)

    def prepare_global_store(self, date_start: int, date_end: int) -> GlobalStore:
        source_dict = self._read_sources(date_start, date_end)
        source_dict = self._apply_single_source_preprocessors(source_dict)
        source_dict = self._apply_multi_source_preprocessor(source_dict)
        return self.global_data_assembler.assemble(source_dict)
