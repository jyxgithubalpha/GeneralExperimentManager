"""
DataModule - 数据模块主类

负责协调数据读取、预处理和组装
"""
from typing import Dict

from .data_dataclasses import SourceSpec, GlobalStore
from .data_reader import DataReader
from .data_preprocessors import SingleSourceDataPreprocessorPipeline, MultiSourceDataPreprocessorPipeline
from .data_assembler import GlobalDataAssembler

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
        self.single_source_data_preprocessor_pipeline_dict = single_source_data_preprocessor_pipeline_dict or {}
        self.multi_source_data_preprocessor_pipeline = multi_source_data_preprocessor_pipeline
        self.global_data_assembler = global_data_assembler
        
    
    def prepare_global_store(self, date_start: int, date_end: int) -> GlobalStore:
        source_dict = self.data_reader.read_sources(self.sourcespec_dict, date_start, date_end)
        
        for name, pipeline in self.single_source_data_preprocessor_pipeline_dict.items():
            if name not in source_dict:
                continue
            source_dict[name] = pipeline.fit_transform(source_dict[name])
        
        if self.multi_source_data_preprocessor_pipeline is not None:
            source_dict = self.multi_source_data_preprocessor_pipeline.fit_transform(source_dict)
        
        return self.global_data_assembler.assemble(source_dict)
    