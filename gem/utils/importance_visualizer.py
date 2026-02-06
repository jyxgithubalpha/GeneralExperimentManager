"""
Importance Vector Visualization Module

提供特征重要性向量的可视化功能，展示所有特征 importance 随 split 的变化。
主要输出：
1. CSV 数据文件 - 所有特征在所有 split 的 importance
2. 全特征热力图 - 可视化所有特征随时间的变化
3. 动画 - 动态展示特征重要性排名变化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ImportanceFrame:
    """单个 split 的 importance 数据"""
    split_id: int
    importance_vector: np.ndarray
    test_date_start: Optional[int] = None
    test_date_end: Optional[int] = None


class ImportanceVisualizer:
    """特征重要性可视化器 - 支持所有特征"""
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 12),
    ):
        """
        Args:
            feature_names: 特征名列表
            figsize: 图形大小
        """
        self.feature_names = feature_names
        self.figsize = figsize
        self.frames: List[ImportanceFrame] = []
        self._importance_df: Optional[pd.DataFrame] = None
    
    def extract_from_results(
        self,
        results: Dict[int, Any],
        splitspec_list: Optional[List[Any]] = None,
    ) -> List[ImportanceFrame]:
        """从 SplitResult 字典中提取 importance 数据"""
        self.frames = []
        
        spec_map = {}
        if splitspec_list:
            for spec in splitspec_list:
                spec_map[spec.split_id] = spec
        
        for split_id in sorted(results.keys()):
            result = results[split_id]
            
            if result.skipped or result.importance_vector is None:
                continue
            
            if len(result.importance_vector) == 0:
                continue
            
            test_date_start = None
            test_date_end = None
            if split_id in spec_map:
                spec = spec_map[split_id]
                if spec.test_date_list:
                    test_date_start = spec.test_date_list[0]
                    test_date_end = spec.test_date_list[-1]
            
            frame = ImportanceFrame(
                split_id=split_id,
                importance_vector=result.importance_vector,
                test_date_start=test_date_start,
                test_date_end=test_date_end,
            )
            self.frames.append(frame)
        
        # 构建 DataFrame
        self._build_importance_dataframe()
        
        return self.frames
    
    def _get_feature_name(self, idx: int) -> str:
        """获取特征名"""
        if self.feature_names and idx < len(self.feature_names):
            return self.feature_names[idx]
        return f"F{idx}"
    
    def _build_importance_dataframe(self):
        """构建完整的 importance DataFrame"""
        if not self.frames:
            self._importance_df = None
            return
        
        n_features = len(self.frames[0].importance_vector)
        feature_names = [self._get_feature_name(i) for i in range(n_features)]
        
        data = {}
        for frame in self.frames:
            col_name = f"split_{frame.split_id}"
            if frame.test_date_start:
                col_name = f"{frame.test_date_start}"
            data[col_name] = frame.importance_vector
        
        self._importance_df = pd.DataFrame(data, index=feature_names)
        self._importance_df.index.name = 'feature'
    
    def get_importance_dataframe(self) -> Optional[pd.DataFrame]:
        """获取完整的 importance DataFrame"""
        return self._importance_df
    
    def export_to_csv(self, output_path: str):
        """导出所有特征的 importance 到 CSV"""
        if self._importance_df is None:
            print("No importance data to export")
            return
        
        self._importance_df.to_csv(output_path)
        print(f"Importance data exported to {output_path}")
        print(f"  - Features: {len(self._importance_df)}")
        print(f"  - Splits: {len(self._importance_df.columns)}")
    
    def plot_full_heatmap(
        self,
        output_path: Optional[str] = None,
        show: bool = True,
        sort_by: str = 'mean',
        normalize: str = 'zscore',
    ):
        """
        绘制所有特征的热力图
        
        Args:
            output_path: 输出文件路径
            show: 是否显示图形
            sort_by: 排序方式 ('mean', 'std', 'max', 'none')
            normalize: 标准化方式 ('zscore', 'minmax', 'rank', 'none')
        """
        if self._importance_df is None or self._importance_df.empty:
            print("No importance data to visualize")
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        df = self._importance_df.copy()
        
        # 标准化 importance
        if normalize == 'zscore':
            # Z-score 标准化 (按特征)
            df = (df.T - df.T.mean()) / (df.T.std() + 1e-10)
            df = df.T
        elif normalize == 'minmax':
            # Min-Max 标准化 (按特征)
            df = (df.T - df.T.min()) / (df.T.max() - df.T.min() + 1e-10)
            df = df.T
        elif normalize == 'rank':
            # 排名标准化 (按列/split)
            df = df.rank(axis=0) / len(df)
        # else: 'none' - 不做标准化
        
        # 排序特征 (使用原始数据排序)
        orig_df = self._importance_df
        if sort_by == 'mean':
            order = orig_df.mean(axis=1).sort_values(ascending=False).index
        elif sort_by == 'std':
            order = orig_df.std(axis=1).sort_values(ascending=False).index
        elif sort_by == 'max':
            order = orig_df.max(axis=1).sort_values(ascending=False).index
        else:
            order = orig_df.index
        
        df = df.loc[order]
        
        # 计算合适的图形大小
        n_features = len(df)
        n_splits = len(df.columns)
        fig_height = max(10, min(100, n_features * 0.05))
        fig_width = max(12, min(30, n_splits * 0.8))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        im = ax.imshow(df.values, aspect='auto', cmap='YlOrRd')
        
        ax.set_xticks(np.arange(n_splits))
        ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=8)
        
        ax.set_yticks([])
        
        ax.set_xlabel('Split / Date', fontsize=12)
        ax.set_ylabel(f'Feature (n={n_features})', fontsize=12)
        norm_str = f", {normalize} normalized" if normalize != 'none' else ""
        ax.set_title(f'All Features Importance Heatmap (sorted by {sort_by}{norm_str})', fontsize=14)
        
        plt.colorbar(im, ax=ax, label='Importance', shrink=0.5)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Full heatmap saved to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, ax
    
    def create_importance_animation(
        self,
        output_path: Optional[str] = None,
        interval: int = 800,
        show: bool = True,
        sort_by: str = 'mean',
    ):
        """
        创建所有特征 importance 变化的动画
        
        X轴: 因子名 (固定顺序)
        Y轴: importance 强度
        
        Args:
            output_path: 输出路径
            interval: 帧间隔 (毫秒)
            show: 是否显示
            sort_by: 特征排序方式 ('mean', 'std', 'max', 'none')
        """
        if not self.frames:
            print("No importance frames to visualize")
            return None
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        n_features = len(self.frames[0].importance_vector)
        
        # 确定特征顺序 (固定)
        all_importance = np.stack([f.importance_vector for f in self.frames])
        mean_importance = np.mean(all_importance, axis=0)
        
        if sort_by == 'mean':
            feature_order = np.argsort(mean_importance)[::-1]
        elif sort_by == 'std':
            feature_order = np.argsort(np.std(all_importance, axis=0))[::-1]
        elif sort_by == 'max':
            feature_order = np.argsort(np.max(all_importance, axis=0))[::-1]
        else:
            feature_order = np.arange(n_features)
        
        # 获取全局最大值
        max_val = np.max(all_importance) * 1.1
        
        # 图形大小根据特征数调整
        fig_width = max(20, min(60, n_features * 0.02))
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # X轴位置
        x_pos = np.arange(n_features)
        
        # 初始化条形图
        bars = ax.bar(x_pos, np.zeros(n_features), width=0.8, color='steelblue', alpha=0.8)
        
        # 设置坐标轴
        ax.set_ylim(0, max_val)
        ax.set_xlim(-0.5, n_features - 0.5)
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_xlabel(f'Feature (sorted by {sort_by}, n={n_features})', fontsize=10)
        
        # 不显示 X 轴标签，特征太多会拥挤
        ax.set_xticks([])
        
        # 添加 split 信息文本
        info_text = ax.text(
            0.02, 0.95, '', 
            transform=ax.transAxes, 
            fontsize=14,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        def init():
            for bar in bars:
                bar.set_height(0)
            info_text.set_text('')
            return list(bars) + [info_text]
        
        def update(frame_idx):
            frame = self.frames[frame_idx]
            # 按固定顺序获取 importance
            importance = frame.importance_vector[feature_order]
            
            for bar, val in zip(bars, importance):
                bar.set_height(val)
            
            # 更新信息文本
            info_str = f"Split {frame.split_id}"
            if frame.test_date_start:
                info_str = f"Date: {frame.test_date_start}"
            info_text.set_text(info_str)
            
            return list(bars) + [info_text]
        
        anim = FuncAnimation(
            fig, update, frames=len(self.frames),
            init_func=init, interval=interval, 
            repeat=True, blit=True
        )
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix == '.gif':
                anim.save(str(output_path), writer='pillow', fps=1000//interval)
                print(f"Animation saved to {output_path}")
            elif output_path.suffix == '.mp4':
                anim.save(str(output_path), writer='ffmpeg', fps=1000//interval)
                print(f"Animation saved to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return anim
    
    def plot_importance_distribution(
        self,
        output_path: Optional[str] = None,
        show: bool = True,
    ):
        """绘制每个 split 的 importance 分布"""
        if self._importance_df is None:
            print("No importance data to visualize")
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.boxplot(
            [self._importance_df[col].values for col in self._importance_df.columns],
            labels=self._importance_df.columns,
        )
        
        ax.set_xlabel('Split / Date', fontsize=12)
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title('Importance Distribution per Split', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Distribution plot saved to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, ax


def visualize_importance_animation(
    results: Dict[int, Any],
    feature_names: Optional[List[str]] = None,
    splitspec_list: Optional[List[Any]] = None,
    output_dir: Optional[str] = None,
    interval: int = 800,
    show: bool = True,
) -> ImportanceVisualizer:
    """
    便捷函数：从结果中提取并可视化所有特征的 importance
    
    输出文件：
    - importance_data.csv: 所有特征在所有 split 的 importance 数据
    - importance_heatmap.png: 全特征热力图
    - importance_animation.gif: 特征排名动画
    - importance_distribution.png: 各 split 的 importance 分布
    """
    visualizer = ImportanceVisualizer(feature_names=feature_names)
    
    visualizer.extract_from_results(results, splitspec_list)
    
    if not visualizer.frames:
        print("No valid importance data found in results")
        return visualizer
    
    n_features = len(visualizer.frames[0].importance_vector)
    print(f"Found {len(visualizer.frames)} splits with {n_features} features")
    
    # 生成输出路径
    csv_path = None
    heatmap_path = None
    gif_path = None
    dist_path = None
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = str(output_dir / "importance_data.csv")
        heatmap_path = str(output_dir / "importance_heatmap.png")
        gif_path = str(output_dir / "importance_animation.gif")
        dist_path = str(output_dir / "importance_distribution.png")
    
    # 1. 导出 CSV
    if csv_path:
        visualizer.export_to_csv(csv_path)
    
    # 2. 全特征热力图
    visualizer.plot_full_heatmap(output_path=heatmap_path, show=show, sort_by='mean')
    
    # 3. 全特征动画 (X轴=因子名固定, Y轴=强度)
    visualizer.create_importance_animation(output_path=gif_path, interval=interval, show=show, sort_by='mean')
    
    # 4. 分布图
    visualizer.plot_importance_distribution(output_path=dist_path, show=show)
    
    return visualizer
