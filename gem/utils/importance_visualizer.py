"""
Importance Vector Visualization Module

Provides visualization functionality for feature importance vectors, showing importance changes across splits.
Main outputs:
1. CSV data file - importance of all features across all splits
2. Full feature heatmap - visualizes feature changes over time
3. Animation - dynamically shows feature importance ranking changes
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ImportanceFrame:
    """Importance data for a single split"""
    split_id: int
    importance_vector: np.ndarray
    test_date_start: Optional[int] = None
    test_date_end: Optional[int] = None


class ImportanceVisualizer:
    """Feature importance visualizer - supports all features"""
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 12),
    ):
        """
        Args:
            feature_names: Feature name list
            figsize: Figure size
        """
        self.feature_names = feature_names
        self.figsize = figsize
        self.frames: List[ImportanceFrame] = []
        self._importance_df: Optional[pl.DataFrame] = None
    
    def extract_from_results(
        self,
        results: Dict[int, Any],
        splitspec_list: Optional[List[Any]] = None,
    ) -> List[ImportanceFrame]:
        """Extract importance data from SplitResult dictionary"""
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
        
        # Build DataFrame
        self._build_importance_dataframe()
        
        return self.frames
    
    def _get_feature_name(self, idx: int) -> str:
        """Get feature name"""
        if self.feature_names and idx < len(self.feature_names):
            return self.feature_names[idx]
        return f"F{idx}"
    
    def _build_importance_dataframe(self):
        """Build complete importance DataFrame"""
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
        
        # Add feature names as a column
        data['feature'] = feature_names
        self._importance_df = pl.DataFrame(data)
    
    def get_importance_dataframe(self) -> Optional[pl.DataFrame]:
        """Get complete importance DataFrame"""
        return self._importance_df
    
    def export_to_csv(self, output_path: str):
        """Export all feature importance to CSV"""
        if self._importance_df is None:
            print("No importance data to export")
            return
        
        self._importance_df.write_csv(output_path)
        print(f"Importance data exported to {output_path}")
        print(f"  - Features: {self._importance_df.height}")
        print(f"  - Splits: {len(self._importance_df.columns) - 1}")  # -1 for feature column
    
    def plot_full_heatmap(
        self,
        output_path: Optional[str] = None,
        show: bool = True,
        sort_by: str = 'mean',
        normalize: str = 'zscore',
    ):
        """
        Plot heatmap for all features
        
        Args:
            output_path: Output file path
            show: Whether to show plot
            sort_by: Sort method ('mean', 'std', 'max', 'none')
            normalize: Normalization method ('zscore', 'minmax', 'rank', 'none')
        """
        if self._importance_df is None or self._importance_df.is_empty():
            print("No importance data to visualize")
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Get numeric columns
        numeric_cols = [c for c in self._importance_df.columns if c != 'feature']
        values = self._importance_df.select(numeric_cols).to_numpy()
        
        # Normalize importance (by row/feature)
        if normalize == 'zscore':
            row_mean = np.mean(values, axis=1, keepdims=True)
            row_std = np.std(values, axis=1, keepdims=True) + 1e-10
            values = (values - row_mean) / row_std
        elif normalize == 'minmax':
            row_min = np.min(values, axis=1, keepdims=True)
            row_max = np.max(values, axis=1, keepdims=True)
            values = (values - row_min) / (row_max - row_min + 1e-10)
        elif normalize == 'rank':
            values = np.apply_along_axis(
                lambda x: np.argsort(np.argsort(x)) / len(x), 
                axis=0, arr=values
            )
        
        # Sort features (using original data for sorting)
        orig_values = self._importance_df.select(numeric_cols).to_numpy()
        if sort_by == 'mean':
            sort_idx = np.argsort(np.mean(orig_values, axis=1))[::-1]
        elif sort_by == 'std':
            sort_idx = np.argsort(np.std(orig_values, axis=1))[::-1]
        elif sort_by == 'max':
            sort_idx = np.argsort(np.max(orig_values, axis=1))[::-1]
        else:
            sort_idx = np.arange(len(orig_values))
        
        values = values[sort_idx]
        
        # Calculate appropriate figure size
        n_features = values.shape[0]
        n_splits = len(numeric_cols)
        fig_height = max(10, min(100, n_features * 0.05))
        fig_width = max(12, min(30, n_splits * 0.8))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        im = ax.imshow(values, aspect='auto', cmap='YlOrRd')
        
        ax.set_xticks(np.arange(n_splits))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=8)
        
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
        Create animation of all feature importance changes
        
        X-axis: Factor names (fixed order)
        Y-axis: Importance intensity
        
        Args:
            output_path: Output path
            interval: Frame interval (milliseconds)
            show: Whether to show
            sort_by: Feature sort method ('mean', 'std', 'max', 'none')
        """
        if not self.frames:
            print("No importance frames to visualize")
            return None
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        n_features = len(self.frames[0].importance_vector)
        
        # Determine feature order (fixed)
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
        
        # Get global maximum
        max_val = np.max(all_importance) * 1.1
        
        # Adjust figure size based on feature count
        fig_width = max(20, min(60, n_features * 0.02))
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # X-axis positions
        x_pos = np.arange(n_features)
        
        # Initialize bar chart
        bars = ax.bar(x_pos, np.zeros(n_features), width=0.8, color='steelblue', alpha=0.8)
        
        # Set axes
        ax.set_ylim(0, max_val)
        ax.set_xlim(-0.5, n_features - 0.5)
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_xlabel(f'Feature (sorted by {sort_by}, n={n_features})', fontsize=10)
        
        # Don't show X-axis labels, too many features would be crowded
        ax.set_xticks([])
        
        # Add split information text
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
            # Get importance in fixed order
            importance = frame.importance_vector[feature_order]
            
            for bar, val in zip(bars, importance):
                bar.set_height(val)
            
            # Update info text
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
        """Plot importance distribution for each split"""
        if self._importance_df is None:
            print("No importance data to visualize")
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        numeric_cols = [c for c in self._importance_df.columns if c != 'feature']
        ax.boxplot(
            [self._importance_df[col].to_numpy() for col in numeric_cols],
            labels=numeric_cols,
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
    Convenience function: extract and visualize all feature importance from results
    
    Output files:
    - importance_data.csv: importance data of all features across all splits
    - importance_heatmap.png: full feature heatmap
    - importance_animation.gif: feature ranking animation
    - importance_distribution.png: importance distribution per split
    """
    visualizer = ImportanceVisualizer(feature_names=feature_names)
    
    visualizer.extract_from_results(results, splitspec_list)
    
    if not visualizer.frames:
        print("No valid importance data found in results")
        return visualizer
    
    n_features = len(visualizer.frames[0].importance_vector)
    print(f"Found {len(visualizer.frames)} splits with {n_features} features")
    
    # Generate output paths
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
    
    # 1. Export CSV
    if csv_path:
        visualizer.export_to_csv(csv_path)
    
    # 2. Full feature heatmap
    visualizer.plot_full_heatmap(output_path=heatmap_path, show=show, sort_by='mean')
    
    # 3. Full feature animation (X-axis: factor names fixed, Y-axis: intensity)
    visualizer.create_importance_animation(output_path=gif_path, interval=interval, show=show, sort_by='mean')
    
    # 4. Distribution plot
    visualizer.plot_importance_distribution(output_path=dist_path, show=show)
    
    return visualizer
