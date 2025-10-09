"""Experiment logging and data collection utilities."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import RunConfig


class ExperimentLogger:
    """Logs experiment data and saves to CSV with optional result files."""
    
    def __init__(self, config: RunConfig, comm: MPI.Intracomm):
        self.config = config
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        # Generate unique experiment ID with short hash
        timestamp = int(time.time())
        size_str = f"{config.width}x{config.height}"
        hash_input = f"{config.schedule}_{config.communication}_{size_str}_{timestamp}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        self.experiment_id = f"{config.schedule}_{config.communication}_{size_str}_{short_hash}"
        
        # Timing data
        self.start_time = time.time()
        self.computation_time = 0.0
        self.communication_time = 0.0
        self.worker_stats: Dict[int, Dict] = {}
        
        # File paths - organize by schedule + communication
        self.experiments_dir = Path("experiments")
        self.combo_dir = self.experiments_dir / f"{config.schedule}_{config.communication}"
        self.combo_dir.mkdir(parents=True, exist_ok=True)
        
        # Plots directory
        self.plots_dir = Path("Plots/images")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def log_computation_time(self, duration: float) -> None:
        """Log time spent in computation."""
        self.computation_time += duration
    
    def log_communication_time(self, duration: float) -> None:
        """Log time spent in communication."""
        self.communication_time += duration
    
    def log_worker_stats(self, rank: int, chunks_processed: int, 
                        comp_time: float, comm_time: float, chunk_ids: List[int] = None) -> None:
        """Log statistics for a specific worker rank."""
        self.worker_stats[rank] = {
            'chunks_processed': chunks_processed,
            'computation_time': comp_time,
            'communication_time': comm_time,
            'chunk_ids': chunk_ids or []
        }
    
    def finalize(self, result_array: Optional[np.ndarray] = None, 
                save_plot: bool = False, save_data: bool = False) -> Optional[str]:
        """Finalize experiment and save all data to CSV."""
        wall_clock_time = time.time() - self.start_time
        
        # Gather worker stats from all ranks
        all_worker_stats = self._gather_worker_stats()
        
        if self.rank != 0:
            return None
        
        # Create DataFrame
        df = self._create_dataframe(wall_clock_time, all_worker_stats)
        
        # Save CSV
        csv_path = self.combo_dir / f"experiment_{self.experiment_id}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save result files if requested
        result_file = None
        plot_file = None
        
        if result_array is not None:
            if save_data:
                result_file = self.combo_dir / f"{self.experiment_id}.npy"
                np.save(result_file, result_array)
            
            if save_plot:
                plot_file = self.plots_dir / f"{self.experiment_id}.pdf"
                self._save_plot(result_array, plot_file)
        
        # Update DataFrame with file paths
        if result_file or plot_file:
            df['result_file'] = str(result_file) if result_file else None
            df['plot_file'] = str(plot_file) if plot_file else None
            df.to_csv(csv_path, index=False)
        
        print(f"Experiment data saved to {csv_path}")
        if result_file:
            print(f"Result array saved to {result_file}")
        if plot_file:
            print(f"Plot saved to {plot_file}")
        
        return str(csv_path)
    
    def _gather_worker_stats(self) -> Dict[int, Dict]:
        """Gather worker statistics from all ranks."""
        if self.rank == 0:
            # Always collect from workers for accurate timing (static and dynamic)
            # Don't use master's measurements as they may include synchronization delays
            all_stats = {0: self.worker_stats.get(0, {})}
            
            for source in range(1, self.size):
                stats_data = self.comm.recv(source=source, tag=2000)
                all_stats[source] = stats_data
            
            return all_stats
        else:
            # Workers send their stats to master
            worker_data = self.worker_stats.get(self.rank, {})
            self.comm.send(worker_data, dest=0, tag=2000)
            return {}
    
    def _create_dataframe(self, wall_clock_time: float, 
                         all_worker_stats: Dict[int, Dict]) -> pd.DataFrame:
        """Create DataFrame with experiment data."""
        rows = []
        
        for rank in range(self.size):
            stats = all_worker_stats.get(rank, {})
            chunks_processed = stats.get('chunks_processed', 0)
            comp_time = stats.get('computation_time', 0.0)
            comm_time = stats.get('communication_time', 0.0)
            chunk_ids = stats.get('chunk_ids', [])
            
            # Core experiment data (no derived/calculated fields)
            row = {
                'experiment_id': self.experiment_id,
                'timestamp': int(self.start_time),
                'schedule': self.config.schedule,
                'communication': self.config.communication,
                'num_processes': self.size,
                'chunk_size': self.config.chunk_size,
                'image_width': self.config.width,
                'image_height': self.config.height,
                'xlim_min': self.config.xlim[0],
                'xlim_max': self.config.xlim[1],
                'ylim_min': self.config.ylim[0],
                'ylim_max': self.config.ylim[1],
                'wall_clock_time': wall_clock_time,
                'computation_time': comp_time,
                'communication_time': comm_time,
                'rank': rank,
                'chunks_processed': chunks_processed,
                'chunk_ids': ','.join(map(str, chunk_ids)),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _save_plot(self, image: np.ndarray, plot_path: Path) -> None:
        """Save plot as PDF."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        extent = (self.config.xlim[0], self.config.xlim[1], 
                 self.config.ylim[0], self.config.ylim[1])
        
        plt.figure(figsize=(10, 8))
        plt.imshow(image.T, extent=extent, origin="lower", cmap='viridis')
        plt.xlabel("x / Re(p_0)")
        plt.ylabel("y / Im(p_0)")
        plt.title(f"Mandelbrot Set - {self.config.schedule} + {self.config.communication}")
        plt.colorbar()
        
        plt.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
        plt.close()