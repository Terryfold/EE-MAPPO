import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt


class ObstaclesGenerator:
    
    def __init__(self, grid_size: float = 5, min_distance: float = 1):
        self.grid_size = grid_size
        self.min_distance = min_distance
        self.boundary_margin = 0
        
    def generate_random_obstacles(self, 
                                num_obstacles: int, 
                                seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            
        obstacles = []
        attempts = 0
        max_attempts = num_obstacles * 100
        
        while len(obstacles) < num_obstacles and attempts < max_attempts:
            x = np.random.uniform(-self.grid_size/2 + self.boundary_margin, 
                                 self.grid_size/2 - self.boundary_margin)
            y = np.random.uniform(-self.grid_size/2 + self.boundary_margin, 
                                 self.grid_size/2 - self.boundary_margin)
            pos = np.array([x, y])
            
            if self._is_valid_position(pos, obstacles):
                obstacles.append(pos)
                
            attempts += 1
            
        if len(obstacles) < num_obstacles:
            print(f"Warning: Only generated {len(obstacles)} obstacles, less than required {num_obstacles}")
            
        return np.array(obstacles)
    
    def generate_grid_obstacles(self, 
                               grid_spacing: float = 2.0, 
                               center_offset: Tuple[float, float] = (0, 0)) -> np.ndarray:
        obstacles = []
        half_size = self.grid_size / 2 - self.boundary_margin
        
        x_range = np.arange(-half_size + center_offset[0], 
                           half_size + center_offset[0], grid_spacing)
        y_range = np.arange(-half_size + center_offset[1], 
                           half_size + center_offset[1], grid_spacing)
        
        for x in x_range:
            for y in y_range:
                pos = np.array([x, y])
                if self._is_valid_position(pos, obstacles):
                    obstacles.append(pos)
                    
        return np.array(obstacles)
    
    def generate_circular_obstacles(self, 
                                   num_obstacles: int, 
                                   radius: float = 3.0, 
                                   center: Tuple[float, float] = (0, 0),
                                   seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            
        obstacles = []
        angle_step = 2 * np.pi / num_obstacles
        
        for i in range(num_obstacles):
            angle = i * angle_step + np.random.uniform(-0.1, 0.1)
            r = radius + np.random.uniform(-0.2, 0.2)
            
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            
            x = np.clip(x, -self.grid_size/2 + self.boundary_margin, 
                       self.grid_size/2 - self.boundary_margin)
            y = np.clip(y, -self.grid_size/2 + self.boundary_margin, 
                       self.grid_size/2 - self.boundary_margin)
            
            pos = np.array([x, y])
            if self._is_valid_position(pos, obstacles):
                obstacles.append(pos)
                
        return np.array(obstacles)
    
    def generate_corridor_obstacles(self, 
                                   corridor_width: float = 2.0,
                                   wall_thickness: float = 0.5) -> np.ndarray:
        obstacles = []
        half_size = self.grid_size / 2 - self.boundary_margin
        
        wall_spacing = wall_thickness * 1.5
        
        for y in np.arange(-half_size, half_size, wall_spacing):
            x = -corridor_width/2
            pos = np.array([x, y])
            if self._is_valid_position(pos, obstacles):
                obstacles.append(pos)
                
        for y in np.arange(-half_size, half_size, wall_spacing):
            x = corridor_width/2
            pos = np.array([x, y])
            if self._is_valid_position(pos, obstacles):
                obstacles.append(pos)
                
        return np.array(obstacles)
    
    def generate_maze_obstacles(self, 
                               maze_size: int = 5,
                               wall_thickness: float = 0.3) -> np.ndarray:
        obstacles = []
        cell_size = self.grid_size / maze_size
        
        for i in range(1, maze_size):
            x = -self.grid_size/2 + i * cell_size
            for y in np.arange(-self.grid_size/2, self.grid_size/2, wall_thickness):
                pos = np.array([x, y])
                if self._is_valid_position(pos, obstacles):
                    obstacles.append(pos)
                    
        for i in range(1, maze_size):
            y = -self.grid_size/2 + i * cell_size
            for x in np.arange(-self.grid_size/2, self.grid_size/2, wall_thickness):
                pos = np.array([x, y])
                if self._is_valid_position(pos, obstacles):
                    obstacles.append(pos)
                    
        return np.array(obstacles)
    
    def generate_custom_pattern(self, 
                               pattern: str = "cross",
                               density: float = 0.1) -> np.ndarray:
        obstacles = []
        half_size = self.grid_size / 2 - self.boundary_margin
        
        if pattern == "cross":
            for t in np.arange(-half_size, half_size, 0.5):
                obstacles.extend([
                    np.array([t, 0]),
                    np.array([0, t])
                ])
                
        elif pattern == "star":
            for angle in np.arange(0, 2*np.pi, np.pi/4):
                for r in np.arange(1, half_size, 1):
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    pos = np.array([x, y])
                    if self._is_valid_position(pos, obstacles):
                        obstacles.append(pos)
                        
        elif pattern == "spiral":
            for t in np.arange(0, 4*np.pi, 0.2):
                r = t * 0.5
                x = r * np.cos(t)
                y = r * np.sin(t)
                pos = np.array([x, y])
                if abs(x) < half_size and abs(y) < half_size:
                    if self._is_valid_position(pos, obstacles):
                        obstacles.append(pos)
                        
        elif pattern == "random_clusters":
            num_clusters = int(5 * density)
            for _ in range(num_clusters):
                cluster_center = np.random.uniform(-half_size, half_size, 2)
                cluster_size = np.random.randint(3, 8)
                
                for _ in range(cluster_size):
                    offset = np.random.uniform(-1, 1, 2)
                    pos = cluster_center + offset
                    if abs(pos[0]) < half_size and abs(pos[1]) < half_size:
                        if self._is_valid_position(pos, obstacles):
                            obstacles.append(pos)
                            
        return np.array(obstacles)
    
    def _is_valid_position(self, pos: np.ndarray, existing_obstacles: List[np.ndarray]) -> bool:
        if abs(pos[0]) > self.grid_size/2 - self.boundary_margin or \
           abs(pos[1]) > self.grid_size/2 - self.boundary_margin:
            return False
            
        for obs in existing_obstacles:
            if np.linalg.norm(pos - obs) < self.min_distance:
                return False
                
        return True
    
    def save_obstacles_to_file(self, 
                              obstacles: np.ndarray, 
                              filename: str = "obstacles.csv") -> None:
        df = pd.DataFrame(obstacles, columns=['x', 'y'])
        df.to_csv(filename, index=False)
        print(f"Obstacles saved to {filename}")
    
    def load_obstacles_from_file(self, filename: str) -> np.ndarray:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
            
        df = pd.read_csv(filename)
        obstacles = df[['x', 'y']].values
        print(f"Loaded {len(obstacles)} obstacles from {filename}")
        return obstacles
    
    def visualize_obstacles(self, 
                           obstacles: np.ndarray, 
                           title: str = "Obstacle Layout",
                           save_path: Optional[str] = None) -> None:
        plt.figure(figsize=(10, 10))
        obstacle_radius = 0.2
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obstacle_radius, 
                                color='red', alpha=0.7, label='Obstacles' if obs is obstacles[0] else "")
            plt.gca().add_patch(circle)
        
        boundary = self.grid_size / 2
        plt.plot([-boundary, boundary, boundary, -boundary, -boundary],
                [-boundary, -boundary, boundary, boundary, -boundary], 
                'k--', alpha=0.5, label='Environment Boundary')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()


def generate_obstacles(method: str = "circular", 
                      num_obstacles: int = 10,
                      grid_size: float = 10.0,
                      **kwargs) -> np.ndarray:
    generator = ObstaclesGenerator(grid_size=grid_size)
    
    if method == "random":
        return generator.generate_random_obstacles(num_obstacles, **kwargs)
    elif method == "grid":
        return generator.generate_grid_obstacles(**kwargs)
    elif method == "circular":
        return generator.generate_circular_obstacles(num_obstacles, **kwargs)
    elif method == "corridor":
        return generator.generate_corridor_obstacles(**kwargs)
    elif method == "maze":
        return generator.generate_maze_obstacles(**kwargs)
    elif method == "custom":
        return generator.generate_custom_pattern(**kwargs)
    else:
        raise ValueError(f"Unknown generation method: {method}")


def load_obstacles_from_file(filename: str, num_obstacles: Optional[int] = None) -> np.ndarray:
    generator = ObstaclesGenerator()
    obstacles = generator.load_obstacles_from_file(filename)
    
    if num_obstacles is not None:
        if len(obstacles) < num_obstacles:
            print(f"Warning: File only contains {len(obstacles)} obstacles, less than required {num_obstacles}")
        else:
            obstacles = obstacles[:num_obstacles]
            
    return obstacles


if __name__ == "__main__":
    generator = ObstaclesGenerator(grid_size=10, min_distance=1)
    
    print("=== Obstacle Generator Test ===")
    
    random_obs = generator.generate_random_obstacles(40, seed=42)
    print(f"Random obstacles: {len(random_obs)}")
    
    grid_obs = generator.generate_grid_obstacles(grid_spacing=2.0)
    print(f"Grid obstacles: {len(grid_obs)}")
    
    circular_obs = generator.generate_circular_obstacles(8, radius=3.0,seed=42)
    print(f"Circular obstacles: {len(circular_obs)}")
    
    corridor_obs = generator.generate_corridor_obstacles(corridor_width=3.0)
    print(f"Corridor obstacles: {len(corridor_obs)}")
    
    maze_obs = generator.generate_maze_obstacles(maze_size=4)
    print(f"Maze obstacles: {len(maze_obs)}")
    
    custom_obs = generator.generate_custom_pattern("cross")
    print(f"Custom obstacles: {len(custom_obs)}")
    
    generator.save_obstacles_to_file(random_obs, "./envs/obstacle_full_list.csv")
    
    generator.visualize_obstacles(random_obs, "Random Obstacle Layout Example")
    
    print("\n=== Convenience Function Test ===")
    obs1 = generate_obstacles("random", 100, grid_size=8.0, seed=123)
    obs2 = generate_obstacles("circular", 8, grid_size=8.0, radius=2.5)
    
    print(f"Convenience function generated: {len(obs1)} random obstacles, {len(obs2)} circular obstacles")
