from turtle import speed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import warnings

from sympy import true
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.radar_system import RadarSystem
from gymnasium import spaces

from rrt_star_pathfinding_point_obstacles import plan_path_for_agent_point_obstacles

class Continuous2DEnv:
    def __init__(self, num_agents=2, agent_size=0.1,
                 num_obstacles=2, obstacle_size = 0.2, 
                 max_cycle= 100, grid_size=10, use_radar=True,
                 radar_rays=16, radar_range=2.0,enable_rrt_guidance=True,
                 easymode=True):
        self.num_agents = num_agents
        self.agent_size = agent_size
        self.num_obstacles = num_obstacles
        self.obstacle_size = obstacle_size
        self.max_cycle = max_cycle
        self.grid_size = grid_size
        self.use_radar = use_radar
        self.radar_rays = radar_rays
        self.radar_range = radar_range
        

        self.enable_rrt_guidance = enable_rrt_guidance  
        self.agent_paths = {}  
        self.agent_waypoints = {} 
        self.agent_waypoint_status = {} 
        self.bounds = [-grid_size/2, grid_size/2, -grid_size/2, grid_size/2]

        self.easymode = easymode
        
        self.use_rrt_reward_shaping = enable_rrt_guidance
        self.waypoint_distance = 0.3  
        self.waypoint_reward_base = 10.0  
        self.rrt_reward_weight = 1.0 

        self.agent_name_list = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.agent_name_list  
        

        self.dt = 0.1 
        self.step_count = 0
        
       
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        
        if self.use_radar:
           
            obs_dim = 6 + self.radar_rays
        else:
           
            obs_dim = 4 + 2 + 6 + 6
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

       
        if self.use_radar:
            self.radar_system = RadarSystem(
                num_rays=self.radar_rays, 
                max_range=self.radar_range
            )
       
      
        self.num_obstacles = num_obstacles
       
        if num_obstacles > 0:
           
            obstacle_file_paths = [
                "obstacle_full_list.csv",
                "envs/obstacle_full_list.csv",
                os.path.join(os.path.dirname(__file__), "obstacle_full_list.csv")
            ]
            obstacles_loaded = False
            for path in obstacle_file_paths:
                if os.path.exists(path):
                    obstacles = np.loadtxt(path, delimiter=',', skiprows=1)
                    #
                    point_obstacles = []
                    for obs in obstacles[:num_obstacles]:
                        point_obstacles.append([obs[0], obs[1], obstacle_size])
                    self.obstacles = np.array(point_obstacles)
                    obstacles_loaded = True
                    break
            if not obstacles_loaded:
                raise FileNotFoundError("Could not find obstacle_full_list.csv in any of the expected locations")
           
            if num_obstacles > 50:
                raise ValueError("Number of obstacles exceeds the maximum limit of 10.")
       
        elif num_obstacles == 0:
            self.obstacles = np.empty((0, 3)) 
        self.reset()    

   
    def reset(self):
        self.agent_pos = []
        self.agent_vel = [] 
        self.goal_pos = [] 
        self.first_goal = {}
        self.agent_death = {f"agent_{i}": 0 for i in range(self.num_agents)}
        self.step_count = 0

        for agent_name in self.agent_name_list:
            self.first_goal[agent_name] = True
        all_pos = [] 
       
        for obs in self.obstacles:
            
            all_pos.append(obs[:2])

        def sample_valid():
            while True:
                p = np.random.uniform(-self.grid_size/2*0.8, self.grid_size/2*0.8, size=(2,))
                if all(np.linalg.norm(p - q) > 0.3 for q in all_pos):
                    all_pos.append(p)
                    return p
        
        if self.easymode:
           
            if self.num_agents == 8:
                
                for i in range(self.num_agents):
                    x_pos = -4 + i * 1.0  
                    self.agent_pos.append(np.array([x_pos, self.grid_size/2-self.agent_size-0.5]))
                
               
                for i in range(self.num_agents):
                    x_pos = -4 + i * 1.0  
                    self.goal_pos.append(np.array([x_pos, -self.grid_size/2+self.agent_size+0.5]))
            else:
                
                for i in range(self.num_agents):
                    self.agent_pos.append(np.array([-4+i*2, self.grid_size/2-self.agent_size]))
                    self.goal_pos.append(np.array([-4+i*2, -self.grid_size/2+self.agent_size]))

        else:
            for _ in range(self.num_agents): 
                self.agent_pos.append(sample_valid())
                self.goal_pos.append(sample_valid())

        self.agent_pos = np.array(self.agent_pos) 
        self.goal_pos = np.array(self.goal_pos) 
        self.agent_vel = np.zeros_like(self.agent_pos) 
       
        
        
        if self.enable_rrt_guidance:
            self.init_rrt_guidance()

        
        return self.get_obs(), {}
 
    # step
    def step(self,actions):
       
        
        if isinstance(actions, dict):
            actions_array = np.zeros((self.num_agents, 2)) 
        for agent_id, action in actions.items():
           
            idx = int(agent_id.split('_')[1])
            actions_array[idx] = action
        actions = actions_array
        self.agent_vel = actions * self.dt 

        next_pos = self.agent_pos + self.agent_vel  
        multi_rewards = {}
        multi_finished = {f"agent_{i}": False for i in range(self.num_agents)}
        multi_truncations = {f"agent_{i}": False for i in range(self.num_agents)}

       
        for i in range(self.num_agents):
            reward = 0
            valid = True
            death = 0
            death_boundary = 0
            done = 0
            
            for j in range(self.num_agents):
                if i != j and self.measure_distance(next_pos[i] ,next_pos[j])<0.25:
                    reward =-5 
                    if self.measure_distance(next_pos[i] ,next_pos[j])<self.agent_size*2:
                        reward =-500 
                        valid = False
                        death = 1
                    

                    #print(f"Agent {i} collided with Agent {j}")
            for obs in self.obstacles:
                
                obs_pos = obs[:2]  
                obs_radius = obs[2] if len(obs) > 2 else self.obstacle_size  
                if self.measure_distance(next_pos[i], obs_pos)<obs_radius+self.agent_size+0.2: 
                    reward =-5
                    if self.measure_distance(next_pos[i], obs_pos)<obs_radius+self.agent_size:
                        reward =-500 
                        valid = False
                        death = 1
                    #print(f"Agent {i} collided with obstacle {obs}")
            
            if not (-self.grid_size/2-0.2 <= next_pos[i][0] <= self.grid_size/2+0.2 and
                    -self.grid_size/2-0.2 <= next_pos[i][1] <= self.grid_size/2+0.2):
                reward -=0
                
                #death = 1
                #print(f"Agent {i} out of bounds")
           
            
             
          
            current_dis_to_goal = np.linalg.norm(self.goal_pos[i] - self.agent_pos[i])
            if valid:
                reward = -0.1
                self.agent_pos[i] = next_pos[i]  
            distance_to_goal = np.linalg.norm(self.agent_pos[i] - self.goal_pos[i])
            
           
            
            if distance_to_goal < 0.8:
                speed = np.linalg.norm(self.agent_vel[i])
                if speed > distance_to_goal*2:
                    speed_penalty = -speed*0.5
                else:
                    speed_penalty = 0
            else:
                speed_penalty = 0

            
           
            
           
            if distance_to_goal< 0.8:

               
                vel_angle = np.arctan2(self.agent_vel[i][1], self.agent_vel[i][0])
                
                goal_angle = np.arctan2(self.goal_pos[i][1] - self.agent_pos[i][1], self.goal_pos[i][0] - self.agent_pos[i][0])
                
                angle_diff = vel_angle - goal_angle
                if abs(angle_diff) < 0.1:
                    reward += 1.5
                elif abs(angle_diff) < 0.3:
                    reward += 1
                else:
                    reward -= 0.1
                reward += (current_dis_to_goal-distance_to_goal)*0.2

                if self.first_goal[f"agent_{i}"] and distance_to_goal<0.3:
                    self.first_goal[f"agent_{i}"] = False
                    reward += 200
                    print(f"Agent {i} reached the goal at the {self.step_count} step")
                    done = 1
                
            multi_finished[f"agent_{i}"] = done 
            self.agent_death[f"agent_{i}"] = death
            potential = self.potential_field(distance_to_goal, self.obstacles, self.agent_pos[i])
            total_rrt_reward = self.get_rrt_reward_shaping(i, next_pos[i], self.agent_pos[i])
            multi_rewards[f"agent_{i}"] = reward  + potential + (-distance_to_goal)*1 +speed_penalty +total_rrt_reward

        self.step_count += 1 
        if self.step_count >= self.max_cycle:  
            multi_truncations = {f"agent_{i}": True for i in range(self.num_agents)}
        
       
        return self.get_obs(), multi_rewards, multi_finished, multi_truncations,self.agent_death
    
    def measure_distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2) 
    def potential_field(self, dis_to_goal, obstacles, agent_pos):
        r_potential = 0
        if dis_to_goal>0.05:
            r_potential += 1/dis_to_goal

        for obs in obstacles:
           
            obs_pos = obs[:2] 
            obs_radius = obs[2] if len(obs) > 2 else self.obstacle_size  
            obs_dist = self.measure_distance(agent_pos, obs_pos)
            if obs_dist<1 and obs_dist>0.001:
                r_potential -= 1/obs_dist

        return r_potential

    
    def get_obs(self):
        obs = {}
        for i in range(self.num_agents):
            if self.use_radar:
               
                agent_pos = self.agent_pos[i]
                goal_pos = self.goal_pos[i]
                
                
                other_agents_pos = [self.agent_pos[j] for j in range(self.num_agents) if j != i]
                
               
                obstacle_positions = [obs[:2] for obs in self.obstacles]
                obstacle_distances = self.radar_system.scan_obstacles_and_boundaries(
                    agent_pos, obstacle_positions, other_agents_pos, self.grid_size, self.obstacle_size, self.agent_size
                )
                
               
                agent_info = np.concatenate([
                    self.agent_pos[i],
                    self.agent_vel[i],
                    goal_pos - agent_pos
                ])
                
                o = np.concatenate([
                    agent_info,
                    obstacle_distances
                ])
                
            else:
                rel_goal = self.goal_pos[i] - self.agent_pos[i]
                
                obstacle_distances = []
                obstacle_relative_positions = []
                for obstacle in self.obstacles:
                    obs_pos = obstacle[:2]
                    rel_pos = obs_pos - self.agent_pos[i]
                    distance = np.linalg.norm(rel_pos)
                    if distance <= 2.0:
                        obstacle_distances.append(distance)
                        obstacle_relative_positions.append(rel_pos)
                
                if len(obstacle_distances) > 0:
                    sorted_indices = np.argsort(obstacle_distances)[:3]
                    nearest_obstacles = [obstacle_relative_positions[idx] for idx in sorted_indices]
                else:
                    nearest_obstacles = []
                
                while len(nearest_obstacles) < 3:
                    nearest_obstacles.append(np.array([0.0, 0.0]))
                
                other_agent_distances = []
                other_agent_relative_positions = []
                for j in range(self.num_agents):
                    if j != i:
                        rel_pos = self.agent_pos[j] - self.agent_pos[i]
                        distance = np.linalg.norm(rel_pos)
                        if distance <= 2.0:
                            other_agent_distances.append(distance)
                            other_agent_relative_positions.append(rel_pos)
                
                if len(other_agent_distances) > 0:
                    sorted_indices = np.argsort(other_agent_distances)[:3]
                    nearest_agents = [other_agent_relative_positions[idx] for idx in sorted_indices]
                else:
                    nearest_agents = []
                
                while len(nearest_agents) < 3:
                    nearest_agents.append(np.array([0.0, 0.0]))
                
                nearest_obstacles_array = np.array(nearest_obstacles).flatten()
                nearest_agents_array = np.array(nearest_agents).flatten()
                
                o = np.concatenate([
                    self.agent_pos[i],
                    self.agent_vel[i],
                    rel_goal,
                    nearest_obstacles_array,
                    nearest_agents_array
                ])
            
            agent_name = f"agent_{i}"
            obs[agent_name] = o.astype(np.float32)
        return obs

    def init_rrt_guidance(self):
        self.agent_paths = {}
        self.agent_waypoints = {}
        self.agent_waypoint_status = {}

        for agent_i in range(self.num_agents):
            path = plan_path_for_agent_point_obstacles(
                agent_pos=self.agent_pos[agent_i],
                goal_pos=self.goal_pos[agent_i],
                obstacles=self.obstacles,
                bounds=self.bounds,
                max_iterations=500,
                agent_radius=self.agent_size,
                obstacle_radius=self.obstacle_size)

            if len(path) > 0:
                self.agent_paths[agent_i] = path

                waypoints = self.generate_waypoints(path,agent_i)
                self.agent_waypoints[agent_i] = waypoints
                self.agent_waypoint_status[agent_i] = [True] * len(waypoints)

    def get_rrt_reward_shaping(self, agent_id, next_pos, current_pos):
        if not self.use_rrt_reward_shaping or agent_id not in self.agent_paths:
            return 0.0
        
        path = self.agent_paths[agent_id]
        if len(path) < 2:
            return 0.0
        
        path_array = np.array(path)
        
        current_distances = np.linalg.norm(path_array - current_pos, axis=1)
        next_distances = np.linalg.norm(path_array - next_pos, axis=1)
        
        current_min_distance = np.min(current_distances)
        next_min_distance = np.min(next_distances)
        
        distance_reward = -next_min_distance * 0.5
        
        waypoint_reward = 0.0
        if agent_id in self.agent_waypoints and agent_id in self.agent_waypoint_status:
            waypoints = self.agent_waypoints[agent_id]
            waypoint_status = self.agent_waypoint_status[agent_id]
            
            
            for i, (waypoint_x, waypoint_y, waypoint_id) in enumerate(waypoints):
                if waypoint_status[i]:
                    waypoint_pos = np.array([waypoint_x, waypoint_y])
                    distance_to_waypoint = np.linalg.norm(next_pos - waypoint_pos)
                    
                    if distance_to_waypoint <= self.waypoint_distance:
                       
                        waypoint_reward = self.waypoint_reward_base * (1.0 + waypoint_id * 1.0)
                        
                        waypoint_status[i] = False
                        
                        for j in range(i):
                            waypoint_status[j] = False
                        
                        break
        
        total_rrt_reward = (distance_reward + waypoint_reward) * self.rrt_reward_weight
        
        return total_rrt_reward   

    def generate_waypoints(self, path, agent_id):
        if len(path) < 2:
            return []
        
        waypoints = []
        waypoint_id = 0
        
        for i in range(len(path) - 1):
            start_point = np.array(path[i])
            end_point = np.array(path[i + 1])
            
            segment_length = np.linalg.norm(end_point - start_point)
            
            num_waypoints_in_segment = max(1, int(segment_length / (self.waypoint_distance * 5)))
            
            for j in range(num_waypoints_in_segment):
                t = (j + 1) / (num_waypoints_in_segment + 1)
                waypoint_pos = start_point + t * (end_point - start_point)
                waypoints.append([waypoint_pos[0], waypoint_pos[1], waypoint_id])
                waypoint_id += 1
        
        return waypoints

def render_frame(env, ax):
    ax.clear()
    ax.set_xlim(-env.grid_size/2, env.grid_size/2)
    ax.set_ylim(-env.grid_size/2, env.grid_size/2)
    ax.set_aspect('equal', adjustable='box')
    
    for obstacle in env.obstacles:
        obs_pos = obstacle[:2]
        obs_radius = obstacle[2] if len(obstacle) > 2 else env.obstacle_size
        circle = plt.Circle(obs_pos, obs_radius, color='gray', alpha=0.5)
        ax.add_artist(circle)
    
    for i, pos in enumerate(env.agent_pos):
        circle = plt.Circle(pos, env.agent_size, color='blue', alpha=0.5)
        ax.add_artist(circle)
        
        if env.use_radar:
            agent_pos = env.agent_pos[i]
            
            other_agents_pos = [env.agent_pos[j] for j in range(env.num_agents) if j != i]
            
            obstacle_positions = [obs[:2] for obs in env.obstacles]
            obstacle_distances = env.radar_system.scan_obstacles_and_boundaries(
                agent_pos, obstacle_positions, other_agents_pos, env.grid_size, env.obstacle_size, env.agent_size
            )
            
            for ray_idx, (angle, direction) in enumerate(zip(env.radar_system.ray_angles, env.radar_system.ray_directions)):
                obs_dist = obstacle_distances[ray_idx]
                
                if obs_dist < env.radar_range:
                    end_point = agent_pos + direction * obs_dist
                    ax.plot([agent_pos[0], end_point[0]], [agent_pos[1], end_point[1]], 
                           'r-', linewidth=1, alpha=0.6)
                else:
                    end_point = agent_pos + direction * env.radar_range
                    ax.plot([agent_pos[0], end_point[0]], [agent_pos[1], end_point[1]], 
                           'k-', linewidth=0.5, alpha=0.2)
            
            radar_circle = plt.Circle(agent_pos, env.radar_range, 
                                    color='red', fill=False, linestyle='--', 
                                    linewidth=1, alpha=0.3)
            ax.add_artist(radar_circle)
    
    for goal in env.goal_pos:
        circle = plt.Circle(goal, env.agent_size, color='green', alpha=0.5)
        ax.add_artist(circle)
    
    if hasattr(env, 'agent_paths') and len(env.agent_paths) > 0:
        for agent_id, path in env.agent_paths.items():
            if len(path) > 1:
                path_array = np.array(path)
                ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, alpha=0.8)
                ax.plot(path_array[:, 0], path_array[:, 1], 'bo', markersize=3)
    
    ax.set_title(f'Step: {env.step_count}')
    ax.grid(True, alpha=0.3)

def animate_environment(env, num_agents=1):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        actions = {}
        for agent_name in env.agents:
            actions[agent_name] = np.random.uniform(-0.5, 0.5, 2)
        
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        render_frame(env, ax)
        
        if any(terminations.values()) or any(truncations.values()):
            print("Episode finished!")
            print(f"Final rewards: {rewards}")
            print(f"Terminations: {terminations}")
            print(f"Truncations: {truncations}")
            print(f"Infos: {infos}")
            print(obs)
            print(rewards)
            print(terminations)

        render_frame(env, ax)

    ani = animation.FuncAnimation(fig, update, frames=env.max_cycle, interval=200)
    plt.show()

if __name__ == "__main__":
    env = Continuous2DEnv(num_agents=1, num_obstacles=20)
    animate_environment(env)
