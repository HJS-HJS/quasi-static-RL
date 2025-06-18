import sys
import os
import time
import yaml
import numpy as np
import random
from typing import List

import quasi_static_push

class Simulation():
    def __init__(self, visualize:str = 'human', state:str = 'image', action_skip:int = 5):
        """
        state : image, information
        """
        # Get initial param
        self.state = state

        # Set display visuality
        if visualize == "human": visualize = True
        elif visualize is None: visualize = False
        else: visualize = False

        ## Get config file
        with open(os.path.dirname(os.path.abspath(__file__)) + "/../config/config.yaml") as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)

        # Set patameters
        self.display_size = np.array([self.config["display"]["WIDTH"], self.config["display"]["HEIGHT"] ]) # Get display size parameter from config.yaml

        ## Set parameters
        # Set pixel unit
        self.unit = self.config["display"]["unit"] #[m/pixel]

        # Set pusher
        _pusher_type = self.config["pusher"]["pusher_type"]

        # Set pusher unit speed
        unit_v_speed = self.config["pusher"]["unit_v_speed"] / 2 # [m/s]
        unit_r_speed = self.config["pusher"]["unit_r_speed"] * 2  # [rad/s]
        unit_w_speed = self.config["pusher"]["unit_w_speed"] * 2  # [m/s]
        self.unit_speed = [unit_v_speed, unit_v_speed, unit_r_speed, unit_w_speed, int(1)]

        # Set slider 
        self.slider_max_num = self.config["auto"]["maximun_number"] # Get sliders number
        self.min_r = self.config["auto"]["minimum_radius"]
        self.max_r = self.config["auto"]["maximum_radius"]

        self.pusher_input = [
            self.config["pusher"]["pusher_num"], self.config["pusher"]["pusher_angle"], _pusher_type["type"], 
            {"a": _pusher_type["a"], "b": _pusher_type["b"], "n": _pusher_type["n"]}, 
            self.config["pusher"]["pusher_distance"], self.config["pusher"]["pusher_d_u_limit"], self.config["pusher"]["pusher_d_l_limit"],
            0.0, 0.0, 0.0
        ]

        self.action_limit = np.array([
            [-1., 1.],
            [-1., 1.],
            [-1., 1.],
            [-1., 1.],
            [0, 1],
        ])

        self.action_space = np.zeros_like(self.unit_speed)
        if state == "image" :   self.observation_space = np.zeros((self.config["display"]["WIDTH"],self.config["display"]["HEIGHT"],3))
        elif state == "gray":   self.observation_space = np.zeros((self.config["display"]["WIDTH"],self.config["display"]["HEIGHT"],1))
        elif state == "linear": self.observation_space = np.zeros(1 + 2 + 4 + 5), np.zeros((2, 5))

        self.simulator = quasi_static_push.SimulationViewer(
            window_width = self.display_size[0],
            window_height = self.display_size[1],
            scale = 1 / self.config["display"]["unit"],
            visualise = visualize,
            frame_rate = self.config["simulator"]["fps"],
            frame_skip = action_skip,
            grid = False,
            recording_enabled = True,
            show_closest_point = False,
            )
        
    def reset(self, 
              table_size:List[float] = None,
              slider_inputs:List[List[float]] = None,
              slider_num:int=None,
              ):
            
            self.state_prev = None
            
            # Table setting
            if table_size is None:
                _table_limit_width  = random.randint(int(self.display_size[0] * 0.36), int(self.display_size[0] * 0.86))
                _table_limit_height = random.randint(int(self.display_size[1] * 0.36), int(self.display_size[1] * 0.86))
                _table_limit = np.array([_table_limit_width, _table_limit_height])
                table_size = _table_limit * self.unit
            else:
                _table_limit = (np.array(table_size) / self.unit).astype(int)
            self.table_limit = _table_limit * self.unit / 2

            # Slider setting
            if slider_inputs is None:
                slider_inputs = []
                _slider_num = random.randint(1, self.slider_max_num) if slider_num is None else np.clip(slider_num, 1, 15)
                points, radius = self.generate_spawn_points(_slider_num)
                for point, _r in zip(points, radius):
                    a = np.clip(random.uniform(0.8, 1.0) * _r, a_min=self.min_r, a_max=_r)
                    b = np.clip(random.uniform(0.75, 1.25) * a, a_min=self.min_r, a_max=_r)
                    r = random.uniform(0, np.pi * 2)
                    slider_inputs.append(("ellipse", [point[0], point[1], r, a, b]))
            slider_num = len(slider_inputs)
            
            # Initial pusher pose
            self.pusher_input[4] = random.uniform(self.pusher_input[5], self.pusher_input[6])    # width
            self.pusher_input[7] = 0   # x
            self.pusher_input[8] = 0   # y
            self.pusher_input[9] = 0   # w

            self.simulator.reset(
                slider_inputs = slider_inputs,
                pusher_input = tuple(self.pusher_input),
                newtableWidth = self.table_limit[0] * 2,
                newtableHeight = self.table_limit[1] * 2,
            )

            return None, {None, None, None}

    def step(self, action):
        if(len(action) != 5): print("Invalid action space")
        
        action = np.clip(action, self.action_limit[:, 0], self.action_limit[:, 1])
        action *= self.unit_speed
        
        result = self.simulator.run(action)

        reward = self.cal_reward(self.state_prev, result.slider_state)
        self.state_prev = result.slider_state

        if self.state == "image": return result.image_state, reward, result.done, result.mode
        elif self.state == "linear": return (result.pusher_state, result.slider_state), reward, result.done, result.mode
        else: return None
    
    def cal_reward(self, state, state_next):
        if state is None: return 0
        ## reward
        reward = 0.0

        return reward
    
    def generate_spawn_points(self, num_points, center_bias=0.75):
        points = []
        x_range = (-self.table_limit[0] + self.min_r * 1.3, self.table_limit[0] - self.min_r * 1.3)
        y_range = (-self.table_limit[1] + self.min_r * 1.3, self.table_limit[1] - self.min_r * 1.3)

        # 첫 번째 점을 랜덤하게 생성
        center_x = random.uniform(*x_range) * 0.9
        center_y = random.uniform(*y_range) * 0.9
        points.append((center_x, center_y))

        # Raduis of inital point
        init_r = random.uniform(self.min_r, self.max_r)
        available_lengh = (init_r + self.min_r, init_r + self.max_r)
        
        # 나머지 점 생성
        candidate_points = []
        for _ in range(num_points - 1):
            # 첫 번째 점 주변에서 가우시안 분포로 점 생성
            if random.random() < center_bias:  # 중심 근처에 생성될 확률
                new_x = np.clip(np.random.normal(center_x, random.uniform(*available_lengh)), *x_range)
                new_y = np.clip(np.random.normal(center_y, random.uniform(*available_lengh)), *y_range)
            else:  # 전체 영역에 균일 분포로 생성
                new_x = random.uniform(*x_range)
                new_y = random.uniform(*y_range)
            candidate_points.append((new_x, new_y))
        
        # 거리 조건을 만족하는 점만 선택
        for point in candidate_points:
            distances = [np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) for p in points]
            if all(d >= (init_r + self.min_r) for d in distances):
                points.append(point)
        
        points = np.array(points)

        min_distances = np.ones(len(points)) * self.min_r
        min_distances[0] = init_r

        for idx, point in enumerate(points):
            if idx == 0: continue
            distances = [np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) for p in points]
            distances = distances - min_distances
            distances[idx] = self.max_r
            min_distances[idx] = min(distances)

        # 첫 번째 점을 포함한 최종 점 리스트
        return points, np.array(min_distances)

class DishSimulation():
    def __init__(self, visualize:str = 'human', state:str = 'image', action_skip:int = 5):
        self.env = Simulation(visualize = visualize, state = state, action_skip = action_skip)
        self._count = 0
        self._setting = None
        self.save_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
        self.skip_frame = 0
    
    def reset(self, mode:str=None, slider_num:int = 15):
        if mode == None: 
            state_curr, _ = self.env.reset(slider_num=slider_num)
            self._setting = None

        elif mode == "continous":
            if (self._setting == None) or (len(self.env.param.sliders) == 0):
                state_curr, self._setting = self.env.reset(slider_num=slider_num)
            else:
                _setting = self.env.get_setting()
                state_curr, _ = self.env.reset(
                    table_size  = _setting["table_size"],
                    slider_inputs = _setting["slider_inputs"],
                    slider_num  = _setting["slider_num"],
                    )
        
        elif mode == "pusher":
            if self._count == 0:
                state_curr, self._setting = self.env.reset(slider_num=slider_num)
                self._count += 1
            else:
                state_curr, _ = self.env.reset(
                    table_size  = self._setting["table_size"],
                    slider_inputs = self._setting["slider_inputs"],
                    slider_num  = self._setting["slider_num"],
                    )
                self._count = (self._count + 1) % 4

        elif mode == "test":
            if (self._setting == None) or (len(self.env.param.sliders) == 0):
                state_curr, self._setting = self.env.reset(slider_num=slider_num)
            else:
                _setting = self.env.get_setting()
                state_curr, _ = self.env.reset(
                    table_size  = _setting["table_size"],
                    slider_inputs = _setting["slider_inputs"],
                    slider_num  = _setting["slider_num"],
                    )
        return state_curr

if __name__=="__main__":
    sim = DishSimulation(state='image', action_skip=8)
    import time
    # state = sim.reset(mode="continous", slider_num=8)
    state = sim.reset(mode=None, slider_num=8)
    action_space = sim.env.action_space.shape[0]
    action = np.zeros(action_space) # Initialize pusher's speed set as zeros 
    while True:
        action = [0.5, 0, 0, 0, 1]
        state_next, reward, done, mode = sim.env.step(action=action)
        state = state_next
        # if reset or done:
        if done:
            sim.reset(mode="continous")



# slider_inputs = [
#     ("circle", [0.0, -0.1, 0.0, 0.13]),
#     ("circle", [0.2, 0.2, 0.0, 0.13]),
#     ("circle", [-0.2, 0.2, 0.0, 0.13]),
#     ("ellipse", [0.0, 0.6, np.random.random(), 0.13, 0.12]),
#     ("ellipse", [0.3, 0.6, np.random.random(), 0.13, 0.10]),
#     ("ellipse", [-0.3, 0.6, np.random.random(), 0.13, 0.16]),
# ]

# # # 푸셔 입력값 (정수, 실수, 문자열, 딕셔너리, 실수 7개)
# pusher_input = (
#     3, 120.0, "superellipse", 
#     {"a": 0.015, "b": 0.03, "n": 10}, 
#     0.10, 0.185, 0.04, 0.0, -0.5, 0.0
# )

# # 새로운 테이블 크기
# newtableWidth = 3 + np.random.random() * 0.5
# newtableHeight = 3 + np.random.random() * 0.5

# while True:
#     viewer.reset(
#         slider_inputs = slider_inputs,
#         pusher_input = pusher_input,
#         newtableWidth = newtableWidth,
#         newtableHeight = newtableHeight
#     )

#     u_input = [0.5, 0.0, 0.000, 0.0, 0]
#     episode_start = time.time()
#     for i in range(1000):
#         if i > 2:
#             u_input[4] = 1
#         start = time.time()
#         result = viewer.run(u_input)
#         print("Time spent [Hz]: {:.2f}".format(1/(time.time() - start)))

#         time.sleep(0.0001)  # CPU 부하 방지
#         if result.done: 
#             # print(result.reasons)
#             time.sleep(1)
#             slider_inputs = []
#             for slider in result.slider_state:
#                 if len(slider) == 4: slider_inputs.append(("circle", [slider[0], slider[1], slider[2], slider[3]]))
#                 else               : slider_inputs.append(("ellipse",[slider[0], slider[1], slider[2], slider[3], slider[4]]))
#             break

#     print("\tTime spent [s]: {:.2f}".format((time.time() - episode_start)))