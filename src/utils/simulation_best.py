import sys
import os
import time
import yaml
import numpy as np
import random
import copy
from typing import List

so_file_path = os.path.abspath("../../cpp/10")
sys.path.append(so_file_path)

from quasi_static_push import SimulationViewer, SimulationResult, SimulationDoneReason, GripperMotion, Player

class Simulation():
    def __init__(self, visualize:str = 'human', state:str = 'image', action_skip:int = 5, record:bool = False, save_dir:str = "recordings"):
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
        with open(os.path.dirname(os.path.abspath(__file__)) + "/../../config/config.yaml") as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)

        # Set patameters
        self.display_size = np.array([self.config["display"]["WIDTH"], self.config["display"]["HEIGHT"] ]) # Get display size parameter from config.yaml

        ## Set parameters
        # Set pixel unit
        # self.unit = self.config["display"]["unit"] #[m/pixel]
        self.unit = 0.001953125

        self.table_range = np.array([0.5, 1.25])
        self.table_pixel_range = (self.table_range / self.unit).astype(int)

        # Set pusher
        _pusher_type = self.config["pusher"]["pusher_type"]

        # Set pusher unit speed
        unit_v_speed = self.config["pusher"]["unit_v_speed"] / 2 * 5 # [m/s]
        unit_r_speed = self.config["pusher"]["unit_r_speed"]     * 5 # [rad/s]
        unit_w_speed = self.config["pusher"]["unit_w_speed"] / 2 * 5 # [m/s]
        self.unit_speed = [unit_v_speed, unit_v_speed, unit_r_speed, unit_w_speed, int(1)]

        self.pusher_width_limit = np.array([self.config["pusher"]["pusher_d_u_limit"], self.config["pusher"]["pusher_d_l_limit"]])

        # Set slider 
        self.slider_max_num = self.config["auto"]["maximun_number"] # Get sliders number
        self.min_r = self.config["auto"]["minimum_radius"]
        self.max_r = self.config["auto"]["maximum_radius"]

        # Set simulation param
        self.fps = self.config["simulator"]["fps"] * 5
        self.action_skip = action_skip * 2
        self.spawn_bias = 0.95

        self.pusher_input = [
            self.config["pusher"]["pusher_num"], self.config["pusher"]["pusher_angle"], _pusher_type["type"], 
            {"a": _pusher_type["a"], "b": _pusher_type["b"], "n": _pusher_type["n"]}, 
            self.config["pusher"]["pusher_distance"], self.pusher_width_limit[0], self.pusher_width_limit[1],
            0.0, 0.0, 0.0
        ]

        self.action_limit = np.array([
            [-1., 1.],
            [-1., 1.],
            [-1., 1.],
            [-1., 1.],
            [0, 1],
        ])
        self.state_prev = None
        self.action_space = np.zeros_like(self.unit_speed)
        if state == "image" :   self.observation_space = np.zeros((self.config["display"]["WIDTH"],self.config["display"]["HEIGHT"],3))
        elif state == "gray":   self.observation_space = np.zeros((self.config["display"]["WIDTH"],self.config["display"]["HEIGHT"],1))
        elif state == "linear": self.observation_space = np.zeros(1 + 2 + 4 + 5), np.zeros((2, 5))

        self.simulator = SimulationViewer(
            window_width = self.display_size[0],
            window_height = self.display_size[1],
            scale = 1 / self.unit,
            headless = not visualize,
            # gripper_movement = GripperMotion.MOVE_TO_TARGET,
            gripper_movement = GripperMotion.MOVE_XY,
            # gripper_movement = GripperMotion.MOVE_FORWARD,
            frame_rate = self.fps,
            frame_skip = self.action_skip,
            # grid = False,
            grid = True,
            recording_enabled = record,
            recording_path = save_dir,
            show_closest_point = False,
            )
        
    def reset(self, 
              table_size:List[float] = None,
              slider_state:List[List[float]] = None,
              slider_num:int=None,
              ):
                        
            # Table setting
            if table_size is None: 
                pass
            else:
                _table_limit = (np.array(table_size) / self.unit).astype(int)
                self.table_limit = _table_limit * self.unit / 2

            # Slider setting
            if slider_state is None:
                for _ in range(1000):
                # while True:
                    # Table
                    _table_limit_width  = random.randint(self.table_pixel_range[0], self.table_pixel_range[1])
                    _table_limit_height = random.randint(self.table_pixel_range[0], self.table_pixel_range[1])
                    _table_limit = np.array([_table_limit_width, _table_limit_height])
                    table_size = _table_limit * self.unit
                    self.table_limit = _table_limit * self.unit / 2
                    # slider
                    slider_inputs = []
                    _slider_num = random.randint(1, self.slider_max_num) if slider_num is None else np.clip(slider_num, 1, 15)
                    points, radius = self.generate_spawn_points(_slider_num)
                    for point, _r in zip(points, radius):
                        a = np.clip(random.uniform(0.8, 1.0) * _r, a_min=self.min_r, a_max=_r)
                        b = np.clip(random.uniform(0.75, 1.25) * a, a_min=self.min_r, a_max=_r)
                        r = random.uniform(0, np.pi * 2)
                        slider_inputs.append(("ellipse", [point[0], point[1], r, a, b]))
                    if len(slider_inputs) == slider_num: break
            else:
                slider_inputs = []
                # print(slider_inputs)
                for slider in slider_state:
                    # print(slider)
                    if len(slider) == 4: slider_inputs.append(("circle", [slider[0], slider[1], slider[2], slider[3]]))
                    else               : slider_inputs.append(("ellipse",[slider[0], slider[1], slider[2], slider[3], slider[4]]))
            slider_num = len(slider_inputs)
            
            # Initial pusher pose
            self.pusher_input[4] = random.uniform(self.pusher_input[5], self.pusher_input[6])    # width
            self.pusher_input[7] = 0   # x
            self.pusher_input[8] = 0   # y
            self.pusher_input[9] = 0   # w

            self.state_prev = None

            self.simulator.reset(
                slider_inputs = slider_inputs,
                pusher_input = tuple(self.pusher_input),
                newtableWidth = self.table_limit[0] * 2,
                newtableHeight = self.table_limit[1] * 2,
            )

            return self.step([0.])

    def image_without_gripper(self):
        self.simulator.renderViewer_(False)
        return self.simulator.getImageState()

    def step(self, action, mode:int = 1):
        if(len(action) == 3): action = np.hstack((action, 1))

        if len(action) == 1:
            state_curr = self.simulator.run([0., 0., 0. ,0., 0.])
            mode = 0
        elif mode == 0:
            if(len(action) != 4): print("Invalid position size")
            action = np.clip(action, self.action_limit[:4, 0], self.action_limit[:4, 1])
            action[:2] *= self.display_size / 2 * self.unit
            action[2] *= np.pi / 3
            action[3] = self.pusher_width_limit[1] + (action[3] / 2 + 0.5) * (self.pusher_width_limit[0] - self.pusher_width_limit[1])
            self.simulator.applyGripperPosition(action)

            state_curr = self.simulator.run([0., 0., 0., 0., 1])

        elif mode == 1:
            if(len(action) != 4): print("Invalid action space")
            
            action = np.clip(action, self.action_limit[:4, 0], self.action_limit[:4, 1])
            action *= self.unit_speed[:4]
            state_curr = self.simulator.run(np.hstack((action, 1)))

        time.sleep(0.01)

        result = self.get_results(self.state_prev, state_curr, mode)
        self.state_prev = state_curr

        return result
    
    def update_state(self):
        state_curr = self.simulator.getState()

        result = self.get_results(self.state_prev, state_curr, 0)
        self.state_prev = state_curr
        
    
    def get_results(self, state_prev: SimulationResult = None, state_curr: SimulationResult = None, mode:int = 0):
        # State
        table_range_range = (self.table_range[1] - self.table_range[0]) / 2
        
        _table = ((self.table_limit - self.table_range[0] / 2) / table_range_range) * 2 - 1

        _pusher = copy.deepcopy(state_curr.pusher_state)
        if state_prev is None:
            _sliders = copy.deepcopy(np.array(state_curr.slider_state))
            _sliders_diff = _sliders - _sliders
        elif len(state_curr.slider_state) == 0 or len(state_curr.slider_state) != len(state_prev.slider_state):
            _sliders = copy.deepcopy(np.array(state_prev.slider_state))
            _sliders_diff = _sliders - _sliders
        else:
            _sliders = copy.deepcopy(np.array(state_curr.slider_state))
            _sliders_diff = _sliders - np.array(state_prev.slider_state)


        _pusher_theta_cos = np.cos(_pusher[2])
        _pusher_theta_sin = np.sin(_pusher[2])
        _sliders_theta_cos = np.cos(_sliders[:,2])
        _sliders_theta_sin = np.sin(_sliders[:,2])

        _pusher[3] = (_pusher[3] - self.pusher_width_limit[1]) / (self.pusher_width_limit[0] - self.pusher_width_limit[1]) * 2 - 1
        _sliders[:,3:5] = (_sliders[:,3:5] - self.min_r) / (self.max_r - self.min_r) * 2 - 1

        relative_pose = (_sliders[0][:2] - _pusher[:2])
        relative_dist = np.linalg.norm(relative_pose)
        relative_pose /= relative_dist

        _state1 = np.hstack((
                _table,
                _pusher[:2] / self.table_limit,
                _pusher_theta_cos,
                _pusher_theta_sin,
                _pusher[3],
                _sliders[0][:2] / self.table_limit,
                [
                    _sliders_theta_cos[0],
                    _sliders_theta_sin[0]
                ],
                _sliders[0][3:5],
                relative_pose,
                ))

        _state2 = np.zeros(((len(_sliders)), 20))

        pusher_theta = _pusher[2]
        pusher_dir = np.array([np.cos(pusher_theta), np.sin(pusher_theta)])

        for idx in range(0, len(_sliders)):

            if idx == 0: _target = -1.0
            else:        _target = 1.0
            
            edge = np.sign(_sliders[idx][:2]) * (self.table_limit - np.abs(_sliders[idx][:2]))

            if mode <= 0:
                relative_pose = _sliders[idx][:2] - _sliders[0][:2]
                relative_dist = 0.0
            else:
                relative_pose = _sliders[idx][:2] - _pusher[:2]
                relative_dist = np.linalg.norm(relative_pose)
                relative_pose /= relative_dist

            # angle difference
            angle_diff = np.arctan2(relative_pose[1], relative_pose[0]) - pusher_theta
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]

            # dot product
            direction_alignment = np.dot(pusher_dir, relative_pose)


            _sliders_movement_diff = _sliders_diff[idx][:2]
            _sliders_movement_diff_dist = np.linalg.norm(_sliders_movement_diff) + 1e-9
            _sliders_movement_diff /= _sliders_movement_diff_dist
            _sliders_angle_diff = (_sliders_diff[idx][2] + np.pi) % (2 * np.pi) - np.pi

            _state2[idx] = np.concatenate([
                _table,
                [_target],
                _sliders[idx][:2] / self.table_limit,
                [
                    _sliders_theta_cos[idx],
                    _sliders_theta_sin[idx]
                ],
                _sliders[idx][3:5],
                relative_pose,
                [relative_dist],
                [angle_diff / np.pi],
                [direction_alignment],
                edge,
                _sliders_movement_diff,
                [_sliders_movement_diff_dist],
                [_sliders_angle_diff / np.pi]
            ])

        state = _state1, _state2

        if self.state == "linear":
            pass
        elif self.state == "image":
            state = state_curr.image_state, _state1, _state2
        else:                        state = None

        # Reward
        if mode == 0: reward = self.cal_init_reward(state_prev, state_curr)
        else:         reward = self.cal_reward(state_prev, state_curr)

        # Mode
        if state_curr.mode < 0:      mode = 0
        else:                        mode = 1

        return state, reward, state_curr.done, mode 

    def cal_init_reward(self, state_prev: SimulationResult, state_curr: SimulationResult):
        ## Reward
        reward = 0.0

        if state_prev is None: return 0
        if state_curr.done & SimulationDoneReason.DONE_GRASP_SUCCESS.value:
            print("DONE_GRASP_SUCCESS")
            return 2.0
        if state_curr.done & SimulationDoneReason.DONE_GRASP_FAILED.value:
            print("DONE_GRASP_FAILED")
            return -2.0

        # Spawn failed penalty
        if state_prev.mode == state_curr.mode:
            return -5.0

        ## Pusher distance from target
        if len(state_curr.slider_state) == 0:
            pusher_distance = (state_curr.pusher_state[:2] - np.array([0, 0]))
        else:
            pusher_distance = (state_curr.pusher_state[:2] - state_curr.slider_state[0][:2])
        pusher_distance = np.linalg.norm(pusher_distance)

        reward = np.clip(2.0 * (1 - pusher_distance / 0.4), -2.0, 2.0)

        return reward

    def cal_reward(self, state_prev: SimulationResult, state_curr: SimulationResult):
        if state_prev is None: return 0

        ## Reward
        reward = 0.0

        ## Failed
        if state_curr.done & SimulationDoneReason.DONE_GRASP_SUCCESS.value:
            print("DONE_GRASP_SUCCESS")
            return 2.0
        if state_curr.done & SimulationDoneReason.DONE_GRASP_FAILED.value:
            print("DONE_GRASP_FAIL")
            return -2.0

        ## Pusher distance from target
        pusher_distance_prev = np.linalg.norm(state_prev.pusher_state[:2] - state_prev.slider_state[0][:2])
        if len(state_curr.slider_state) == 0:
            pusher_distance_curr = pusher_distance_prev
        else:
            pusher_distance_curr = np.linalg.norm(state_curr.pusher_state[:2] - state_curr.slider_state[0][:2])

        # distance
        reward += -0.1
        reward += max(0.4 * (1 - pusher_distance_curr / 0.3) - 0.4, -2.0)

        ## Slider
        if len(state_prev.slider_state) != len(state_curr.slider_state):
            reward += -1.0
        else:
            slider_distance = np.linalg.norm((np.array(state_prev.slider_state)[:,:2] - np.array(state_curr.slider_state)[:,:2]), axis=1)
            slider_distance_diff = slider_distance * (self.fps / 5) / self.action_skip

            if len(np.where(np.abs(slider_distance_diff[1:]) - 1e-5 > 0)[0]) > 0:
                reward += -0.20
            
            if slider_distance_diff[0] - 1e-5 > 0:
                reward += -0.75
            
            # Simulation break case
            if np.max(np.abs(slider_distance_diff)) > 0.2:
                print("SIMULATION BREAK")
                reward = -1000

            # Check most danger
            danger_list = np.array(state_curr.slider_state)[:,:2]

            danger_list1 = ((np.abs(danger_list) + self.min_r * 5.0 - self.table_limit) / (self.min_r * 5.0)).reshape(-1)
            danger_list1[np.where(danger_list1 < 0.0)[0]] = 0
            reward += -0.5 * (np.max(danger_list1))

            reward = np.clip(reward, -2.0, 2.0) / 2

            # Check danger movement
            slider_distance = (
                (np.abs(np.array(state_prev.slider_state)[:,:2]) - np.abs(np.array(state_curr.slider_state)[:,:2]))
                ).reshape(-1)
            
            danger_list2 = ((np.abs(danger_list) + self.min_r * 4.0 - self.table_limit) / (self.min_r * 4.0)).reshape(-1)
            danger_list2[np.where(danger_list2 < 0.0)[0]] = 0
            danger_list2[np.where(slider_distance + 1e-9 > 0)[0]] = 0
            reward += -np.max(danger_list2) * 0.60
            danger_list3 = ((np.abs(danger_list) + self.min_r * 1.5 - self.table_limit) / (self.min_r * 1.5)).reshape(-1)
            danger_list3[np.where(danger_list3 < 0.0)[0]] = 0
            danger_list3[np.where(slider_distance + 1e-9 > 0)[0]] = 0
            reward += -np.max(danger_list3) * 1.00

        ## Failed
        if state_curr.done & SimulationDoneReason.DONE_FALL_OUT.value:
            print("DONE_FALL_OUT")
            reward += -2.0
        # return reward
        return reward
    
    def get_state(self):
        if self.state_prev is None: return None
        table_size = self.table_limit * 2
        slider_state = self.state_prev.slider_state
        return {"table_size":table_size, "slider_state":slider_state}

    def generate_spawn_points(self, num_points):
        points = []
        x_range = ((-self.table_limit[0] + self.min_r * 1.3) * 0.9, (self.table_limit[0] - self.min_r * 1.3) * 0.9)
        y_range = ((-self.table_limit[1] + self.min_r * 1.3) * 0.9, (self.table_limit[1] - self.min_r * 1.3) * 0.9)

        # 첫 번째 점을 랜덤하게 생성
        center_x = random.uniform(*x_range)
        center_y = random.uniform(*y_range)
        points.append((center_x, center_y))

        # Raduis of inital point
        init_r = random.uniform(self.min_r, self.max_r)
        available_lengh = (init_r + self.min_r, init_r + self.max_r)
        
        # 나머지 점 생성
        candidate_points = []
        for _ in range(num_points - 1):
            # 첫 번째 점 주변에서 가우시안 분포로 점 생성
            if random.random() < self.spawn_bias:  # 중심 근처에 생성될 확률
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

        min_distances = np.array(min_distances)

        if random.random() < 0.4:
            indices = np.random.permutation(len(points))
            # 동일한 순서로 두 배열 섞기
            points = points[indices]
            min_distances = min_distances[indices]

        # 첫 번째 점을 포함한 최종 점 리스트
        return points, min_distances

class DishSimulation():
    def __init__(self, visualize:str = 'human', state:str = 'image', action_skip:int = 5, record:bool = False, save_dir:str = "recordings"):
        self.env = Simulation(visualize = visualize, state = state, action_skip = action_skip, record = record, save_dir=save_dir)
        self._count = 0
        self.save_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
        self.skip_frame = 0
        self.reset()
    
    def reset(self, mode:str=None, slider_num:int = 5):
        if mode == None: 
            state_curr = self.env.reset(slider_num=slider_num)
            return state_curr

        elif mode == "continuous":
            _setting = self.env.get_state()
            _dish_prev = len(_setting["slider_state"])
            self.env.update_state()
            _setting = self.env.get_state()
            _dish_curr = len(_setting["slider_state"])
            if (_setting == None) or (_dish_curr == 0) or (_dish_curr == _dish_prev):
                state_curr = self.env.reset(slider_num=slider_num)
            else:
                state_curr = self.env.reset(slider_num=1)
                state_curr = self.env.reset(
                    table_size  = _setting["table_size"],
                    slider_state = _setting["slider_state"],
                    )
            return state_curr

    def keyboard_control(self):
        return self.env.simulator.keyboard_input()

    def replay_video(self):
        import cv2
        player = Player("recordings", GripperMotion.MOVE_XY)
        state_prev = None
        for is_new, state, action in player:
            if is_new: 
                state_prev = state
                continue

            # cv2.imshow("Replay", state.image_state)
            # time.sleep(1/30)
            # if cv2.waitKey(30) == 27:  # ESC 키로 종료
            #     break
            if state.mode < 0: continue
            yield action[:4], state_prev, self.env.get_results(state_prev, state, state.mode + 1)
            state_prev = state

        cv2.destroyAllWindows()
        del player
        return

if __name__=="__main__":
    # sim = DishSimulation(state='linear', action_skip=8, record=True)
    sim = DishSimulation(state='linear', action_skip=8)
    print("start")

    # for action, (state_next, reward, done, mode) in sim.replay_video():
    #     print(state_next)
    #     pass
    # exit()

    import time
    state = sim.reset(mode=None, slider_num=2)
    state_curr, _, _, mode = sim.reset(mode="continuous", slider_num=2)
    state_next, reward, done, mode_next = sim.env.step([random.choice([0.9, -0.9]), random.choice([0.9, -0.9]), 0, 0.5], mode)
    # state = sim.reset(mode=None, slider_num=8)
    action_space = sim.env.action_space.shape[0]
    action = np.zeros(action_space) # Initialize pusher's speed set as zeros 
    step = 0
    while True:
        action, escape, reset = sim.keyboard_control()
        if any(action) != 0.:
            # if action[4]>0.5:
            #     action[4] = np.random.random() * 0.5 + 0.5
            # else:
            #     action[4] = np.random.random() * 0.5
            # action[:4] *= np.random.random(4)
            # action[4] = np.random.random() * 0.5
        
            step += 1
            if step % 10 == 0:
                state_next1, state_next2 = state_next
                _, reward, action = sim.env.augment_init_data(state_next1)
                state_next, reward, done, mode = sim.env.step(action=action, mode=0)
            else:
                state_next, reward, done, mode = sim.env.step(action=action[:4])
                
            state = state_next
            time.sleep(0.01)
            if reset or done:
                state_curr, _, _, mode = sim.reset(mode="continuous")
                state_next, reward, done, mode_next = sim.env.step([random.choice([0.9, -0.9]), random.choice([0.9, -0.9]), 0, 0.5], mode)
                step = 0
                time.sleep(1)
            if escape:
                exit()
        else:
            time.sleep(0.01)
            if reset:
                state_curr, _, _, mode = sim.reset(mode="continuous")
                state_next, reward, done, mode_next = sim.env.step([random.choice([0.9, -0.9]), random.choice([0.9, -0.9]), 0, 0.5], mode)
                step = 0
            if escape:
                exit()
