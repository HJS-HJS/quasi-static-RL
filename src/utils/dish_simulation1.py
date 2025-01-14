import os
import sys
import copy
import random
import yaml
import numpy as np
import pygame

from third_party.quasi_static_push.scripts.utils.diagram         import Circle, Ellipse, SuperEllipse, RPolygon, SmoothRPolygon
from third_party.quasi_static_push.scripts.utils.object_obstacle import ObjectObstacle
from third_party.quasi_static_push.scripts.utils.object_pusher   import ObjectPusher
from third_party.quasi_static_push.scripts.utils.object_slider   import ObjectSlider
from third_party.quasi_static_push.scripts.utils.param_function  import ParamFunction
from third_party.quasi_static_push.scripts.utils.quasi_state_sim import QuasiStateSim
from third_party.quasi_static_push.scripts.utils.color import COLOR

class Simulation():
    def __init__(self, visualize:str = 'human', state:str = 'image', random_place:bool = True, action_skip:int = 5):
        """
        state : image, information
        """
        # Get initial param
        self.state = state
        self.random = random_place
        self.action_skip = action_skip
        self.gripper_on = False
        self.is_failed = False

        self.distance_buffer = [0] * 1

        # Set pygame display
        if visualize == "human":
            print("[Info] simulator is visulaized")
            os.environ["SDL_VIDEODRIVER"] = "x11"
        elif visualize is None:
            print("[Info] simulator is NOT visulaized")
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            print("[Info] simulator is visulaized")
        
        if (visualize == "human") or (state != "linear"):
            self.visualization = True
        else: self.visualization = False

        ## Get config file
        with open(os.path.dirname(os.path.abspath(__file__)) + "/../../config/config.yaml") as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)

        # Set patameters
        WIDTH, HEIGHT = self.config["display"]["WIDTH"], self.config["display"]["HEIGHT"] # Get pygame display size parameter from config.yaml
        self.display_center = np.array([WIDTH/2, HEIGHT/2])                          # Get center pixel of pygame display

        ## Set parameters
        # Set pixel unit
        self.unit = self.config["display"]["unit"] #[m/pixel]

        # Set pusher
        pusher_num      = self.config["pusher"]["pusher_num"]
        pusher_angle    = np.deg2rad(self.config["pusher"]["pusher_angle"])
        pusher_type     = self.config["pusher"]["pusher_type"]
        pusher_distance = self.config["pusher"]["pusher_distance"]
        self.pusher_d_u_limit= self.config["pusher"]["pusher_d_u_limit"]
        self.pusher_d_l_limit= self.config["pusher"]["pusher_d_l_limit"] 

        pusher_position = self.config["pusher"]["pusher_position"]
        pusher_rotation = np.deg2rad(self.config["pusher"]["pusher_rotation"])

        # Set pusher speed
        u_input = np.zeros(3)                            # Initialize pusher's speed set as zeros 
        self.width   = pusher_distance                        # Initialize pusher's speed set as zeros 
        unit_v_speed = self.config["pusher"]["unit_v_speed"]  # [m/s]
        unit_r_speed = self.config["pusher"]["unit_r_speed"]  # [rad/s]
        unit_w_speed = self.config["pusher"]["unit_w_speed"]  # [m/s]
        self.unit_speed = [unit_v_speed, unit_v_speed, unit_r_speed, unit_w_speed]

        # Set slider 
        self.slider_num = self.config["auto"]["maximun_number"] # Get sliders number
        self.min_r = self.config["auto"]["minimum_radius"]
        self.max_r = self.config["auto"]["maximum_radius"]

        # Set simulate param
        fps = self.config["simulator"]["fps"]      # Get simulator fps from config.yaml
        self.frame = 1 / fps                       # 1 frame = 1/fps
        sim_step = self.config["simulator"]["sim_step"] # Maximun LCP solver step
        self.dist_threshold = float(self.config["simulator"]["dist_threshold"]) # Distance to decide whether to calculate parameters


        ## Generate objects
        # Generate pushers
        self.pushers = ObjectPusher(pusher_num, pusher_angle, pusher_type, pusher_distance, self.pusher_d_u_limit, self.pusher_d_l_limit, pusher_position[0], pusher_position[1], pusher_rotation)

        # Generate quasi-static simulation class
        self.simulator = QuasiStateSim(sim_step)

        self.action_limit = np.array([
            [-1., 1.],
            [-1., 1.],
            [-1., 1.],
            [-1., 1.],
            [0, 2],
        ])
        self.param = None
        state, _, _ = self.reset()

        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=state.shape, dtype=np.float32)
        # self.action_space = Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = state
        self.action_space = np.zeros(5)


    def reset(self, slider_num:int=None):
        del self.param
        
        self.dist = 0.
        self.is_failed = False
        
        ## Set pygame display settings
        WIDTH, HEIGHT = self.config["display"]["WIDTH"], self.config["display"]["HEIGHT"] # Get pygame display size parameter from config.yaml
        # Initialize pygame
        pygame.init()                                       # Initialize pygame
        pygame.display.set_caption("Quasi-static pushing")  # Set pygame display window name
        clock = pygame.time.Clock()                         # Generate pygame clock for apply proper fps
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))   # Set pygame display size
        if not self.random:
            self.table_limit = np.array([WIDTH, HEIGHT])
        else:
            rand_x = random.randint(WIDTH // 3, int(WIDTH * 0.8))
            rand_y = random.randint(HEIGHT // 3, int(HEIGHT * 0.8))
            self.table_limit = np.array([rand_x, rand_y])
        self.backgound = self.create_background_surface(WIDTH, HEIGHT, self.table_limit, self.unit, grid=False) # Generate pygame background surface
        self.table_limit = self.table_limit*self.unit/2

        ## Generate objects
        # Initialize sliders
        if not self.random:
            # Set sliders
            slider_diagram = self.config["sliders"]             # Get sliders property (type, pose, parameters)
            # Set obstacles
            obstacle_diagram = self.config["obstacles"]         # Get sliders property (type, pose, parameters)
            # Generate sliders
            sliders = ObjectSlider()
            for slider in slider_diagram:
                if   slider["type"] == "circle":       sliders.append(Circle(slider['q'], slider['r']))
                elif slider["type"] == "ellipse":      sliders.append(Ellipse(slider['q'], slider['a'], slider['b']))
                elif slider["type"] == "superellipse": sliders.append(SuperEllipse(slider['q'], slider['a'], slider['b'], slider['n']))
                elif slider["type"] == "rpolygon":     sliders.append(RPolygon(slider['q'], slider['a'], slider['k']))
                elif slider["type"] == "srpolygon":    sliders.append(SmoothRPolygon(slider['q'], slider['a'], slider['k']))
            # Generate Obstacles
            obstacles = ObjectObstacle()
            for obstacle in obstacle_diagram:
                if   obstacle["type"] == "circle":       obstacles.append(Circle(obstacle['q'], obstacle['r']))
                elif obstacle["type"] == "ellipse":      obstacles.append(Ellipse(obstacle['q'], obstacle['a'], obstacle['b']))
                elif obstacle["type"] == "superellipse": obstacles.append(SuperEllipse(obstacle['q'], obstacle['a'], obstacle['b'], obstacle['n']))
                elif obstacle["type"] == "rpolygon":     obstacles.append(RPolygon(obstacle['q'], obstacle['a'], obstacle['k']))
                elif obstacle["type"] == "srpolygon":    obstacles.append(SmoothRPolygon(obstacle['q'], obstacle['a'], obstacle['k']))
        else:
            # Generate sliders
            sliders = ObjectSlider()
            _margin  = self.min_r
            if slider_num is None: _slider_num = random.randint(1, self.slider_num) # Get sliders number
            else: _slider_num = np.clip(slider_num, 1, 15)
            points, radius = self.generate_spawn_points(_slider_num, self.min_r, self.max_r, self.table_limit, _margin)
            for point, _r in zip(points, radius):
                a = np.clip(random.uniform(0.8, 1.0) * _r, a_min=self.min_r, a_max=_r)
                b = np.clip(random.uniform(0.75, 1.25) * a, a_min=self.min_r, a_max=_r)
                r = random.uniform(0, np.pi * 2)
                sliders.append(Ellipse(np.hstack((point,[r])), a, b))
            # Generate dummy object
            obstacles = ObjectObstacle()

        self.state_max_num = len(self.pushers.q) + _slider_num * 3
        self._state_idx = random.sample(range(1, 15), len(sliders) - 1)
        self._state_idx.insert(0, 0)

        # Pusher initial pose at corner
        _q = np.sign(sliders[0].q[0:2]) * 0.85 * np.array([WIDTH, HEIGHT]) / 2 * self.unit

        # Initialize pusher position and velocity
        self.pushers.apply_q([_q[0] * random.choice([1, -1]), _q[1] * random.choice([1, -1]), 0., random.uniform(self.pusher_d_l_limit, self.pusher_d_u_limit)])
        # self.pushers.apply_q([_q[0], _q[1], 0., random.uniform(self.pusher_d_l_limit, self.pusher_d_u_limit)])
        self.pushers.apply_v([0., 0., 0., 0.])

        # Generate pygame object surfaces
        for pusher in self.pushers: pusher.polygon = self.create_polygon_surface(pusher.torch_points.cpu().numpy().T, COLOR["RED"],   self.unit) # Generate pygame pushers surface
        for slider in sliders:      slider.polygon = self.create_polygon_surface(slider.torch_points.cpu().numpy().T, COLOR["BLUE"],  self.unit) # Generate pygame sliders surface
        for obstac in obstacles:    obstac.polygon = self.create_polygon_surface(obstac.torch_points.cpu().numpy().T, COLOR["YELLOW"], self.unit) # Generate pygame sliders surface
        sliders[0].polygon                         = self.create_polygon_surface(sliders[0].torch_points.cpu().numpy().T, COLOR["GREEN"],  self.unit) # Generate pygame sliders surface
        self.pusher_center                         = self.create_bead_surface()

        # Quasi-static simulation class
        # Generate parameter functions
        self.param = ParamFunction(sliders,
                                   self.pushers, 
                                   obstacles, 
                                   self.dist_threshold,
                                   )
        
        self._simulate_once(action=[0., 0., 0., 0., 1.], repeat=1)
        return self.generate_result()

    def step(self, action):
        """
        action: 
        """
        success, phi = self._simulate_once(action=action, repeat=self.action_skip)

        # Update pygame display
        self._visualize_update()

        return self.generate_result(success, target_phi=phi[:len(self.param.pushers)], obs_phi=phi[len(self.param.pushers):len(self.param.pushers)*len(self.param.sliders)])
    
    def _visualize_update(self):
        if self.visualization:
            # Bliting background
            self.screen.blit(self.backgound, (0, 0))
            # Bliting sliders
            for slider in self.param.sliders:
                _center = slider.q
                _surface = slider.surface([
                    int(_center[0]/self.unit + self.display_center[0]), 
                    int(-_center[1]/self.unit + self.display_center[1]), 
                    _center[2]
                    ])
                self.screen.blit(_surface[0], _surface[1])
            # Bliting pushers
            for pusher in self.param.pushers:
                _center = pusher.q
                _surface = pusher.surface([
                    int(_center[0]/self.unit + self.display_center[0]), 
                    int(-_center[1]/self.unit + self.display_center[1]), 
                    _center[2]
                    ])
                self.screen.blit(_surface[0], _surface[1])
            # Show updated pygame display
            _q = (int(self.param.pushers.q[0] / self.unit + self.display_center[0] - 16),
                  int(-self.param.pushers.q[1] / self.unit + self.display_center[1] - 16))
            self.screen.blit(self.pusher_center, _q)
            pygame.display.flip()
            return
        else: 
            return

    def _simulate_once(self, action, repeat:int = 1):
        if(len(action) == 4):
            action = np.hstack((action, 1))
        _fail_count = 0

        # Limit pusher speed
        action = np.clip(action, self.action_limit[:, 0], self.action_limit[:, 1])

        action[:4] *= self.unit_speed
        for _ in range(repeat):
            # Update parameters for quasi-state simulation
            self.param.update_param()
            # Get parameters for simulations
            _qs, _qp, _phi, _JNS, _JNP, _JTS, _JTP, _mu, _A, _B = self.param.get_simulate_param()
            if action[4]:
                if not self.gripper_on:
                    if np.min(_phi[:len(self.param.pushers) * len(self.param.sliders)]) < -0.001: 
                        success = False
                        break
                self.gripper_on = True
                _vec = self.param.sliders[0].q[:2] - self.param.pushers.q[:2]
                _vec = np.arctan2(_vec[1], _vec[0]) - np.pi / 2
                _rot = np.array([
                    [-np.sin(_vec), -np.cos(_vec)],
                    [ np.cos(_vec), -np.sin(_vec)]
                    ])
                # Run quasi-static simulator
                _action = action[:4] + random.choice([1., 0.5, -0.5, -1.]) * 1e-6
                _action[:2] = _rot@_action[:2]
                qs, qp, success = self.simulator.run(
                    u_input = _action * self.frame,
                    qs  = _qs,
                    qp  = _qp,
                    phi = _phi,
                    JNS = _JNS,
                    JNP = _JNP,
                    JTS = _JTS,
                    JTP = _JTP,
                    mu  = _mu,
                    A   = _A,
                    B   = _B,
                    perfect_u_control = False
                    )

                ## Update simulation results
                if success:
                    self.param.sliders.apply_v((qs - _qs) / self.frame) # Update slider velocity
                    self.param.sliders.apply_q(qs)                      # Update slider position
                    qp[:2] = np.clip(qp[:2], -self.display_center * 0.85 * self.unit + self.param.pushers.q[3] / 3, self.display_center * 0.85 * self.unit - self.param.pushers.q[3] / 3)
                    self.param.pushers.apply_v((qp - _qp) / self.frame) # Update pusher velocity
                    self.param.pushers.apply_q(qp)                      # Update pusher position
                else: 
                    _fail_count += 1
            else:
                _rot = np.array([
                    [-np.sin(self.param.pushers.rot), -np.cos(self.param.pushers.rot)],
                    [np.cos(self.param.pushers.rot),  -np.sin(self.param.pushers.rot)]
                    ])
                qp = _qp + np.hstack([_rot@action[:2], action[2], action[3]]) * self.frame
                qp[:2] = np.clip(qp[:2], -self.display_center * 0.85 * self.unit + self.param.pushers.q[3] / 3, self.display_center * 0.85 * self.unit - self.param.pushers.q[3] / 3)
                self.param.pushers.apply_q(qp)
                success = True
                self.gripper_on = False
        
        if _fail_count != 0: print("\trecover pusher input")

        for _ in range(_fail_count * 5):
            # Update parameters for quasi-state simulation
            self.param.update_param()
            # Get parameters for simulations
            _qs, _qp, _phi, _JNS, _JNP, _JTS, _JTP, _mu, _A, _B = self.param.get_simulate_param()
            if not self.gripper_on:
                if np.min(_phi[:len(self.param.pushers) * len(self.param.sliders)]) < -0.001: 
                    success = False
                    break
            self.gripper_on = True
            _vec = self.param.sliders[0].q[:2] - self.param.pushers.q[:2]
            _vec = np.arctan2(_vec[1], _vec[0]) - np.pi / 2
            _rot = np.array([
                [-np.sin(_vec), -np.cos(_vec)],
                [ np.cos(_vec), -np.sin(_vec)]
                ])
            # Run quasi-static simulator
            _action = action[:4] + random.choice([1., 0.5, -0.5, -1.]) * 1e-6
            _action[:2] = _rot@_action[:2]
            qs, qp, success = self.simulator.run(
                u_input = _action * self.frame / 5,
                qs  = _qs,
                qp  = _qp,
                phi = _phi,
                JNS = _JNS,
                JNP = _JNP,
                JTS = _JTS,
                JTP = _JTP,
                mu  = _mu,
                A   = _A,
                B   = _B,
                perfect_u_control = False
                )

            ## Update simulation results
            if success:
                self.param.sliders.apply_v((qs - _qs) / self.frame) # Update slider velocity
                self.param.sliders.apply_q(qs)                      # Update slider position
                qp[:2] = np.clip(qp[:2], -self.display_center * 0.85 * self.unit + self.param.pushers.q[3] / 3, self.display_center * 0.85 * self.unit - self.param.pushers.q[3] / 3)
                self.param.pushers.apply_v((qp - _qp) / self.frame) # Update pusher velocity
                self.param.pushers.apply_q(qp)                      # Update pusher position
            else: 
                print("\t\tfailed")
        
        return success, _phi

    def generate_result(self, success:bool = True, target_phi = [10., 10., 10.], obs_phi = [10.,]):
        """
        state, reward, done
        """
        done = False
        if success:
            self.is_failed = False
        else:
            if self.is_failed: done = True
            else: self.is_failed = True
        ## state
        if self.state == 'image':
            # image 
            surface = pygame.display.get_surface()
            state = pygame.surfarray.array3d(surface)
        elif self.state == 'gray':
            # image 
            surface = pygame.display.get_surface()
            img = pygame.surfarray.array3d(surface)
            gray_img = np.dot(img[..., :], [0.299, 0.587, 0.114])
            state = np.expand_dims(gray_img, axis=2)
        else:
            # Linear
            # Table size    2 (x,y) [m]
            # Pusher pose   4 (x,y,r,width)
            # target data   5 (x,y,r,a,b)
            # obstacle data 5 (x,y,r,a,b) * N (max 15)

            _table   = copy.deepcopy(self.table_limit)
            _pusher  = copy.deepcopy(self.param.qp)
            _sliders = np.zeros(len(self.param.sliders)*5)

            for idx in range(len(self.param.sliders)):
                _slider = self.param.sliders[idx]
                _sliders[5*idx:5*idx+5] = np.hstack((_slider.q, _slider.a, _slider.b))

            # Normalize
            _pusher[3]     = (_pusher[3] - self.pusher_d_l_limit) / (self.pusher_d_u_limit - self.pusher_d_l_limit) - 0.5
            _pusher[2]     = ((_pusher[2]     + np.pi) % (2 * np.pi)) / np.pi - 1
            _sliders[2::5] = ((_sliders[2::5] + np.pi) % (2 * np.pi)) / np.pi - 1

            state1 = np.hstack((_table, _pusher, _sliders[:5]))
            state2 = _sliders[5:].reshape(-1,5)
            if len(state2) < 2:
                state2 = np.vstack((state2, np.zeros((2 - len(state2), 5))))
            state = state1, state2

        ## reward
        reward = 0.0
        # distance
        dist = np.linalg.norm(self.param.sliders[0].q[0:2] - self.param.pushers.q[0:2])
        
        # reward += int((1 - (dist * 2))*100) / 1000
        
        # reward += int((1 - (dist * 2))*100) / 1000 / 2
        # if dist > 0.3:
        #     reward += -(dist - 0.3)

        # reward += (-dist**2 + 0.4**2) * 0.1 / 0.4**2 - 0.1

        # if (dist - self.dist) < 1e-3:
        #     reward += 0.1
        # else: reward += -0.2
        # if (dist - self.dist) < -1e-2:
        # # if (dist - self.dist) < -2e-2:
        #     # reward += 0.1
        #     pass
        # else: reward += -0.1
        reward += -0.1

        self.dist = dist

        if dist < 0.01:
            
            _width = 10.
            print("\t\ttry grasp")
            while True:

                # Check gripper width changed
                if np.abs(_width - self.param.pushers.q[3]) < 0.0001:
                    print("\t\t\twidth not changed")
                    done = True
                    break
                else: _width = self.param.pushers.q[3]

                dist = np.linalg.norm(self.param.sliders[0].q[0:2] - self.param.pushers.q[0:2])
                if (max(target_phi) < 0.003) and (dist < 0.01):
                    print("\t\t\tgrasp")
                    done = True
                    break

                success, phi = self._simulate_once(action=np.array([0., 0., 0., -1.]), repeat=1)
                self._visualize_update()
                target_phi = phi[:len(self.param.pushers)]
                obs_phi=phi[len(self.param.pushers):len(self.param.pushers)*len(self.param.sliders)]

        ## done
        for slider in self.param.sliders:
            if np.any(np.abs(slider.q[0:2]) > self.table_limit):
                done = True
                # reward -= 0.05
                reward -= 0.5
                print("\t\t\tdish fall out")
                break
        if max(target_phi) < 0.015:
            print("\t\t\tgrasp successed!!")
            done = True
            reward = +10
            # reward = +75
            obs_phi = obs_phi[np.where(obs_phi > 0)]
            if len(obs_phi) > 0:
                reward += np.log(min(obs_phi)) * 5 / 10
                # reward += np.log(min(obs_phi)) * 5
        return state, reward, done
    
    def generate_spawn_points(self, num_points, min_r, max_r, limit, margin, center_bias=0.8):
        points = []
        x_range = (-limit[0] + margin, limit[0] - margin)
        y_range = (-limit[1] + margin, limit[1] - margin)

        # 첫 번째 점을 랜덤하게 생성
        center_x = random.uniform(*x_range)
        center_y = random.uniform(*y_range)
        points.append((center_x, center_y))

        # Raduis of inital point
        init_r = random.uniform(min_r, max_r)
        available_lengh = (init_r + min_r, init_r + max_r)
        
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
            if all(d >= (init_r + min_r) for d in distances):
                points.append(point)
        
        points = np.array(points)

        min_distances = np.ones(len(points)) * min_r
        min_distances[0] = init_r

        for idx, point in enumerate(points):
            if idx == 0: continue
            distances = [np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) for p in points]
            distances = distances - min_distances
            distances[idx] = max_r
            min_distances[idx] = min(distances)

        # 첫 번째 점을 포함한 최종 점 리스트
        return points, np.array(min_distances)

    def create_background_surface(self, width, height, table_size, unit, grid:bool = False):
        # Generate pygame surface
        background_surface = pygame.Surface((width, height))    # Generate pygame surface with specific size
        background_surface.fill(COLOR["BLACK"])                          # Fill surface as white

        # Draw white rectangle at the center
        center_x = width // 2
        center_y = height // 2
        table_rect = pygame.Rect(
            center_x - table_size[0] // 2,  # Top-left x
            center_y - table_size[1] // 2,  # Top-left y
            table_size[0],                  # Width
            table_size[1]                   # Height
        )
        pygame.draw.rect(background_surface, COLOR["WHITE"], table_rect)  # Fill the table area with black

        # Draw gridlines
        # 0.1m spacing
        if grid:
            gap = 1 / unit / 10  # Guideline lengh
            for y_idx in range(int(height / gap)): pygame.draw.line(background_surface, COLOR["LIGHTGRAY"], (0, y_idx * gap), (width, y_idx * gap), 2)  # horizontal gridlines
            for x_idx in range(int(width  / gap)): pygame.draw.line(background_surface, COLOR["LIGHTGRAY"], (x_idx * gap, 0), (x_idx * gap, height), 2) # vertical gridlines
            # 1m spacing
            gap = 1 / unit      # Guideline lengh
            for y_idx in range(int(height / gap)): pygame.draw.line(background_surface, COLOR["DARKGRAY"], (0, y_idx * gap), (width, y_idx * gap), 2)   # horizontal gridlines
            for x_idx in range(int(width  / gap)): pygame.draw.line(background_surface, COLOR["DARKGRAY"], (x_idx * gap, 0), (x_idx * gap, height), 2)  # vertical gridlines
        return background_surface

    def create_polygon_surface(self, points, color, unit):
        # Convert polygon points coordinate to pygame display coordinate\
        _points = points.T / unit

        w_l = np.abs(np.min(_points[:,0])) if np.abs(np.min(_points[:,0])) > np.max(_points[:,0]) else np.max(_points[:,0])
        h_l = np.abs(np.min(_points[:,1])) if np.abs(np.min(_points[:,1])) > np.max(_points[:,1]) else np.max(_points[:,1])

        _points[:,0] =  1.0 * _points[:,0] + w_l
        _points[:,1] = -1.0 * _points[:,1] + h_l

        # Set pygame surface size
        width  = int(w_l * 2)
        height = int(h_l * 2)
        # Generate pygame surface
        polygon_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        # Draw
        pygame.draw.polygon(polygon_surface, color, _points.astype(int).tolist())                               # Draw polygon
        pygame.draw.line(polygon_surface, COLOR["WHITE"], (width / 4, height / 2), (width * 3 / 4, height / 2), 3)   # Draw horizontal line
        pygame.draw.line(polygon_surface, COLOR["WHITE"], (width / 2, height / 4), (width / 2, height * 3 / 4), 3)   # Draw vertical line
        return polygon_surface

    def create_bead_surface(self, radius = 12, width = 4):
        # Generate pygame surface
        lengh = (radius + width) * 2
        polygon_surface = pygame.Surface((lengh,lengh), pygame.SRCALPHA)
        center = lengh // 2
        # Draw
        pygame.draw.circle(polygon_surface, color=COLOR["DARKGRAY"], center=[center,center] , radius=radius, width = width)
        return polygon_surface

    def close(self):
        print("close")
        pygame.quit()
        del(self.param)

    def __del__(self):
        print("del")
        pygame.quit()
        del(self.param)

class DishSimulation():
    def __init__(self, visualize:str = 'human', state:str = 'image', random_place:bool = True, action_skip:int = 5):
        self.env = Simulation(visualize = visualize, state = state, random_place = random_place, action_skip = action_skip)

    def keyboard_input(self, action):
        # Keyboard event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        # Keyboard input
        keys = pygame.key.get_pressed()

        ## Keyboard input response
        # Move pusher center in y-axis (ws)
        if keys[pygame.K_w]:   action[0] +=  1/10  # Move forward      (w)
        elif keys[pygame.K_s]: action[0] += -1/10  # Move backward     (s)
        else:                  action[0]  =  0                # Stop
        # Move pusher center in x-axis (ad)
        if keys[pygame.K_a]:   action[1] +=  1/10  # Move left         (a)
        elif keys[pygame.K_d]: action[1] += -1/10  # Move right        (d)
        else:                  action[1]  =  0                # Stop
        # Rotate pusher center (qe)
        if keys[pygame.K_q]:   action[2] +=  1/10  # Turn ccw          (q)
        elif keys[pygame.K_e]: action[2] += -1/10  # Turn cw           (e)
        else:                  action[2]  =  0                # Stop
        # Control gripper width (left, right)
        if keys[pygame.K_LEFT]:    action[3] += -1/10  # Decrease width
        elif keys[pygame.K_RIGHT]: action[3] +=  1/10  # Increase width
        else:                      action[3]  = 0

        if keys[pygame.K_SPACE]:   action[4] = 1  # Gripper On
        else:                      action[4] = 0  # Gripper Off

        if keys[pygame.K_r]: 
            return np.zeros_like(action), True

        return action, False
    
    def __del__(self):
        del self.env

if __name__=="__main__":
    sim = DishSimulation(state='linear', action_skip=1)
    # sim = DishSimulation()
    observ_space = sim.env.observation_space.shape[0]
    action_space = sim.env.action_space.shape[0]
    action = np.zeros(action_space) # Initialize pusher's speed set as zeros 
    while True:
        action, reset = sim.keyboard_input(action)
        state, reward, done = sim.env.step(action=action[:4])
        print(state)
        if reset or done:
            # sim.env.reset(slider_num=1)
            sim.env.reset(  )