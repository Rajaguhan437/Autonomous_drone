
#import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address="127.0.0.1", step_length=1, image_shape=(84,84,1), destination=(-5, -10, -10, 10)):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.destination = destination
        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.moveToPositionAsync(-10, -10, -10, 10).join()
        self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
        #self.drone.simTestLineOfSightToPoint(self.destination)
        #self.drone.simSetTraceLine(.3)

    def transform_obs(self, responses):
        '''
        img1d = np.array(responses[0].image_data_float, dtype=np.float64)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])
        '''
        responses = self.drone.simGetImages([airsim.ImageRequest("4", airsim.ImageType.DepthPerspective, False, False)])
        response = responses[0]
    # get numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 
    # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)

    # original image is fliped vertically
        #img_rgb = np.flipud(img_rgb)
        from PIL import Image

        image = Image.fromarray(img_rgb)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
        ).join()

    
    def reward(self):
        beta = 1
        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )
        x = (self.destination[0] - quad_pt[0])**2
        y = (self.destination[1] - quad_pt[1])**2
        z = (self.destination[2] - quad_pt[2])**2
        dist = (x+y+z)**1/2
        
        if self.state["collision"]:
            reward = -100000
        else:
            reward_dist = (-(dist)**2/1000)+10
            reward_speed = (
                    np.linalg.norm(
                        [
                            self.state["velocity"].x_val,
                            self.state["velocity"].y_val,
                            self.state["velocity"].z_val,
                        ]
                    )
                    - 0.5
                )
            reward = reward_dist + reward_speed
        
        done = False
        if dist <= 1 or self.state["collision"]:
            done = True
        self.drone.simPrintLogMessage("DIST:"+str(dist)+" Reward:"+str(reward)+" "+str(done))
        
        return reward, done
            
    def _compute_reward(self):
        thresh_dist = 7
        beta = 1

        z = -10
        pts = [
            np.array([-0.55265, -31.9786, -19.0225]),
            np.array([48.59735, -63.3286, -60.07256]),
            np.array([193.5974, -55.0786, -46.32256]),
            np.array([369.2474, 35.32137, -62.5725]),
            np.array([541.3474, 143.6714, -32.07256]),
        ]

        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )

        if self.state["collision"]:
            reward = -100
        else:
            dist = 10000000
            for i in range(0, len(pts) - 1):
                dist = min(
                    dist,
                    np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1])))
                    / np.linalg.norm(pts[i] - pts[i + 1]),
                )

            if dist > thresh_dist:
                reward = -10
            else:
                reward_dist = math.exp(-beta * dist) - 0.5
                reward_speed = (
                    np.linalg.norm(
                        [
                            self.state["velocity"].x_val,
                            self.state["velocity"].y_val,
                            self.state["velocity"].z_val,
                        ]
                    )
                    - 0.5
                )
                reward = reward_dist + reward_speed

        done = False
        if reward <= -10:
            done = True

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self.reward()
        info = self.state
        truncated = False
        return obs, reward, done, info

    def reset(self, *,seed=None, options=None):
        self._setup_flight()
        obs, reward, done, info = self.step((0, 0, 0))
        return obs

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset
