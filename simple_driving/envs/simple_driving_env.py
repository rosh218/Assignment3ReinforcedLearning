import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt
import time
from collections import defaultdict

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'fp_camera', 'tp_camera']}

    def __init__(self, isDiscrete=True, renders=False):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40], dtype=np.float32),
            high=np.array([40, 40], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self._envStepCounter = 0
        

    def step(self, epsilon, Q):
        # Feed action to the car and get observation of car's state
        if (self._isDiscrete):
            action = SimpleDrivingEnv.epsilon_greedy(self, self.car.get_observation(), Q, epsilon, 1, 0)
        else:
            # Implement continuous action space here
            pass
        self.car.apply_action(action)
        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

            carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
            goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
            car_ob = self.getExtendedObservation()

            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

            # Compute reward as L2 change in distance to goal
            dist_to_goal = math.sqrt(((carpos[0] - goalpos[0]) ** 2 +
                                    (carpos[1] - goalpos[1]) ** 2))
            reward = -dist_to_goal
            self.prev_dist_to_goal = dist_to_goal

            # Done by reaching goal
            if dist_to_goal < 1.5 and not self.reached_goal:
                #print("reached goal")
                self.done = True
                self.reached_goal = True

            ob = car_ob

        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0

        # Set the goal to a random target
        x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        self.goal = (x, y)
        self.done = False
        self.reached_goal = False

        # Visual element of the goal
        self.goal_object = Goal(self._p, self.goal)

        # Get observation to return
        carpos = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((carpos[0] - self.goal[0]) ** 2 +
                                           (carpos[1] - self.goal[1]) ** 2))
        car_ob = self.getExtendedObservation()
        return np.array(car_ob, dtype=np.float32)

    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.2

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
            # self.rendered_img.set_data(frame)
            # plt.draw()
            # plt.pause(.00001)

        elif mode == "tp_camera":
            car_id = self.car.get_ids()
            base_pos, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                    distance=20.0,
                                                                    yaw=40.0,
                                                                    pitch=-35,
                                                                    roll=0,
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        # self._observation = []  #self._racecar.getObservation()
        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        goalPosInCar, goalOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, goalpos, goalorn)

        observation = [goalPosInCar[0], goalPosInCar[1]]
        return observation

    def _termination(self):
        return self._envStepCounter > 2000

    def close(self):
        self._p.disconnect()


    def epsilon_greedy(env, state, Q, epsilon, episodes, episode):
        """Selects an action to take based on a uniformly random sampled number.
        If this number is greater than epsilon then returns action with the largest
        Q-value at the current state. Otherwise it returns a random action.

        Args:
            env: gym object.
            state: current state
            Q: Q-function. This is a dictionary that is indexed by the state and
            returns an array of Q-values for each action at that state. For example,
            Q[0] will return an array of Q-values for state 0 where the index of
            the array corresponds to the action.
            epsilon: control how often you explore random actions versus focusing on
                    high value state and actions
            episodes: maximum number of episodes (used in other epsilon greedy variant later)
            episode: number of episodes played so far (used in other epsilon greedy variant later)

        Returns:
            Action to be executed for next step.
        """
        if np.random.uniform(0, 1) > epsilon:
            #### return the action with the highest Q value at the given state  ####
            return np.argmax(Q[state])
            ########################################################################
        else:
            return env.action_space.sample()
        

    def simulate(env, Q, max_episode_length, epsilon, episodes, episode):
        """Rolls out an episode of actions to be used for learning.

        Args:
            env: gym object.
            Q: state-action value function
            epsilon: control how often you explore random actions versus focusing on
                    high value state and actions
            episodes: maximum number of episodes
            episode: number of episodes played so far

        Returns:
            Dataset of episodes for training the RL agent containing states, actions and rewards.
        """
        D = []
        state = env.reset()                                                     # line 2 - note we don't sample the start state since this is predefined
        done = False
        for step in range(max_episode_length):                                  # line 3
            action = SimpleDrivingEnv.epsilon_greedy(env, state, Q, epsilon, episodes, episode)  # line 4
            next_state, reward, done, info = env.step(action)                   # line 5
            #### change reward so that a negative reward is given if the agent falls down a hole #######
            if env.desc[int(next_state/env.ncol), int(next_state % env.ncol)] == b'H':  # fell in hole
                reward = -1.0
            ############################################################################################
            D.append([state, action, reward, next_state])                       # line 7
            state = next_state                                                  # line 8
            if done:                                                            # if we fall into a hole or reach treasure then end episode
                break
        return D                                                                # line 10

    def q_learning(env, gamma, episodes, max_episode_length, epsilon, step_size):
        """Main loop of Q-learning algorithm.

        Args:
            env: gym object.
            gamma: discount factor - determines how much to value future actions
            episodes: number of episodes to play out
            max_episode_length: maximum number of steps for episode roll out
            epsilon: control how often you explore random actions versus focusing on
                    high value state and actions
            step_size: learning rate - controls how fast or slow to update Q-values
                    for each iteration.

        Returns:
            Q-function which is used to derive policy.
        """
        Q = defaultdict(lambda: np.zeros(env.action_space.n))                       # line 2
        total_reward = 0
        for episode in range(episodes):                                             # slightly different to line 3, we just run until maximum episodes played out
            D = SimpleDrivingEnv.simulate(env, Q, max_episode_length, epsilon, episodes, episode)    # line 4
            for data in D:                                                          # data = [state, action, reward, next_state]  (line 5)
                # print(data)
                ####################### update Q value (line 6) #########################
                Q[data[0]][data[1]] = (1 - step_size) * Q[data[0]][data[1]] + step_size * (data[2] +  gamma * np.max(Q[data[3]]))  # line 6
                #########################################################################
                total_reward += data[2]
                # input()
            if episode % 100 == 0:
                print("average total reward per episode batch since episode ", episode, ": ", total_reward/ float(100))
                total_reward = 0
        return Q  # line 9