import numpy as np
import airsim
import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
import logging
import math
logger = logging.getLogger()

logger.setLevel(logging.INFO)

class AirSimDroneEnv(AirSimEnv):




    def __init__(self, target_z=-20):
        super().__init__()
        self.drone = airsim.MultirotorClient(ip="127.0.0.1",port=41453)
        self.drone.confirmConnection()
        self._setup_flight()

        self.start_position = [0, 0, 4]
        self.goal_position = [80, 5, 4]
        self.goal_distance = None

        self.goal_x= self.goal_position[0]
        self.goal_y= self.goal_position[1]

        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0

        self.yaw = 0
        self.vxy_speed = 0
        self.yaw_speed = 0


        self.v_xy_max = 5
        self.v_xy_min = -1
        self.yaw_rate_max_deg = 1
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.max_vertical_difference = 5


        self.action_space = spaces.Box(low=np.array([self.v_xy_min, -self.yaw_rate_max_rad]),
                                           high=np.array([self.v_xy_max, self.yaw_rate_max_rad]),
                                           dtype=np.float32)
        self.crash_distance = 1
        self.goal_radius=2


        self.previous_dist = 0
        self.max_depth = 10
        self.dt=0.1
        self.max_episode_steps = 1000

        self.depth_image_size=5
        self.state_feature_length = 2
        self.lidar_feature_length = 3
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(1, self.depth_image_size + self.state_feature_length + self.lidar_feature_length),
            dtype=np.float32
        )
        self.lidar_flag=False


    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)


    def set_action(self, action):

        self.vxy_speed = action[0] * 0.7
        self.yaw_speed = action[-1] * 2

        self.yaw = self.get_attitude()[2]
        self.yaw_sp = self.yaw + self.yaw_speed * self.dt

        if self.yaw_sp > math.radians(180):
            self.yaw_sp -= math.pi * 2
        elif self.yaw_sp < math.radians(-180):
            self.yaw_sp += math.pi * 2

        vx_local_sp = self.vxy_speed * math.cos(self.yaw_sp)
        vy_local_sp = self.vxy_speed * math.sin(self.yaw_sp)
        self.drone.simPrintLogMessage("Yaw Rate",str(self.yaw_speed))
        self.drone.simPrintLogMessage("Velocity X",str((vx_local_sp,vy_local_sp)))
        self.drone.simPrintLogMessage("Goal",str((self.goal_position,self.get_distance_to_goal())))

        self.drone.moveByVelocityZAsync(vx_local_sp, vy_local_sp, -self.start_position[2], self.dt,
                                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_speed))).join()

    def get_attitude(self):
        self.state_current_attitude = self.drone.simGetVehiclePose().orientation
        return airsim.to_eularian_angles(self.state_current_attitude)


    def get_lidar_data(self):
        lidar_data = self.drone.getLidarData()

        if len(lidar_data.point_cloud) > 2:
            pts=np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
            pts=np.reshape(pts, (int(pts.shape[0]/3), 3))
            return pts
        return np.zeros(3)

    def filter_directional_points(self, points):
        front, left, right = [], [], []
        for point in points:
            ang = np.rad2deg(np.arctan2(point[1], point[0]))
            if -30 <= ang < 30:
                front.append(point)
            elif -90 <= ang < -30:
                left.append(point)
            elif 30 <= ang < 90:
                right.append(point)

        return np.array(front), np.array(left), np.array(right)

    def obstacles_check(self,segmented_points):
        obstacle_directions = {}

        for key in segmented_points:
            if len(segmented_points[key]) == 0:
                obstacle_directions[key]=10
                continue
            distances=np.sqrt(np.sum(segmented_points[key][:, :2]**2, axis=1))
            obstacle_directions[key] = np.mean(distances)
        return obstacle_directions

    def get_depth_image(self):

        responses = self.drone.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
        ])

        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.drone.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
        ])

        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float, responses[0].width,
            responses[0].height)

        depth_meter = depth_img * 100

        return depth_meter

    def _get_obs(self):

        image = self.get_depth_image()
        self.min_distance_to_obstacles = image.min()
        # print(image.min())

        image_scaled = np.clip(image, 0, self.max_depth) / self.max_depth * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        image_obs = image_uint8
        split_row = 1
        split_col = 5

        v_split_list = np.vsplit(image_obs, split_row)

        split_final = []
        for i in range(split_row):
            h_split_list = np.hsplit(v_split_list[i], split_col)
            for j in range(split_col):
                split_final.append(h_split_list[j].max())

        img_feature = np.array(split_final) / 255.0

        state_feature = self._get_state_feature() / 255
        # print("State Feature Shape",state_feature.shape)


        lidar_points = self.get_lidar_data()
        front, left, right = self.filter_directional_points(lidar_points)
        obstacles = self.obstacles_check({'front': front, 'left': left, 'right': right})

        # print("LIDARR DATA WITHOUT NORM",obstacles)

        obstacles=np.array(list(obstacles.values()))
        if self.min_distance_to_obstacles>np.min(obstacles):
            # print("MIN LIDAR DISTANCE")
            self.lidar_flag=True
            self.min_distance_to_obstacles = np.min(obstacles)

        obstacles = (np.clip(obstacles, 0, 10)*255 / 10)/255
        # print("LIDARR DATA WITH NORM",obstacles)

        feature_all = np.concatenate((img_feature, state_feature, obstacles), axis=0)
        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all


    def _get_state_feature(self):

        dist = self.get_distance_to_goal()
        relative_yaw = self._get_relative_yaw()
        relative_pose_z = self.get_position()[2] - self.goal_position[2]
        vertical_dist_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255
        dist_norm = dist / self.goal_distance * 255

        velocity = self.get_velocity()
        linear_velocity_xy = velocity[0]
        linear_velocity_z = velocity[1]
        self.state_raw = np.array([dist, relative_pose_z,  math.degrees(
            relative_yaw), linear_velocity_xy, linear_velocity_z,  math.degrees(velocity[2])])

        state_norm = np.array([dist_norm, vertical_dist_norm])
        state_norm = np.clip(state_norm, 0, 255)

        self.state_norm = state_norm

        return state_norm

    def get_velocity(self):
        states = self.drone.getMultirotorState()
        lin_velocity = states.kinematics_estimated.linear_velocity
        ang_velocity = states.kinematics_estimated.angular_velocity

        velocity_xy = math.sqrt(pow(lin_velocity.x_val, 2) + pow(lin_velocity.y_val, 2))
        velocity_z = lin_velocity.z_val
        yaw_rate = ang_velocity.z_val

        return [velocity_xy, -velocity_z, yaw_rate]


    def _get_relative_yaw(self):

        current_position = self.get_position()

        relative_pose_x = self.goal_position[0] - current_position[0]
        relative_pose_y = self.goal_position[1] - current_position[1]
        ang = math.atan2(relative_pose_y, relative_pose_x)


        current_yaw = self.get_attitude()[2]

        yaw_error = ang - current_yaw
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error

    def step(self, action):
        self.drone.simPrintLogMessage("Position:",str(self.drone.simGetVehiclePose()))
        self.set_action(action)
        observation=self._get_obs()
        done=self.is_done()
        reward= self.cal_reward(done)

        info = {
            'is_success': self.is_near_goal(),
            'is_crash': self.is_crashed(),
            'step_num': self.step_num,
            "Goal Distance":self.get_distance_to_goal(),
            "Position": self.get_position(),
            "Lidar Flag":self.lidar_flag,
        }
        if done:
            print(info)
        self.step_num += 1
        self.total_step += 1

        return observation, reward, done, info


    def is_done(self):
        episode_done = False

        has_reached_des_pose = self.is_near_goal()
        too_close_to_obstable = self.is_crashed()

        episode_done = has_reached_des_pose or\
            too_close_to_obstable or\
            self.step_num >= self.max_episode_steps

        # print("Crashed", too_close_to_obstable)
        # print("Episode Done",episode_done)
        return episode_done

    def is_near_goal(self):
        in_desired_pose = False
        if self.get_distance_to_goal() < self.goal_radius:
            in_desired_pose = True

        return in_desired_pose

    def is_crashed(self):
        crashed_flag = False
        collision_info = self.drone.simGetCollisionInfo()
        if collision_info.has_collided or self.min_distance_to_obstacles < self.crash_distance:

            crashed_flag = True

        return crashed_flag

    def get_distance_to_goal(self):
        current_pose = self.get_position()
        goal_pose = self.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2))

    def get_position(self):
        position = self.drone.simGetVehiclePose().position
        return [position.x_val, position.y_val, -position.z_val]


    def cal_reward(self, done):
        reward = 0
        reward_reach = 10
        reward_crash = -20


        distance_reward_coef = 50

        if not done:
            # goal dist reward
            distance_now = self.get_distance_to_goal()
            reward_distance = distance_reward_coef * (self.previous_dist - distance_now) / \
                self.goal_distance
            self.previous_dist = distance_now
            #Position punishment
            current_pose = self.get_position()
            goal_pose = self.goal_position
            x = current_pose[0]
            y = current_pose[1]
            x_g = goal_pose[0]
            y_g = goal_pose[1]

            punishment_xy = np.clip(self.getDis(
                x, y, 0, 0, x_g, y_g) / 10, 0, 1)


            punishment_pose = punishment_xy

            if self.min_distance_to_obstacles < 10:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 5, 0, 1)
            else:
                punishment_obs = 0

            punishment_action = 0


            yaw_error = self.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)

            reward = reward_distance - 0.1 * punishment_pose - 0.2 * \
                punishment_obs - 0.1 * punishment_action - 0.5 * yaw_error_cost
        else:
            if self.is_near_goal():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash


        return reward




    def reset(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)


        pose=self.drone.simGetVehiclePose()
        pose.position.x_val = self.start_position[0]
        pose.position.y_val = self.start_position[1]
        pose.position.z_val = - self.start_position[2]

        self.drone.simSetVehiclePose(pose, True)

        self.drone.takeoffAsync().join()
        self.drone.moveToZAsync(-self.start_position[2], 2).join()

        self.episode_num += 1
        self.step_num = 0
        self.goal_distance = self.get_distance_to_goal()
        self.previous_dist = self.goal_distance
        obs = self._get_obs()

        return obs


    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):

        a = lineY2-lineY1
        b = lineX1-lineX2
        c = lineX2*lineY1-lineX1*lineY2
        dis = (math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b, 0.5))

        return dis

    def render(self, mode='human'):
        pass

    def close(self):
        self.drone.enableApiControl(False)
