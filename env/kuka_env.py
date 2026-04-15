import pybullet as p
import pybullet_data
import numpy as np
import time


class KukaEnv:
    def __init__(self, render=False):
        self.render = render
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.time_step = 1. / 240.
        p.setTimeStep(self.time_step)

        self.max_steps = 300
        self.step_counter = 0

        self.num_joints = 7
        self.action_scale = 0.2
        self.difficulty = 0  # 0: easy, 1: medium, 2: hard

        self.target_visual = None  # NEW

        self.reset()

    # ---------------- RESET ----------------
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Load plane + robot
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(
            "kuka_iiwa/model.urdf",
            useFixedBase=True
        )

        # Reset joints
        for i in range(self.num_joints):
            p.resetJointState(self.robot, i, 0)

        self.step_counter = 0

        # Sample target
        self.target_pos = self._sample_target()

        # -------- CREATE TARGET VISUAL --------
        self._create_target_visual()

        return self._get_state()

    # ---------------- TARGET VISUAL ----------------
    def _create_target_visual(self):
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[1, 0, 0, 1]  # red
        )

        self.target_visual = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_pos
        )

    # ---------------- TARGET ----------------
    def _sample_target(self):

        if self.difficulty == 0:  # EASY
            x_range = (0.45, 0.5)
            y_range = (0.4, 0.4)
            z_range = (0.5, 0.5)

        elif self.difficulty == 1:  # MEDIUM
            x_range = (0.4, 0.5)
            y_range = (0.3, 0.4)
            z_range = (0.4, 0.45)

        else:  # HARD
            x_range = (0.3, 0.5)
            y_range = (0.3, 0.4)
            z_range = (0.4, 0.5)

        return np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            np.random.uniform(*z_range)
        ])

    # ---------------- STATE ----------------
    def _get_state(self):
        joint_states = p.getJointStates(self.robot, range(self.num_joints))

        joint_angles = np.array([s[0] for s in joint_states])
        joint_vels = np.array([s[1] for s in joint_states])

        ee_pos = np.array(
            p.getLinkState(self.robot, self.num_joints - 1)[0]
        )

        state = np.concatenate([
            joint_angles,
            joint_vels,
            ee_pos,
            self.target_pos
        ])

        return state

    # ---------------- STEP ----------------
    def step(self, action):
        action = np.clip(action, -1, 1)

        # Apply action
        for i in range(self.num_joints):
            current_angle = p.getJointState(self.robot, i)[0]
            target_angle = current_angle + action[i] * self.action_scale

            p.setJointMotorControl2(
                self.robot,
                i,
                p.POSITION_CONTROL,
                targetPosition=target_angle,
                force=200
            )

        # Step simulation
        p.stepSimulation()
        if self.render:
            time.sleep(self.time_step)

        self.step_counter += 1

        # Get state
        state = self._get_state()

        ee_pos = state[14:17]
        distance = np.linalg.norm(ee_pos - self.target_pos)

        # -------- DEBUG LINE (TCP → TARGET) --------
        if self.render:
            p.addUserDebugLine(
                lineFromXYZ=ee_pos,
                lineToXYZ=self.target_pos,
                lineColorRGB=[0, 1, 0],
                lifeTime=0.1
            )

        # Reward
        reward = -distance * 2.0

        # Done condition
        done = False
        if distance < 0.07:
            reward += 10.0
            done = True

        if self.step_counter >= self.max_steps:
            done = True

        return state, reward, done, {"distance": distance}

    # ---------------- CURRICULUM ----------------
    def update_difficulty(self, success_rate):
        if self.difficulty == 0 and success_rate > 0.5:
            self.difficulty = 1
            print("⬆️ Difficulty increased to MEDIUM")

        elif self.difficulty == 1 and success_rate > 0.6:
            self.difficulty = 2
            print("⬆️ Difficulty increased to HARD")

    # ---------------- CLOSE ----------------
    def close(self):
        p.disconnect(self.client)