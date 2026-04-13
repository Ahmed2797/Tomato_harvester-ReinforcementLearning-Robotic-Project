# ros_nodes/moveit_control.py
# import moveit_commander

# class RobotArm:

#     def __init__(self):
#         moveit_commander.roscpp_initialize([])
#         self.arm = moveit_commander.MoveGroupCommander("manipulator")

#     def go_to(self, pos):
#         pose = self.arm.get_current_pose().pose

#         pose.position.x = pos[0]
#         pose.position.y = pos[1]
#         pose.position.z = pos[2]

#         self.arm.set_pose_target(pose)
#         self.arm.go()


class RobotArm:

    def __init__(self):
        pass  # ROS MoveIt connection here

    def move_to(self, x, y, z):
        print(f"Moving robot to {x}, {y}, {z}")

    def pick(self):
        print("Gripper close")

    def release(self):
        print("Gripper open")