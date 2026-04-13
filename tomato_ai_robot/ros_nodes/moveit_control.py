

class RobotArm:

    def __init__(self):
        pass  # ROS MoveIt connection here

    def move_to(self, x, y, z):
        print(f"Moving robot to {x}, {y}, {z}")

    def pick(self):
        print("Gripper close")

    def release(self):
        print("Gripper open")