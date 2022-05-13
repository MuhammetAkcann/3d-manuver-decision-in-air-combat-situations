import random


class UAV:
    """docstring for uav"""

    def __init__(self):
        self.position = [random.randint(0, 30), random.randint(0, 30), random.randint(0, 30)]
        self.roll = random.randint(-23, 23)
        self.yaw = random.randint(0, 360)
        self.pitch = random.randint(-23, 23)
        self.speed = random.randint(4, 20)
        self.max_roll = 23  # degrees
        self.max_pitch = 23  # degrees
        self.max_speed = 20  # m/s
        self.min_speed = 4  # m/s
        self.der_roll = 45  # degrees/s
        self.der_pitch = 20  # degrees/s
        self.der_speed = 5  # m/s^2
        self.der_x = self.der_y = self.der_z = 0  # m/s^2
