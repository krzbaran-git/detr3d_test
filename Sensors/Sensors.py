from abc import ABC, abstractmethod


class Sensor(ABC):
    def __init__(self, token = None, sensor_token = None, channel = None, translation = None, rotation = None):
        self.token = token
        self.sensor_token = sensor_token
        self.channel = channel
        self.translation = translation
        self.rotation = rotation

    def get_token(self):
        return self.token

    @abstractmethod
    def get_calib(self):
        pass


class LidarSensor(Sensor):
    def __init__(self, token = None, sensor_token = None, channel = None, translation = None, rotation = None):
        super().__init__(token, sensor_token, channel, translation, rotation)

    def get_calib(self) -> dict:
        return {'translation': self.translation, 'rotation': self.rotation}

class CameraSensor(Sensor):
    def __init__(self, token = None, sensor_token = None, channel = None, translation = None, rotation = None, camera_intrinsic = None):
        super().__init__(token, sensor_token, channel, translation, rotation)
        self.camera_intrinsic = camera_intrinsic

    def get_calib(self) -> dict:
        return {'translation': self.translation, 'rotation': self.rotation, 'camera_intrinsic': self.camera_intrinsic}

class RadarSensor(Sensor):
    def __init__(self, token = None, sensor_token = None, channel = None, translation = None, rotation = None):
        super().__init__(token, sensor_token, channel, translation, rotation)

    def get_calib(self) -> dict:
        return {'translation': self.translation, 'rotation': self.rotation}

