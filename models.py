from dataclasses import dataclass
import numpy as np
import gtsam
from typing import Tuple

@dataclass
class VehicleParams:
    """Vehicle parameters from the documentation"""
    L: float = 2.83  # Vehicle wheelbase (meters)
    a: float = 0.95  # Distance from rear axle to GPS/laser
    b: float = 0.5   # Half of rear axle length
    H: float = 0.76  # Height of GPS/laser

class MotionModel:
    """
    Implements the vehicle motion model.
    See documentation Fig 1 and motion model equations.
    """
    def __init__(self, vehicle_params: VehicleParams):
        self.params = vehicle_params
        
    def predict(self, pose: gtsam.Pose2, velocity: float, steering: float, dt: float) -> gtsam.Pose2:
        """
        TODO: Implement discrete motion model:
        x(k+1) = f(x,u)
        
        Hint:
        - Use vehicle parameters (L, a, b)
        - Remember to transform velocity from wheel to center
        - Account for GPS/laser position in calculations
        """
        x = pose.x()
        y = pose.y()
        theta = pose.theta()

        L = self.params.L

        # From motion model in paper
        dx = velocity*np.cos(theta)*dt
        dy = velocity*np.sin(theta)*dt
        dtheta = velocity/L*np.tan(steering)

        return pose.compose(gtsam.Pose2(dx, dy, dtheta))

    
    def get_noise_model(self, velocity: float, steering: float) -> gtsam.noiseModel.Diagonal:
        """
        TODO: Create appropriate noise model for motion
        
        Hint:
        - Noise should increase with velocity and steering angle
        - Consider effects of wheel slip at higher speeds/steering
        """
        # Base noise levels
        base_xy = 0.05  # 5cm base noise
        base_theta = 0.02  # ~1 degree base noise
        
        # Scale noise with velocity and steering
        sigma_x = base_xy + 0.1 * abs(velocity)
        sigma_y = base_xy + 0.1 * abs(velocity)
        sigma_theta = base_theta + 0.1 * abs(steering) + 0.05 * abs(velocity * np.tan(steering)/self.params.L)
        
        return gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_x, sigma_y, sigma_theta]))


class ObservationModel:
    """
    Implements the laser observation model.
    See documentation Fig 2 and measurement equations.
    """
    def __init__(self, vehicle_params: VehicleParams):
        self.params = vehicle_params
        
    def predict(self, pose: gtsam.Pose2, landmark: np.ndarray) -> Tuple[float, float]:
        """
        TODO: Implement measurement prediction:
        z = h(x)
        
        Hint:
        - Calculate expected range and bearing
        - Remember coordinate transformations
        - Normalize angles properly
        """
        x = pose.x()
        y = pose.y()
        theta = pose.theta()

        a = self.params.a
        b = self.params.b
        L = self.params.L

        x_L = x + a + L
        y_L = y + b

        x_i = landmark[0]
        y_i = landmark[1]

        delta_x = x_i - x_L
        delta_y = y_i - y_L

        range_pred = np.sqrt((delta_x)**2 + (delta_y)**2)
        bearing_pred = theta - a*np.tan(-(delta_y)/(delta_x) + np.pi/2)

        return gtsam.Point2(range_pred, bearing_pred)
    
    def get_noise_model(self, measured_range: float) -> gtsam.noiseModel.Diagonal:
        """
        TODO: Create appropriate noise model for measurements
        
        Hint:
        - Range noise typically increases with distance
        - Bearing noise might also increase with range
        """
        # Range noise increases with distance
        sigma_range = 0.1 + 0.01 * measured_range  # 10cm + 1% of range
        
        # Bearing noise also slightly increases with range
        # (harder to accurately measure bearing to distant objects)
        sigma_bearing = (0.5 + 0.01 * measured_range) * np.pi / 180.0  # Base 0.5 degrees
        
        return gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_bearing, sigma_range]))

    def measurement_likelihood(self, measured: Tuple[float, float], 
                             predicted: Tuple[float, float], 
                             noise_model: gtsam.noiseModel.Diagonal) -> float:
        """
        Compute likelihood of measurement given prediction
        Useful for data association
        
        Args:
            measured: (range, bearing) measurement
            predicted: (range, bearing) prediction
            noise_model: Current noise model
            
        Returns:
            likelihood score
        """
        # Extract measurements and predictions
        r_meas, b_meas = measured
        r_pred, b_pred = predicted
        
        # Compute normalized bearing difference
        bearing_diff = (b_meas - b_pred + np.pi) % (2 * np.pi) - np.pi
        
        # Get noise sigmas
        sigma_bearing, sigma_range = noise_model.sigmas()
        
        # Compute Mahalanobis distance
        range_error = (r_meas - r_pred) / sigma_range
        bearing_error = bearing_diff / sigma_bearing
        
        # Return negative squared error (higher is better)
        return -(range_error**2 + bearing_error**2)