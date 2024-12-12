import numpy as np
from Controllers.abstract_controller import AbstractController

class PIDController(AbstractController):
    """
    Represents a PID controller object, written in such a way that it can be readily adapted to any target system.

    """
    def __init__(self, K=[4.0,0.006,0.2], T=15, lb=-0.1, ub=20, dynamic_gains=None, antiwindup=False, parity = 1):

        self.previous_error = 0.0           # Error computed in previous time-step. Used for derivative computation
        self.previous_control = 0.0         # Control signal used in previous time-step. Used for back-calculation (anti-windup)

        self.integral = {"value": 0.0,  # Integrator value, protected to prevent
                         "lower_bound": lb,  # Lower bound for integrator, set to -np.inf for unbounded
                         "upper_bound": ub,  # Upper bound for integrator, set to np.inf for unbounded
                         }

        self.Kp = K[0]
        self.Ki = K[1]
        self.Kd = K[2]
        self.dt = T
        self.parity = parity

        self.enable_gain_scheduling = False
        self.enable_antiwindup = antiwindup

        if dynamic_gains is not None:
            # Enable gain scheduling with provided parameters
            self.setup_gain_scheduling(dynamic_gains=dynamic_gains)

    def setup_gain_scheduling(self, dynamic_gains=None):
        """
        Enable gain scheduling for the PID controller. The P-gain, I-gain and D-gain are linearly interpolated for the
        input values as a function of *operating_points*.

        Parameters
        ------
        :param dynamic_gains: Dictionary, containing elements:
            :param Kp: (1xN) sequence (list/array), defining the P-gain at location of operating points
            :param Ki: (1xN) sequence (list/array), defining the I-gain at location of operating_points
            :param Kd: (1xN) sequence (list/array), defining the D-gain at location of operating_points
            :param operating_points: (1xN) sequence (list/array, defining the operating points for the PID gains
        """

        self.enable_gain_scheduling = True

        if dynamic_gains is None:
            # Enable gain scheduling with default parameters
            self.dynamic_Kp = np.array([180/2, 90/2, 9, 5, 3])
            self.dynamic_Ki = np.array([1/2, 0.5/2, 0.05, 0.04, 0.02])
            self.dynamic_Kd = np.array([6/2, 3/2, 0.3, 0.2, 0.2])
            self.operating_points = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
 

        else:
            # Enable gain scheduling with input parameters
            self.dynamic_Kp = dynamic_gains["Kp"]
            self.dynamic_Ki = dynamic_gains["Ki"]
            self.dynamic_Kd = dynamic_gains["Kd"]
            self.operating_points = dynamic_gains["operating_points"]

    def step(self, measurement, target, prev_control_applied = None, scheduling_variable=None):
        """
        Computes the PID control output given a measurement and reference value.

        Parameters
        ------
        :param measurement: Measurement value on which the PID controller should act
        :param target: Reference value to control toward

        Notes
        -----
        .. math:: e = target - measurement
        .. math:: u = K_p  e + K_i  \int e dt + K_d \frac{de}{dt}s
        """
        # Get the current gains for the PID controller
        Kp, Ki, Kd, Kb = self.get_gains(scheduling_variable)

        # Compute difference between target measurement and current measurement
        e = self.parity * (target - measurement)

        # Update integrator
        if prev_control_applied is None or not self.enable_antiwindup:
            # Do not use back-calculation
            self.integral["value"] += e * self.dt
        else:
            # Use back-calculation
            self.integral["value"] += (e + Kb * (prev_control_applied - self.previous_control)) * self.dt

        # Clip lower and upper bounds of integrator
        self.integral["value"] = np.clip(self.integral["value"], self.integral["lower_bound"], self.integral["upper_bound"])

        # Compute derivative

        derivative = (e - self.previous_error) / self.dt
        self.previous_error = e

        # Return PID control output
        return Kp * e + Ki * self.integral["value"] + Kd * derivative

    def get_gains(self, scheduling_variable=None):
        """
        Return a tuple of (Kp, Ki, Kd), based on the constants defined as parameters of the PID controller class, or if
        a scheduling variable is provided, use it for gain scheduling. Note: gain scheduled values are not related to
        the constant parameters self.Kp, self.Ki or self.Kd
        """

        if scheduling_variable is None or not self.enable_gain_scheduling:
            # If no scheduling variable is provided or gain scheduling is disabled, use constant parameters
            Kb = self.Ki / self.Kp
            return self.Kp, self.Ki, self.Kd, Kb
        else:
            # Gain schedule based on scheduling variable
            Kp = np.interp(scheduling_variable, self.operating_points, self.dynamic_Kp)
            Ki = np.interp(scheduling_variable, self.operating_points, self.dynamic_Ki)
            Kd = np.interp(scheduling_variable, self.operating_points, self.dynamic_Kd)
            print('Kp: ',Kp)
            return Kp, Ki, Kd, Ki / Kp
