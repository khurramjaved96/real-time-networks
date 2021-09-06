import numpy as np


class MountainCarFixed:
    """Implements the fixed mountaincar agent"""

    def predict(self, observation):
        # https://github.com/ZhiqingXiao/OpenAIGymSolution/blob/master/MountainCar-v0_close_form/mountaincar_v0_close_form.ipynb
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2  # push right
        else:
            action = 0  # push left
        return action
