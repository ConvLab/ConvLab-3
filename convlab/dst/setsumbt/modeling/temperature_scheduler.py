# -*- coding: utf-8 -*-
# Copyright 2022 DSML Group, Heinrich Heine University, Düsseldorf
# Authors: Carel van Niekerk (niekerk@hhu.de)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Linear Temperature Scheduler Class"""


# Temp scheduler class for ensemble distillation
class LinearTemperatureScheduler:
    """
    Temperature scheduler object used for distribution temperature scheduling in distillation

    Attributes:
        state (dict): Internal state of scheduler
    """
    def __init__(self,
                 total_steps: int,
                 base_temp: float = 2.5,
                 cycle_len: float = 0.1):
        """
        Args:
            total_steps (int): Total number of training steps
            base_temp (float): Starting temperature
            cycle_len (float): Fraction of total steps used for scheduling cycle
        """
        self.state = dict()
        self.state['total_steps'] = total_steps
        self.state['current_step'] = 0
        self.state['base_temp'] = base_temp
        self.state['current_temp'] = base_temp
        self.state['cycles'] = [int(total_steps * cycle_len / 2), int(total_steps * cycle_len)]
        self.state['rate'] = (self.state['base_temp'] - 1.0) / (self.state['cycles'][1] - self.state['cycles'][0])
    
    def step(self):
        """
        Update temperature based on the schedule
        """
        self.state['current_step'] += 1
        assert self.state['current_step'] <= self.state['total_steps']
        if self.state['current_step'] > self.state['cycles'][0]:
            if self.state['current_step'] < self.state['cycles'][1]:
                self.state['current_temp'] -= self.state['rate']
            else:
                self.state['current_temp'] = 1.0
    
    def temp(self):
        """
        Get current temperature

        Returns:
            temp (float): Current temperature for distribution scaling
        """
        return float(self.state['current_temp'])
    
    def state_dict(self):
        """
        Return scheduler state

        Returns:
            state (dict): Dictionary format state of the scheduler
        """
        return self.state
    
    def load_state_dict(self, state_dict: dict):
        """
        Load scheduler state from dictionary

        Args:
            state_dict (dict): Dictionary format state of the scheduler
        """
        self.state = state_dict
