# -*- coding: utf-8 -*-
# Copyright 2021 DSML Group, Heinrich Heine University, Düsseldorf
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
"""Temperature Scheduler Class"""
import torch

# Temp scheduler class for ensemble distillation
class TemperatureScheduler:

    def __init__(self, total_steps, base_temp=2.5, cycle_len=0.1):
        self.state = {}
        self.state['total_steps'] = total_steps
        self.state['current_step'] = 0
        self.state['base_temp'] = base_temp
        self.state['current_temp'] = base_temp
        self.state['cycles'] = [int(total_steps * cycle_len / 2), int(total_steps * cycle_len)]
    
    def step(self):
        self.state['current_step'] += 1
        assert self.state['current_step'] <= self.state['total_steps']
        if self.state['current_step'] > self.state['cycles'][0]:
            if self.state['current_step'] < self.state['cycles'][1]:
                rate = (self.state['base_temp'] - 1.0) / (self.state['cycles'][1] - self.state['cycles'][0])
                self.state['current_temp'] -= rate
            else:
                self.state['current_temp'] = 1.0
    
    def temp(self):
        return float(self.state['current_temp'])
    
    def state_dict(self):
        return self.state
    
    def load_state_dict(self, sd):
        self.state = sd


# if __name__ == "__main__":
#     temp_scheduler = TemperatureScheduler(100)
#     print(temp_scheduler.state_dict())

#     temp = []
#     for i in range(100):
#         temp.append(temp_scheduler.temp())
#         temp_scheduler.step()
    
#     temp_scheduler.load_state_dict(temp_scheduler.state_dict())
#     print(temp_scheduler.state_dict())

#     print(temp)
