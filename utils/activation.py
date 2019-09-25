import numpy as np

"""
  Violation types are defined by 1) gate state (up, down, ascending, descending), 2) violating entity (vehicle, pedestrian), 3) direction of travel (southeast, northwest). Violations only occur when the gate lights are flashing.
  
  Given a matrix of probabilities and a set of tests, we want to know if any of 
  the tests pass at least once. To implement, we might use numpy apply_along_axis 
"""
class Violation:
  def __init__(self):
    pass

"""
  An activation is initiated when gate lights start flashing and terminated
  when gate lights stop flashing, regardless of gate motion or violation
  occurrence.
"""
class Activation:
  def __init__(self, start_clip_number, end_clip_number, probability_array, frames_per_second=30,
               frames_per_clip=64):
    self.frames_per_second = frames_per_second
    self.frames_per_clip = frames_per_clip * 2  # because we're sub-sampling
    self.start_time = np.round(
      start_clip_number * self.frames_per_clip / self.frames_per_second).astype(
      np.int32)
    self.end_time = np.round(
      end_clip_number * self.frames_per_clip / self.frames_per_second).astype(
      np.int32)
    self.probability_array = probability_array
    self.violation_list = []

    def find_violations():
      pass