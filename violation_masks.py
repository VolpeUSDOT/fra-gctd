import numpy as np

# Activation starts when the lights begin to flash and ends when the gates finish their ascent to a vertical position and the lights stop flashing
def _does_contain_activation(classifications):
  gates_are_descending = classifications[4]
  gates_are_down = classifications[2]
  gates_are_ascending = classifications[3]
  gate_lights_are_flashing = classifications[5]
  return gate_lights_are_flashing and (
    gates_are_descending or gates_are_down or gates_are_ascending)

# COMPUTE FUNCTIONS BELOW ONLY DURING ACTIVATION

# Vehicle traversed a crossing while lights were flashing but before gates started descending
def _does_contain_northwest_vehicle_warning_violation_type_1(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_up = classifications[1]
  vehicle_is_traveling_northwest_on_crossing = np.any(
    classifications[[34, 40, 43, 49]])
  return gate_lights_are_flashing and gates_are_up \
         and vehicle_is_traveling_northwest_on_crossing

# Vehicle traversed a crossing while gates were descending
def _does_contain_northwest_vehicle_warning_violation_type_2(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_descending = classifications[4]
  vehicle_is_on_crossing = np.any(
    classifications[[34, 40, 43, 49]])
  return gate_lights_are_flashing and gates_are_descending and vehicle_is_on_crossing

# Vehicle traversed a crossing while gates were fully horizontal
def _does_contain_northwest_vehicle_warning_violation_type_3(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_down = classifications[2]
  vehicle_is_on_crossing = np.any(
    classifications[[34, 40, 43, 49]])
  return gate_lights_are_flashing and gates_are_down and vehicle_is_on_crossing

# Vehicle traversed a crossing while gates were ascending
def _does_contain_northwest_vehicle_warning_violation_type_4(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_ascending = classifications[3]
  vehicle_is_on_crossing = np.any(
    classifications[[34, 40, 43, 49]])
  return gate_lights_are_flashing and gates_are_ascending \
         and vehicle_is_on_crossing

# Vehicle traversed a crossing while lights were flashing but before gates started descending
def _does_contain_southeast_vehicle_warning_violation_type_1(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_up = classifications[1]
  vehicle_is_on_crossing = np.any(
    classifications[[31, 37, 43, 49]])
  return gate_lights_are_flashing and gates_are_up and vehicle_is_on_crossing

# Vehicle traversed a crossing while gates were descending
def _does_contain_southeast_vehicle_warning_violation_type_2(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_descending = classifications[4]
  vehicle_is_on_crossing = np.any(
    classifications[[31, 37, 43, 49]])
  return gate_lights_are_flashing and gates_are_descending \
         and vehicle_is_on_crossing

# Vehicle traversed a crossing while gates were fully horizontal
def _does_contain_southeast_vehicle_warning_violation_type_3(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_down = classifications[2]
  vehicle_is_on_crossing = np.any(
    classifications[[31, 37, 43, 49]])
  return gate_lights_are_flashing and gates_are_down and vehicle_is_on_crossing

# Vehicle traversed a crossing while gates were ascending
def _does_contain_southeast_vehicle_warning_violation_type_4(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_ascending = classifications[3]
  vehicle_is_on_crossing = np.any(
    classifications[[31, 37, 43, 49]])
  return gate_lights_are_flashing and gates_are_ascending \
         and vehicle_is_on_crossing

# Pedestrian traversed a crossing while lights were flashing but before gates
# started descending
def _does_contain_north_pedestrian_warning_violation_type_1(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_up = classifications[1]
  pedestrian_is_on_north_crossing = np.any(
    classifications[[69, 79, 80, 83, 85, 86, 89, 91, 92, 95]])
  return gate_lights_are_flashing and gates_are_up \
         and pedestrian_is_on_north_crossing

# Pedestrian traversed a crossing while gates were descending
def _does_contain_north_pedestrian_warning_violation_type_2(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_descending = classifications[4]
  pedestrian_is_on_north_crossing = np.any(
    classifications[[69, 79, 80, 83, 85, 86, 89, 91, 92, 95]])
  return gate_lights_are_flashing and gates_are_descending \
         and pedestrian_is_on_north_crossing

# Pedestrian traversed a crossing while gates were fully horizontal
def _does_contain_north_pedestrian_warning_violation_type_3(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_down = classifications[2]
  pedestrian_is_on_north_crossing = np.any(
    classifications[[69, 79, 80, 83, 85, 86, 89, 91, 92, 95]])
  return gate_lights_are_flashing and gates_are_down \
         and pedestrian_is_on_north_crossing

# Pedestrian traversed a crossing while gates were ascending
def _does_contain_north_pedestrian_warning_violation_type_4(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_ascending = classifications[3]
  pedestrian_is_on_crossing = np.any(
    classifications[[69, 79, 80, 83, 85, 86, 89, 91, 92, 95]])
  return gate_lights_are_flashing and gates_are_ascending \
         and pedestrian_is_on_crossing

# Pedestrian traversed a crossing while lights were flashing but before gates
# started descending
def _does_contain_south_pedestrian_warning_violation_type_1(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_up = classifications[1]
  pedestrian_is_on_south_crossing = np.any(
    classifications[[68, 78, 81, 82, 84, 87, 88, 90, 93, 94]])
  return gate_lights_are_flashing and gates_are_up and pedestrian_is_on_south_crossing

# Pedestrian traversed a crossing while gates were descending
def _does_contain_south_pedestrian_warning_violation_type_2(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_descending = classifications[4]
  pedestrian_is_on_south_crossing = np.any(
    classifications[[68, 78, 81, 82, 84, 87, 88, 90, 93, 94]])
  return gate_lights_are_flashing and gates_are_descending \
         and pedestrian_is_on_south_crossing

# Pedestrian traversed a crossing while gates were fully horizontal
def _does_contain_south_pedestrian_warning_violation_type_3(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_down = classifications[2]
  pedestrian_is_on_south_crossing = np.any(
    classifications[[68, 78, 81, 82, 84, 87, 88, 90, 93, 94]])
  return gate_lights_are_flashing and gates_are_down and pedestrian_is_on_south_crossing

# Pedestrian traversed a crossing while gates were ascending
def _does_contain_south_pedestrian_warning_violation_type_4(classifications):
  gate_lights_are_flashing = classifications[5]
  gates_are_ascending = classifications[3]
  pedestrian_is_on_south_crossing = np.any(
    classifications[[68, 78, 81, 82, 84, 87, 88, 90, 93, 94]])
  return gate_lights_are_flashing and gates_are_ascending and pedestrian_is_on_south_crossing

# COMPUTE FUNCTIONS BELOW REGARDLESS OF ACTIVATION STATE

# A vehicle is stopped in the dynamic envelope zone regardless of traffic or activation states.
def _does_contain_stopped_on_crossing_violation(classifications):
  vehicle_is_stopped_on_crossing = np.any(
    classifications[[55, 58, 61, 64]])
  return vehicle_is_stopped_on_crossing

# A vehicle is stopped in the dynamic envelope zone regardless of traffic or activation states.
def _does_contain_vehicle_right_of_way_incursion_violation(classifications):
  vehicle_is_stopped_on_crossing = np.any(
    classifications[[30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63]])
  return vehicle_is_stopped_on_crossing

# A vehicle is stopped in the dynamic envelope zone regardless of traffic or activation states.
def _does_contain_pedestrian_right_of_way_incursion_violation(classifications):
  vehicle_is_stopped_on_crossing = np.any(
    classifications[[66, 67]])
  return vehicle_is_stopped_on_crossing


def _train_is_present(classifications):
  train_is_present = np.any(
    classifications[[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                     22, 23, 24, 25, 26, 27, 28, 29, 30, 31]])
  return train_is_present
