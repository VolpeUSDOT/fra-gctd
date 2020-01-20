import numpy as np
import os
from utils.io import IO

path = os.path


class Feature:
  def __init__(self, feature_id, class_id, class_name,
               start_clip_number, end_clip_number, event_id=None):
    """Create a new 'Feature' object.
    
    Args:
      feature_id: The position of the feature in the source video relative to
        other features
      event_id: The id of the event to which this feature was assigned
      start_clip_number: The number of the first frame in which the feature
        occurred
      end_clip_number: The number of the last frame in which the feature
        occurred
    """
    self.feature_id = feature_id
    self.class_id = class_id
    self.class_name = class_name
    self.start_clip_number = start_clip_number
    self.end_clip_number = end_clip_number
    self.event_id = event_id

    # the number of consecutive frames over which the feature occurs
    # if self.end_clip_number and self.start_clip_number:
    self.length = self.end_clip_number - self.start_clip_number
    # else:
    #   self.length = None

  def __str__(self):
    print_string = '\tfeature_id: ' + str(self.feature_id) + '\n'
    print_string += '\tevent_id: ' + str(self.event_id) + '\n'
    print_string += '\tclass_id: ' + str(self.class_id) + '\n'
    print_string += '\tclass_name: ' + str(self.class_name) + '\n'
    print_string += '\tstart_clip_number: ' + str(self.start_clip_number) + \
                    '\n'
    print_string += '\tend_clip_number: ' + str(self.end_clip_number) + '\n'
    print_string += '\tlength: ' + str(self.length)

    return print_string


class ActivationEvent:
  def __init__(self, event_id, target_feature_list,
               preceding_feature=None, following_feature=None):
    """Create a new 'Event' object.
    
    Args:
      event_id: int. The position of the event in the source video relative to
        other events.
      target_feature_list: Feature List. A list of features the event of
      interest could contain.
      preceding_feature: Feature. An auxiliary feature strictly different in
        type from the target feature that should be included in the event if it
        occurs just before the target feature in the source video.
      following_feature: Feature. An auxiliary feature strictly different in
        type from the target feature that should be included in the event if it
        occurs just after the target feature in the source video.
    """
    self.event_id = event_id

    self.target_feature_list = target_feature_list

    for target_feature in self.target_feature_list:
      target_feature.event_id = self.event_id

    self.start_clip_number = self.target_feature_list[0].start_clip_number
    self.end_clip_number = self.target_feature_list[-1].end_clip_number

    if preceding_feature:
      self.start_clip_number = preceding_feature.start_clip_number

    if following_feature:
      self.end_clip_number = following_feature.end_clip_number

    self.length = self.end_clip_number - self.start_clip_number

    self.length = self.end_clip_number - self.start_clip_number

    self._preceding_feature = preceding_feature
    self._following_feature = following_feature

    self.contains_nw_veh_warning_type_1 = None
    self.contains_nw_veh_warning_type_2 = None
    self.contains_nw_veh_warning_type_3 = None
    self.contains_nw_veh_warning_type_4 = None
    self.contains_se_veh_warning_type_1 = None
    self.contains_se_veh_warning_type_2 = None
    self.contains_se_veh_warning_type_3 = None
    self.contains_se_veh_warning_type_4 = None

    self.contains_north_ped_warning_type_1 = None
    self.contains_north_ped_warning_type_2 = None
    self.contains_north_ped_warning_type_3 = None
    self.contains_north_ped_warning_type_4 = None
    self.contains_south_ped_warning_type_1 = None
    self.contains_south_ped_warning_type_2 = None
    self.contains_south_ped_warning_type_3 = None
    self.contains_south_ped_warning_type_4 = None

    self.contains_ped_arnd_se_ped_gate = None
    self.contains_ped_arnd_ne_ped_gate = None
    self.contains_ped_arnd_ne_veh_gate = None
    self.contains_ped_arnd_sw_ped_gate = None
    self.contains_ped_arnd_sw_veh_gate = None
    self.contains_ped_arnd_nw_ped_gate = None
    self.contains_ped_over_se_ped_gate = None
    self.contains_ped_over_ne_ped_gate = None
    self.contains_ped_over_ne_veh_gate = None
    self.contains_ped_over_sw_ped_gate = None
    self.contains_ped_over_sw_veh_gate = None
    self.contains_ped_over_nw_ped_gate = None
    self.contains_ped_undr_se_ped_gate = None
    self.contains_ped_undr_ne_ped_gate = None
    self.contains_ped_undr_ne_veh_gate = None
    self.contains_ped_undr_sw_ped_gate = None
    self.contains_ped_undr_sw_veh_gate = None
    self.contains_ped_undr_nw_ped_gate = None

    self.train_is_present = None

  def find_violations(self, classifications):
    # find vehicle violations
    violation_state = np.any(classifications[:, [34, 40, 43, 49]], axis=1)

    # Vehicle traversed a crossing while lights were flashing but before gates 
    # started descending
    gate_state = classifications[:, 1]
    self.contains_nw_veh_warning_type_1 = np.any(
      np.logical_and(gate_state, violation_state))

    # Vehicle traversed a crossing while gates were descending
    gate_state = classifications[:, 4]
    self.contains_nw_veh_warning_type_2 = np.any(
      np.logical_and(gate_state, violation_state))
    
    # Vehicle traversed a crossing while gates were fully horizontal
    gate_state = classifications[:, 2]
    self.contains_nw_veh_warning_type_3 = np.any(
      np.logical_and(gate_state, violation_state))
    
    # Vehicle traversed a crossing while gates were ascending
    gate_state = classifications[:, 3]
    self.contains_nw_veh_warning_type_4 = np.any(
      np.logical_and(gate_state, violation_state))

    violation_state = np.any(classifications[:, [31, 37, 46, 52]], axis=1)

    # Vehicle traversed a crossing while lights were flashing but before gates
    # started descending
    gate_state = classifications[:, 1]
    self.contains_se_veh_warning_type_1 = np.any(
      np.logical_and(gate_state, violation_state))
    
    # Vehicle traversed a crossing while gates were descending
    gate_state = classifications[:, 4]
    self.contains_se_veh_warning_type_2 = np.any(
      np.logical_and(gate_state, violation_state))
    
    # Vehicle traversed a crossing while gates were fully horizontal
    gate_state = classifications[:, 2]
    self.contains_se_veh_warning_type_3 = np.any(
      np.logical_and(gate_state, violation_state))
    
    # Vehicle traversed a crossing while gates were ascending
    gate_state = classifications[:, 3]
    self.contains_se_veh_warning_type_4 = np.any(
      np.logical_and(gate_state, violation_state))

    # find pedestrian violations
    violation_state = np.any(
      classifications[:, [69, 79, 80, 83, 85, 86, 89, 91, 92, 95]], axis=1)
    
    # Pedestrian traversed a crossing while lights were flashing but before
    # gates started descending
    gate_state = classifications[:, 1]
    self.contains_north_ped_warning_type_1 = np.any(
      np.logical_and(gate_state, violation_state))
    
    # Pedestrian traversed a crossing while gates were descending
    gate_state = classifications[:, 4]
    self.contains_north_ped_warning_type_2 = np.any(
      np.logical_and(gate_state, violation_state))
    
    # Pedestrian traversed a crossing while gates were fully horizontal
    gate_state = classifications[:, 2]
    self.contains_north_ped_warning_type_3 = np.any(
      np.logical_and(gate_state, violation_state))
    
    # Pedestrian traversed a crossing while gates were ascending
    gate_state = classifications[:, 3]
    self.contains_north_ped_warning_type_4 = np.any(
      np.logical_and(gate_state, violation_state))

    violation_state = np.any(
      classifications[:, [68, 78, 81, 82, 84, 87, 88, 90, 93, 94]], axis=1)

    # Pedestrian traversed a crossing while lights were flashing but before gates started descending
    gate_state = classifications[:, 1]
    self.contains_south_ped_warning_type_1 = np.any(
      np.logical_and(gate_state, violation_state))
    
    # Pedestrian traversed a crossing while gates were descending
    gate_state = classifications[:, 4]
    self.contains_south_ped_warning_type_2 = np.any(
      np.logical_and(gate_state, violation_state))
    
    # Pedestrian traversed a crossing while gates were fully horizontal
    gate_state = classifications[:, 2]
    self.contains_south_ped_warning_type_3 = np.any(
      np.logical_and(gate_state, violation_state))
    
    # Pedestrian traversed a crossing while gates were ascending
    gate_state = classifications[:, 3]
    self.contains_south_ped_warning_type_4 = np.any(
      np.logical_and(gate_state, violation_state))

    self.contains_ped_arnd_se_ped_gate = np.any(classifications[:, 78])
    self.contains_ped_arnd_ne_ped_gate = np.any(classifications[:, 79])
    self.contains_ped_arnd_ne_veh_gate = np.any(classifications[:, 80])
    self.contains_ped_arnd_sw_ped_gate = np.any(classifications[:, 81])
    self.contains_ped_arnd_sw_veh_gate = np.any(classifications[:, 82])
    self.contains_ped_arnd_nw_ped_gate = np.any(classifications[:, 83])
    self.contains_ped_over_se_ped_gate = np.any(classifications[:, 84])
    self.contains_ped_over_ne_ped_gate = np.any(classifications[:, 85])
    self.contains_ped_over_ne_veh_gate = np.any(classifications[:, 86])
    self.contains_ped_over_sw_ped_gate = np.any(classifications[:, 87])
    self.contains_ped_over_sw_veh_gate = np.any(classifications[:, 88])
    self.contains_ped_over_nw_ped_gate = np.any(classifications[:, 89])
    self.contains_ped_undr_se_ped_gate = np.any(classifications[:, 90])
    self.contains_ped_undr_ne_ped_gate = np.any(classifications[:, 91])
    self.contains_ped_undr_ne_veh_gate = np.any(classifications[:, 92])
    self.contains_ped_undr_sw_ped_gate = np.any(classifications[:, 93])
    self.contains_ped_undr_sw_veh_gate = np.any(classifications[:, 94])
    self.contains_ped_undr_nw_ped_gate = np.any(classifications[:, 95])

    self.train_is_present = np.any(
      classifications[:, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]])

  @property
  def preceding_feature(self):
    return self._preceding_feature
  
  @preceding_feature.setter
  def preceding_feature(self, preceding_feature):
    self._preceding_feature = preceding_feature
  
  @property  
  def following_feature(self):
    return self._following_feature
  
  @following_feature.setter
  def following_feature(self, following_feature):
    self._following_feature = following_feature
    # if this event's following feature is being reassigned to a later event,
    # the 'following_feature' argument will be None
    if self.following_feature:
      self.end_clip_number = self.following_feature.end_clip_number
    else:
      self.end_clip_number = self.target_feature_list[-1].end_clip_number

  def __str__(self):
    print_string = 'SHRP2 NDS Video Event\n\n'

    if self.preceding_feature:
      print_string += 'Preceding Feature:\n{}\n\n'.format(
        self.preceding_feature)

    print_string += 'Target Features:\n{}\n\n'.format(self.target_feature_list)

    if self.following_feature:
      print_string += 'Following Feature:\n{}\n\n'.format(
        self.following_feature)

    return print_string


class StoppedOnCrossingIncursionEvent:
  def __init__(self, event_id, target_feature_list,
               preceding_feature=None, following_feature=None):
    """Create a new 'Event' object.

    Args:
      event_id: int. The position of the event in the source video relative to
        other events.
      target_feature_list: Feature List. A list of features the event of
      interest could contain.
      preceding_feature: Feature. An auxiliary feature strictly different in
        type from the target feature that should be included in the event if it
        occurs just before the target feature in the source video.
      following_feature: Feature. An auxiliary feature strictly different in
        type from the target feature that should be included in the event if it
        occurs just after the target feature in the source video.
    """
    self.event_id = event_id

    self.target_feature_list = target_feature_list

    for target_feature in self.target_feature_list:
      target_feature.event_id = self.event_id

    self.start_clip_number = self.target_feature_list[0].start_clip_number
    self.end_clip_number = self.target_feature_list[-1].end_clip_number

    self.length = self.end_clip_number - self.start_clip_number

    self._preceding_feature = preceding_feature
    self._following_feature = following_feature

    # self.contains_stopped_on_crossing_violation = None

    self.contains_veh_std_on_se_crsg = None
    self.contains_veh_std_on_ne_crsg = None
    self.contains_veh_std_on_sw_crsg = None
    self.contains_veh_std_on_nw_crsg = None

    self.train_is_present = None

  def find_violations(self, classifications):
    # find vehicle violations
    self.contains_veh_std_on_se_crsg = np.any(classifications[:, 55])
    self.contains_veh_std_on_ne_crsg = np.any(classifications[:, 58])
    self.contains_veh_std_on_sw_crsg = np.any(classifications[:, 61])
    self.contains_veh_std_on_nw_crsg = np.any(classifications[:, 64])

    self.train_is_present = np.any(
      classifications[:,
      [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]])

  @property
  def preceding_feature(self):
    return self._preceding_feature

  @preceding_feature.setter
  def preceding_feature(self, preceding_feature):
    self._preceding_feature = preceding_feature
    # self.start_timestamp = self.preceding_feature.start_timestamp
    # self.start_clip_number = self.preceding_feature.start_clip_number

  @property
  def following_feature(self):
    return self._following_feature

  @following_feature.setter
  def following_feature(self, following_feature):
    self._following_feature = following_feature
    # if this event's following feature is being reassigned to a later event,
    # the 'following_feature' argument will be None
    if self.following_feature:
      self.end_timestamp = self.following_feature.end_timestamp
      self.end_clip_number = self.following_feature.end_clip_number
    else:
      self.end_timestamp = self.target_feature_list[-1].end_timestamp
      self.end_clip_number = self.target_feature_list[-1].end_clip_number

  def __str__(self):
    print_string = 'SHRP2 NDS Video Event\n\n'

    if self.preceding_feature:
      print_string += 'Preceding Feature:\n{}\n\n'.format(
        self.preceding_feature)

    print_string += 'Target Features:\n{}\n\n'.format(self.target_feature_list)

    if self.following_feature:
      print_string += 'Following Feature:\n{}\n\n'.format(
        self.following_feature)

    return print_string


class VehicleRightOfWayIncursionEvent:
  def __init__(self, event_id, target_feature_list,
               preceding_feature=None, following_feature=None):
    """Create a new 'Event' object.

    Args:
      event_id: int. The position of the event in the source video relative to
        other events.
      target_feature_list: Feature List. A list of features the event of
      interest could contain.
      preceding_feature: Feature. An auxiliary feature strictly different in
        type from the target feature that should be included in the event if it
        occurs just before the target feature in the source video.
      following_feature: Feature. An auxiliary feature strictly different in
        type from the target feature that should be included in the event if it
        occurs just after the target feature in the source video.
    """
    self.event_id = event_id

    self.target_feature_list = target_feature_list

    for target_feature in self.target_feature_list:
      target_feature.event_id = self.event_id

    self.start_clip_number = self.target_feature_list[0].start_clip_number
    self.end_clip_number = self.target_feature_list[-1].end_clip_number

    self.length = self.end_clip_number - self.start_clip_number

    self._preceding_feature = preceding_feature
    self._following_feature = following_feature

    # self.contains_stopped_on_crossing_violation = None

    self.contains_veh_adv_on_se_corr = None
    self.contains_veh_adv_on_ne_corr = None
    self.contains_veh_adv_on_sw_corr = None
    self.contains_veh_adv_on_nw_corr = None
    self.contains_veh_rec_on_se_corr = None
    self.contains_veh_rec_on_ne_corr = None
    self.contains_veh_rec_on_sw_corr = None
    self.contains_veh_rec_on_nw_corr = None
    self.contains_veh_std_on_se_corr = None
    self.contains_veh_std_on_ne_corr = None
    self.contains_veh_std_on_sw_corr = None
    self.contains_veh_std_on_nw_corr = None

    self.train_is_present = None

  def find_violations(self, classifications):
    # find vehicle violations
    self.contains_veh_adv_on_se_corr = np.any(classifications[:, 30])
    self.contains_veh_adv_on_ne_corr = np.any(classifications[:, 33])
    self.contains_veh_adv_on_sw_corr = np.any(classifications[:, 36])
    self.contains_veh_adv_on_nw_corr = np.any(classifications[:, 39])
    self.contains_veh_rec_on_se_corr = np.any(classifications[:, 42])
    self.contains_veh_rec_on_ne_corr = np.any(classifications[:, 45])
    self.contains_veh_rec_on_sw_corr = np.any(classifications[:, 48])
    self.contains_veh_rec_on_nw_corr = np.any(classifications[:, 51])
    self.contains_veh_std_on_se_corr = np.any(classifications[:, 54])
    self.contains_veh_std_on_ne_corr = np.any(classifications[:, 57])
    self.contains_veh_std_on_sw_corr = np.any(classifications[:, 60])
    self.contains_veh_std_on_nw_corr = np.any(classifications[:, 63])

    self.train_is_present = np.any(
      classifications[:,
      [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]])

  @property
  def preceding_feature(self):
    return self._preceding_feature

  @preceding_feature.setter
  def preceding_feature(self, preceding_feature):
    self._preceding_feature = preceding_feature
    self.start_clip_number = self.preceding_feature.start_clip_number

  @property
  def following_feature(self):
    return self._following_feature

  @following_feature.setter
  def following_feature(self, following_feature):
    self._following_feature = following_feature
    # if this event's following feature is being reassigned to a later event,
    # the 'following_feature' argument will be None
    if self.following_feature:
      self.end_clip_number = self.following_feature.end_clip_number
    else:
      self.end_clip_number = self.target_feature_list[-1].end_clip_number

  def __str__(self):
    print_string = 'SHRP2 NDS Video Event\n\n'

    if self.preceding_feature:
      print_string += 'Preceding Feature:\n{}\n\n'.format(
        self.preceding_feature)

    print_string += 'Target Features:\n{}\n\n'.format(self.target_feature_list)

    if self.following_feature:
      print_string += 'Following Feature:\n{}\n\n'.format(
        self.following_feature)

    return print_string


class PedestrianRightOfWayIncursionEvent:
  def __init__(self, event_id, target_feature_list,
               preceding_feature=None, following_feature=None):
    """Create a new 'Event' object.

    Args:
      event_id: int. The position of the event in the source video relative to
        other events.
      target_feature_list: Feature List. A list of features the event of
      interest could contain.
      preceding_feature: Feature. An auxiliary feature strictly different in
        type from the target feature that should be included in the event if it
        occurs just before the target feature in the source video.
      following_feature: Feature. An auxiliary feature strictly different in
        type from the target feature that should be included in the event if it
        occurs just after the target feature in the source video.
    """
    self.event_id = event_id

    self.target_feature_list = target_feature_list

    for target_feature in self.target_feature_list:
      target_feature.event_id = self.event_id

    self.start_clip_number = self.target_feature_list[0].start_clip_number
    self.end_clip_number = self.target_feature_list[-1].end_clip_number

    self.length = self.end_clip_number - self.start_clip_number

    self._preceding_feature = preceding_feature
    self._following_feature = following_feature

    # self.contains_stopped_on_crossing_violation = None

    self.contains_ped_on_sth_corr = None
    self.contains_ped_on_nth_corr = None

    self.train_is_present = None

  def find_violations(self, classifications):
    # find vehicle violations
    self.contains_ped_on_sth_corr = np.any(classifications[:, 66])
    self.contains_ped_on_sth_corr = np.any(classifications[:, 67])

    self.train_is_present = np.any(
      classifications[:,
      [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]])

  @property
  def preceding_feature(self):
    return self._preceding_feature

  @preceding_feature.setter
  def preceding_feature(self, preceding_feature):
    self._preceding_feature = preceding_feature
    self.start_clip_number = self.preceding_feature.start_clip_number

  @property
  def following_feature(self):
    return self._following_feature

  @following_feature.setter
  def following_feature(self, following_feature):
    self._following_feature = following_feature
    # if this event's following feature is being reassigned to a later event,
    # the 'following_feature' argument will be None
    if self.following_feature:
      self.end_clip_number = self.following_feature.end_clip_number
    else:
      self.end_clip_number = self.target_feature_list[-1].end_clip_number

  def __str__(self):
    print_string = 'SHRP2 NDS Video Event\n\n'

    if self.preceding_feature:
      print_string += 'Preceding Feature:\n{}\n\n'.format(
        self.preceding_feature)

    print_string += 'Target Features:\n{}\n\n'.format(self.target_feature_list)

    if self.following_feature:
      print_string += 'Following Feature:\n{}\n\n'.format(
        self.following_feature)

    return print_string


class Trip:
  def __init__(self, report_clip_numbers,
               report_probs, class_name_map, non_event_weight_scale=0.05,
               minimum_event_length=1):
    self.class_names = class_name_map
    self.class_ids = {value: key for key, value in self.class_names.items()}

    report_class_ids = np.apply_along_axis(
      func1d=self.feature_fn, axis=1, arr=report_probs)

    self.feature_sequence = []

    feature_id = 0
    class_id = report_class_ids[0]

    start_clip_number = report_clip_numbers[0]

    for i in range(1, len(report_class_ids)):
      if report_class_ids[i] != class_id:
        end_clip_number = report_clip_numbers[i - 1]

        # the beginning of the next feature has been reached.
        # create an object for the preceding feature.
        self.feature_sequence.append(Feature(
          feature_id, class_id, self.class_names[class_id],
          start_clip_number, end_clip_number))

        feature_id += 1

        class_id = report_class_ids[i]

        start_clip_number = report_clip_numbers[i]

        if i == len(report_class_ids) - 1:
          self.feature_sequence.append(Feature(
            feature_id, class_id, self.class_names[class_id],
            start_clip_number, start_clip_number))
      elif i == len(report_class_ids) - 1:
        end_clip_number = report_clip_numbers[i]

        self.feature_sequence.append(Feature(
          feature_id, class_id, self.class_names[class_id],
          start_clip_number, end_clip_number))

    self.weight_scale = non_event_weight_scale
    self.minimum_event_length = minimum_event_length

  def feature_fn(self, clip_probs):
    gates_ascending = self.class_ids['gates_ascending']
    gates_descending = self.class_ids['gates_descending']
    if clip_probs[gates_ascending] >= .25 \
        and clip_probs[gates_ascending] > clip_probs[gates_descending]:
      return gates_ascending
    if clip_probs[gates_descending] >= .25 \
        and clip_probs[gates_descending] > clip_probs[gates_ascending]:
      return gates_descending
    gates_down = self.class_ids['gates_down']
    gates_up = self.class_ids['gates_up']
    if clip_probs[gates_descending] < .25 \
        and clip_probs[gates_ascending] < .25 \
        and clip_probs[gates_down] >= .6 \
        and clip_probs[gates_up] < clip_probs[gates_down] / 3:
      return gates_down
    else:
      return gates_up

  def get_stopped_on_crossing_incursion_feature_sequence(
      self, report_clip_numbers, smooth_probs,
      smoothing_factor):
    feature_sequence = []
    
    # extract features for use in finding gate stopped_on_crossing_incursions
    incursion_states = self.report_probs[:, [55, 58, 61, 64]]

    if smooth_probs:
      incursion_states = IO.smooth_probs(
        incursion_states, smoothing_factor)

    incursion_states = np.round(incursion_states).astype(np.uint8)

    feature_id = 0

    current_state = np.any(incursion_states[0])

    start_clip_number = report_clip_numbers[0]

    for i in range(1, len(incursion_states)):
      ith_gate_light_state = np.any(incursion_states[i])
      if ith_gate_light_state != current_state:
        end_clip_number = report_clip_numbers[i - 1]

        # the beginning of the next feature has been reached.
        # create an object for the preceding feature.
        feature_sequence.append(Feature(
          feature_id, current_state,
          start_clip_number, end_clip_number))

        feature_id += 1

        current_state = np.any(incursion_states[i])

        start_clip_number = report_clip_numbers[i]

        if i == len(incursion_states) - 1:
          print('current_state: {}'.format(current_state))

          feature_sequence.append(Feature(
            feature_id, current_state,
            start_clip_number, start_clip_number))
      elif i == len(incursion_states) - 1:
        end_clip_number = report_clip_numbers[i]

        print('current_state: {}'.format(current_state))

        feature_sequence.append(Feature(
          feature_id, current_state,
          start_clip_number, end_clip_number))

    return feature_sequence

  def find_stopped_on_crossing_incursion_events(self):
    events = []

    event_id = 0

    i = 0

    weight = 0.0

    while i < len(self.stopped_on_crossing_incursion_feature_sequence):
      current_feature = self.stopped_on_crossing_incursion_feature_sequence[i]
      i += 1
      if current_feature.state:
        target_feature_list = [current_feature]
        longest_target_feature_gap = 0
        weight += current_feature.length

        while i < len(self.stopped_on_crossing_incursion_feature_sequence):
          current_feature = self.stopped_on_crossing_incursion_feature_sequence[i]
          i += 1
          if current_feature.state:
            current_feature_gap = current_feature.start_clip_number - \
                    target_feature_list[-1].end_clip_number
            if longest_target_feature_gap < current_feature_gap:
              longest_target_feature_gap = current_feature_gap

            target_feature_list.append(current_feature)
            weight += current_feature.length
          else:
            weight -= self.weight_scale * current_feature.length

          if weight <= 0:
            break

        current_event = StoppedOnCrossingIncursionEvent(event_id=event_id,
                              target_feature_list=target_feature_list)

        weight = 0

        if current_event.length >= self.minimum_event_length:
          events.append(current_event)
          event_id += 1

    for event in events:
      print('start_clip_number', event.start_clip_number)
      print('end_clip_number', event.end_clip_number)
      classifications = np.round(
        self.report_probs[event.start_clip_number - 1:event.end_clip_number]
      ).astype(np.uint8)
      print('classifications', classifications)

      event.find_violations(classifications)

    return events
  
  def get_ped_right_of_way_incursion_feature_sequence(
      self, report_clip_numbers, smooth_probs,
      smoothing_factor):
    feature_sequence = []
    
    # extract features for use in finding gate ped_right_of_way_incursions
    incursion_states = self.report_probs[:, [66, 67]]

    if smooth_probs:
      incursion_states = IO.smooth_probs(
        incursion_states, smoothing_factor)

    incursion_states = np.round(incursion_states).astype(np.uint8)

    feature_id = 0

    current_state = np.any(incursion_states[0])

    start_clip_number = report_clip_numbers[0]

    for i in range(1, len(incursion_states)):
      ith_gate_light_state = np.any(incursion_states[i])
      if ith_gate_light_state != current_state:
        end_clip_number = report_clip_numbers[i - 1]

        print('current_state: {}'.format(current_state))

        # the beginning of the next feature has been reached.
        # create an object for the preceding feature.
        feature_sequence.append(Feature(
          feature_id, current_state,
          start_clip_number, end_clip_number))

        feature_id += 1

        current_state = np.any(incursion_states[i])

        start_clip_number = report_clip_numbers[i]

        if i == len(incursion_states) - 1:
          print('current_state: {}'.format(current_state))

          feature_sequence.append(Feature(
            feature_id, current_state,
            start_clip_number, start_clip_number))
      elif i == len(incursion_states) - 1:
        end_clip_number = report_clip_numbers[i]

        print('current_state: {}'.format(current_state))

        feature_sequence.append(Feature(
          feature_id, current_state,
          start_clip_number, end_clip_number))

    return feature_sequence

  def find_ped_right_of_way_incursion_events(self):
    events = []

    event_id = 0

    i = 0

    weight = 0.0

    while i < len(self.ped_right_of_way_incursion_feature_sequence):
      current_feature = self.ped_right_of_way_incursion_feature_sequence[i]
      i += 1
      if current_feature.state:
        target_feature_list = [current_feature]
        longest_target_feature_gap = 0
        weight += current_feature.length

        while i < len(self.ped_right_of_way_incursion_feature_sequence):
          current_feature = self.ped_right_of_way_incursion_feature_sequence[i]
          i += 1
          if current_feature.state:
            current_feature_gap = current_feature.start_clip_number - \
                    target_feature_list[-1].end_clip_number
            if longest_target_feature_gap < current_feature_gap:
              longest_target_feature_gap = current_feature_gap

            target_feature_list.append(current_feature)
            weight += current_feature.length
          else:
            weight -= self.weight_scale * current_feature.length

          if weight <= 0:
            break

        current_event = PedestrianRightOfWayIncursionEvent(event_id=event_id,
                              target_feature_list=target_feature_list)

        weight = 0

        if current_event.length >= self.minimum_event_length:
          events.append(current_event)
          event_id += 1

    for event in events:
      print('start_clip_number', event.start_clip_number)
      print('end_clip_number', event.end_clip_number)
      classifications = np.round(
        self.report_probs[event.start_clip_number - 1:event.end_clip_number]
      ).astype(np.uint8)
      print('classifications', classifications)

      event.find_violations(classifications)

    return events
  
  def get_veh_right_of_way_incursion_feature_sequence(
      self, report_clip_numbers, smooth_probs,
      smoothing_factor):
    feature_sequence = []
    
    # extract features for use in finding gate veh_right_of_way_incursions
    incursion_states = self.report_probs[:, [30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63]]

    if smooth_probs:
      incursion_states = IO.smooth_probs(
        incursion_states, smoothing_factor)

    incursion_states = np.round(incursion_states).astype(np.uint8)

    feature_id = 0

    current_state = np.any(incursion_states[0])

    start_clip_number = report_clip_numbers[0]

    for i in range(1, len(incursion_states)):
      ith_gate_light_state = np.any(incursion_states[i])
      if ith_gate_light_state != current_state:
        end_clip_number = report_clip_numbers[i - 1]

        print('current_state: {}'.format(current_state))

        # the beginning of the next feature has been reached.
        # create an object for the preceding feature.
        feature_sequence.append(Feature(
          feature_id, current_state,
          start_clip_number, end_clip_number))

        feature_id += 1

        current_state = np.any(incursion_states[i])

        start_clip_number = report_clip_numbers[i]

        if i == len(incursion_states) - 1:
          feature_sequence.append(Feature(
            feature_id, current_state,
            start_clip_number, start_clip_number))
      elif i == len(incursion_states) - 1:
        end_clip_number = report_clip_numbers[i]

        feature_sequence.append(Feature(
          feature_id, current_state,
          start_clip_number, end_clip_number))
        
    return feature_sequence

  def find_veh_right_of_way_incursion_events(self):
    events = []

    event_id = 0

    i = 0

    weight = 0.0

    while i < len(self.veh_right_of_way_incursion_feature_sequence):
      current_feature = self.veh_right_of_way_incursion_feature_sequence[i]
      i += 1
      if current_feature.state:
        target_feature_list = [current_feature]
        longest_target_feature_gap = 0
        weight += current_feature.length

        while i < len(self.veh_right_of_way_incursion_feature_sequence):
          current_feature = self.veh_right_of_way_incursion_feature_sequence[i]
          i += 1
          if current_feature.state:
            current_feature_gap = current_feature.start_clip_number - \
                    target_feature_list[-1].end_clip_number
            if longest_target_feature_gap < current_feature_gap:
              longest_target_feature_gap = current_feature_gap

            target_feature_list.append(current_feature)
            weight += current_feature.length
          else:
            weight -= self.weight_scale * current_feature.length

          if weight <= 0:
            break

        current_event = VehicleRightOfWayIncursionEvent(event_id=event_id,
                              target_feature_list=target_feature_list)

        weight = 0

        if current_event.length >= self.minimum_event_length:
          events.append(current_event)
          event_id += 1

    for event in events:
      classifications = np.round(
        self.report_probs[event.start_clip_number - 1:event.end_clip_number]
      ).astype(np.uint8)

      event.find_violations(classifications)

    return events


  def find_events(
      self, target_feature_class_ids, target_feature_class_names=None,
      preceding_feature_class_id=None, preceding_feature_class_name=None,
      following_feature_class_id=None, following_feature_class_name=None):
    if target_feature_class_ids is None:
      if target_feature_class_names is None:
        raise ValueError('target_feature_class_ids and target_'
                         'feature_class_names cannot both be None')
      else:
        target_feature_class_ids = [self.class_ids[name]
                                    for name in target_feature_class_names]

    if preceding_feature_class_id is None and \
            preceding_feature_class_name is not None:
      preceding_feature_class_id = self.class_ids[preceding_feature_class_name]

    if preceding_feature_class_id in target_feature_class_ids:
      raise ValueError('preceding_feature_class_id cannot be equal to any '
                       'target_feature_class_id')

    if following_feature_class_id is None and \
            following_feature_class_name is not None:
      following_feature_class_id = self.class_ids[following_feature_class_name]

    if following_feature_class_id in target_feature_class_ids:
      raise ValueError('following_feature_class_id cannot be equal to any '
                       'target_feature_class_id')

    events = []

    previous_event = None

    event_id = 0

    i = 0

    weight = 0.0

    if preceding_feature_class_id and following_feature_class_id:
      previous_preceding_feature = None
      previous_following_feature = None

      while i < len(self.feature_sequence):
        current_feature = self.feature_sequence[i]
        i += 1
        if current_feature.class_id in target_feature_class_ids:
          target_feature_list = [current_feature]
          longest_target_feature_gap = 0
          weight += current_feature.length

          while i < len(self.feature_sequence) and current_feature.class_id \
              not in [preceding_feature_class_id, following_feature_class_id]:
            current_feature = self.feature_sequence[i]
            i += 1
            if current_feature.class_id in target_feature_class_ids:
              current_feature_gap = current_feature.start_clip_number - \
                      target_feature_list[-1].end_clip_number
              if longest_target_feature_gap < current_feature_gap:
                longest_target_feature_gap = current_feature_gap

              target_feature_list.append(current_feature)
              weight += current_feature.length
            else:
              weight -= self.weight_scale * current_feature.length

            if weight <= 0:
              break

          # if a detected event begins or ends in a frame from which the timestamp
          # could not be read or syntehsized, just ignore the event.
          # if target_feature_list[0].start_timestamp != -1 \
          #     or target_feature_list[-1].end_timestamp != -1:
          current_event = ActivationEvent(
            event_id=event_id, target_feature_list=target_feature_list,
            preceding_feature=previous_preceding_feature)

          if current_event.length >= self.minimum_event_length:
            weight = 0

            # if two consecutive events share a common following/preceding
            # feature, and that feature is closer to the current event than the
            # previous event, reassign it to the current event.
            if previous_preceding_feature:
              if current_event.start_clip_number - \
                  previous_preceding_feature.end_clip_number < \
                  longest_target_feature_gap * 10:
                if previous_preceding_feature.event_id:
                  previous_target_feature = events[
                    previous_preceding_feature.event_id].target_feature_list[-1]

                  previous_target_feature_distance = \
                    previous_preceding_feature.start_clip_number - \
                    previous_target_feature.end_clip_number

                  assert previous_target_feature_distance >= 0

                  current_feature_distance = \
                    current_event.target_feature_list[0].start_clip_number - \
                    previous_preceding_feature.end_clip_number

                  assert current_feature_distance >= 0

                  if current_feature_distance < previous_target_feature_distance:
                    previous_event.following_feature = None
                    current_event.preceding_feature = previous_preceding_feature
                    previous_preceding_feature.event_id = event_id
                else:
                  current_event.preceding_feature = previous_preceding_feature
                  previous_preceding_feature.event_id = event_id

                if previous_preceding_feature == previous_following_feature:
                  previous_following_feature = None

                previous_preceding_feature = None

            events.append(current_event)
            event_id += 1
            previous_event = current_event

        if current_feature.class_id == preceding_feature_class_id:
          previous_preceding_feature = current_feature

        if current_feature.class_id == following_feature_class_id:
          previous_following_feature = current_feature

          if previous_event and \
                  previous_event.following_feature is None:
            previous_event.following_feature = previous_following_feature
            previous_following_feature.event_id = previous_event.event_id
            previous_following_feature = None
    elif not preceding_feature_class_id and following_feature_class_id:
      while i < len(self.feature_sequence):
        current_feature = self.feature_sequence[i]
        i += 1
        if current_feature.class_id in target_feature_class_ids:
          target_feature_list = [current_feature]

          while i < len(self.feature_sequence) \
              and current_feature.class_id != following_feature_class_id:
            current_feature = self.feature_sequence[i]
            i += 1
            if current_feature.class_id in target_feature_class_ids:
              target_feature_list.append(current_feature)

          current_event = ActivationEvent(
              event_id=event_id, target_feature_list=target_feature_list)

          if current_event.length >= self.minimum_event_length:
            events.append(current_event)
            event_id += 1
            previous_event = current_event

        if current_feature.class_id == following_feature_class_id:
          previous_following_feature = current_feature

          if previous_event and \
                  previous_event.following_feature is None:
            previous_event.following_feature = previous_following_feature
            previous_following_feature.event_id = previous_event.event_id
    elif preceding_feature_class_id and not following_feature_class_id:
      previous_preceding_feature = None

      while i < len(self.feature_sequence):
        current_feature = self.feature_sequence[i]
        i += 1

        if current_feature.class_id in target_feature_class_ids:
          target_feature_list = [current_feature]

          while i < len(self.feature_sequence) \
              and current_feature.class_id != preceding_feature_class_id:
            current_feature = self.feature_sequence[i]
            i += 1

            if current_feature.class_id in target_feature_class_ids:
              target_feature_list.append(current_feature)

          current_event = ActivationEvent(
              event_id=event_id, target_feature_list=target_feature_list)

          # if two consecutive events share a common following/preceding
          # feature, and that feature is closer to the current event than the
          # previous event, reassign it to the current event.
          if previous_preceding_feature:
            current_event.preceding_feature = previous_preceding_feature
            previous_preceding_feature.event_id = event_id
            previous_preceding_feature = None

          if current_event.length >= self.minimum_event_length:
            events.append(current_event)
            event_id += 1

        if current_feature.class_id == preceding_feature_class_id:
          previous_preceding_feature = current_feature
    else:
      while i < len(self.feature_sequence):
        current_feature = self.feature_sequence[i]
        i += 1
        if current_feature.class_id in target_feature_class_ids:
          target_feature_list = [current_feature]
          longest_target_feature_gap = 0
          weight += current_feature.length

          while i < len(self.feature_sequence):
            current_feature = self.feature_sequence[i]
            i += 1
            if current_feature.class_id in target_feature_class_ids:
              current_feature_gap = current_feature.start_clip_number - \
                      target_feature_list[-1].end_clip_number
              if longest_target_feature_gap < current_feature_gap:
                longest_target_feature_gap = current_feature_gap

              target_feature_list.append(current_feature)
              weight += current_feature.length
            else:
              weight -= self.weight_scale * current_feature.length

            if weight <= 0:
              break

          current_event = ActivationEvent(
              event_id=event_id, target_feature_list=target_feature_list)

          weight = 0

          if current_event.length >= self.minimum_event_length:
            events.append(current_event)
            event_id += 1

    return events

  def find_activation_events(self):
    return self.find_events(
      target_feature_class_ids=[self.class_ids['gates_down']],
      target_feature_class_names=['gates_down'],
      preceding_feature_class_id=self.class_ids['gates_descending'],
      preceding_feature_class_name='gates_descending',
      following_feature_class_id=self.class_ids['gates_ascending'],
      following_feature_class_name='gates_ascending'
    )

class TripFromReportFile(Trip):
  def __init__(self, report_file_path, class_names_file_path,
               smooth_probs=False, smoothing_factor=16):
    class_name_map = IO.read_class_names(class_names_file_path)

    class_header_names = [class_name + '_probability'
                          for class_name in class_name_map.values()]

    header_mask = ['clip_number', 'start_time']
    header_mask.extend(class_header_names)

    report_header, report_data, data_col_range = IO.read_report(
      report_file_path, clip_col_num=1, start_time_col_num=2,
      header_mask=header_mask, return_data_col_range=True)

    report_clip_numbers = report_data['clip_numbers']
    report_clip_numbers = report_clip_numbers.astype(np.int32)

    report_probs = report_data['probabilities']
    report_probs = report_probs.astype(np.float32)

    if smooth_probs:
      report_probs = IO.smooth_probs(report_probs, smoothing_factor)

    Trip.__init__(self, report_clip_numbers, report_probs, class_name_map)
