import csv
import json
import logging
import numpy as np
import os
import subprocess as sp

path = os.path


class IO:
  @staticmethod
  def _invoke_subprocess(command):
    completed_subprocess = sp.run(
      command, stdout=sp.PIPE, stderr=sp.PIPE, timeout=60)

    if len(completed_subprocess.stderr) > 0:
      std_err = str(completed_subprocess.stderr, encoding='utf-8')

      raise Exception(std_err)

    return str(completed_subprocess.stdout, encoding='utf-8')

  @staticmethod
  def get_device_ids():
    command = ['nvidia-smi', '--query-gpu=index', '--format=csv']
    output = IO._invoke_subprocess(command)
    line_list = output.rstrip().split('\n')
    return line_list[1:]

  @staticmethod
  def read_class_names(class_names_path):
    meta_map = IO._read_meta_file(class_names_path)
    return {int(key): value for key, value in meta_map.items()}

  @staticmethod
  def read_node_names(io_node_names_path):
    meta_map = IO._read_meta_file(io_node_names_path)
    return {key: value + ':0' for key, value in meta_map.items()}

  @staticmethod
  def read_video_file_names(video_file_dir_path):
    included_extenstions = ['avi', 'mp4', 'asf', 'mkv', 'm4v', 'mpeg', 'mov']
    return sorted([fn for fn in os.listdir(video_file_dir_path) if any(
      fn.lower().endswith(ext) for ext in included_extenstions)])

  @staticmethod
  def _div_odd(n):
    return n // 2, n // 2 + 1

  @staticmethod
  def get_processing_duration(end_time, msg):
    minutes, seconds = divmod(end_time, 60)
    hours, minutes = divmod(minutes, 60)
    hours = int(round(hours))
    minutes = int(round(minutes))
    milliseconds = int(round(end_time * 1000))
    return '{} {:02d}:{:02d}:{:05.2f} ({:d} ms)'.format(
      msg, hours, minutes, seconds, milliseconds)

  @staticmethod
  def _read_meta_file(file_path):
    meta_lines = [line.rstrip().split(':')
                  for line in open(file_path).readlines()]
    return {line[0]: line[1] for line in meta_lines}

  @staticmethod
  def get_video_dimensions(video_file_path, ffprobe_path):
    command = [ffprobe_path, '-show_streams', '-print_format',
               'json', '-loglevel', 'warning', video_file_path]
    output = IO._invoke_subprocess(command)
    try:
      json_map = json.loads(output)
    except Exception as e:
      logging.error('encountered an exception while parsing ffprobe JSON file.')
      logging.debug('received raw ffprobe response: {}'.format(output))
      logging.debug('will raise exception to caller.')
      raise e
    return int(json_map['streams'][0]['width']),\
           int(json_map['streams'][0]['height']),\
           int(json_map['streams'][0]['nb_frames'])

  @staticmethod
  def _get_gauss_weight_and_window(smoothing_factor):
    window = smoothing_factor * 2 - 1
    weight = np.ndarray((window,))
    for i in range(window):
      frac = (i - smoothing_factor + 1) / window
      weight[i] = 1 / np.exp((4 * frac) ** 2)
    return weight, window

  @staticmethod
  def _smooth_class_prob_sequence(
      probs, weight, weight_sum, indices, head_padding_len, tail_padding_len):
    smoothed_probs = weight * probs[indices]
    smoothed_probs = np.sum(smoothed_probs, axis=1)
    smoothed_probs = smoothed_probs / weight_sum
    head_padding = np.ones((head_padding_len, )) * smoothed_probs[0]
    tail_padding = np.ones((tail_padding_len, )) * smoothed_probs[-1]
    smoothed_probs = np.concatenate(
      (head_padding, smoothed_probs, tail_padding))
    return smoothed_probs

  @staticmethod
  def smooth_probs(class_probs, smoothing_factor):
    weight, window = IO._get_gauss_weight_and_window(smoothing_factor)
    weight_sum = np.sum(weight)
    indices = np.arange(class_probs.shape[0] - window)
    indices = np.expand_dims(indices, axis=1) + np.arange(weight.shape[0])
    head_padding_len, tail_padding_len = IO._div_odd(window)
    smoothed_probs = np.ndarray(class_probs.shape)
    for i in range(class_probs.shape[1]):
      smoothed_probs[:, i] = IO._smooth_class_prob_sequence(
        class_probs[:, i], weight, weight_sum, indices,
        head_padding_len, tail_padding_len)
    return smoothed_probs

  @staticmethod
  def _expand_class_names(class_names, appendage):
    return class_names + [class_name + appendage for class_name in class_names]

  @staticmethod
  def _binarize_probs(class_probs):
    # since numpy rounds 0.5 to 0.0, identify occurrences of 0.5 and replace
    # them with 1.0. If a prob has two 0.5s, replace them both with 1.0
    binarized_probs = class_probs.copy()
    uncertain_prob_indices = binarized_probs == 0.5
    binarized_probs[uncertain_prob_indices] = 1.0
    binarized_probs = np.round(binarized_probs)

    return binarized_probs

  @staticmethod
  def open_report(report_file_path):
    report_file = open(report_file_path, newline='')

    return csv.reader(report_file)

  @staticmethod
  def read_report_header(
      report_reader, clip_col_num=None, start_time_col_num=None,
      data_col_range=None, header_mask=None, return_data_col_range=False):
    if data_col_range is None and header_mask is None:
      raise ValueError('data_col_range and header_mask cannot both be None.')

    csv_header = next(report_reader)

    report_header = []
    
    if clip_col_num:
      report_header.append(csv_header[clip_col_num])
    
    if start_time_col_num:
      report_header.append(csv_header[start_time_col_num])
        
    if len(report_header) == 0:
      raise ValueError(
        'clip_col_num and start_time_col_num cannot both be None.')

    if data_col_range is None:
      data_col_indices = [csv_header.index(data_col_name)
                          for data_col_name in header_mask[len(report_header):]]
      data_col_range = (data_col_indices[0], data_col_indices[-1] + 1)

    report_header.extend(csv_header[data_col_range[0]: data_col_range[1]])

    if header_mask and report_header != header_mask:
      raise ValueError(
        'report header: {} was expected to match header mask: {}\ngiven '
        'clip_col_num: {}, start_time_col_num: {} and data_col_range: '
        '{}'.format(report_header, header_mask, clip_col_num, 
                    start_time_col_num, data_col_range))
    
    if return_data_col_range:
      return report_header, data_col_range
    else:
      return report_header

  @staticmethod
  def read_report_data(report_reader, clip_col_num=None,
                       start_time_col_num=None, data_col_range=None):
    if clip_col_num and start_time_col_num and data_col_range:
      clip_numbers = []
      timestamps = []
      probabilities = []

      for row in report_reader:
        clip_numbers.append(row[clip_col_num])
        timestamps.append(row[start_time_col_num])
        probabilities.append(row[data_col_range[0]:data_col_range[1]])

      report_data = {'clip_numbers': np.array(clip_numbers),
                     'clip_timestamps': np.array(timestamps),
                     'probabilities': np.array(probabilities)}
    elif clip_col_num and data_col_range:
      clip_numbers = []
      probabilities = []

      for row in report_reader:
        clip_numbers.append(row[clip_col_num])
        probabilities.append(row[data_col_range[0]:data_col_range[1]])

      report_data = {'clip_numbers': np.array(clip_numbers),
                     'probabilities': np.array(probabilities)}
    elif start_time_col_num and data_col_range:
      timestamps = []
      probabilities = []

      for row in report_reader:
        timestamps.append(row[start_time_col_num])
        probabilities.append(row[data_col_range[0]:data_col_range[1]])

      report_data = {'clip_timestamps': np.array(timestamps),
                     'probabilities': np.array(probabilities)}
    elif data_col_range:
      probabilities = []

      for row in report_reader:
        probabilities.append(row[data_col_range[0]:data_col_range[1]])

      report_data = {'probabilities': np.array(probabilities)}
    else:
      report_data = np.array([row for row in report_reader])

    return report_data

  @staticmethod
  def read_report(report_file_path, clip_col_num=None, start_time_col_num=None,
                  data_col_range=None, header_mask=None,
                  return_data_col_range=False):
    report_reader = IO.open_report(report_file_path)

    if return_data_col_range:
      report_header, data_col_range = IO.read_report_header(
        report_reader, clip_col_num=clip_col_num,
        start_time_col_num=start_time_col_num,
        data_col_range=data_col_range, header_mask=header_mask,
        return_data_col_range=True)
    else:
      report_header = IO.read_report_header(
        report_reader, clip_col_num=clip_col_num,
        start_time_col_num=start_time_col_num,
        data_col_range=data_col_range, header_mask=header_mask,
        return_data_col_range=False)

    report_data = IO.read_report_data(
      report_reader, clip_col_num=clip_col_num,
      start_time_col_num=start_time_col_num,
      data_col_range=data_col_range)

    if return_data_col_range:
      return report_header, report_data, data_col_range
    else:
      return report_header, report_data

  @staticmethod
  def write_csv(file_path, header, rows):
    with open(file_path, 'w', newline='') as file:
      csv_writer = csv.writer(file)
      csv_writer.writerow(header)
      csv_writer.writerows(rows)

  # TODO: confirm that the csv can be opened after writing
  @staticmethod
  def write_inference_report(
      report_file_name, report_dir_path, class_probs, class_name_map,
      clip_length, smooth_probs=False, smoothing_factor=0, binarize_probs=False):
    class_names = ['{}_probability'.format(class_name)
                   for class_name in class_name_map.values()]

    if smooth_probs and smoothing_factor > 1:
      class_names = IO._expand_class_names(class_names, '_smoothed')
      smoothed_probs = IO.smooth_probs(class_probs, smoothing_factor)
      class_probs = np.concatenate((class_probs, smoothed_probs), axis=1)

    if binarize_probs:
      class_names = IO._expand_class_names(class_names, '_binarized')
      binarized_probs = IO._binarize_probs(class_probs)
      class_probs = np.concatenate((class_probs, binarized_probs), axis=1)

    # if timestamp_strings is not None:
    #   header = ['file_name', 'clip_number', 'clip_timestamp', 'qa_flag'] + \
    #            class_names
    #   rows = [[report_file_name, '{:d}'.format(i + 1), timestamp_strings[i],
    #            qa_flags[i]] + ['{0:.4f}'.format(cls) for cls in class_probs[i]]
    #           for i in range(len(class_probs))]
    # else:
    header = ['file_name', 'clip_number', 'start_time'] + class_names

    rows = []

    for i in range(len(class_probs)):
      mins, secs = divmod(i * clip_length * 2 / 30, 60)
      hours, minutes = divmod(mins, 60)

      rows.append(
        [report_file_name, '{:d}'.format(i + 1), '{:02d}:{:02d}:{:02d}'.format(
          round(hours), round(mins), round(secs))]
        + ['{0:.4f}'.format(cls) for cls in class_probs[i]])

    report_dir_path = path.join(report_dir_path, 'inference_reports')

    if not path.exists(report_dir_path):
      os.makedirs(report_dir_path)

    report_file_path = path.join(
      report_dir_path, report_file_name + '.csv')

    IO.write_csv(report_file_path, header, rows)

  # TODO: confirm that the csv can be opened after writing
  @staticmethod
  def write_activation_event_report(
      report_file_name, report_dir_path, events, clip_length):
    report_dir_path = path.join(report_dir_path, 'activation_reports')

    if not path.exists(report_dir_path):
      os.makedirs(report_dir_path)

    report_file_path = path.join(
      report_dir_path, report_file_name + '.csv')

    header = [
      'file_name', 
      'sequence_number', 
      'start_clip_number', 
      'end_clip_number',
      'start_time',
      'end_time',
      'nw_veh_warning_type_1', 
      'nw_veh_warning_type_2', 
      'nw_veh_warning_type_3', 
      'nw_veh_warning_type_4',
      'se_veh_warning_type_1', 
      'se_veh_warning_type_2', 
      'se_veh_warning_type_3',
      'se_veh_warning_type_4', 
      'north_ped_warning_type_1', 
      'north_ped_warning_type_2', 
      'north_ped_warning_type_3', 
      'north_ped_warning_type_4', 
      'south_ped_warning_type_1', 
      'south_ped_warning_type_2', 
      'south_ped_warning_type_3', 
      'south_ped_warning_type_4', 
      'ped_arnd_se_ped_gate',
      'ped_arnd_ne_ped_gate',
      'ped_arnd_ne_veh_gate',
      'ped_arnd_sw_ped_gate',
      'ped_arnd_sw_veh_gate',
      'ped_arnd_nw_ped_gate',
      'ped_over_se_ped_gate',
      'ped_over_ne_ped_gate',
      'ped_over_ne_veh_gate',
      'ped_over_sw_ped_gate',
      'ped_over_sw_veh_gate',
      'ped_over_nw_ped_gate',
      'ped_undr_se_ped_gate',
      'ped_undr_ne_ped_gate',
      'ped_undr_ne_veh_gate',
      'ped_undr_sw_ped_gate',
      'ped_undr_sw_veh_gate',
      'ped_undr_nw_ped_gate', 
      'train_is_present'
    ]

    rows = []

    for event in events:
      mins, secs = divmod((event.start_clip_number * clip_length * 2 - 1) / 30, 60)
      hours, minutes = divmod(mins, 60)
      start_time = '{:02d}:{:02d}:{:02d}'.format(int(hours), int(mins), int(secs))

      mins, secs = divmod((event.end_clip_number * clip_length * 2 - 1) / 30, 60)
      hours, minutes = divmod(mins, 60)
      end_time = '{:02d}:{:02d}:{:02d}'.format(int(hours), int(mins), int(secs))

      rows.append(
        [report_file_name,
         event.event_id + 1,
         event.start_clip_number,
         event.end_clip_number,
         start_time,
         end_time,
         event.contains_nw_veh_warning_type_1,
         event.contains_nw_veh_warning_type_2,
         event.contains_nw_veh_warning_type_3,
         event.contains_nw_veh_warning_type_4,
         event.contains_se_veh_warning_type_1,
         event.contains_se_veh_warning_type_2,
         event.contains_se_veh_warning_type_3,
         event.contains_se_veh_warning_type_4,
         event.contains_north_ped_warning_type_1,
         event.contains_north_ped_warning_type_2,
         event.contains_north_ped_warning_type_3,
         event.contains_north_ped_warning_type_4,
         event.contains_south_ped_warning_type_1,
         event.contains_south_ped_warning_type_2,
         event.contains_south_ped_warning_type_3,
         event.contains_south_ped_warning_type_4,
         event.contains_ped_arnd_se_ped_gate,
         event.contains_ped_arnd_ne_ped_gate,
         event.contains_ped_arnd_ne_veh_gate,
         event.contains_ped_arnd_sw_ped_gate,
         event.contains_ped_arnd_sw_veh_gate,
         event.contains_ped_arnd_nw_ped_gate,
         event.contains_ped_over_se_ped_gate,
         event.contains_ped_over_ne_ped_gate,
         event.contains_ped_over_ne_veh_gate,
         event.contains_ped_over_sw_ped_gate,
         event.contains_ped_over_sw_veh_gate,
         event.contains_ped_over_nw_ped_gate,
         event.contains_ped_undr_se_ped_gate,
         event.contains_ped_undr_ne_ped_gate,
         event.contains_ped_undr_ne_veh_gate,
         event.contains_ped_undr_sw_ped_gate,
         event.contains_ped_undr_sw_veh_gate,
         event.contains_ped_undr_nw_ped_gate,
         event.train_is_present])

    IO.write_csv(report_file_path, header, rows)

    # TODO: confirm that the csv can be opened after writing
  @staticmethod
  def write_stopped_on_crossing_incursion_event_report(
      report_file_name, report_dir_path, events, clip_length):
    report_dir_path = path.join(
      report_dir_path, 'stopped_on_crossing_incursion_reports')

    if not path.exists(report_dir_path):
      os.makedirs(report_dir_path)

    report_file_path = path.join(
      report_dir_path, report_file_name + '.csv')

    header = [
      'file_name', 
      'sequence_number', 
      'start_clip_number', 
      'end_clip_number',
      'start_time',
      'end_time',
      'veh_std_on_se_crsg',
      'veh_std_on_ne_crsg',
      'veh_std_on_sw_crsg',
      'veh_std_on_nw_crsg',
      'train_is_present'
    ]

    rows = []

    for event in events:
      mins, secs = divmod(((event.start_clip_number + 1) * clip_length * 2 - 1) / 30, 60)
      hours, minutes = divmod(mins, 60)
      start_time = '{:02d}:{:02d}:{:02d}'.format(
          round(hours), round(mins), round(secs))

      mins, secs = divmod(((event.end_clip_number + 1) * clip_length * 2 - 1) / 30, 60)
      hours, minutes = divmod(mins, 60)
      end_time = '{:02d}:{:02d}:{:02d}'.format(
          round(hours), round(mins), round(secs))

      rows.append(
        [report_file_name, 
         event.event_id + 1, 
         event.start_clip_number,
         event.end_clip_number,
         start_time,
         end_time,
         event.contains_veh_std_on_se_crsg,
         event.contains_veh_std_on_ne_crsg,
         event.contains_veh_std_on_sw_crsg,
         event.contains_veh_std_on_nw_crsg,
         event.train_is_present])

    IO.write_csv(report_file_path, header, rows)

  # TODO: confirm that the csv can be opened after writing
  @staticmethod
  def write_ped_right_of_way_incursion_event_report(
      report_file_name, report_dir_path, events, clip_length):
    report_dir_path = path.join(
      report_dir_path, 'ped_right_of_way_incursion_reports')

    if not path.exists(report_dir_path):
      os.makedirs(report_dir_path)

    report_file_path = path.join(
      report_dir_path, report_file_name + '.csv')

    header = [
      'file_name', 
      'sequence_number', 
      'start_clip_number', 
      'end_clip_number',
      'start_time',
      'end_time',
      'ped_on_sth_corr',
      'ped_on_nth_corr', 
      'train_is_present'
    ]

    rows = []

    for event in events:
      mins, secs = divmod(((event.start_clip_number + 1) * clip_length * 2 - 1) / 30, 60)
      hours, minutes = divmod(mins, 60)
      start_time = '{:02d}:{:02d}:{:02d}'.format(
          round(hours), round(mins), round(secs))

      mins, secs = divmod(((event.end_clip_number + 1) * clip_length * 2 - 1) / 30, 60)
      hours, minutes = divmod(mins, 60)
      end_time = '{:02d}:{:02d}:{:02d}'.format(
          round(hours), round(mins), round(secs))

      rows.append(
        [report_file_name, 
         event.event_id + 1, 
         event.start_clip_number,
         event.end_clip_number,
         start_time,
         end_time,
         event.contains_ped_on_sth_corr,
         event.contains_ped_on_nth_corr,
         event.train_is_present])

    IO.write_csv(report_file_path, header, rows)

  # TODO: confirm that the csv can be opened after writing
  @staticmethod
  def write_veh_right_of_way_incursion_event_report(
      report_file_name, report_dir_path, events, clip_length):
    report_dir_path = path.join(
      report_dir_path, 'veh_right_of_way_incursion_reports')

    if not path.exists(report_dir_path):
      os.makedirs(report_dir_path)

    report_file_path = path.join(
      report_dir_path, report_file_name + '.csv')

    header = [
      'file_name', 
      'sequence_number', 
      'start_clip_number', 
      'end_clip_number', 
      'start_time',
      'end_time',
      'veh_adv_on_se_corr',
      'veh_adv_on_ne_corr',
      'veh_adv_on_sw_corr',
      'veh_adv_on_nw_corr',
      'veh_rec_on_se_corr',
      'veh_rec_on_ne_corr',
      'veh_rec_on_sw_corr',
      'veh_rec_on_nw_corr',
      'veh_std_on_se_corr',
      'veh_std_on_ne_corr',
      'veh_std_on_sw_corr',
      'veh_std_on_nw_corr', 
      'train_is_present'
    ]

    rows = []

    for event in events:
      mins, secs = divmod(((event.start_clip_number + 1) * clip_length * 2 - 1) / 30, 60)
      hours, minutes = divmod(mins, 60)
      start_time = '{:02d}:{:02d}:{:02d}'.format(
          round(hours), round(mins), round(secs))

      mins, secs = divmod(((event.end_clip_number + 1) * clip_length * 2 - 1) / 30, 60)
      hours, minutes = divmod(mins, 60)
      end_time = '{:02d}:{:02d}:{:02d}'.format(
          round(hours), round(mins), round(secs))

      rows.append(
        [report_file_name, 
         event.event_id + 1, 
         event.start_clip_number,
         event.end_clip_number, 
         start_time,
         end_time,
         event.contains_veh_adv_on_se_corr,
         event.contains_veh_adv_on_ne_corr,
         event.contains_veh_adv_on_sw_corr,
         event.contains_veh_adv_on_nw_corr,
         event.contains_veh_rec_on_se_corr,
         event.contains_veh_rec_on_ne_corr,
         event.contains_veh_rec_on_sw_corr,
         event.contains_veh_rec_on_nw_corr,
         event.contains_veh_std_on_se_corr,
         event.contains_veh_std_on_ne_corr,
         event.contains_veh_std_on_sw_corr,
         event.contains_veh_std_on_nw_corr,
         event.train_is_present])

    IO.write_csv(report_file_path, header, rows)
