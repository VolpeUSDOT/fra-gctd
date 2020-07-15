import logging
from multiprocessing import Process
import numpy as np
from subprocess import PIPE, Popen
import tensorflow as tf

class VideoAnalyzer(Process):
  def __init__(
      self, clip_shape, num_frames, num_classes, batch_size, model_input_size,
      model_path, device_type, num_processes_per_device, cpu_count,
      node_names_map, gpu_memory_fraction, should_crop, crop_x, crop_y,
      crop_width, crop_height, ffmpeg_command, child_interrupt_queue,
      result_queue, name, prefetch_size=tf.data.experimental.AUTOTUNE):
    super(VideoAnalyzer, self).__init__(name=name)

    #### TF session variables ####
    graph_def = tf.GraphDef()

    with open(model_path, 'rb') as file:
      graph_def.ParseFromString(file.read())

    self.input_node, self.output_node = tf.import_graph_def(
      graph_def, return_elements=[node_names_map['input_node_name'],
                                  node_names_map['output_node_name']])

    self.device_type = device_type
    
    if self.device_type == 'gpu':
      gpu_options = tf.GPUOptions(
        allow_growth=True,
        per_process_gpu_memory_fraction=gpu_memory_fraction)

      self.session_config = tf.ConfigProto(allow_soft_placement=True,
                                           gpu_options=gpu_options)
    else:
      self.session_config = None

    #### clip generator variables ####
    self.clip_shape = clip_shape
    self.should_crop = should_crop

    if self.should_crop:
      self.crop_x = crop_x
      self.crop_y = crop_y
      self.crop_width = crop_width
      self.crop_height = crop_height

    self.model_input_size = model_input_size
    self.num_processes_per_device = num_processes_per_device
    self.cpu_count = cpu_count
    self.batch_size = batch_size
    self.prefetch_size = prefetch_size
    self.ffmpeg_command = ffmpeg_command

    self.prob_array = np.ndarray(
      (int(np.ceil(num_frames / (self.clip_shape[0] * 2)) * 2), num_classes),
      dtype=np.float32)
    tf.logging.info('num_frames: {}'.format(num_frames))
    tf.logging.info('prob_array: {}'.format(self.prob_array.shape))
    self.child_interrupt_queue = child_interrupt_queue
    self.result_queue = result_queue

    logging.debug('opening video clip pipe')

    self.frame_string_len = 1

    for dim in self.clip_shape[1:]:
      self.frame_string_len *= dim

    self.clip_string_len = self.frame_string_len * self.clip_shape[0]

    buffer_scale = 2

    while buffer_scale < self.clip_string_len:
      buffer_scale *= 2

    self.clip_pipe = Popen(
      self.ffmpeg_command, stdout=PIPE, stderr=PIPE, bufsize=buffer_scale)

    logging.debug('video clip pipe created with pid: {}'.format(
      self.clip_pipe.pid))

    self.num_processed_frames = 0
    self.num_processed_clips = 0
    self.num_ingested_clips = 0
    self.odd_clip = None

  # feed the tf.data input pipeline one image at a time and, while we're at it,
  # extract timestamp overlay crops for later mapping to strings.
  def generate_clips(self):
    while True:
      try:
        try:
          _ = self.child_interrupt_queue.get_nowait()
          logging.warning('closing video clip pipe following interrupt signal')
          if self.clip_pipe is not None:
            self.clip_pipe.stdout.close()
            self.clip_pipe.stderr.close()
            self.clip_pipe.terminate()
          if self.num_ingested_clips % 2 == 1:
            self.num_ingested_clips += 1
            tf.logging.info('num_ingested_clips: {}'.format(
              self.num_ingested_clips))
            yield self.odd_clip
          return
        except:
          pass

        # if self.should_terminate:
        #     logging.debug('closing video clip pipe following end of stream')
        #     self.clip_pipe.stdout.close()
        #     self.clip_pipe.stderr.close()
        #     self.clip_pipe.terminate()
        #     return

        if self.num_ingested_clips % 2 == 1:
          self.num_ingested_clips += 1
          tf.logging.info('num_ingested_clips: {}'.format(
            self.num_ingested_clips))
          yield self.odd_clip
        else:
          clip_string = self.clip_pipe.stdout.read(self.clip_string_len * 2)

          if not clip_string:
            logging.debug('closing video clip pipe following end of stream')
            self.clip_pipe.stdout.close()
            self.clip_pipe.stderr.close()
            self.clip_pipe.terminate()
            return
          else:
            clip_array = np.fromstring(clip_string, dtype=np.uint8)
            # clip_string_len = len(clip_string)
            clip_array_len = len(clip_array)
            # tf.logging.info('clip_array: {}'.format(clip_array.shape))

            # if clip_string_len < self.clip_string_len:
            if clip_array_len < self.clip_string_len * 2: # only one clip left, don't increment num_ingested_clips
              # clip_string += b''.join(
              #   [b'0' for _ in range(self.clip_string_len - clip_string_len)])
              appendage = np.zeros(
                (self.clip_string_len * 2 - clip_array_len,), dtype=np.uint8)

              # tf.logging.info('appendage: {}'.format(appendage.shape))

              clip_array = np.concatenate((clip_array, appendage))

              # tf.logging.info('appended_clip_array: {}'.format(clip_array.shape))

            # tf.logging.info('clip_array_len: {}'.format(len(clip_array)))

            # self.num_processed_frames += int(
            #   clip_string_len / self.clip_string_len * self.clip_shape[0])
            self.num_processed_frames += int(clip_array_len / (
              self.clip_string_len * 2) * (self.clip_shape[0] * 2))
            tf.logging.info('num_processed_frames: {}'.format(
              self.num_processed_frames))

          clip_array = clip_array.reshape(
            (self.clip_shape[0] * 2, self.frame_string_len))
          # tf.logging.info('clip_array: {}'.format(clip_array.shape))

          self.odd_clip = clip_array[1::2]
          # tf.logging.info('odd_clip: {}'.format(self.odd_clip.shape))
          # self.odd_clip = self.odd_clip.reshape(
          #   (self.frame_string_len * self.clip_shape[0],))
          self.odd_clip = self.odd_clip.reshape(self.clip_shape)
          # self.odd_clip = np.ravel(self.odd_clip)

          if self.should_crop:
            self.odd_clip = self.odd_clip[:, self.crop_y:self.crop_y + self.crop_height,
                    self.crop_x:self.crop_x + self.crop_width]

          even_clip = clip_array[0::2]
          # even_clip = even_clip.reshape(
          #   (self.frame_string_len * self.clip_shape[0],))
          even_clip = even_clip.reshape(self.clip_shape)
          # tf.logging.info('even_clip: {}'.format(even_clip.shape))
          # even_clip = np.ravel(even_clip)

          if self.should_crop:
            even_clip = even_clip[:, self.crop_y:self.crop_y + self.crop_height,
                    self.crop_x:self.crop_x + self.crop_width]
          self.num_ingested_clips += 1
          tf.logging.info('num_ingested_clips: {}'.format(
            self.num_ingested_clips))
          yield even_clip
      except Exception as e:
        logging.error(
          'met an unexpected error after processing {} clips.'.format(
            self.num_processed_frames))
        logging.error(e)
        if self.clip_pipe is not None:
          logging.error(
            'ffmpeg reported:\n{}'.format(self.clip_pipe.stderr.readlines()))
          logging.debug('closing video clip pipe following raised exception')
          self.clip_pipe.stdout.close()
          self.clip_pipe.stderr.close()
          self.clip_pipe.terminate()
        logging.debug('raising exception to caller.')
        raise e

  def _preprocess_clips(self, video):
    # video = tf.decode_raw(video, tf.uint8)
    # video = tf.reshape(video, self.clip_shape)
    # if self.should_crop:
    #   video = video[:, self.crop_y:self.crop_y + self.crop_height,
    #           self.crop_x:self.crop_x + self.crop_width]
    video = tf.image.convert_image_dtype(video, dtype=tf.float32)
    video = tf.image.resize_bilinear(
      video, [self.model_input_size, self.model_input_size])
    video = tf.subtract(video, 0.5)
    video = tf.multiply(video, 2.0)
    return video

  # assumed user specification of numperdeviceprocesses has been validated,
  # to be <= cpu cores when in --cpuonly mode
  def _get_num_parallel_calls(self):
    if self.device_type == 'gpu':
      return int(self.cpu_count / self.num_processes_per_device)
    else:
      if self.num_processes_per_device == 1:
        return self.cpu_count - 1
      elif self.num_processes_per_device == self.cpu_count:
        return self.cpu_count
      else:
        return int((self.cpu_count - self.num_processes_per_device) / 
                   self.num_processes_per_device)

  def run(self):
    logging.info('started inference.')

    # TODO: merge map and batch, batchify preprocess_clips fn
    with tf.device('/cpu:0') if self.device_type == 'cpu' else tf.device(None):
      with tf.Session(config=self.session_config) as session:
        # clip_dataset = tf.data.Dataset.from_generator(
        #   self.generate_clips, tf.string, tf.TensorShape([]))
        clip_dataset = tf.data.Dataset.from_generator(
          self.generate_clips, tf.uint8, tf.TensorShape(self.clip_shape))
        clip_dataset = clip_dataset.map(self._preprocess_clips,
                                          self._get_num_parallel_calls())
        clip_dataset = clip_dataset.batch(self.batch_size)
        clip_dataset = clip_dataset.prefetch(self.prefetch_size)
        next_batch = clip_dataset.make_one_shot_iterator().get_next()

        while True:
          try:
            clip_batch = session.run(next_batch)
            probs = session.run(self.output_node, {self.input_node: clip_batch})

            self.prob_array[self.num_processed_clips:\
              self.num_processed_clips + probs.shape[0]] = probs

            self.num_processed_clips += probs.shape[0]

            tf.logging.info('num_processed_clips: {}'.format(
              self.num_processed_clips))
          except tf.errors.OutOfRangeError:
            logging.info('completed inference.')
            break

    #since ffprobe may overestimate (not sure if it underestimates) the number
    # of frames, recalculate the number of probs we should have received and
    # return a truncated array
    logging.info('pre prob_array: {}'.format(self.prob_array.shape))
    self.prob_array = self.prob_array[:self.num_ingested_clips, :]
    logging.info('truncated prob_array: {}'.format(self.prob_array.shape))

    # average pairs of adjacent clips (due to sub-sampling every other frame in
    # a 128-frame clip, yielding two 64-frame clips)
    even_probs = np.expand_dims(self.prob_array[0::2], axis=0)
    odd_probs = np.expand_dims(self.prob_array[1::2], axis=0)

    self.prob_array = np.concatenate((even_probs, odd_probs), axis=0)
    self.prob_array = np.average(self.prob_array, axis=0)

    logging.info('post prob_array: {}'.format(self.prob_array.shape))
    self.result_queue.put((self.num_processed_frames, self.prob_array))
    # self.result_queue.put((count, self.prob_array, self.num_processed_framesmestamp_array))
    self.result_queue.close()

  def __del__(self):
    if self.clip_pipe is not None and self.clip_pipe.returncode is None:
      logging.debug(
        'video clip pipe with pid {} remained alive after being instructed to '
        'temrinate and had to be killed'.format(self.clip_pipe.pid))
      self.clip_pipe.kill()
