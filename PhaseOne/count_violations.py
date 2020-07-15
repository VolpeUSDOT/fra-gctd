import argparse
import os
from utils.event import TripFromReportFile
from utils.io import IO

parser = argparse.ArgumentParser()

parser.add_argument(
  '--classnamesfilepath', '-cnfp',
  default='/home/franklin/PycharmProjects/fra-gctd/class_names.txt',
  help='Path to the class ids/names text file.')
parser.add_argument('--inputdir', '-id', required=True)
parser.add_argument('--outputdir', '-od', required=True)
parser.add_argument('--smoothprobs', '-sp', action='store_true',
                    help='Apply class-wise smoothing across video frame class'
                         ' probability distributions.')
parser.add_argument('--smoothingfactor', '-sf', type=int, default=8,
                    help='The class-wise probability smoothing factor.')

args = parser.parse_args()

for report_name in sorted(os.listdir(args.inputdir)):
  try:
    report_path = os.path.join(args.inputdir, report_name)

    trip = TripFromReportFile(report_path, args.classnamesfilepath,
                              args.smoothprobs, args.smoothingfactor)

    activation_events = trip.find_activation_events()

    IO.write_activation_event_report(
      report_name.split('.')[0], args.outputdir, activation_events, 64)
  except Exception() as e:
    print(e)