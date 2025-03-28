"""
synthseg - Neural Network-Based Brain MRI Segmentation

Part of the micaflow processing pipeline for neuroimaging data.

This module provides an interface to SynthSeg, a deep learning-based tool for automated 
brain MRI segmentation that works across different MRI contrasts without retraining. 
SynthSeg segments brain anatomical structures in T1w, T2w, FLAIR, and other MR contrasts, 
providing both whole-brain segmentation and optional cortical parcellation.

Features:
--------
- Contrast-agnostic segmentation working across different MRI acquisition types
- Whole-brain anatomical structure segmentation with 37 labels
- Optional cortical parcellation (up to 95 additional regions)
- Multiple execution modes: standard, robust (higher quality), and fast
- Volumetric analysis with CSV output for region-wise measurements
- Quality control metrics for assessing segmentation reliability
- GPU acceleration with optional CPU-only execution

API Usage:
---------
micaflow synthseg 
    --i <path/to/image.nii.gz>
    --o <path/to/segmentation.nii.gz>
    [--parc]
    [--robust]
    [--fast]
    [--vol <path/to/volumes.csv>]
    [--qc <path/to/qc_scores.csv>]
    [--threads <num_threads>]

Python Usage:
-----------
>>> from micaflow.scripts.synthseg import main
>>> main({
...     'i': 'input_image.nii.gz',
...     'o': 'segmentation.nii.gz',
...     'parc': True,
...     'robust': False,
...     'fast': True,
...     'vol': 'volumes.csv',
...     'threads': 4
... })

"""

# python imports
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from colorama import init, Fore, Style
init()

def print_extended_help():
    """Print extended help message with examples and usage instructions."""
    # ANSI color codes
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL
    
    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                         SYNTHSEG                               ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script runs the SynthSeg neural network-based tool for brain MRI
    segmentation. It provides automated segmentation of anatomical structures
    even across different contrasts and acquisition types.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow synthseg {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--i{RESET} PATH       : Input image(s) to segment (file or folder)
      {YELLOW}--o{RESET} PATH       : Output segmentation file(s) or folder
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--parc{RESET}         : Enable cortical parcellation
      {YELLOW}--robust{RESET}       : Use robust mode (slower but better quality)
      {YELLOW}--fast{RESET}         : Faster processing (less postprocessing)
      {YELLOW}--threads{RESET} N    : Set number of CPU threads (default: 1)
      {YELLOW}--cpu{RESET}          : Force CPU processing (instead of GPU)
      {YELLOW}--vol{RESET} PATH     : Output volumetric CSV file
      {YELLOW}--qc{RESET} PATH      : Output quality control scores CSV file
      {YELLOW}--post{RESET} PATH    : Output posterior probability maps
      {YELLOW}--resample{RESET} PATH: Output resampled images
      {YELLOW}--crop{RESET} N [N ...]: Size of 3D patches to analyze (default: 192)
      {YELLOW}--ct{RESET}           : Clip intensities for CT scans [0,80]
      {YELLOW}--v1{RESET}           : Use SynthSeg 1.0 instead of 2.0
    
    {CYAN}{BOLD}────────────────── EXAMPLE USAGE ────────────────────────{RESET}
    
    {BLUE}# Basic segmentation{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} t1w_scan.nii.gz \\
      {YELLOW}--o{RESET} segmentation.nii.gz
    
    {BLUE}# With cortical parcellation{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} t1w_scan.nii.gz \\
      {YELLOW}--o{RESET} segmentation.nii.gz \\
      {YELLOW}--parc{RESET}
    
    {BLUE}# Batch processing with volume calculation{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} input_folder/ \\
      {YELLOW}--o{RESET} output_folder/ \\
      {YELLOW}--vol{RESET} volumes.csv
    
    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} SynthSeg works with any MRI contrast without retraining
    {MAGENTA}•{RESET} GPU acceleration is used by default for faster processing
    {MAGENTA}•{RESET} The robust mode provides better quality but is slower
    {MAGENTA}•{RESET} For batch processing, input and output paths must be folders
    """
    print(help_text)
    
def main(args):
  synthseg_home = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(synthseg_home)
  model_dir = os.path.join(synthseg_home, 'models')
  labels_dir = os.path.join(synthseg_home, 'data/labels_classes_priors')
  # The rest of your code remains unchanged
  # print SynthSeg version and checks boolean params for SynthSeg-robust
  if args['robust']:
      args['fast'] = True
      assert not args['v1'], 'The flag --v1 cannot be used with --robust since SynthSeg-robust only came out with 2.0.'
      version = 'SynthSeg-robust 2.0'
  else:
      version = 'SynthSeg 1.0' if args['v1'] else 'SynthSeg 2.0'
      if args['fast']:
          version += ' (fast)'
  print('\n' + version + '\n')

  # enforce CPU processing if necessary
  if args['cpu']:
      print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  # limit the number of threads to be used if running on CPU
  import tensorflow as tf
  args['threads'] = int(args['threads'])
  if args['threads'] == 1:
      print('using 1 thread')
  else:
      print('using %s threads' % args['threads'])
  tf.config.threading.set_inter_op_parallelism_threads(args['threads'])
  tf.config.threading.set_intra_op_parallelism_threads(args['threads'])


  # path models
  if args['robust']:
      args['path_model_segmentation'] = os.path.join(model_dir, 'synthseg_robust_2.0.h5')
  else:
      args['path_model_segmentation'] = os.path.join(model_dir, 'synthseg_2.0.h5')
  args['path_model_parcellation'] = os.path.join(model_dir, 'synthseg_parc_2.0.h5')
  args['path_model_qc'] = os.path.join(model_dir, 'synthseg_qc_2.0.h5')

  # path labels
  args['labels_segmentation'] = os.path.join(labels_dir, 'synthseg_segmentation_labels_2.0.npy')
  args['labels_denoiser'] = os.path.join(labels_dir, 'synthseg_denoiser_labels_2.0.npy')
  args['labels_parcellation'] = os.path.join(labels_dir, 'synthseg_parcellation_labels.npy')
  args['labels_qc'] = os.path.join(labels_dir, 'synthseg_qc_labels_2.0.npy')
  args['names_segmentation_labels'] = os.path.join(labels_dir, 'synthseg_segmentation_names_2.0.npy')
  args['names_parcellation_labels'] = os.path.join(labels_dir, 'synthseg_parcellation_names.npy')
  args['names_qc_labels'] = os.path.join(labels_dir, 'synthseg_qc_names_2.0.npy')
  args['topology_classes'] = os.path.join(labels_dir, 'synthseg_topological_classes_2.0.npy')
  args['n_neutral_labels'] = 19

  # use previous model if needed
  if args['v1']:
      args['path_model_segmentation'] = os.path.join(model_dir, 'synthseg_1.0.h5')
      args['labels_segmentation'] = args['labels_segmentation'].replace('_2.0.npy', '.npy')
      args['labels_qc'] = args['labels_qc'].replace('_2.0.npy', '.npy')
      args['names_segmentation_labels'] = args['names_segmentation_labels'].replace('_2.0.npy', '.npy')
      args['names_qc_labels'] = args['names_qc_labels'].replace('_2.0.npy', '.npy')
      args['topology_classes'] = args['topology_classes'].replace('_2.0.npy', '.npy')
      args['n_neutral_labels'] = 18

  from lamar.SynthSeg.predict_synthseg import predict
  # run prediction
  predict(path_images=args['i'],
          path_segmentations=args['o'],
          path_model_segmentation=args['path_model_segmentation'],
          labels_segmentation=args['labels_segmentation'],
          robust=args['robust'],
          fast=args['fast'],
          v1=args['v1'],
          do_parcellation=args['parc'],
          n_neutral_labels=args['n_neutral_labels'],
          names_segmentation=args['names_segmentation_labels'],
          labels_denoiser=args['labels_denoiser'],
          path_posteriors=args['post'],
          path_resampled=args['resample'],
          path_volumes=args['vol'],
          path_model_parcellation=args['path_model_parcellation'],
          labels_parcellation=args['labels_parcellation'],
          names_parcellation=args['names_parcellation_labels'],
          path_model_qc=args['path_model_qc'],
          labels_qc=args['labels_qc'],
          path_qc_scores=args['qc'],
          names_qc=args['names_qc_labels'],
          cropping=args['crop'],
          topology_classes=args['topology_classes'],
          ct=args['ct'])

if __name__ == '__main__':
    # Check if help flags are provided or no arguments
  if len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv:
      print_extended_help()
      sys.exit(0)

  # parse arguments
  parser = ArgumentParser(
      description="SynthSeg: Deep learning tool for brain MRI segmentation", 
      epilog="For more details see: https://github.com/BBillot/SynthSeg",
      formatter_class=RawDescriptionHelpFormatter
  )

  # input/outputs
  parser.add_argument("--i", help="Image(s) to segment. Can be a path to an image or to a folder.")
  parser.add_argument("--o", help="Segmentation output(s). Must be a folder if --i designates a folder.")
  parser.add_argument("--parc", action="store_true", help="(optional) Whether to perform cortex parcellation.")
  parser.add_argument("--robust", action="store_true", help="(optional) Whether to use robust predictions (slower).")
  parser.add_argument("--fast", action="store_true", help="(optional) Bypass some postprocessing for faster predictions.")
  parser.add_argument("--ct", action="store_true", help="(optional) Clip intensities to [0,80] for CT scans.")
  parser.add_argument("--vol", help="(optional) Path to output CSV file with volumes (mm3) for all regions and subjects.")
  parser.add_argument("--qc", help="(optional) Path to output CSV file with qc scores for all subjects.")
  parser.add_argument("--post", help="(optional) Posteriors output(s). Must be a folder if --i designates a folder.")
  parser.add_argument("--resample", help="(optional) Resampled image(s). Must be a folder if --i designates a folder.")
  parser.add_argument("--crop", nargs='+', type=int, help="(optional) Size of 3D patches to analyse. Default is 192.")
  parser.add_argument("--threads", type=int, default=1, help="(optional) Number of cores to be used. Default is 1.")
  parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")
  parser.add_argument("--v1", action="store_true", help="(optional) Use SynthSeg 1.0 (updated 25/06/22).")

  # parse commandline
  args = vars(parser.parse_args())
  main(args)