#!/usr/bin/env python3
"""
LaMAR: Label Augmented Modality Agnostic Registration
Command-line interface
"""

import argparse
import sys
import os
from lamar.scripts.lamar import lamareg
from lamar.scripts import synthseg, coregister, apply_warp


def main():
    """Main entry point for the LaMAR CLI."""
    parser = argparse.ArgumentParser(
        description="LaMAR: Label Augmented Modality Agnostic Registration"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # WORKFLOW 1: Full registration pipeline
    register_parser = subparsers.add_parser(
        "register", 
        help="Perform full registration pipeline with SynthSeg parcellation"
    )
    register_parser.add_argument("--moving", required=True, help="Input moving image to be registered")
    register_parser.add_argument("--fixed", required=True, help="Reference fixed image (target space)")
    register_parser.add_argument("--output", required=True, help="Output registered image")
    register_parser.add_argument("--moving-parc", required=True, help="Output path for moving image parcellation")
    register_parser.add_argument("--fixed-parc", required=True, help="Output path for fixed image parcellation")
    register_parser.add_argument("--registered-parc", required=True, help="Output path for registered parcellation")
    register_parser.add_argument("--affine", required=True, help="Output path for affine transformation")
    register_parser.add_argument("--warpfield", required=True, help="Output path for warp field")
    register_parser.add_argument("--inverse-warpfield", required=True, help="Output path for inverse warp field")
    register_parser.add_argument("--inverse-affine", required=True, help="Output path for inverse affine transformation")
    register_parser.add_argument("--registration-method", default="SyNRA", help="Registration method")
    register_parser.add_argument("--synthseg-threads", type=int, default=1, 
                                help="Number of threads to use for SynthSeg segmentation (default: 1)")
    register_parser.add_argument("--ants-threads", type=int, default=1,
                                help="Number of threads to use for ANTs registration (default: 1)")
    
    # WORKFLOW 2: Generate warpfield only
    warpfield_parser = subparsers.add_parser(
        "generate-warpfield", 
        help="Generate registration warpfield without applying it"
    )
    warpfield_parser.add_argument("--moving", required=True, help="Input moving image")
    warpfield_parser.add_argument("--fixed", required=True, help="Reference fixed image")
    warpfield_parser.add_argument("--moving-parc", required=True, help="Output path for moving image parcellation")
    warpfield_parser.add_argument("--fixed-parc", required=True, help="Output path for fixed image parcellation")
    warpfield_parser.add_argument("--registered-parc", required=True, help="Output path for registered parcellation")
    warpfield_parser.add_argument("--affine", required=True, help="Output path for affine transformation")
    warpfield_parser.add_argument("--warpfield", required=True, help="Output path for warp field")
    warpfield_parser.add_argument("--inverse-warpfield", required=True, help="Output path for inverse warp field")
    warpfield_parser.add_argument("--inverse-affine", required=True, help="Output path for inverse affine transformation")
    warpfield_parser.add_argument("--registration-method", default="SyNRA", help="Registration method")
    warpfield_parser.add_argument("--synthseg-threads", type=int, default=1, 
                                 help="Number of threads to use for SynthSeg segmentation (default: 1)")
    warpfield_parser.add_argument("--ants-threads", type=int, default=1,
                                 help="Number of threads to use for ANTs registration (default: 1)")
    
    # WORKFLOW 3: Apply existing warpfield
    apply_parser = subparsers.add_parser(
        "apply-warpfield", 
        help="Apply existing warpfield to an image"
    )
    apply_parser.add_argument("--moving", required=True, help="Input image to transform")
    apply_parser.add_argument("--fixed", required=True, help="Reference space image")
    apply_parser.add_argument("--output", required=True, help="Output registered image")
    apply_parser.add_argument("--warpfield", required=True, help="Path to warp field")
    apply_parser.add_argument("--affine", required=True, help="Path to affine transformation")
    apply_parser.add_argument("--ants-threads", type=int, default=1,
                             help="Number of threads to use for ANTs transformation (default: 1)")
    
    # DIRECT TOOL ACCESS: SynthSeg
    synthseg_parser = subparsers.add_parser(
        "synthseg",
        help="Run SynthSeg brain MRI segmentation directly"
    )
    synthseg_parser.add_argument("--i", required=True, help="Input image")
    synthseg_parser.add_argument("--o", required=True, help="Output segmentation")
    synthseg_parser.add_argument("--parc", action="store_true", help="Output parcellation")
    synthseg_parser.add_argument("--cpu", action="store_true", help="Use CPU")
    synthseg_parser.add_argument("--threads", type=int, default=1, help="Number of threads")
    # Add other SynthSeg arguments as needed
    
    # DIRECT TOOL ACCESS: Coregister
    coregister_parser = subparsers.add_parser(
        "coregister",
        help="Run coregistration directly"
    )
    
    # DIRECT TOOL ACCESS: Apply Warp
    apply_warp_parser = subparsers.add_parser(
        "apply-warp",
        help="Apply transformation to an image directly"
    )
    
    # Parse known args, leaving the rest for the subcommands
    args, unknown_args = parser.parse_known_args()
    
    # Handle command routing
    if args.command == "register":
        lamareg(
            input_image=args.moving,
            reference_image=args.fixed,
            output_image=args.output,
            input_parc=args.moving_parc,
            reference_parc=args.fixed_parc, 
            output_parc=args.registered_parc,
            affine_file=args.affine,
            warp_file=args.warpfield,
            inverse_warp_file=args.inverse_warpfield,
            inverse_affine_file=args.inverse_affine,
            registration_method=args.registration_method,
            synthseg_threads=args.synthseg_threads,
            ants_threads=args.ants_threads
        )
    elif args.command == "generate-warpfield":
        lamareg(
            input_image=args.moving,
            reference_image=args.fixed,
            output_image=None,  # No output image for generate-warpfield
            input_parc=args.moving_parc,
            reference_parc=args.fixed_parc,
            output_parc=args.registered_parc,
            affine_file=args.affine,
            warp_file=args.warpfield,
            inverse_warp_file=args.inverse_warpfield,
            inverse_affine_file=args.inverse_affine,
            generate_warpfield=True,
            registration_method=args.registration_method,
            synthseg_threads=args.synthseg_threads,
            ants_threads=args.ants_threads
        )
    elif args.command == "apply-warpfield":
        lamareg(
            input_image=args.moving,
            reference_image=args.fixed,
            output_image=args.output,
            apply_warpfield=True,
            affine_file=args.affine,
            warp_file=args.warpfield,
            ants_threads=args.ants_threads,
            synthseg_threads=1  # Not used in this workflow but needed for the function
        )
    elif args.command == "synthseg":
        # Create a clean dictionary with the args provided by the parser
        synthseg_args = {}
        
        # Add explicit arguments from argparse
        if hasattr(args, 'i') and args.i:
            synthseg_args['i'] = args.i
        if hasattr(args, 'o') and args.o:
            synthseg_args['o'] = args.o
        
        # Add flag arguments 
        for flag in ['parc', 'cpu']:
            if flag in unknown_args or f'--{flag}' in unknown_args:
                synthseg_args[flag] = True
        
        # Parse remaining arguments from command line
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i].lstrip('-')
            if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith('-'):
                synthseg_args[arg] = unknown_args[i+1]
                i += 2
            else:
                # It's a flag
                synthseg_args[arg] = True
                i += 1
        
        # Set ALL required defaults for SynthSeg
        synthseg_args.setdefault('parc', True)
        synthseg_args.setdefault('cpu', True)
        synthseg_args.setdefault('robust', False)
        synthseg_args.setdefault('v1', False)
        synthseg_args.setdefault('fast', True)
        synthseg_args.setdefault('post', None)
        synthseg_args.setdefault('resample', None)
        synthseg_args.setdefault('ct', None)
        synthseg_args.setdefault('vol', None)
        synthseg_args.setdefault('qc', None)
        synthseg_args.setdefault('device', None)
        synthseg_args.setdefault('crop', None)

        if hasattr(args, 'threads') and args.threads:
            synthseg_args['threads'] = str(args.threads)
        else:
            synthseg_args['threads'] = '1'
        
        try:
            synthseg.main(synthseg_args)
        except Exception as e:
            print(f"SynthSeg error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "coregister":
        # If no additional arguments are provided, print help
        if not unknown_args:
            coregister.print_help()
            sys.exit(0)
        # Forward arguments to coregister
        sys.argv = [sys.argv[0]] + unknown_args
        coregister.main()
    elif args.command == "apply-warp":
        # If no additional arguments are provided, print help
        if not unknown_args:
            apply_warp.print_help()
            sys.exit(0)
        # Forward arguments to apply_warp
        sys.argv = [sys.argv[0]] + unknown_args
        apply_warp.main()
    elif args.command is None:
        parser.print_help()
        sys.exit(0)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()