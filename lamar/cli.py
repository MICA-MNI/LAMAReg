#!/usr/bin/env python3
"""
LaMAR: Label Augmented Modality Agnostic Registration
Command-line interface
"""

import argparse
import sys
from lamar.scripts.synthseg_registration import main as registration_main


def main():
    """Main entry point for the LaMAR CLI."""
    parser = argparse.ArgumentParser(
        description="LaMAR: Label Augmented Modality Agnostic Registration"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Register command - uses same parameters as synthseg_registration
    register_parser = subparsers.add_parser(
        "register", 
        help="Register images using parcellation-based registration"
    )
    
    # Add the same arguments as in synthseg_registration.py
    register_parser.add_argument("--moving", help="Input moving image to be registered")
    register_parser.add_argument("--fixed", help="Reference fixed image (target space)")
    register_parser.add_argument("--output", required=True, help="Output registered image")
    register_parser.add_argument("--moving-parc", required=True, help="Moving image parcellation")
    register_parser.add_argument("--fixed-parc", required=True, help="Fixed image parcellation")
    register_parser.add_argument("--output-parc", required=True, help="Output registered parcellation")
    register_parser.add_argument("--registration-method", default="SyNRA", help="Registration method")
    

    # Register command - uses same parameters as synthseg_registration
    warpfield_parser = subparsers.add_parser(
        "generate-warpfield", 
        help="Register images using parcellation-based registration"
    )
    
    # Add the same arguments as in synthseg_registration.py
    warpfield_parser.add_argument("--moving", required=True, help="Input moving image to be registered")
    warpfield_parser.add_argument("--fixed", required=True, help="Reference fixed image (target space)")
    warpfield_parser.add_argument("--moving-parc", help="Input moving parcellation")
    warpfield_parser.add_argument("--fixed-parc", help="Reference fixed parcellation")
    warpfield_parser.add_argument("--output-parc", help="Output registered parcellation")
    warpfield_parser.add_argument("--registration-method", default="SyNRA", help="Registration method")
    
    # Register command - uses same parameters as synthseg_registration
    register_parser = subparsers.add_parser(
        "apply-warpfield", 
        help="Register images using parcellation-based registration"
    )
    
    # Add the same arguments as in synthseg_registration.py
    register_parser.add_argument("--moving", required=True, help="Input moving image to be registered")
    register_parser.add_argument("--fixed", required=True, help="Reference fixed image (target space)")
    register_parser.add_argument("--output", required=True, help="Output registered image")
    register_parser.add_argument("--apply-warpfield", action="store_true", help="Apply warp field to moving image")
    register_parser.add_argument("--registration-method", default="SyNRA", help="Registration method")
    


    args = parser.parse_args()
    
    # Handle command routing
    if args.command == "register":
        # Use the existing registration main function
        sys.argv = ["synthseg_registration"] + sys.argv[2:]
        registration_main()
    if args.command == "generate-warpfield":
        # Use the existing registration main function
        sys.argv = ["synthseg_registration"] + sys.argv[2:]
        registration_main()
    if args.command == "apply-warpfield":
        # Use the existing registration main function
        sys.argv = ["synthseg_registration"] + sys.argv[2:]
        registration_main()
    elif args.command is None:
        parser.print_help()
        sys.exit(0)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()