"""
Y2M CLI Module

Command Line Interface for the YOLO to Mobile conversion pipeline.
"""

import sys
import argparse
import logging
from pathlib import Path

from .utils import (
    validate_model_path,
    create_output_directory,
    create_metadata,
    cleanup_intermediate_files,
    ERR_FILE_NOT_FOUND,
    ERR_INVALID_EXTENSION,
    ERR_FILE_EMPTY,
)
from .converter import YOLOConverter
from .optimizer import quantize_with_ultralytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Exit codes
EXIT_SUCCESS = 0
EXIT_VALIDATION_ERROR = 1
EXIT_CONVERSION_ERROR = 2
EXIT_QUANTIZATION_ERROR = 3


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog='y2m',
        description='Y2M: YOLO to Mobile Conversion Pipeline',
        epilog='Example: python -m y2m.cli --weights best.pt --output ./converted_models --quantize'
    )
    
    parser.add_argument(
        '--weights', '-w',
        type=str,
        required=True,
        help='Path to the YOLO .pt model file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./converted_models',
        help='Output directory for converted models (default: ./converted_models)'
    )
    
    parser.add_argument(
        '--quantize', '-q',
        action='store_true',
        help='Apply Int8 quantization for smaller model size'
    )
    
    parser.add_argument(
        '--input-size', '-s',
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=('HEIGHT', 'WIDTH'),
        help='Input image size (default: 640 640)'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove intermediate files (.onnx) after conversion'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def main(args=None) -> int:
    """
    Main entry point for the Y2M CLI.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Set logging level
    if parsed_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print("\n" + "="*50)
    print("   Y2M - YOLO to Mobile Conversion Pipeline")
    print("="*50 + "\n")
    
    # Step 1: Validate input
    logger.info("Step 1/4: Validating input...")
    is_valid, error_code = validate_model_path(parsed_args.weights)
    
    if not is_valid:
        print(f"\n[ERROR] Validation failed: {error_code}")
        return EXIT_VALIDATION_ERROR
    
    # Step 2: Setup output directory
    logger.info("Step 2/4: Setting up output directory...")
    output_dir = create_output_directory(parsed_args.output)
    
    # Step 3: Convert to TFLite
    logger.info("Step 3/4: Converting model...")
    input_size = tuple(parsed_args.input_size)
    
    converter = YOLOConverter(parsed_args.weights, parsed_args.output)
    
    if not converter.load_model():
        print("\n[ERROR] Failed to load model")
        return EXIT_CONVERSION_ERROR
    
    tflite_path = converter.export_to_tflite(input_size=input_size)
    
    if tflite_path is None:
        print("\n[ERROR] TFLite conversion failed")
        return EXIT_CONVERSION_ERROR
    
    # Step 4: Quantization (optional)
    int8_path = None
    if parsed_args.quantize:
        logger.info("Step 4/4: Applying Int8 quantization...")
        int8_path = quantize_with_ultralytics(parsed_args.weights, parsed_args.output)
        
        if int8_path is None:
            logger.warning("Int8 quantization failed, continuing with float32 only")
    else:
        logger.info("Step 4/4: Skipping quantization (use --quantize to enable)")
    
    # Generate metadata
    create_metadata(
        model_name=Path(parsed_args.weights).name,
        output_dir=output_dir,
        class_names=converter.get_class_names(),
        input_size=input_size,
        quantized=int8_path is not None
    )
    
    # Cleanup intermediate files
    if parsed_args.cleanup:
        logger.info("Cleaning up intermediate files...")
        # Clean from both source and output directories
        cleanup_intermediate_files(Path(parsed_args.weights).parent)
        cleanup_intermediate_files(output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("   [SUCCESS] Conversion Complete!")
    print("="*50)
    print(f"\nOutput directory: {output_dir}")
    print(f"   +-- {tflite_path.name}")
    if int8_path:
        print(f"   +-- {int8_path.name}")
    print(f"   +-- metadata.json")
    print()
    
    return EXIT_SUCCESS


if __name__ == '__main__':
    sys.exit(main())
