import argparse
from pathlib import Path
import sys
import _util as util

from api import align_face, visualize_facial_lines

def align_face_cli(imagep, outputp, quiet=False):
    try:
        # api call to align image
        aligned_image = align_face(filepath=str(imagep.absolute()))

        if outputp is sys.stdout:
            sys.stdout.buffer.write(util.encode_image_to_buffer(aligned_image))
        else:
            util.write_image(aligned_image, str(outputp))
            # Message to stdout, success
            if not quiet:
                print('"{}" aligned. Save to: "{}"'.format(image, outputp))
    except AssertionError as err:
        print('Unable to align image ' + image, file=sys.stderr)
        print(str(err), file=sys.stderr)

def visualize_face_alignments_cli(imagep, outputp, quiet=False):
    try:
        # api call to visualize image
        visualized_image = visualize_facial_lines(filepath=str(imagep.absolute()))

        if outputp is sys.stdout:
            sys.stdout.buffer.write(util.encode_image_to_buffer(visualized_image))
        else:
            util.write_image(visualized_image, str(outputp))
            # Message to stdout, success
            if not quiet:
                print('"{}" visualized. Save to: "{}"'.format(image, outputp))
    except AssertionError as err:
        print('Unable to visuals image ' + image, file=sys.stderr)
        print(str(err), file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligns Facial Features of Animated Characters')
    parser.add_argument('image', metavar='Image', type=Path,
        help='Filepath to image with facial features to be aligned.')
    parser.add_argument('destination', metavar='Destination', type=Path, nargs='?', default='.',
        help='Output path. (default: current working directory)')
    parser.add_argument('-o', '--output-redirect', dest='is_to_stdout', action='store_true',
        help='Output image as binary data to stdout')
    parser.add_argument('-V', '--visualize', dest='visualize', action='store_true', 
        help='Draw the Visual Landmark Features')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true',
        help='Do not output success message.')

    args = parser.parse_args()
    image = args.image.absolute()
    destp = args.destination.absolute()
    is_to_stdout = args.is_to_stdout

    # determine paths
    imagep = Path(image)
    if is_to_stdout:
        outputp = sys.stdout
    else:
        outputp = destp if destp.suffix else destp.joinpath(imagep.name)

    if args.visualize:
        visualize_face_alignments_cli(imagep, outputp, args.quiet)
    else:
        align_face_cli(imagep, outputp, args.quiet)
    
    exit()
