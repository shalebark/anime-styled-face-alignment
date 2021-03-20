import argparse
from pathlib import Path
import sys
from api import get_and_align_all_faces
import cv2
from _util import write_image, encode_image_to_buffer

def __write_image(image, outputp, quiet=False):
    # output to stdout
    if outputp is sys.stdout:
        sys.stdout.buffer.write(encode_image_to_buffer(image))
    # output to file
    else:
        write_image(image, str(outputp))

def __cli_call(imagep, outputp, quiet=False):
    try:
        # api call to align image
        faces = get_and_align_all_faces(imagepath=str(imagep))

        for i, face in enumerate(faces):
            __write_image(faces[0], outputp)

    except AssertionError as err:
        print('Unable to align image ' + image, file=sys.stderr)
        print(str(err), file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligns Facial Features of Animated Characters')
    parser.add_argument('image', metavar='Image', type=Path,
        help='Filepath to image with facial features to be aligned.')
    parser.add_argument('destination', metavar='Destination', type=Path, nargs='?', default='.',
        help='Output path. (default: current working directory)')
    parser.add_argument('-o', '--output-redirect', dest='is_to_stdout', action='store_true',
        help='Output image as binary data to stdout')
    # parser.add_argument('-V', '--visualize', dest='visualize', action='store_true',
    #     help='Draw the Visual Landmark Features')
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

    __cli_call(imagep, outputp, args.quiet)
