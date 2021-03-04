# Anime Styled Face Alignment

An tool for face alignment for anime-styled faces.

I made this tool because I needed a tool to align anime-styled faces. Sadly, I was able to find any freely available.
The difficulty being most of facial alignment tools uses human facial landmark detection neural networks, and those are incompatible with anime styled faces.

This package is meant to help alleviate that a bit, it's currently uses an anime-styled feature detection network from 2009. 
Alignment is currently only on the z-axis and the alignment is currently done using eye landmarks.

There are other alignment techniques that uses other facial features, but I didn't find much success with them.

## Face Detection

Face Detection and Feature detection uses this python-animeface package: https://github.com/nya3jp/python-animeface
The package is based on the Imager AnimeFace library http://anime.udp.jp/imager-animeface.html

## Installation

### On Ubuntu

```
apt-get install libgl1-mesa-dev libglib2.0-0
python3 -m pip install -r requirements.txt
```

## Usage

```
python cli.py {path/to/my/image/image.jpg} {/path/to/output/image.jpg}
```

## Usage with Docker

If you have docker installed, you can run the image through docker.

### Build the docker image

```
docker build -t anime-style-face-align .
```

### Docker example with your image

```
docker run -v {parent-directory-to-your-image}:/root/ anime-style-face-align -o {your-image-name} > {output-file.jpg}
```

For more information on flags, use --help.

## Visualize the facial features

python cli.py --visualize {path/to/my/image/image.jpg} {/path/to/output/image.jpg}

## API Example

```
# add file to your path
import os, sys
sys.path.append(os.path.abspath('anime_face_alignment'))

# importing the library
from anime_face_alignment import align_face as anime_face_align
# used to save the imagefile
import cv2

# aligning the image from filepath
img = anime_face_align(filepath='filepath')

# aligning the image from image
img = cv2.imread('filepath')

cv2.imwrite('outputpath', img)
```

# Results

![](https://github.com/shalebark/anime-styled-face-alignment/raw/master/examples/test.jpg)

![](https://github.com/shalebark/anime-styled-face-alignment/raw/master/examples/test_results.jpg)

![](https://github.com/shalebark/anime-styled-face-alignment/raw/master/examples/test_visuals.jpg)