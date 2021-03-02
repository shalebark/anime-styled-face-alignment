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
python face-align.py {path/to/my/image/image.jpg} --destination={/path/to/output/image.jpg}
```

## Usage with Docker

If you have docker installed, you can run the image through docker.

### Build the docker image

```
docker build -t anime-style-face-align .
```

### Example using example image

```
docker run anime-style-face-align -p /app/examples/test.jpg > output.jpg
```

### Example with your image

```
docker run anime-style-face-align -v {path-to-your-image}:/root/ {your-image-name} > {output-file.jpg}
```

For more information on flags, use --help.

