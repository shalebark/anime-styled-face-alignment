# Anime Styled Face Alignment

An tool for face alignment for anime-styled faces.

I made this tool because I needed a tool to align anime-styled faces. Sadly, I was able to find any freely available.
The difficulty being most of facial alignment tools uses human facial landmark detection neural networks, and those are incompatible with anime styled faces.

This package is meant to help alleviate that a bit, it's currently uses an anime-styled feature detection network from 2009. 
Alignment is currently only on the z-axis and the alignment is currently done using eye landmarks.

There are other alignment techniques that uses other facial features, but I didn't find much success with them.

## Face Detection

Face Detection and Feature detection uses the python-animeface package: https://github.com/nya3jp/python-animeface
The package is based on the Imager AnimeFace library [http://anime.udp.jp/imager-animeface.html]

## Usage

```
python face-align.py {my-image.jpg} --dest={aligned-image-directory/}
```


