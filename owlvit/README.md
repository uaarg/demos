# OWL-ViT Demo

The [OWL-ViT model from Google](https://huggingface.co/google/owlv2-base-patch16)
is capable of bounding box object detection in an image using only a text
description of the object.

This demo looks for red and yellow circles in a target image. You can run it with:

```
python3 -m venv venv
source venv/bin/activate (or venv/Scripts/activate on Windows)
pip install -r requirements.txt

python3 detect.py --image $input --output result.jpg --texts 'a red circle' 'a yellow circle'
# result.jpg will be a copy of the input image annotated with the detected bounding boxes
```
