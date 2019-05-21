# BallTrack

A tracking interface using python and `opencv`.

## File description

videoProcessor: a track interface which manage parameters

balltrack: tracking method using `opencv` trackers

tools: other tools used in tracking

## Installation

```bash
python setup.py bdist_wheel
cd dist
pip install balltrack-xxx-xxx.wheel
```



Reinstall with the current version 

```
pip install --upgrade --no-deps --force-reinstall balltrack
```

## Usage

```python
from balltrack import balltrack
setting = {"skiptime": 5,
            "tracker": "csrt",
            "feature": "Hough",
            "fps": 25,
            "size": (1920, 1080),
            "resize": 4}
test = balltrack("test.avi",setting=setting)
test.process()
```

