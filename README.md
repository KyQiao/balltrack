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
pip install -U balltrack-xxx-xxx.wheel
```

## Usage

```python
from balltrack import balltrack
setting = {"skiptime": 5,
            "tracker": "csrt",
            "feature": "HT",
            "fps": 25,
            "size": (1920, 1080),
            "resize": 4}
test = balltrack("test.avi",setting=setting)
test.process()
```

### trackers
`csrt`, `kcf`, `boosting`, `mil`, `tld`, `medianflow`, `mosse`,