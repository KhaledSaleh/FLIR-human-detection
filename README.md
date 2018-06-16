# Human Detection in Low Resolution Thermal Image Cameras (FLIR Lepton)
This code demonstartes a hand-crafted image processing algorithm for detecting humans in low-res thermal images that are usually captured using FLIR cameras such as FLIR Lepton and FLIR One.

## Requirements
- Python 2.x
- Numpy 1.13.x
- OpenCV 2.4.x
- Tqdm 4.14.0

## Content
- `data_utils.py`: Utlitiy functions used for serving the main functionality of the code.
- `human_detector.py`: The building blocks of human detection algorithm.
- `main.py`: Sample code to test the functionality of the human detection algorithm on a given folder of raw radimetry data in .txt files.

## Usage
- In order to test the human detection algorithm on your raw radimoetry data folder, run the `main.py` script. For more information on the available options/argument to pass to the script, run the following command:
```
python main.py --help
