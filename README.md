# Bodybuilding Pose Classifier

This project uses computer vision, powered by Google's MediaPipe library, to detect, analyze, and classify bodybuilding poses from images. It's capable of learning new poses and classifying images based on a prototype model.

## Setup and Installation

```bash
pip install -r requirements.txt
```

## Usage

All actions are run through the central `main.py` script.

### Training the Model
This command processes images in `data/raw/`, saves skeletons and prototypes, and generates all visualizations.
```bash
python main.py train
```

### Predicting a specific pose
To classify a new image, use the `predict` command. It will compare the image's pose against all trained prototypes.
```bash
python main.py predict path/to/your/test_image.jpg
```
The terminal will display the analysis, showing the most likely pose and the similarity score.

### Predicting all poses at once
This command runs prediction on all images in `data/validation/`, prints a summary report with accuracy stats to the terminal, and saves a bar chart (`accuracy_report.png`) in the `reports/` folder.
```bash
python main.py validate
```

### Visualizing a Single Skeleton
To quickly check the skeleton on any image without running the full training process.
```bash
python main.py visualize path/to/any/image.jpg
```
The output image will be saved in `reports/visualizations/`.

### Real-time pose detection
Uses Web-Cam to detect poses in real-time. Certify your whole body is being framed in camera.
```bash
python main.py real-time
```

### Cleaning Generated Files
To delete all processed files (`prototypes`, `skeletons`, `visualizations`) and start fresh.
```bash
python main.py clean
```