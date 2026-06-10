# Technical Specifications

The aim of the project is to develop and/or fine-tune a computer vision model for the automatic recognition of 0.5-litre Coca-Cola and Pepsi bottles in photos and videos.

The following tools were used to carry out the project:
 - Python version 3.12.6 for writing the code
 - Roboflow for annotating and augmenting the image dataset
 - The YOLOv8m model for training object detection in video. The model was trained and deployed using the computer’s GPU via CUDA drivers.


## 1. Data collection and annotation
The dataset for training the computer vision model was collected by searching for images in open sources (ready-made Roboflow datasets, web scraping) and by frame-by-frame segmentation of personally recorded videos using the ffmpeg tool.
The image set was uploaded to the Roboflow website for further processing (annotation) of each photo.

<img width="400" height="218" alt="image" src="https://github.com/user-attachments/assets/bc9e82d9-2e02-4146-9fba-4901a0772205" />
<img width="400" height="218" alt="image" src="https://github.com/user-attachments/assets/4609d17c-eeda-4630-b5ff-34f7489fc5f6" />




## 2. Creating the final dataset
After annotation, we created a new version of the dataset, split the photos into training and validation sets, and added automatic image rotation and augmentation by applying random noise to the photos.
The final dataset consists of 805 images, which was sufficient to train the YOLO model to recognise and classify two classes of objects in the video.
In the project repository, we create a Jupyter Notebook file and import the dataset from Roboflow.


## 3. Loading the image set
Let’s install and import the libraries we need for our work:
```
pip install -q ultralytics roboflow opencv-python
```
To work with a local GPU, you also need to install special versions of PyTorch with CUDA support; the installation command can be found on the official PyTorch website (https://pytorch.org/get-started/locally/).
We’ll write the code to load the image dataset, specifying our personal Roboflow API key, the name of the workspace and project, the desired dataset version, and the machine vision model being used:
```
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("your-dataset-name") 
version = project.version(5)
dataset = version.download("yolov5")
```


## 4. Training the model
The ready dataset is downloaded to the repository in a separate folder; we rename it to Dataset_v5. It contains images and labels with bounding box coordinates, organised into the train and valid folders, as well as a file specifying which data should be used for training the model.
We initialise the weights of the yolov8m model – a mid-range model in the yolov8 family, which provides high-quality detection with relatively moderate hardware requirements.


## 5. Model inference
The model’s accuracy allows it to classify and detect objects in images with near-perfect precision. Let’s write a function that takes an image as input, detects the required objects in it, and outputs the confidence score for each one. The model can easily find all the required objects and determine their class with reasonable confidence:

<img width="400" height="314" alt="image" src="https://github.com/user-attachments/assets/96c8046e-01e2-47cc-86b0-bbf6c6939793" />
<img width="400" height="314" alt="image" src="https://github.com/user-attachments/assets/f94bd358-88f3-4147-a45e-abd0a46ca3e5" />



## 6. Inference for video
The programme is also capable of identifying objects in videos with a high degree of accuracy. The built-in video handling function allows you to load a video file from the `input_videos` folder, stream it in a separate window, and save it to the `output_videos` folder if required. Let’s copy the trained model to a separate file (ColaPepsiNet.pt), initialise it, switch to fuse() mode, and initialise the video processing function process_video_with_tracking()

Object detection in videos is just as accurate as in photos. The function also displays the FPS value for the video to assess the model’s performance. Depending on the lighting, the quality of the video stream and its load, the average FPS is 35–40 frames per second, which is quite comfortable for people to watch.

<img width="650" height="400" alt="image" src="https://github.com/user-attachments/assets/3ee74eca-32c7-4313-af26-08f3a340521d" />


The input can be either a video file saved in a folder or a live video feed from a laptop camera

<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/7b740c2e-cf35-451f-bd68-a14eff2c0f21" />

