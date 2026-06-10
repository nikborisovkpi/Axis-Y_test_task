# Technical Specifications No. 1

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

<img width="500" height="370" alt="image" src="https://github.com/user-attachments/assets/7b740c2e-cf35-451f-bd68-a14eff2c0f21" />


-----

# Technical Specification No. 2

**Objective**: To identify and count the number of Coca-Cola and Pepsi bottles in zones A and B in images or videos.

 - Input data: A video stream or individual images with two defined zones for analysis (A, B), identified by highlighted polygons.
 - Output: The number of Coca-Cola and Pepsi bottles in each zone. Output format: JSON structure.
 - Visualisation: outlined zones + a label indicating the number of objects in each zone.
 - Functional requirements: use a trained YOLO model to detect and classify bottles (Coca-Cola and Pepsi); for each detected object, determine whether it falls within zone A or B.
 - Algorithm: check whether the centre of the object is inside the polygon.


## Key project updates
### Image processing:
 - Two polygons have been created to count the number of objects inside them (zone_A and zone_B) and visualise them on the image.
 - The `counts` counter, which records the number of objects detected inside the polygons.
 - Logic for detecting objects within polygons: checking via `cv2.pointPolygonTest()` whether the centre of the object is inside the polygon.
 - Outputting data from the `counts` counter in JSON format.

<img width="483" height="318" alt="image" src="https://github.com/user-attachments/assets/acf3e253-67c0-4a84-b673-d8116d7c5e39" />

Reading data from the meter:

<img width="170" height="211" alt="image" src="https://github.com/user-attachments/assets/52a941a7-9826-4b62-affe-7635754b0899" />


### Video proccessing:
 - Two polygons have been created to count the number of objects inside them (zone_A and zone_B) and to visualise them on the image.
 - The `counts` counter records the number of objects detected inside the polygons during a single frame.
 - A total counter, total_counts, has also been added, which records the number of unique objects detected inside the polygons over the duration of the entire video. This can be useful when it is necessary to count the total number of detected objects, for example, on a conveyor belt.
 - Logic for detecting objects within polygons: checking via cv2.pointPolygonTest() whether the centre of the object is inside the polygon.

<img width="662" height="374" alt="image" src="https://github.com/user-attachments/assets/b66bf20e-19f7-4128-a57d-3db83665dfa4" />




Data from the total_counts counter, containing the number of unique objects detected within the polygons, is output after the entire video has been processed.
Logic for detecting unique objects: two dictionaries, tracked_objects_A and tracked_objects_B, are created corresponding to the two zones. If an object is detected in one of the polygons, its object_id is written to the corresponding dictionary, and +1 object is added to the total_counts counter for the corresponding zone.
When the next object is detected in a specific zone, the system checks whether the object’s identifier (object_id) has been recorded in the corresponding dictionary; if not, the system adds it to the overall statistics, and if so, it is clear that this object has already been detected in this zone previously and there is no point in recording it again – we skip it.

<img width="170" height="215" alt="image" src="https://github.com/user-attachments/assets/24aad390-4675-481c-903e-6fe8f29f16a6" />




The updated system has delivered high-quality results with input images and video files. The average frame rate when processing video from a downloaded file was 33 frames per second, whilst processing video from the laptop’s camera in real time yielded 25 frames.

<img width="541" height="306" alt="image" src="https://github.com/user-attachments/assets/cc35607e-df57-4949-8143-9df04cf118dc" />


