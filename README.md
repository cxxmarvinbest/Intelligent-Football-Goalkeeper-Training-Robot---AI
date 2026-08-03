# Intelligent-Football-Goalkeeper-Training-Robot---AI
## Project Overview
This project is an **intelligent football goalkeeper training penalty kick robot** based on artificial intelligence visual recognition technology and precision motor control system.  
The equipment is fixed and deployed in the center of the penalty spot (**11 meters**), facing the goal; The system divides the goal area into **3 × 6 = 18** grid training areas. All serve parameters, manual, and AI intelligent serve are based on 18 divisions to realize landing point control, and comprehensively simulate shots from all angles and high and low positions of the goal. Training the goalkeeper's ability to fight.  
AI is the core intelligent capability of the equipment. Based on the goalkeeper's real-time position, it automatically adapts to the serve strategy and manually adjusts parameters to realize fully automatic intelligent confrontation training, which greatly improves the professionalism and actual combat of training.  
##   Mode
### AI follow serve
The system recognizes the goalkeeper's position in real time, automatically matches the goal corresponding to 18 sub-areas, follows the personnel's position to dynamically serve the landing point, and always sends the ball to the goalkeeper's controllable training area to ensure the continuity of training and avoid empty balls and invalid serves. The height of the ball (**low, medium, high**).  
### AI reverse serve
It is specially designed for goalkeeper confrontation training. According to the goalkeeper's position, it intelligently reverses the position to serve, forcibly mobilizes the goalkeeper to move and save, and trains lateral movement speed and instantaneous response:  
• The goalkeeper is positioned to the left → intelligently serve the goal area on the right;  
• The goalkeeper is positioned to the right → intelligently serve the left goal area;  
• The goalkeeper stands in the middle → the left/right/random area can be manually preset to be given priority, and the training plan can be flexibly adapted;  
• The height of the ball supports selection: low, medium, high, random.  
All reverse serve landing points correspond to 16 grid divisions of the goal, with accurate landing points and scientific training intensity.  
## Technologies Used
**Core Logic Language:** Python 3  
**Computer Vision and Image Processing:** OpenCV + Numpy  
**Target detection:** YOLOv8n  
**Algorithm:** EMA exponential moving average  
**Video streaming:** RTSP protocol live camera  
**Web Framework:** FastAPI + Uvicorn  
**Data check:** Pydantic BaseModel  
**Video decoding:** FFmpeg + subprocess (H265-> BGR24)  
**Multithreading:** Python threading + QThread  
**Communication protocol:** RTSP, HTTP, RS485 (serial port) JSON protocol  
**Image processing:** Zoom  
## Usage
1.Create a virtual environment(conda create -n football python=3.12)  
2.Use `extract.ipynb` to split videos into individual images based on a specified number of frames  
3.Use **MakeSense** to annotate the image, dividing it into `0 goalkeeper`, `1 crossbar`, `2 post_left`, `3 post_right`, `4 ball`  
4.Split the `images` and `Annotation` into `train` and `val`, and place them into `datasets`  
5.Configure and specify the paths of the dataset and category information `football.yaml`  
6.Train model:`yolov8-train.py`  
7.Generative model:`train/best.pt`  
8.`test_trajectory_qt` includes videos used for testing  
9.`trajectory_qt.py`: Show qt interface and test  
10.`11m_position.py`:Its HTTP protocol is available for front-end use  
![AI智能模式界面](https://github.com/cxxmarvinbest/repository/blob/main/images/example.png)
