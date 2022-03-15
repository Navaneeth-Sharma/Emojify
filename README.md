# Emojify
Emojify is a fun weekend project built for purpose of learning computer vision and opencv tools. The tech stack involves following 
- Pytorch (for building model)
- dlib (for face detection)
- Open CV (almost entire project)
- And basic packages like Pandas, Numpy, Matplotlib, and PIL

### Small Demo
https://user-images.githubusercontent.com/63489382/158441186-d012df9b-b94e-4f22-8b05-506244235631.mp4

### How to Run use this ?

1. Clone the project

    ```bash
    git clone https://github.com/Navaneeth-Sharma/Emojify/
    ```
2. Move to Emojify directory
3. Download the requirements
    ```bash
    pip install -r requirements.txt
    ```
4. For training 
    - Generate Data using following commands
  
        ```bash
        cd src/data
        ./generate_data.sh
        cv ..
        ```
    - Train the model
  
        ```bash
        python train.py
        ```
5. For Running the system
    - Download the pretrained models (edit the model.pth for not downlading the emotion recognition model)
   
        ```bash
        cd src/models
        ./download_models.sh
        cd ..
        ```
    - Run the main python file
        ```bash
        python main.py
        ```
       
### The Block Diagram of Model for Emotion Recognition
<img src="https://github.com/Navaneeth-Sharma/Emojify/blob/main/docs/emotionnet.png" height="300" width="550" >

###
