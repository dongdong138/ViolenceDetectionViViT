<h1 align="center">
Violence Detection ViViT
</h1>

### Installation
1. Clone this repository.
   
    ```
    git clone https://github.com/dongdong138/ViolenceDetectionViViT.git
    ```  
3. Create a virtual environment and activate it.
   
    ```
    python3 -m venv .env
    ```  
3. Install Tensorflow with CUDA [follow this Tensorflow installation](https://www.tensorflow.org/install).
4. Install requirement libraries.
   
    ```
    pip install -r requirements.txt
    ```

### How to run
  To crop videos go to project directory and run *cropvideo.py* like below.
  ```
  python cropvideo.py
  ```
  To train models go to project directory and run *training.py* like below.
  ```
  python training.py
  ```
  To run demo go to project directory and run *inference.py* like below.
  ```
  python inference.py
  ```


Reference: ViViT: [A Video Vision Transformer](https://arxiv.org/abs/2103.15691)
