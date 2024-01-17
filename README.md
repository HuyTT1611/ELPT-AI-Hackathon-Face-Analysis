# AI-Hackathon-Face-Analysis

Huy T. Tran, Phuong D. Do

In this repository, we introduce two methods for face-analysis problem. First method is based Transformer for highest accuracy in public and private test. 
Second method is based CNN (lightweight) for deploy in cpu and mobile. For face detection we use YOLOv8, output of this model is input of two methods above.
In the next round of competition, we will use distillation between first method(tearcher) and second method(student) two improve accuracy of second 
method with fast inference speed on cpu and gpu. 

## 1. Prepare the dataset from AI-Hackathon-Face-Analysis

Dataset for this hackathon is not public


## 2. Installing 

Using Ubuntu or Linux to run this repository.

```bash
conda create -n face_analysis  python=3.8
conda activate face_analysis
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
## 3. Training

### YOLOv8 detection 
```yaml
# Change path of data in ./configs/detect.yaml to your directory
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ./AI_Hackathon/face_detect  # dataset root dir
train: train  # train images (relative to 'path') 
val: valid  # val images (relative to 'path') 
test:  # test images (optional)

# Classes
names:
  0: face
```
```bash
python train_detect.py
```

### Transformer face analysis model 
```yaml
# Change PHASE1_ROOT_PATH (train.csv) PHASE2_ROOT_PATH (valid.csv) ROOT_PATH (images path) in ./configs/face.yaml to your directory
DATASET:
  TYPE: 'face'
  NAME: 'HACKATHON'
  TRAIN_SPLIT: 'trainval'
  VAL_SPLIT: 'test'
  ZERO_SHOT: False
  LABEL: 'eval'
  HEIGHT: 256
  WIDTH: 128
  PHASE1_ROOT_PATH: './face_attributes'
  PHASE2_ROOT_PATH: './face_attributes'
  ROOT_PATH: './face_attributes/images'
```
```bash
python train_face_analysis.py
```

### Lightweight face analysis model 

```bash
cd lightweight
# Change root path, train path, valid path to your directory
python train.py --root_path ./face_attributes/images --train_path ./face_attributes/train.csv --valid_path ./face_attributes/valid.csv
```

## 4. Inference

### Transformer face analysis model 

Download pretrained YOLOv8 model from this [link](https://drive.google.com/file/d/1140s4fia8e8b3N_v2Ea_TstBF4BUN4_v/view?usp=sharing) and move it to  ```./checkpoints ```

Download pretrained backbone model from this [link](https://drive.google.com/file/d/1B4ttu-VcXHttOLHZEDkH0468xL61be-7/view?usp=sharing) and move it to  ```./pretrained ```

Download final Transformer face analysis model from this [link](https://drive.google.com/file/d/15NFNiMTnzQqr35q51HjiL48kgJ7uW8Sw/view?usp=sharing) and move it to  ```./exp_result/HACKATHON/result/img_model/checkpoints ```

```bash
python inference.py --images_dir "your images dir" --json_dir "path of json file of public or private test" --phase "public/private"
```

### Lightweight face analysis model

Download pretrained YOLOv8 model from this [link](https://drive.google.com/file/d/1140s4fia8e8b3N_v2Ea_TstBF4BUN4_v/view?usp=sharing) and move it to  ```./checkpoints ```

Download pretrained Lightweight face analysis model from this [link](https://drive.google.com/file/d/1dBG3yAkMEk4uHW6oFQGlvWgsaJCMV-td/view?usp=sharing) and move it to  ```./lightweight```

```bash
cd lightweight
python infer.py --root_path "your images dir"
```





