# Tunnel Length Detection From Traffic Signs

## Project Description and Objective

The goal of this project is to detect tunnel signs and extract the tunnel length written in it.

## Core features

- A framework for training and evaluating object detection models.
- An add-on feature that runs OCR on detected object

## Deliverables

- A containerized python CLI for launching model training or prediction 
- A collection of pre-trained model checkpoints (to use for prediction)

## Solution Architecture

- Modular framework that uses TensorFlow Object Detection API for loading models and training them
- The framework will use config files to run the experiments
- OCR feature that can be activated/deactivated from the configs

## Download the data

- Download the data from [here](https://drive.google.com/drive/folders/1_GNqpPk47ABSVBV_fWOp-M4QZ6Qh_OsL?usp=sharing) and set your paths in configs/deliver/configs.yaml file.

## Create a new experiment and run the training
First, build the docker image for this project:
- CPU image:
```
docker build -t tensorflow_object_detection:latest -f ./docker/Dockerfile .
```
- GPU image:
```
docker build -t tensorflow_object_detection:latest -f ./docker/Dockerfile.gpu .
```
To run the docker container use this command:
```
docker run -it -v $PWD:/computer_vision_uc -w /computer_vision_uc --name computer_vision_uc tensorflow_object_detection:latest
```

**NB:** 
- For windows users, change `$PWD` with an absolute path.
- Add `--gpus all` if you have access to gpus.

Once gone inside the running container, use the following command to create a new experiment and run the corresponding training after choosing one of the **pre-trained models** listed below:

- SSD EfficientDet D0 512x512
- SSD EfficientDet D1 640x640
- SSD EfficientDet D2 768x768
- SSD EfficientDet D3 896x896
- SSD EfficientDet D4 1024x1024
- SSD EfficientDet D5 1280x1280
- SSD EfficientDet D6 1280x1280
- SSD EfficientDet D7 1536x1536
- SSD MobileNet V1 FPN 640x640
- SSD MobileNet V2 FPNLite 320x320
- SSD MobileNet V2 FPNLite 640x640
- SSD ResNet50 V1 FPN 640x640 (RetinaNet50)
- SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)
- SSD ResNet101 V1 FPN 640x640 (RetinaNet101)
- SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)
- SSD ResNet152 V1 FPN 640x640 (RetinaNet152)
- SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)

```
python execute.py --experiment_name="test_exp" --experiments_dir="experiment_test" --config_file="/computer_vision_uc/configs/deliver/config.yaml" --mode="train" --model="SSD MobileNet V2 FPNLite 320x320" --visualize=False
```
where:
* **experiment name**: the name of the experiment you want to create.
* **experiments_dir**: the root folder of all the experiments (created if it doesn't exist).
* **config_file**: path of your config file.
* **mode**: the mode you want to run (train in our case).
* **model**: the full model name for the model that you want to download and use.
* **visualize**: this argument is optional but set it to True if you want to visualize the split of the used data according to to the specified mode.



Running the command above without the model argument will display a list of available pre-trained models from which one has to be chosen to be fine tuned with our data. The checkpoints of the pre-trained model will be automatically downloaded and stored in the /tmp directory.

Once the fine tuning is done, you should find a new folder **"experiment_test"** which is the folder that will contain our experiments. So inside that folder another folder is created with the name of the experiment **"test_exp"** in which you will find:
* **logs**: the folder that contains the tensorboard logs.
* **model_checkpoints**: the folder that contains the checkpoints of the output model.
* **output_images**: this is where the model's output images will be saved after inference.
* **config.yaml**: the config file.

## Evaluate the trained model from a checkpoint
To evaluate the trained model's performance with new data use the following command:
```
python execute.py --experiment_path="the path of the fine-tuned experiment" --image_dir="the directory containing your evaluation data" --annot_file_path="the path to your evaluation annotations"--mode="eval" --iou_threshold=0.5 --score_threshold=0.5
```
where:
* **experiment_path**: the path of the fine-tuned experiment.
* **image_dir**: the test image directory path.
* **annot_file_path**: the path of the annotation file for evaluation.
* **mode**: the mode you want to run (eval in our case).
* **iou_threshold**: the intersection over union thresold.
* **score_threshold**: the score threshold to filter the valid bounding boxes.

**NB:** 
- You can lower the score_threshold if you haven't trained your model on enough number of batches to select more predicted bounding boxes.

## Run the inference pipeline from a checkpoint on new data
To execute the inference pipeline on new data use the following command:
```
python execute.py --experiment_path="the path of the fine-tuned experiment" --image_dir="the directory containing your inference data" --mode="inference" --score_threshold=0.5
```
where:
* **experiment_path**: the path of the fine-tuned experiment.
* **image_dir**: the test image directory path.
* **mode**: the mode you want to run (inference in our case).
* **score_threshold**: the score threshold to filter the valid bounding boxes.

Once the inference is done, a file called **"output.csv"**  containing the bounding boxes predictions and the OCR predictions for each photo will be saved in the experiment's folder, and the output images will be saved in the **output_images** folder in the experiment's directory. 

## Serve the model
To serve the model, you need to build tensorflow_object_detection image as it is explained above. Than you need to create the serving docker image with the following command:
```
docker build -t detection_server:latest -f ./docker/Dockerfile.server .
```
Once the image is built, run the following command:
```
docker run -it -v $PWD:/computer_vision_uc -w /computer_vision_uc --name server -p 5000:5000 --expose 5000 detection_server:latest serving/config.yaml "experiment_path"
```
where:
* **experiment_path**: the path of the fine-tuned experiment.

To simulate the client and send images to the server run the following command:
```
python -m serving.client --image_dir="image_dir" --config_file=serving/config.yaml --score_threshold=0.5 --result_folder_path='result_folder_path'
```
where:
* **image_dir**: the test image directory path.
* **result_folder_path**: optional parameter. If filled the server will return the image with the detections and the client will save them in the desired path.


## Train a customized PaddleOCR model
To finetune pretrained models (hosted by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)), please follow the following steps.

### Prepare a Custom dataset

To use your own data for training, please refer to the following to organize your data.

- Training set

It is recommended to put the training images in the same folder, and use a txt file (rec_gt_train.txt) to store the image path and label. The contents of the txt file are as follows:

* Note: by default, the image path and image label are split with \t, if you use other methods to split, it will cause training error

```
" Image file name           Image annotation "

train_data/rec/train/sign_001.jpg   1.1km
train_data/rec/train/sign_002.jpg   1770m
...
```

The final training set should have the following file structure:

```
|-train_data
    |- rec_gt_train.txt
    |- train
        |- sign_001.png
        |- sign_002.jpg
        |- sign_003.jpg
        | ...
```

- Test set

Similar to the training set, the test set also needs to be provided a folder containing all images (test) and a rec_gt_test.txt. The structure of the test set is as follows:

```
|-train_data
    |- rec_gt_test.txt
    |- test
        |- sign_001.png
        |- sign_002.jpg
        |- sign_003.jpg
        | ...
```

### Dictionary

Finally, a dictionary ({word_dict_name}.txt) needs to be provided so that when the model is trained, all the characters that appear can be mapped to the dictionary index.

Therefore, the dictionary needs to contain all the characters that you want to be recognized correctly. {word_dict_name}.txt needs to be written in the following format and saved in the `utf-8` encoding format:

```
0
1
2
3
4
5
6
7
8
9
k
m
,
```

In `{word_dict_name}.txt`, there is a single word in each line, which maps characters and numeric indexes together, e.g "1,1km" will be mapped to [1 12 1 10 11]

### Train Script
After preparing the data, please run the following script:

```
cd external/train_paddleocr
bash train.sh </full/path/to/created/dataset> <full/path/of/{word_dict_name}.txt>
```
### Use the trained model for inference

After the bash script runs successfully, there are three files in the model save directory:
```
./external/train_paddleocr/PaddleOCR/inference/model/
    ├── inference.pdiparams         # The parameter file of recognition inference model
    ├── inference.pdiparams.info    # The parameter information of recognition inference model, which can be ignored
    └── inference.pdmodel           # The program file of recognition model
```

To use this model, you should use this path in the config-file as: 

```
paddle_pretrained="./external/train_paddleocr/PaddleOCR/inference/model/"
```