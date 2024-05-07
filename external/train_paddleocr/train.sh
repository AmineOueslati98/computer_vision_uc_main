#!/bin/bash
# train.bash - Demo script to train a customized PaddleOCR Model
# ------------------------------------------------------

#Global Variables
ocr_data=$1
dict=$2
#"paddledb.zip"
config_file="configs/rec/rec_icdar15_train.yml"

#Save Current directory
cwd=$(pwd)

#Check if the paddleocr repo exists; else clone it !
if [ -d "PaddleOCR" ] 
then
    echo "PaddleOCR Git Already cloned!" 
else
    echo "Cloning PaddleOCR repo:"
    git clone https://github.com/PaddlePaddle/PaddleOCR
fi
#Change directory to the PaddleOCR
cd PaddleOCR

#DATA PREPARATION
#Create data folders as required
if [ -d "train_data" ] 
then
    rm train_data 
fi

ln -sf $ocr_data ./train_data

#Unzip ocr_data folder
#unzip $cwd/$ocr_data -d train_data


#PATCH CONFIG FILE
export YAML_PATH=$config_file
export DICT_FILE=$dict


python << END
import yaml
import os

with open(os.environ['YAML_PATH'], "r") as file:
    parameters = yaml.safe_load(file)

parameters["Global"]["save_model_dir"]="./output/rec/sign/"
parameters["Global"]["character_dict_path"]=os.environ['DICT_FILE']
parameters["Global"]["character_type"]="en"
parameters["Global"]["save_res_path"]="./output/rec/predicts_sign.txt"
parameters["Global"]["eval_batch_step"]=[0, 1900]
parameters["Global"]["epoch_num"]=3

parameters["Train"]["dataset"]["data_dir"]="./train_data/"
parameters["Train"]["dataset"]["label_file_list"]=["./train_data/rec_gt_train.txt"]
parameters["Train"]["loader"]["batch_size_per_card"]=4

parameters["Eval"]["dataset"]["data_dir"]="./train_data/"
parameters["Eval"]["dataset"]["label_file_list"]=["./train_data/rec_gt_test.txt"]
parameters["Eval"]["loader"]["batch_size_per_card"]=4

with open(os.environ['YAML_PATH'], "w") as f:
    yaml.dump(parameters, f)
END
#Install Requirements
pip install -r requirements.txt
#Install PaddlePaddle-GPU
python -m pip install paddlepaddle-gpu==2.1.0.post112 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

#Download Pretrained Model If it does not exist
if [ -d "pretrain_models/rec_mv3_none_bilstm_ctc_v2.0_train" ] 
then
    echo "pretrain_models Already downloaded!" 
else
    wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar
    # Decompress model parameters
    cd pretrain_models
    tar -xf rec_mv3_none_bilstm_ctc_v2.0_train.tar && rm -rf rec_mv3_none_bilstm_ctc_v2.0_train.tar
    cd ..
fi

#Train Model
python  tools/train.py  -c configs/rec/rec_icdar15_train.yml

#Export Model
python tools/export_model.py -c configs/rec/rec_icdar15_train.yml -o Global.pretrained_model=./output/rec/sign/best_accuracy  Global.save_inference_dir=./inference/model/
cp os.environ['DICT_FILE'] ./inference/model/dict.txt