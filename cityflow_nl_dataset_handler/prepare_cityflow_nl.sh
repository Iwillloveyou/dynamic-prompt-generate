#!/bin/bash

set -e  # 出错时停止

echo "=== CityFlow-NL 数据预处理开始 ==="

# 设置路径
BASE_DIR="./cityflow-nl"
DATA_DIR="$BASE_DIR/data"
IMAGES_DIR="$BASE_DIR/images"
ANNOTATIONS_DIR="$BASE_DIR/annotations"

# 创建目录
mkdir -p $IMAGES_DIR $ANNOTATIONS_DIR

# 1. 下载数据（如果还没有）
if [ ! -d "$DATA_DIR" ]; then
    echo "下载数据集..."
    cd $BASE_DIR
    git clone https://github.com/fredfung007/cityflow-nl.git temp
    mv temp/data ./
    rm -rf temp
fi

# 2. 提取视频帧
echo "提取视频帧..."
cd $DATA_DIR

# 检查是否有提取脚本
if [ -f "extract_vdo_frms.py" ]; then
    python extract_vdo_frms.py --dataset_path ./ --output_path ./
else
    # 手动提取
    for video in $(find ./train -name "*.avi" 2>/dev/null || echo ""); do
        if [ -f "$video" ]; then
            scene_cam=$(echo $video | grep -o "S[0-9]\+/c[0-9]\+" || echo "")
            if [ ! -z "$scene_cam" ]; then
                mkdir -p "../../images/$scene_cam"
                ffmpeg -i "$video" -q:v 2 -start_number 0 "../../images/$scene_cam/frame_%06d.jpg" -loglevel error
                echo "  已提取: $scene_cam"
            fi
        fi
    done
fi

# 3. 解析标注
echo "解析标注文件..."
ann_file="$ANNOTATIONS_DIR/cityflow-nl.json"
python parse_annotations.py $DATA_DIR $ann_file

# 4. 划分训练/验证集
python split_dataset.py $ann_file $ANNOTATIONS_DIR