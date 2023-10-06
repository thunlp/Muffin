## Download UniMM-Chat Data
bash download_data.sh

## Prepare Environment
echo "Creating conda environment"
conda create -n muffin python=3.10
conda activate muffin

echo "Installing dependencies"
pip install transformers
pip install openai
pip install timm torchscale
pip install opencv-python
pip install protobuf sentencepiece

echo "Installing flash-attention"
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=8 python setup.py install
cd ..

