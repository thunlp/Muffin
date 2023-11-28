## Download UniMM-Chat Data
bash download_data.sh

## Prepare Environment
echo "Creating conda environment"
conda create -n muffin python=3.10
conda activate muffin

echo "Installing dependencies"
pip install -e .

echo "Installing flash-attention"
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# uncomment the following line if you have CUDA version < 11.6
# git checkout 757058d

MAX_JOBS=8 python setup.py install
cd ..

