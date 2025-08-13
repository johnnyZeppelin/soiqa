pip install -U -r requirements.txt

# for vmamba
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0

