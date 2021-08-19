Install environment following the requirement.txt
To run the code in Deeplabv3_Ensemble you also need apex and segmentation_models_pytorch and ttach.
Please install segmentation_models_pytorch and ttach with pip, 
and download source code of apex using git clone https://github.com/NVIDIA/apex,
then cd apex ; pip install -v --disable-pip-version-check --no-cache-dir ./

To repeat our result please go to code/Deeplabv3_Ensemble folder;
then decompress the data to data folder with
cd data && tar -xvf Agriculture-Vision.tar.gz
run
./ensemble_test.sh
You can edit ensemble_test.sh for your own environment.
