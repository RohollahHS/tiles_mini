mkdir data
mkdir checkpoints
mkdir outputs

gdown '1-89C2dX4pZEazjGa3JmIJ42RpoVReY4C&confirm=True'
mv tiles_mini.zip data
unzip data/tiles_mini.zip -d data/
mv 'data/content/drive/MyDrive/datsets/tiles_mini' 'data/'
rm -r 'data/content'
rm data/tiles_mini.zip

wget 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
wget 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
mv resnet50-11ad3fa6.pth checkpoints/
mv resnet18-f37072fd.pth checkpoints/
