gdown '1-89C2dX4pZEazjGa3JmIJ42RpoVReY4C&confirm=True'
mv tiles_mini.zip data
unzip -x tiles_mini.zip


wget 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
mv resnet50-11ad3fa6.pth checkpoints/