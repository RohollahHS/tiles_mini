gdown '1-89C2dX4pZEazjGa3JmIJ42RpoVReY4C&confirm=True'
!mv tiles_mini.zip data
!rm data/tiles_mini.zip
!unzip data/tiles_mini.zip -d data/
!rm -r 'data/content'

wget 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
mv resnet50-11ad3fa6.pth checkpoints/
