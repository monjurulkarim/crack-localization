# Surface Crack Classification with Visual explanation using Pytorch

The whole procedure for Crack classification and Grad-CAM visualization can be found in this <a href="https://medium.com/@raju.monjurulkarim/tutorial-on-surface-crack-classification-with-visual-explanation-part-1-14542d2ea7ac"> Medium</a> article.</p> 

## Requirements
- Pytorch (>=1.9)
- torchvision (>=0.4)
- cv2
- pil
- matplotlib

## Getting started
To train the network:
```shell
python train.py
```
An example inference command:
```shell
python inference.py
```
To generate grad-cam visual explanation (heat maps) run the following:
```shell
python xai.py
```

## Customizing dataset
The procedure for using your own dataset is very simple. Just prepare the dataset in the same way shown inside the directory folder `data/`. You can also increase the number of classes as you need. All you need to change the final layer of the architecture.

## Sample classification Results
<div align=center>
  <img src="temp.png" alt="Visualization Demo" width="800"/>
</div>

## Sample grad-cam visualization
<div align=center>
  <img src="gradcam.png" alt="Visualization grad-cam" width="800"/>
</div>
