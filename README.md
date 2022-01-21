# Weakly supervised surface crack localization using Pytorch

This approach uses a traditional CNN for crack classification and Grad-CAM for crack localization. The whole procedure can be found in this <a href="https://medium.com/@raju.monjurulkarim/tutorial-on-surface-crack-classification-with-visual-explanation-part-1-14542d2ea7ac"> Medium</a> article.</p> 

## Requirements
- Pytorch (>=1.9)
- torchvision (>=0.4)
- cv2
- pil
- matplotlib
<br>
To create class activation map using the grad-cam method you need to install the package simply by typing:

```shell
pip install grad-cam
```

## Getting started
At first clone this repository and install the required dependencies. <br>
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
