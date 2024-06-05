# Fusion of Semantic Segmentation Models for Vehicle Perception Tasks

# Abstract
In self-navigation problems for autonomous vehicles, the variability of environmental conditions, complex scenes with vehicles and pedestrians, and the high-dimensional or real-time nature of tasks make segmentation challenging. Sensor fusion can representatively improve performances. Thus, this work highlights a late fusion concept used for semantic segmentation tasks in such perception systems. It is based on two approaches for merging information coming from two neural networks, one trained for camera data and one for LiDAR frames. The first approach involves fusing probabilities along with calculating partial conflicts and redistributing data. The second technique focuses on making individual decisions based on sources and fusing them later with weighted Shannon entropies. The two segmentation models are trained and evaluated on a particular KITTI semantic dataset. In the realm of multi-class segmentation tasks, the two fusion techniques are compared and evaluated with illustrative examples. Intersection over union metric and quality of decision are computed to assess the performance of each methodology. 	

# Repository info
This repository represents some approaches of fusing two identical segmentation models. They are both convolutional neural network based, inspired from a cross-fusion model. One represents the neural network architecture that is trained with camera images, and the same one is used to learns features from dense map Lidar data. 

# Method 1: Bayesian PCR6+ Fusion

# Method 2: Fused Decision Obtained with Shannon Entropy

![arch_fusion_segmentation_models](https://github.com/vasigiurgi/fusing-segmentation-models/assets/49117053/dcc178fd-b369-48a6-83c3-a8305965040a)


![fusion_global](https://github.com/vasigiurgi/fusing-segmentation-models/assets/49117053/a5ffc1c9-96fe-4efa-800b-e6da6ac0e036)

# Code to be uploaded once the work is recognized representative and the writing advances in the publishing proccedure. 
