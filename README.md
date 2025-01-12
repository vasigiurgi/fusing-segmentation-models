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

# Bayesian Decision Fusion with Weighted Entropies

The second approach works by making some decisions based on the Bayesian output of the architectures, considering entropies thereafter to check how consistent the information is. Suppose that for a camera model, a pixel \((i,j)\), is considered with the following mass values for each class:

[m<sub>1</sub>(R) = 0.80,   m<sub>1</sub>(V) = 0.15,   m<sub>1</sub>(B) = 0.05]

In this situation, taking the decision for pixel \((i,j) = R\) (from the camera model) can be relevant, but not 100% sure because m<sub>1</sub>(R) < 1. Similarly, for a LiDAR frame, suppose a pixel with mass values accordingly:

[m<sub>2</sub>(R) = 0.55,   m<sub>2</sub>(V) = 0.25,   m<sub>2</sub>(B) = 0.20]

The decision will be the same, pixel \((i,j) = R\), which can again be relevant, but the decision tends to be riskier as m<sub>2</sub>(R) is just above 0.5. Instead of fusing probabilities directly, another way is to fuse weighted decisions by their quality, calculated from entropy. 

## Weighted Decision Fusion

In the previous example, based on m<sub>1</sub>, the early state of the decision will represent \(R\) (road) class for the camera segmentation model:

[ md<sub>1</sub>(R) = 1,   md<sub>1</sub>(V) = 0,   md<sub>1</sub>(B) = 0]

Then, according to the weight, the decision will be updated. The weight of source 1 for this pixel is calculated by the quality measure as:

w<sub>1</sub> = 1 - H(m<sub>1</sub>)/H<sup>max</sup>

where H(m<sub>1</sub>) is the entropy of m<sub>1</sub> because m<sub>1</sub> is Bayesian. (For a more general (non-probabilistic) context when working with non-Bayesian BBAs, we could use the generalized entropy for belief functions defined in [DezertEntropy](https://ieeexplore.ieee.org/document/5711937?denied=).) Therefore, H(m<sub>1</sub>) corresponds to Shannon entropy, while H<sup>max</sup> is the maximum of Shannon entropy obtained for a uniform probability mass function.

Based on m<sub>2</sub>, the \(R\) class will be decided. Therefore, the judgment based on LiDAR data is:

[md<sub>2</sub>(R) = 1,   md<sub>2</sub>(V) = 0,   md<sub>2</sub>(B) = 0]

with the weight of source 2 (LiDAR) provided by the quality:

w<sub>2</sub> = 1 - H(m<sub>2</sub>)/H<sup>max</sup>

## Final Fused Decisions

The decisions are fused by a simple weighted averaging rule as follows:

md(R) = (w<sub>1</sub> / (w<sub>1</sub> + w<sub>2</sub>)) * md<sub>1</sub>(R) + (w<sub>2</sub> / (w<sub>1</sub> + w<sub>2</sub>)) * md<sub>2</sub>(R)
md(V) = (w<sub>1</sub> / (w<sub>1</sub> + w<sub>2</sub>)) * md<sub>1</sub>(V) + (w<sub>2</sub> / (w<sub>1</sub> + w<sub>2</sub>)) * md<sub>2</sub>(V)
md(B) = (w<sub>1</sub> / (w<sub>1</sub> + w<sub>2</sub>)) * md<sub>1</sub>(B) + (w<sub>2</sub> / (w<sub>1</sub> + w<sub>2</sub>)) * md<sub>2</sub>(B)

In this simple example, _Theta = 3_ since the frame of discernment (_FoD_) has three singletons only. Therefore, w<sub>1</sub> will have a greater value than w<sub>2</sub> due to the lower entropy of H(m<sub>1</sub>). Consequently, the camera source shows greater confidence.
