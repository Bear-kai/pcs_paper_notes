# Paper from sotawhat
search by key words

## Search keywords: Point cloud sequence

- [[arxiv 2022](https://arxiv.org/abs/2207.04673)] [`seg`] Learning **Spatial and Temporal** Variations for 4D Point Cloud Segmentation. 
    
    We design a temporal variation-aware interpolation module and a temporal voxel-point refiner to capture the temporal variation in the 4D point cloud. Specifically, our method achieves 52.5\% in mIoU (+5.5% against previous best) on the multiple scan segmentation task on SemanticKITTI, and 63.0% on SemanticPOSS (+2.8% against previous best)

- [[arxiv 2022](https://arxiv.org/abs/2205.05979)] [`det`] MPPNet: Multi-Frame Feature Intertwining with Proxy Points for 3D **Temporal Object Detection**.

    We propose a novel three-hierarchy framework with proxy points for multi-frame feature encoding and interactions. The three hierarchies conduct per-frame feature encoding, short-clip feature fusion, and whole-sequence feature aggregation, respectively. On largeWaymo Open dataset, our approach outperforms SOTA methods with large margins when applied to both short (e.g., 4-frame) and long (e.g., 16-frame) point cloud sequences.

- [[CVPR 2022](https://arxiv.org/abs/2203.11590)] [`interp`] IDEA-Net: Dynamic 3D Point Cloud Interpolation via Deep Embedding Alignment. 

    This paper investigates the problem of temporally interpolating dynamic 3D point clouds with large non-rigid deformation. We formulate the problem as estimation of point-wise trajectories (i.e., smooth curves) 

- [[arxiv 2021]()] [`rec`] Garment4D: Garment Reconstruction from Point Cloud Sequences (衣物重建-简单了解)

- [[arxiv 2021]()] [`seg`] Point Cloud Segmentation Using Sparse Temporal Local Attention. 

- [[arxiv 2021](https://arxiv.org/abs/2111.08492)] [`action-cls`]  SequentialPointNet: A strong frame-level parallel point cloud sequence network for 3D action recognition.

    we propose a strong frame-level parallel point cloud sequence network referred to as SequentialPointNet for 3D action recognition. The key to our approach is to divide the main modeling operations into frame-level units executed in parallel.

- [[arxiv 2021](https://arxiv.org/pdf/2012.10860.pdf)] [`cls`, `seg`] Anchor-Based **Spatio-Temporal** Attention 3D Convolutional Networks for Dynamic 3D Point Cloud Sequences.

    Anchor-based Spatio-Temporal Attention 3D Convolutional Neural Networks (ASTA3DCNNs) are built for classification and segmentation tasks based on the proposed ASTA3DConv and evaluated on action recognition and semantic segmentation tasks. The experiments and ablation studies on MSRAction3D and Synthia datasets demonstrate the superior performance and effectiveness of our method for dynamic 3D point cloud sequences.

- [[arxiv 2021]()] [`compres`] Real-Time Spatio-Temporal LiDAR Point Cloud Compression.(点云压缩-简单了解)

- [[arxiv 2020]()] [`pred`] Inverting the Pose Forecasting Pipeline with SPF2: Sequential Pointcloud Forecasting for Sequential Pose Forecasting.

    Instead of detecting, tracking and then forecasting the objects, we propose to first forecast 3D sensor data (e.g., point clouds with 100k points) and then detect/track objects on the predicted point cloud sequences to obtain future poses, i.e., a forecast-then-detect pipeline.

- [[arxiv 2022]()] [`compres`] Lossless Compression of Point Cloud Sequences Using Sequence Optimized CNN Models. (点云压缩-简单了解)

- [[arxiv 2022]()] [`compres`] 4DAC: Learning Attribute Compression for Dynamic Point Clouds. (点云压缩-简单了解)

- [[arxiv 2022]()] [`seg`] **Spatiotemporal** Transformer Attention Network for 3D Voxel Level Joint Segmentation and Motion Prediction in Point Cloud.

- [[arxiv 2022]()] [`track`] **PTTR**: Relational 3D Point Cloud Object Tracking with Transformer (已下载)

- [[arxiv 2021]()] [`seg`, `cls`] **Spatial-Temporal** Transformer for 3D Point Cloud Sequences.

    In this paper, we propose a novel framework named Point Spatial-Temporal Transformer (PST2) to learn spatial-temporal representations from dynamic 3D point cloud sequences. We test the effectiveness our PST2 with two different tasks on point cloud sequences, i.e., 4D semantic segmentation and 3D action recognition

- [[arxiv 2021]()] Sequential Point Cloud Prediction in Interactive Scenarios: A Survey (已下载)

- [[arxiv 2021]()] [`rep`] **Spatio-temporal** Self-Supervised Representation Learning for 3D Point Clouds. (自监督；表达学习; Song-Chun Zhu)

    STRL takes two temporally-correlated frames from a 3D point cloud sequence as the input, transforms it with the spatial data augmentation, and learns the invariant representation self-supervisedly.

- [[arxiv 2021]()] [`pred`] **Spatio-temporal** Graph-RNN for Point Cloud Prediction.

    In this paper, we propose an end-to-end learning network to predict future frames in a point cloud sequence.

## Search keywords: rgbd sequence

- [[arxiv 2021]()] [`track`] Occlusion-robust Deformable Object Tracking without Physics Simulation. 

    we propose an occlusion-robust RGBD sequence tracking framework based on Coherent Point Drift (CPD). To mitigate the effects of occlusion, our method 1) Uses a combination of locally linear embedding and constrained optimization to regularize the output of CPD, thus enforcing topological consistency when occlusions create disconnected pieces of the object; 2) Reasons about the free-space visible by an RGBD sensor to better estimate the prior on point location and to detect tracking failures during occlusion; and 3) Uses shape descriptors to find the most relevant previous state of the object to use for tracking after a severe occlusion.

- [[arxiv 2021]()] [`rec`] CurveFusion: Reconstructing Thin Structures from RGBD Sequences.

- [[arxiv 2017]()] [`track`] Deep 6-DOF Tracking.

## Search keywords: rgbd video 
略看
- [[arxiv 2022]()] [] Learning Dynamic View Synthesis With Few RGBD Cameras.

    The dataset consists of 43 multi-view RGBD video sequences of everyday activities, capturing complex interactions between human subjects and their surroundings.

- [[arxiv 2021]()] [] Robust Event Detection based on Spatio-Temporal Latent Action Unit using Skeletal Information.

- [[arxiv 2012]()] [] Tracking Revisited using RGBD Camera: Baseline and Benchmark. (by Shuran Song，略看)

- [[arxiv 2021]()] [] IKEA Object State Dataset: A 6DoF object pose estimation dataset and benchmark for multi-state assembly objects. (关注下)

    We present IKEA Object State Dataset, a new dataset that contains IKEA furniture 3D models, RGBD video of the assembly process, the 6DoF pose of furniture parts and their bounding box.

## Search keywords: pose tracking 

- [[arxiv 2021]()] [`track`] **ROFT**: Real-Time Optical Flow-Aided 6D Object Pose and Velocity Tracking.

- [[arxiv 2022]()] [`track`] A Wide-area, Low-latency, and Power-efficient 6-DoF Pose Tracking System for Rigid Objects.

- [[arxiv 2021]()] [`track`] Self-supervised Keypoint Correspondences for Multi-Person Pose Estimation and Tracking in Videos

- [[arxiv 2020]()] [`track`] Multi-person Pose Tracking using Sequential Monte Carlo with Probabilistic Neural Pose Predictor.

- [[arxiv 2019]()] [`track`] FastPose: Towards Real-time Pose Estimation and Tracking via Scale-normalized Multi-task Networks.
    
    This paper addresses the task of articulated multi-person pose estimation and tracking towards real-time speed. 

- [[arxiv 2019]()] [`track`] Efficient Circle-Based Camera Pose Tracking Free of PnP.

- [[arxiv 2019]()] [`track`] Multi-person Articulated Tracking with Spatial and Temporal Embeddings.

- [[arxiv 2019]()] [`track`] Human Pose Estimation using Motion Priors and Ensemble Models.

- [[arxiv 2019]()] [`track`] A Top-down Approach to Articulated Human Pose Estimation and Tracking.

    We take part in two ECCV 18 PoseTrack challenges: pose estimation and pose tracking.

- [[arxiv 2019]()] [`track`] Explicit Spatiotemporal Joint Relation Learning for Tracking Human Pose.

    By explicitly learning and exploiting these joint relationships, our system achieves state-of-the-art performance on standard benchmarks for various pose tracking tasks including 3D body pose tracking in RGB video, 3D hand pose tracking in depth sequences, and 3D hand gesture tracking in RGB video.

- [[arxiv 2018]()] [`track`] Joint Flow: Temporal Flow Fields for Multi Person Tracking.

# Paper from Arxiv 
作为sotawhat补充

- [[arxiv 2022](https://arxiv.org/abs/2203.16482)] [`rec`, `flow`] **RFNet-4D**: Joint Object Reconstruction and Flow Estimation from 4D Point Clouds.

    To prove this ability, we design a temporal vector field learning module using an unsupervised learning approach for flow estimation, leveraged by supervised learning of spatial structures for object reconstruction. 

- [[CVPR 2022](https://arxiv.org/abs/2203.13394)] [`det`] **Point2Seq**: Detecting 3D Objects as Sequences.

    We view each 3D object as a sequence of words and reformulate the 3D object detection task as decoding words from 3D scenes in an auto-regressive manner. 

- [[CVPR 2022](https://arxiv.org/abs/2203.11113)] [`det`] **No Pain, Big Gain**: Classify Dynamic Point Cloud Sequences with Static Models by Fitting Feature-level Space-time Surfaces.

    To capture 3D motions without explicitly tracking correspondences, we propose a kinematics-inspired neural network (Kinet) by generalizing the kinematic concept of ST-surfaces to the feature space. 

- [[CVPR 2022](https://arxiv.org/abs/2203.10314)] [`det`] **Voxel Set Transformer**: A Set-to-Set Approach to 3D Object Detection from Point Clouds.

    We propose a novel voxel-based architecture, namely Voxel Set Transformer (VoxSeT), to detect 3D objects from point clouds. It can be used as a good alternative to the convolutional and point-based backbones. VoxSeT reports competitive results on the KITTI and Waymo detection benchmarks. 

- [[arxiv 2022](https://arxiv.org/abs/2203.00138)] [`seg`, `pred`] **Spatiotemporal** Transformer Attention Network for 3D Voxel Level Joint Segmentation and Motion Prediction in Point Cloud.

    We propose a novel spatiotemporal attention network based on a transformer self-attention mechanism for joint semantic segmentation and motion prediction within a point cloud at the voxel level. The network is trained to simultaneously outputs the voxel level class and predicted motion by learning directly from a sequence of point cloud datasets.

- [[arxiv 2022](https://arxiv.org/abs/2202.13377)] [`seg`] **Meta-RangeSeg**: LiDAR Sequence Semantic Segmentation Using Multiple Feature Aggregation.

    We propose a novel approach to semantic segmentation for LiDAR sequences named Meta-RangeSeg, where a novel range residual image representation is introduced to capture the spatial-temporal information.

- [[ICRA 2022](https://arxiv.org/abs/2202.03084)] [`complet`] **Temporal Point Cloud Completion** with Pose Disturbance.

    With the help of gated recovery units(GRU) and attention mechanisms as temporal units, we propose a point cloud completion framework that accepts a sequence of unaligned and sparse inputs, and outputs consistent and aligned point clouds. 

- [[arxiv 2022](https://arxiv.org/abs/2201.10326)] [`complet`] **ShapeFormer**: Transformer-based Shape Completion via Sparse Representation

    We present ShapeFormer, a transformer-based network that produces a distribution of object completions, conditioned on incomplete, and possibly noisy, point clouds. The resultant distribution can then be sampled to generate likely completions, each exhibiting plausible shape details while being faithful to the input.

- [[arxiv 2022](https://arxiv.org/abs/2201.06304)] [`action-cls`] Action Keypoint Network for Efficient Video Recognition. (勉强相关)

    This paper proposes to integrate temporal and spatial selection into an Action Keypoint Network (AK-Net). From different frames and positions, AK-Net selects some informative points scattered in arbitrary-shaped regions as a set of action keypoints and then transforms the video recognition into point cloud classification.

- [[arxiv 2021](https://arxiv.org/abs/2111.08755)] [`flow`, `pred`] **Learning Scene Dynamics from Point Cloud Sequences**.

    we propose a novel problem -- sequential scene flow estimation (SSFE) -- that aims to predict 3D scene flow for all pairs of point clouds in a given sequence. This is unlike the previously studied problem of scene flow estimation which focuses on two frames. This approach can be effectively modified for sequential point cloud forecasting (SPF).

- [[arxiv 2021](https://arxiv.org/abs/2111.08492)] [`action-cls`]  
    

# Paper from Github

- Awesome Object Pose Estimation and Reconstruction [[link](https://github.com/ZhongqunZHANG/awesome-6d-object)] （主要是单帧pose或跟踪；不筛选了，抽空顺着扫读）

- awesome-point-cloud-analysis [[link](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2022)] （按新旧顺序扫读，先2022-->2021-->...）

    先挑出sequence相关:
    - ASAP-Net: Attention and Structure Aware Point Cloud Sequence Segmentation.
    - Unsupervised Sequence Forecasting of 100,000 Points for Unsupervised Trajectory Forecasting.
    - Any Motion Detector: Learning Class-agnostic Scene Dynamics from a Sequence of LiDAR Point Clouds.
    - PointMotionNet: Point-Wise Motion Learning for Large-Scale LiDAR Point Clouds Sequences

- Vision-based Robotic Grasping: Papers and Codes [[link](https://github.com/GeorgeDu/vision-based-robotic-grasping)]

    + Self-Supervised Learning of Part Mobility from Point Cloud Sequence
    + Seeing Behind Objects for 3D Multi-Object Tracking in RGB-D Sequences

- awesome-Automanous-3D-detection-methods [[link](https://github.com/tyjiang1997/awesome-Automanous-3D-detection-methods)]
