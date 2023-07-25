# Object Pose Estimation

## Instance Level

### Year 2023

- **MV-Keypoints** Alan Li (University of Toronto), Multi-View Keypoints for Reliable 6D Object Pose Estimation. [[ICRA 2023]](https://arxiv.org/pdf/2303.16833.pdf)
    - 基于PVNet的多视图关键点_细节不清_测ROBI数据集
    - 相关 **MV6D** Fabian Duffhauss (Bosch), MV6D: Multi-View 6D Pose Estimation on RGB-D Frames
Using a Deep Point-wise Voting Network. [[IROS 2022]](https://arxiv.org/pdf/2208.01172.pdf)[cite 2]


- **CheckerPose** Ruyi Lian (美国石溪大学), CheckerPose: Progressive Dense Keypoint Localization for Object Pose Estimation with Graph Neural Network. [[arxiv 2023]](https://arxiv.org/abs/2303.16874)


- **TexPose** Hanzhi Chen, Fabian Manhardt (TUM), TexPose: Neural Texture Learning for Self-Supervised 6D Object Pose Estimation. [[CVPR 2023](https://arxiv.org/pdf/2212.12902.pdf)]


- **---** xxx (), xxx [[]]()


### Year 2022

- **`suo-slam`** Nathaniel Merrill (特拉华大学&TUM), **S**ymmetry and **U**ncertainty-Aware **O**bject **SLAM** for 6DoF Object Pose Estimation. [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Merrill_Symmetry_and_Uncertainty-Aware_Object_SLAM_for_6DoF_Object_Pose_Estimation_CVPR_2022_paper.pdf)] [[github](https://github.com/rpng/suo_slam)] [cite 6]


- **RNNPose** Yan Xu (CUHK), RNNPose: Recurrent 6-DoF Object Pose Refinement with Robust Correspondence Field Estimation and Pose Optimization. [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_RNNPose_Recurrent_6-DoF_Object_Pose_Refinement_With_Robust_Correspondence_Field_CVPR_2022_paper.pdf)] [[github](https://github.com/DecaYale/RNNPose)] [cite 2]


- **OSOP** Ivan Shugurov (慕尼黑工大), OSOP: A Multi-Stage One Shot Object Pose Estimation Framework [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Shugurov_OSOP_A_Multi-Stage_One_Shot_Object_Pose_Estimation_Framework_CVPR_2022_paper.pdf)] 


- **ZebraPose** Yongzhi Su (DFKI), ZebraPose: Coarse to Fine Surface Encoding for 6DoF Object Pose Estimation. [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Su_ZebraPose_Coarse_To_Fine_Surface_Encoding_for_6DoF_Object_Pose_CVPR_2022_paper.pdf)] [[github]( https://github.com/suyz526/ZebraPose)] [cite 1]


- **SurfEmb** SurfEmb: Dense and Continuous Correspondence Distributions
for Object Pose Estimation with Learnt Surface Embeddings. [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Haugaard_SurfEmb_Dense_and_Continuous_Correspondence_Distributions_for_Object_Pose_Estimation_CVPR_2022_paper.pdf)] [[github](https://surfemb.github.io/)] [cite 3]
    - 相关 **EPOS**： Tomas Hodan (捷克理工), EPOS: Estimating 6D Pose of Objects with Symmetries. [[CVPR 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hodan_EPOS_Estimating_6D_Pose_of_Objects_With_Symmetries_CVPR_2020_paper.pdf)[[page]](http://cmp.felk.cvut.cz/epos/)[cite 174]


- **RCV-Pose** Yangzheng Wu (Queen’s University), Vote from the Center: 6 DoF Pose Estimation in RGB-D Images by Radial Keypoint Voting. [[ECCV 2022](https://arxiv.org/pdf/2104.02527.pdf)] [cite 6]
    - 关键点投票，可看作是KDFNet的3D版本。
    - **KDFNet** Xingyu Liu (CMU), KDFNet: Learning Keypoint Distance Field for 6D Object Pose Estimation. [[IROS 2021](https://arxiv.org/pdf/2109.10127)] [cite 1]


- **YOLOPose** YOLOPose: Transformer-based Multi-Object 6D Pose Estimation using Keypoint Regression. [[arxiv 2022](https://arxiv.org/pdf/2205.02536)]


- **OVE6D** Dingding Cai (Tampere University), OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation. [[CVPR 2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_OVE6D_Object_Viewpoint_Encoding_for_Depth-Based_6D_Object_Pose_Estimation_CVPR_2022_paper.pdf)[[github]](https://github.com/dingdingcai/OVE6D-pose)[cite 9]

- **SC6D** Dingding Cai (Tampere University), SC6D: Symmetry-agnostic and Correspondence-free
6D Object Pose Estimation. [[3DV 2022](https://arxiv.org/pdf/2208.02129.pdf)]


- **ROPE** Bo Chen (University of Adelaide), Occlusion-Robust Object Pose Estimation with Holistic Representation. [[WACV 2022]](https://openaccess.thecvf.com/content/WACV2022/papers/Chen_Occlusion-Robust_Object_Pose_Estimation_With_Holistic_Representation_WACV_2022_paper.pdf)[[github]](http://github.com/BoChenYS/ROPE)[cite 7]


- **MegaPose** Yann Labbe (Inria), Lucas Manuelli (Nvidia), MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare. [[arxiv 2022]](https://arxiv.org/pdf/2212.06870.pdf)[[github]](https://megapose6d.github.io/)[cite 11]


### Year 2021 and before


- **`TemporalFusion`** Fengjun Mu (中科大), TemporalFusion: Temporal Motion Reasoning with Multi-Frame Fusion for 6D Object Pose Estimation. [[IROS 2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9636583)] [[github](https://github.com/mufengjun260/TemporalFusion21)] [cite 0]
    - **扩展版**: Rui Huang (中科大), Estimating 6D Object Poses with Temporal Motion Reasoning for Robot Grasping in Cluttered Scenes. [[RAL 2022](https://ieeexplore.ieee.org/abstract/document/9699040/)] [[github](https://github.com/mufengjun260/H-MPose)] [cite 0]


- **VideoPose** Apoorva Beedu (佐治亚理工), VideoPose: Estimating 6D object pose from videos. [[arxiv 2021](https://arxiv.org/abs/2111.10677)]


- **SO-Pose** Yan Di (TUM), Fabian Manhardt2 (Google), Gu Wang, Xiangyang Ji (Tsinghua), SO-Pose: Exploiting Self-Occlusion for Direct 6D Pose Estimation. [[ICCV 2021]]()[cite 54]

- **InstancePose** Lee Aing (台湾中正大学), InstancePose: Fast 6DoF Pose Estimation for Multiple Objects from a Single RGB Image. [[ICCV-W 2021]](https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/papers/Aing_InstancePose_Fast_6DoF_Pose_Estimation_for_Multiple_Objects_From_a_ICCVW_2021_paper.pdf)[cite 3]


- **GDR-Net** Gu Wang (Tsinghua), Fabian Manhardt (TUM), GDR-Net: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation. [[CVPR 2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_GDR-Net_Geometry-Guided_Direct_Regression_Network_for_Monocular_6D_Object_Pose_CVPR_2021_paper.pdf)[cite 160]


- **Morefusion** Kentaro Wada (帝国理工), Morefusion: Multi-object reasoning for 6d pose estimation from volumetric fusion, [[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wada_MoreFusion_Multi-object_Reasoning_for_6D_Pose_Estimation_from_Volumetric_Fusion_CVPR_2020_paper.pdf)] [cite 48]


- **PVN3D** Yisheng He (港科大), Wei Sun (旷视) Haibin Huang (快手), PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose
Estimation. [[CVPR 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_PVN3D_A_Deep_Point-Wise_3D_Keypoints_Voting_Network_for_6DoF_CVPR_2020_paper.pdf)[[github]](https://github.com/ethnhe/PVN3D.git)[cite 317]


- **G2L-Net** Wei Chen (国防科大&伯明翰), G2L-Net: Global to Local Network for Real-time 6D Pose Estimation with Embedding Vector Features. [[CVPR 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_G2L-Net_Global_to_Local_Network_for_Real-Time_6D_Pose_Estimation_CVPR_2020_paper.pdf)[[github]](https://github.com/DC1991/G2L_Net)[cite 63]

- **PointPoseNet** Wei Chen (国防科大&伯明翰), PointPoseNet: Point Pose Network for Robust 6D Object Pose Estimation. [[WACV 2020]](https://openaccess.thecvf.com/content_WACV_2020/papers/Chen_PonitPoseNet_Point_Pose_Network_for_Robust_6D_Object_Pose_Estimation_WACV_2020_paper.pdf)[cite 27]


- **Cosypose** Cosypose: Consistent multi-view multi-object 6d pose estimation. [[ECCV 2020](https://arxiv.org/pdf/2008.08465)] [[page](https://www.di.ens.fr/willow/research/cosypose/)] [cite 133]


- **Self6D** Gu Wang (Tsinghua), Fabian Manhardt (TUM), Self6D: Self-Supervised Monocular 6D Object Pose Estimation. [[ECCV 2020]](https://arxiv.org/pdf/2004.06468.pdf)[cite 97]
    - 扩展篇Self6D++：Occlusion-Aware Self-Supervised Monocular 6D Object Pose Estimation. [[TPAMI 2021]](https://arxiv.org/pdf/2203.10339.pdf)[cite 15]


- **PPR-Net** Zhikai Dong (商汤&清华), PPR-Net:Point-wise Pose Regression Network for Instance Segmentation and 6D Pose Estimation in Bin-picking Scenarios. [[IROS 2019](https://ieeexplore.ieee.org/abstract/document/8967895/)] [[github](https://github.com/lvwj19/PPR-Net-plus)] [cite 26]


- **PVNet** Sida Peng, Xiaowei Zhou (浙大CAD), PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation. [[CVPR 2019]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Peng_PVNet_Pixel-Wise_Voting_Network_for_6DoF_Pose_Estimation_CVPR_2019_paper.pdf)[[github]](https://zju3dv.github.io/pvnet/)[cite 782]


- **DPOD** Sergey Zakharov (TUM), DPOD: 6D Pose Object Detector and Refiner. [[ICCV 2019]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zakharov_DPOD_6D_Pose_Object_Detector_and_Refiner_ICCV_2019_paper.pdf)[cite 346]
    - 扩展篇: Ivan Shugurov, Sergey Zakharov (TUM), DPODv2: Dense Correspondence-Based 6 DoF Pose Estimation.[[TPAMI 2021]](https://arxiv.org/pdf/2207.02805.pdf)[cite 17]


- **CDPN** Zhigang Li, Gu Wang, Xiangyang Ji (Tsinghua), CDPN: Coordinates-Based Disentangled Pose Network for Real-Time RGB-Based 6-DoF Object Pose Estimation. [[ICCV 2019]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_CDPN_Coordinates-Based_Disentangled_Pose_Network_for_Real-Time_RGB-Based_6-DoF_Object_ICCV_2019_paper.pdf)[cite 292]


- **PoseCNN** Yu Xiang (University of Washington), Tanner Schmidt (Nvidia), PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes [[RSS 2018]](https://arxiv.org/pdf/1711.00199.pdf)[[page]](https://rse-lab.cs.washington.edu/projects/posecnn/)[cite 1559]


- **AAE** Martin Sundermeyer (German Aerospace Center), Implicit 3D Orientation Learning for 6D Object Detection from RGB Images. [[ECCV 2018 **Best Paper**]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Martin_Sundermeyer_Implicit_3D_Orientation_ECCV_2018_paper.pdf)[cite 550]


- **DOPE** Jonathan Tremblay, Yu Xiang, Dieter Fox (Nvidia), Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects. [[CoRL 2018]](https://arxiv.org/pdf/1809.10790.pdf)[[page]](https://research.nvidia.com/publication/2018-09_Deep-Object-Pose)[cite 606]
    - The first deep network trained only on synthetic data.
    - 域泛化（Domain randomization）


- TCLCH, Real-Time Monocular Pose Estimation of 3D Objects using Temporally Consistent Local Color Histograms. [[ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tjaden_Real-Time_Monocular_Pose_ICCV_2017_paper.pdf)] [cite 90]
（传统方法，RBOT的三篇论文之一）


- **SSD-6D** Wadim Kehl, Fabian Manhardt (TUM), SSD-6D: Making RGB-Based 3D Detection and 6D Pose Estimation Great Again. [[ICCV 2017]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Kehl_SSD-6D_Making_RGB-Based_ICCV_2017_paper.pdf)[[github]](https://wadimkehl.github.io/)[cite 928]


## Category Level

### Year 2023

- **IST-Net** Jianhui Liu (CUHK), Prior-free Category-level Pose Estimation with Implicit Space Transformation.[[arxiv 2023]](https://arxiv.org/pdf/2303.13479.pdf)


- **SLO-LocNet** Junyi Wang (北航|鹏城), Simultaneous Scene-independent Camera Localization and Category-level Object Pose Estimation via Multi-level Feature Fusion. [[VR 2023]](https://ieeexplore.ieee.org/abstract/document/10108437)


- **Self-Pose** Kaifeng Zhang, 王小龙组 (UCSD), Self-Supervised Geometric Correspondence for Category-Level 6D Object Pose Estimation in the Wild. [[ICLR 2023]](https://arxiv.org/pdf/2210.07199.pdf)[[github]](https://kywind.github.io/self-pose)[cite 5]
    - 相关 **RePoNet** Yang Fu, 王小龙组 (UCSD), Category-Level 6D Object Pose Estimation in the Wild: A Semi-Supervised Learning Approach and
    A New Dataset. [[NIPS 2022]](https://proceedings.neurips.cc/paper_files/paper/2022/file/afe99e55be23b3523818da1fefa33494-Paper-Conference.pdf)[[github]](https://oasisyang.github.io/semi-pose/)[cite 6]


- **GCASP** Guanglin Li (浙大CAD), Generative Category-Level Shape and Pose Estimation with Semantic Primitives. [[PMLR 2023]](https://proceedings.mlr.press/v205/li23d/li23d.pdf) [[github]](https://zju3dv.github.io/gCasp) [cite 4]


- **---** xxx (), xxx [[]]()


### Year 2022

- **GPV-Pose** Yan Di (慕尼黑工大), GPV-Pose: Category-level Object Pose Estimation via Geometry-guided Point-wise Voting. [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Di_GPV-Pose_Category-Level_Object_Pose_Estimation_via_Geometry-Guided_Point-Wise_Voting_CVPR_2022_paper.pdf)] [[github](https://github.com/lolrudy/GPV_Pose)] [cite 24]


- **Gen6D** Gen6D: Generalizable Model-Free 6-DoF Object Pose Estimation from RGB Images [[arxiv 2022](https://arxiv.org/pdf/2204.10776.pdf)] [[github](https://liuyuan-pal.github.io/Gen6D/)]


- **FS6D** Yisheng He (HKUST & 旷视), FS6D: Few-Shot 6D Pose Estimation of Novel Objects. [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/He_FS6D_Few-Shot_6D_Pose_Estimation_of_Novel_Objects_CVPR_2022_paper.pdf)] [[github](https://fs6d.github.io/)] [cite 1]


- **UDA-COPE** Taeyeop Lee (KAIST), UDA-COPE: Unsupervised Domain Adaptation for Category-level Object Pose Estimation. [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_UDA-COPE_Unsupervised_Domain_Adaptation_for_Category-Level_Object_Pose_Estimation_CVPR_2022_paper.pdf)] [cite 1]


- **OnePose** Jiaming Sun (浙大&商汤), OnePose: One-Shot Object Pose Estimation without CAD Models. [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_OnePose_One-Shot_Object_Pose_Estimation_Without_CAD_Models_CVPR_2022_paper.pdf)] [[github](https://zju3dv.github.io/onepose/)] 
    - 扩展篇 OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models. [[arxiv 2023](https://arxiv.org/pdf/2301.07673.pdf)]


- **TemplatePose** Van Nguyen Nguyen (CNRS, France), Templates for 3D Object Pose Estimation Revisited: Generalization to New Objects and Robustness to Occlusions. [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Nguyen_Templates_for_3D_Object_Pose_Estimation_Revisited_Generalization_to_New_CVPR_2022_paper.pdf)] [[github](https://github.com/nv-nguyen/template-pose)] [cite 2]


- **SAR-Net** Haitao Lin (复旦付彦伟组), SAR-Net: Shape Alignment and Recovery Network for Category-level 6D Object Pose and Size Estimation. [[CVPR 2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_SAR-Net_Shape_Alignment_and_Recovery_Network_for_Category-Level_6D_Object_CVPR_2022_paper.pdf)[[github]](https://hetolin.github.io/SAR-Net)[cite 15]


- **CenterSnap** Muhammad Zubair Irshad (乔治亚理工 & 丰田研究院), CenterSnap: Single-Shot Multi-Object 3D Shape Reconstruction and Categorical 6D Pose and Size Estimation. [[ICRA 2022](https://arxiv.org/pdf/2203.01929)] [[github](https://github.com/zubair-irshad/CenterSnap)] [cite 2]



- **OLD-Net** Zhaoxin Fan (人大), Object level depth reconstruction for category level 6d object pose estimation from monocular RGB image. [[ECCV 2022](https://arxiv.org/pdf/2204.01586.pdf)] [[github]()] [cite 6]
    <details>
    <summary> note </summary>
        1. 输入RGB，同时预测物体级深度图和NOCS表示，并将两者对齐(umeyama)得到物体Pose！具体做法暂略。
        2. 另一个可以用来预测深度的工具是：arxiv2022_GCVD_Globally Consistent Video Depth and Pose Estimation。
    </details>


- **NeRF-Pose** Fu Li (国防科大&TUM), Nerf-pose: A first-reconstruct-then-regress approach for weakly-supervised 6d object pose estimation. [[arxiv 2022](https://arxiv.org/pdf/2203.04802.pdf)] [cite 8]




### Year 2021 and before

- **StablePose** Yifei Shi (国防科大), StablePose: Learning 6D Object Poses from Geometrically Stable Patches.[[CVPR 2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Shi_StablePose_Learning_6D_Object_Poses_From_Geometrically_Stable_Patches_CVPR_2021_paper.pdf) [cite 20]


- **FS-Net** Wei Chen (Birmingham), FS-Net: Fast Shape-based Network for Category-Level 6D Object Pose Estimation with Decoupled Rotation Mechanism. [[CVPR 2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_FS-Net_Fast_Shape-Based_Network_for_Category-Level_6D_Object_Pose_Estimation_CVPR_2021_paper.pdf)[[github]](https://github.com/DC1991/FS-Net)[cite 89]


- **Equi-Pose** Xiaolong Li, Yijia Weng (王鹤&&弋力组), Leveraging SE(3) Equivariance for Self-Supervised Category-Level Object Pose Estimation [[NIPS 2021]](https://proceedings.neurips.cc/paper_files/paper/2021/file/81e74d678581a3bb7a720b019f4f1a93-Paper.pdf)[[github]](https://dragonlong.github.io/equi-pose/)[cite 33]


- **DualPoseNet** Jiehong Lin,...,Kui Jia, (华南理工 & 华为), DualPoseNet: Category-level 6D Object Pose and Size Estimation
Using Dual Pose Network with Refined Learning of Pose Consistency. [[ICCV 2021](http://openaccess.thecvf.com/content/ICCV2021/papers/Lin_DualPoseNet_Category-Level_6D_Object_Pose_and_Size_Estimation_Using_Dual_ICCV_2021_paper.pdf)] [[github]()] [cite 17]


- **SGPA**: Kai Chen, Qi Dou (CUHK), Structure-Guided Prior Adaptation for
Category-Level 6D Object Pose Estimation. [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_SGPA_Structure-Guided_Prior_Adaptation_for_Category-Level_6D_Object_Pose_Estimation_ICCV_2021_paper.pdf)] [[page](https://www.cse.cuhk.edu.hk/˜kaichen/projects/sgpa/sgpa.html)] [cite 14]


- **CenterPose** Yunzhi Lin (佐治亚理工), Single-Stage Keypoint-Based Category-Level
Object Pose Estimation from an RGB Image. [[arxiv 2021](https://arxiv.org/abs/2111.10677)]
    <details>
    <summary> 姊妹篇：CenterPoseTrack </summary>

    [CenterPoseTrack](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9811720&casa_token=jl6si6ZQoFEAAAAA:bXXs-DU7uhCrkHNO_vAHUCePwNRSUJEFvZymA_6eO_jZdh6LTVx2n4Z0vUUIt4pnGTiEv4cAAk_q)：Keypoint-Based Category-Level Object Pose Tracking from an RGB Sequence with Uncertainty Estimation. (基于CenterPose网络，补充贝叶斯滤波和卡尔曼滤波，具体暂略)
    </details>


- Objectron: A large scale dataset of object-centric videos in the wild with pose
annotations. [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Ahmadyan_Objectron_A_Large_Scale_Dataset_of_Object-Centric_Videos_in_the_CVPR_2021_paper.pdf)] [cite 45]


- **Latentfusion** Latentfusion: End-to-end differentiable reconstruction and rendering for unseen object pose estimation. [[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_LatentFusion_End-to-End_Differentiable_Reconstruction_and_Rendering_for_Unseen_Object_Pose_CVPR_2020_paper.pdf)] [[github](https://keunhong.com/publications/latentfusion/)] [cite 54]


- **RLLG** Ming Cai (阿德莱德大学), Reconstruct locally, localize globally: A model free method for object pose estimation. [[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cai_Reconstruct_Locally_Localize_Globally_A_Model_Free_Method_for_Object_CVPR_2020_paper.pdf)] [cite 9]


- **CASS** Dengsheng Chen (国防科大), Learning **Ca**nonical **S**hape **S**pace for Category-Level 6D Object Pose and Size Estimation. [[CVPR 2020]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Learning_Canonical_Shape_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2020_paper.pdf)[cite 120]


- **Shape Prior** Meng Tian (新加坡国立NUS), Shape Prior Deformation for Categorical 6D Object Pose and Size Estimation. [[ECCV 2020]](https://arxiv.org/pdf/2007.08454.pdf)[[github]](https://github.com/mentian/object-deformnet)[cite 105]


- **CPS++** Fabian Manhardt (TUM), Gu Wang (Tsinghua), CPS++: Improving Class-level 6D Pose and Shape Estimation From Monocular Images With Self-Supervised Learning. [[arxiv 2020]](https://arxiv.org/pdf/2003.05848.pdf)[cite 35]


- **Neural Object Fitting** - Category level object pose estimation via neural analysis-by-synthesis. [[ECCV 2020](https://arxiv.org/pdf/2008.08145)] [cite 78]


## Others


- **MaskFusion** Martin Runz (伦敦学院大学), MaskFusion: Real-Time Recognition, Tracking and Reconstruction of Multiple Moving Objects. [[ISMAR 2018](https://arxiv.org/pdf/1804.09194)] [[page](http://visual.cs.ucl.ac.uk/pubs/maskfusion/)] [cite 207]（物体级的语义动态SLAM）


- **POMNet** Pose for Everything: Towards **C**ategory-**A**gnostic **P**ose **E**stimation. [[ECCV 2022]()] [[github](https://github.com/luminxu/Pose-for-Everything)] [cite ]

    <details>
    <summary> notes </summary>

    1. 关键词：**小样本设置** (metric-learning based)；2D关键点检测; transformer-based.（注：标题中pose其实指关键点）

    2. 对比：作者称最相关的是[StarMap (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xingyi_Zhou_Category-Agnostic_Semantic_Keypoint_ECCV_2018_paper.pdf)，因为都是关注类别无关的关键点！只不过StarMap要用到3D CAD model并标注3D关键点，而本文是关注2D关键点。

    3. 概述StarMap：StarMap即单通道heatmap，可得所有关键点的2D像素坐标，同时预测DepthMap和CanViewFeature，分别得各像素对应的depth值，和canonical标准物体坐标系下的3D坐标；2D像素和标准3D坐标可以直接PnP，文中是2D像素+depth先恢复到cam下的3D坐标，再和标准3D坐标得物体6D pose！
    
    4. 评StarMap：其实这不就是**NOCS**了嚒！只不过StarMap是focus关键点，而NOCS是对物体所有可见像素，且NOCS的depth是传感器值而非预测值！另外，ICCV'19_**Pix2Pose**也是预测像素的标注3D坐标，与NOCS如出一辙，只不过它暂只关注了instance-level！

    </details>


- **--** [[]]() [[]]() [cite ]


## Pose Tracking


- **TP-AE** Linfang Zheng (南方科大 & 英国伯明翰大学), TP-AE: Temporally Primed 6D Object Pose Tracking with Auto-Encoders. [[ICRA 2022](https://research.birmingham.ac.uk/files/164770788/_ICRA_TP_AE_6D_Object_Tracking.pdf)] [[github](https://github.com/Lynne-Zheng-Linfang/TP-AE_Object_tracking)] [cite 2] 


- **6-Pack**: Chen Wang (SJTU), 6-pack: Category-level 6d pose tracker with anchor-based keypoints.[[ICRA 2020]](https://arxiv.org/pdf/1910.10750.pdf)[[github]](https://sites.google.com/view/6packtracking)[cite 120]


- **BundleTrack** Bowen Wen (罗格斯大学), BundleTrack: 6D Pose Tracking for Novel Objects without Instance or Category-Level 3D Models.[[IROS 2021]](https://arxiv.org/pdf/2108.00516.pdf) [[github](https://github.com/wenbowen123/BundleTrack)] [cite 48]


- **RBOT** Henning Tjaden (RheinMain University), A Region-based Gauss-Newton Approach to Real-Time Monocular Multiple Object Tracking. [[TPAMI 2018]](https://arxiv.org/pdf/1807.02087.pdf) [[github]](https://github.com/henningtjaden/RBOT) [cite 75]


- **ICG** Iterative Corresponding Geometry: Fusing Region and Depth for Highly Efficient 3D Tracking of Textureless Objects.[[CVPR 2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Stoiber_Iterative_Corresponding_Geometry_Fusing_Region_and_Depth_for_Highly_Efficient_CVPR_2022_paper.pdf)[cite 9]


- **CAPTRA** Yijia Weng (北大王鹤组), CAPTRA: CAtegory-Level Pose Tracking for Rigid and Articulated Objects From Point Clouds. [[ICCV 2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Weng_CAPTRA_CAtegory-Level_Pose_Tracking_for_Rigid_and_Articulated_Objects_From_ICCV_2021_paper.pdf) [[github]](https://yijiaweng.github.io/CAPTRA/) [cite 39]


- **3DObjectTracking Repo** [[link]](https://github.com/DLR-RM/3DObjectTracking)


- **--** CatTrack: Single-Stage Category-Level 6D Object Pose Tracking via Convolution and Vision Transformer. [[]]() [[]]() [cite ]


- **--** Cluster-Based 3D Keypoint Detection for Category-Agnostic 6D Pose Tracking. [[]]() [[]]() [cite ]

- **--** [[]]() [[]]() [cite ]


## Paper notes


<details>
<summary> <b> RNNPose (CVPR 2022) - 迭代pose refine </b> </summary>

- 需要提供物体的CAD模型，和初始的pose；注意它的目的是做pose refinement!
- 对我而言，它的优点是较好地将RAFT框架和pose微调任务结合起来了，并且利用了render和非线性优化技术得到end2end模型，以及明确了光流场和3D刚体变换之间的关系(Eq.1)；缺点是，如作者本人所言，该模型是object-specific，对于novel object, pose refinement module需要进一步被微调！

- 摘要：本文提出一种方法，从**单目图像**中估计物体的6-Dof位姿，采用了基于RNN的框架，能较鲁棒地应对erroneous初始pose和遮挡问题。在循环迭代中，基于估计的匹配场（correspondence field），物体的pose优化被建模为非线性最小二乘问题，然后基于可微的LM优化算法求解，可实现端到端训练，其中，匹配场的估计和位姿优化这两个步骤是交替进行的。在LINEMOD，Occlusion-LINEMOD和YCB-Video上达到了SOTA效果。

- 算法架构：3D model表示成从各个角度渲染的2D query template的集合。
    ![RNNPose_archi](assets_pose/RNNPose_archi.png)

- 详细框架：其中render是基于pytorch3d；render包括将3D model根据init_pose渲染为图片，还有将3D features渲染为2D context feature map。
![RNNPose_fig2](assets_pose/RNNPose_fig2.png)

</details>


<details>
<summary> <b> GPV-Pose (CVPR 2022) - 冗余预测：R/t/s回归 + bbox投票 </b> </summary>

1. 虽然自称是depth-based的方法，但实际上也是要用rgb图片，用比如maskRCNN处理rgb图得到目标物体的mask，然后结合depth图得到物体对应的点云！实验对比说是优于DualPoseNet，并且是一个模型训练所有类别？！
2. 自以为，该方法的核心是进行**冗余预测**，包括直接回归出R,t和s得到pose，和预测逐点到bbox的6个面的方向和距离，投票出bbox的位置朝向大小从而得到pose；除了预测Pose，还搞了个重构分支，反正它们提取特征的backbone是共享的，故宣称这样做利于特征的学习！
3. 然后加了2个几何约束，一个约束思路和建模都比较直观，比如对于旋转分量，就是让回归分支预测的rx和ry，跟bbox投票分支得到的平面法向一致，对于平移分量，就是利用点法线公式，构建bbox平面到原点距离，跟bbox的size之间的关系！另一个约束的思路还算直观，但是建模太tricky，暂略；
- 摘要：利用几何信息来增强类别级pose估计的特征学习，一者引入解耦的置信度驱动的旋转表达，二来提出几何引导的逐点投票进行3D bbox估计。最后，利用不同的输出流，加入几何一致性约束，可以进一步提升性能。GPV-Pose的推理速度能达到20FPS。

- 网络结构：下面$r_x$和$r_y$是平面法向；预测的残差平移，通过加上输入点云的均值，得到最终的平移量；预测的残差size，通过加上预先计算的类均值size，得到最终的size；对称考虑镜像对称和旋转对称；对于逐点bbox投票，给每个点预测它到每个面的方向，距离和置信度，因此每个点要预测的维度就是$(3+1+1)\times 6 = 30$；置信度感知的损失函数（论文中Eq.(1)和Eq.(6)有点意思）。
    ![GPV_archi](assets_pose/GPV_archi.png)

</details>


<details>
<summary> <b> Gen6D (arxiv 2022) - 小样本：学好特征 </b> </summary>

- 摘要：Existing generalizable pose estimators either need the high-quality object models or require additional depth maps or object masks in test time, which significantly limits their
application scope. In contrast, our pose estimator only requires some posed images of the unseen object and is able to accurately predict poses of the object in arbitrary environments. Gen6D consists of an object detector, a viewpoint selector and a pose refiner, all of which do not require
the 3D object model and can generalize to unseen objects. 

- 作者argue一个pose估计器应该具有的性质包括：1) 泛化性，指泛化到任何物体；2) model-free；3) simple-inputs,只须输入rgb，无需物体mask或depth map。考虑到基于回归的旋转和平移预测，受限于特定的实例或者类别，无法泛化到任意未见物体；并且，由于缺乏3D model，无法构建2D-3D匹配，因此基于PnP的方法也不能用，因此作者采用基于图像匹配的方式，进行coarse-to-fine的pose估计。

- 网络结构：模型输入是一张query image，和一堆reference image，并且参考图像中的物体pose是已知的，感觉这个要求也不实用啊？！文中说的是："Given Nr images of an object with known camera poses"，搞不清到底是已知谁的pose；另外Data normalization中提到利用三角化来估计物体的size，存疑，三角化不是存在尺度不确定性问题嚒？！

- 大致流程：基于correlation的object location，定位出query图片中的物体位置，得出大致的平移分量；然后基于网络学习相似性度量，挑选ref图像中view最接近的图像，结合预测的in-plane rotation得粗糙的旋转分量；最后pose refiner利用3D CNN和transformer的特征信息融合方式，进行输出微调。

- **这个应该是Few-shot setting，相当于通过少量标注样本，就可以泛化到该instance，另参FS6D**。虽然无须3DCAD model, depth和mask信息，但要提供一些ref图片！

    ![Gen6D_vis](assets_pose/Gen6D_vis.png)
    ![Gen6D_archi](assets_pose/Gen6D_archi.png)

</details>


<details>
<summary> <b> OSOP (CVPR 2022) </b> - 小样本 </summary>

- 输入RGB + 3D CAD model；2D-2D匹配 + 2D-3D匹配（PnP with RANSAC）；
- 摘要：We present a novel **one-shot** method for object detection and 6 DoF pose estimation, that does not require training on target objects. At test time, it **takes as input a target image and a textured 3D query model**. The core idea is to represent a 3D model with a number of 2D templates rendered from different viewpoints. This enables CNN-based direct dense feature extraction and matching. The object is
first localized in 2D, then its approximate viewpoint is estimated, followed by dense 2D-3D correspondence prediction. The final pose is computed with PnP. We evaluate the method on LineMOD, Occlusion, Homebrewed, YCB-V and TLESS datasets.

- 网络结构
    ![OSOP_pipe](assets_pose/OSOP_pipe.png)

- 注：**Kabsch算法**：A solution for the best rotation to relate two sets of vectors（1976）；在"点云累积"论文ECCV2022_Dynamic 3D Scene Analysis by Point Cloud Accumulation的Eq.(3)中也用到了，看上去是带权的最小二乘问题。

</details>


<details>
<summary> <b> ZebraPose (CVPR 2022) </b> </summary>

- 摘要：we present a discrete descriptor, which can represent the object surface
densely. By incorporating a hierarchical binary grouping, we can encode the object surface very efficiently. Moreover,
we propose a coarse to fine training strategy, which enables fine-grained correspondence prediction. Finally, by matching predicted codes with object surface and using a PnP solver, we estimate the 6DoF pose. **In summary, we propose ZebraPose, a two-stage RGBbased approach that defines the matching of dense 2D-3D correspondence as a hierarchical classification task**.

- Surface encoding：以简单情形为例：编码长度为$d$，即对object的顶点进行$d$次分组（提到用kmeans），每一次迭代分组，每个顶点都被赋一个类别id（binary取值），最后把$d$个类id堆叠起来就是顶点的code，可知一个group内的顶点共享code。对于每个3D model，都建立这样的表达并存储起来。

- 网络结构：自大致理解，基于网络预测像素的code，然后跟3D model预先建立的顶点code进行比较，即可构建2D-3D的匹配关系，然后PnP求解pose。
    ![ZebraPose_vis](assets_pose/ZebraPose_vis.png)
    ![ZebraPose_archi](assets_pose/ZebraPose_archi.png)

</details>


<details>
<summary> <b> FS6D (CVPR 2022) - 小样本：学好特征 </b> </summary>

- 没有太大意思，RGBD input，基于FFB6D的特征提取网络，加上transformer进行特征增强，构建support view和query之间的匹配关系，论文废话有点多，部分细节没说清楚，自分析应该是能得到depth point之间的匹配关系，然后利用Umeyama算法即得query相对于support的delta pose，再根据support的pose恢复出query的Pose。噢，还搞了一个PBR类型的rgbd数据集。

- 摘要：We study a new open set problem; the few-shot 6D object poses estimation: estimating the 6D pose of an unknown object by a few support views without extra training. We propose a dense prototypes matching framework by extracting and matching dense RGBD prototypes with transformers. We propose a large-scale RGBD photorealistic dataset (ShapeNet6D) for network pre-training.

- 网络结构
    ![FS6D_data](assets_pose/FS6D_data.png)
    ![FS6D_pipe](assets_pose/FS6D_pipe.png)

</details>


<details>
<summary> <b> UDA-COPE (CVPR 2022) </b> </summary>

- 是第一个对基于RGBD的类级别物体姿态估计做无监督域适应的工作；
- 摘要：The proposed method exploits
a teacher-student self-supervised learning scheme to train a pose estimation network without using target domain pose labels. We also introduce a bidirectional filtering method between the predicted normalized object coordinate space (NOCS) map and observed point cloud, to not only make
our teacher network more robust to the target domain but also to provide more reliable pseudo labels for the student network training.

- 网络结构：(1) 先在合成数据上对教师网络进行有监督训练，再在真实数据上进行无监督域适应；(2) Fig.1是Fig.2中model的网络结构，注意2D特征是与点云有效匹配的特征，是从特征图中采样（4.1节提到把图像块resize为192x192的大小，再随机采样1024个点，应该是指对应这些点的2D特征）；(3) 为了让合成数据上训练的教师网络，在真实数据上的预测更好，即学生网络的伪标签更好，提出Fig.3所示的双向点滤波，看上去比较简单，就是基于老师网络的初始预测，将depth点云对齐到NOCS map，然后计算对齐后二者的逐点距离，设定阈值，分别从两个方向上过滤掉距离值大的异常点；(4) 联合训练老师网络和学生网络，具体地，利用过滤后的NOCS图作为学生网络的伪标签；同时，老师网络基于自监督开始学习真实数据上的知识，作者提出利用几何一致性，通过交叉熵损失，约束过滤后的对齐点云（可能只是原始点云的一个很小子集）要和老师网络自己预测的NOCS图一致！
    ![UDA_archi](assets_pose/UDA_archi.png)
    ![UDA_vis](assets_pose/UDA_vis.png)

</details>


<details>
<summary> <b> TemplatePose (CVPR 2022) </b> </summary>

- 关键词：Model-based；图像匹配；泛化到长的很不一样的物体上。
- 摘要：Our method requires neither a training phase on these objects nor real images depicting them, only their CAD models. It relies on a small set of training objects to learn local object representations, which allow us to locally match the input image to a set of “templates”, rendered images of the CAD models for the new objects. As a result, we are the first to show generalization without retraining on the LINEMOD and
Occlusion-LINEMOD datasets.

- 网络结构
    ![TemplatePose_vis](assets_pose/TemplatePose_vis.png)
    ![TemplatePose_archi](assets_pose/TemplatePose_archi.png)

</details>



<details>
<summary> <b> CenterSnap (ICRA 2022) - center系单阶段！AE形状重建 </b> </summary>

- 要解决：现有的基于“标准坐标回归”和“直接pose回归”方案，计算量大，并且在复杂的多目标场景中性能不好。Existing approaches mainly follow a complex multi-stage pipeline which first localizes and detects each object instance in the image and then regresses to either their 3D meshes or 6D poses. These approaches suffer from high-computational cost and low performance in complex multi-object scenarios, where occlusions can be present. 

- 摘要：同时进行多目标3D重建和基于单视图RGB-D的pose估计，参考CenterNet将目标表示为点。 This paper studies the complex task of simultaneous multi-object 3D reconstruction, 6D pose and size estimation from a single-view RGB-D observation. Our method treats object instances as spatial centers where each center denotes the complete shape of an object along with its 6D pose and size.

- 单阶段和两阶段框架对比
    ![CenterSnap_compare](assets_pose/CenterSnap_compare.png)
- 网络结构：
  1. 直接输入RGB-D图片，提取多尺度FPN特征。这里直接resnet处理depth图好像不太常见！
  2. FPN特征分别输入2个head网络，其中heatmap head用于定位物体的center，param head用于输出全部3D信息，包括shape的128维latent code，和13维的Pose信息（即9维R，3维t，1维s），至于3维的size，可以从latent code重建的标准化的物体点云的bbox获取到，再乘以scale缩放到原来尺寸！
  3. shape的latent code的ground truth，是通过自编码器AE预先训练学习的！
    ![CenterSnap_archi](assets_pose/CenterSnap_archi.png)

</details>


<details>
<summary> <b> CenterPose (ICRA 2022)  </b> </summary>

- **关键词**：CenterNet inspired; Keypoint-based; Category-level 6-DoF pose.
- **摘要**：The proposed network performs 2D object detection, detects 2D keypoints, estimates 6-DoF pose, and regresses relative bounding cuboid dimensions. These quantities are estimated in a sequential fashion, leveraging the recent idea of convGRU for **propagating information from easier tasks to those that
are more difficult**... on the challenging Objectron benchmark...

- **网络结构**：输入RGB, 一个分支做检测，得到物体中心坐标和bbox；一个分支做关键点预测（基于2种方式），**关键点是3Dbbox的8个角点**；一个分支预测bbox的相对大小，这样可以得到标准物体坐标系下的3D关键点坐标 (up to scale)。于是有了2D关键点和3D关键点，就可以PnP得pose了。（细节存疑暂略：两种方式预测的2D关键点一起用于Levenberg-Marquardt version of PnP）
    ![CenterPose_pipe](assets_pose/CenterPose_pipe.png)

</details>


<details>
<summary> <b> YOLOPose (arxiv 2022)  </b> </summary>

- **摘要**：We propose YOLOPose, a **Transformer-based multi-object monocular 6D pose estimation** method based on **keypoint regression**. In contrast to the standard heatmaps for predicting keypoints in an image, we directly regress the keypoints.  Additionally, we employ a learnable orientation estimation module to predict the orientation from the keypoints. Our model is **end-to-end** differentiable and is suitable for **real-time** applications. ...test on the YCBVideo dataset.

- 注：32 keypoints (the eight corners of the 3D bounding box and the 24 intermediate bounding box keypoints)

- **示意图**
    ![YOLOPose_vis](assets_pose/YOLOPose_vis.png)

- **网络结构**
    ![YOLOPose_archi](assets_pose/YOLOPose_archi.png)

</details>


<details>
<summary> <b> OnePose (CVPR 2022) - SFM重建 + 转为相机定位任务  </b> </summary>

- **OnePose**：One-shot之意！因涉及重建，自然model-free，要现成的2D检测器提供bbox！
- **思路**：基于传统的定位pipeline来做物体pose任务，即“offline mapping + online localization”，mapping就是要先给定一段物体的video scan，利用SFM进行稀疏物体点云重建（物体点云看作不动的scene）；localization就是对于query img，通过特征匹配，获取相机的pose，注意这个pose是相机相对于scene的，这里也即相对于物体的，所以相机pose的逆，就是最终要求的物体的pose！关于特征匹配，传统是2D到2D，作者提出3D-2D，即先把ref video frame中的2D特征点的描述子，基于注意力聚合为对应的3D地图点的描述子，然后基于描述子，匹配3D地图点和query img中的2D特征点，有了匹配，就可以通过RANSAC PNP求解位姿！
- **其他**: 论文中是基于ARKit/ARCore工具标注出物体的bbox（相当于指定标准的物体坐标系）和每帧的相机位姿，拍摄ref video时假设物体竖直放置于平面，且保持静止，故bbox限于绕竖直的z轴旋转；OnePose的位姿估计模块仅处理关键帧，所以还有位姿tracking模块处理每一帧，这部分在补充材料中，暂略~
- **缺点**：如作者所言，依赖于局部特征匹配（特征如SIFT, SuperPoint，匹配器如最近邻，SuperGlue），匹配限于3D bbox内的重建点云，和query img上的2D bbox内的特征点，所以对于低纹理的物体可能失败，当训练和测试的seq差异太大时，也可能失败！
- **扩展**: 作者自行扩展了OnePose++，丢掉了基于关键点+描述子的匹配策略！提升了处理低纹理物体的能力！具体内容暂略。

- **摘要**：OnePose draws the idea from visual localization and only requires a simple RGB video scan of the object to build a sparse SfM model of the object. We propose a new graph attention network that directly matches 2D interest points in the query image with the 3D points in the SfM model, resulting in efficient and robust pose estimation. ...run in real-time. ... test on self-collected dataset that consists of 450 sequences of 150 objects.

- 关注related works章节；摘录对NOCS系列方法的评价： A limitation of this line of work is that the shape and the appearance of some instances could vary significantly even they belong to the same category, thus the generalization capabilities of trained networks over these instances are questionable. Moreover, accurate CAD models are still required
for ground-truth NOCS map generation during training, and different networks need to be trained for different categories. 总结就是NOCS只是测试阶段不需要CAD models，训练阶段仍需要，因此在本文中仍被划分为Model-Based方法。

- **对比各种设置的示意图**
    ![OnePose_vis](assets_pose/OnePose_vis.png)

- **算法流程**
    ![OnePose_overview](assets_pose/OnePose_overview.png)

</details>


<details>
<summary> <b> SurfEmb (CVPR 2022) </b> </summary>

- **关键词**：基于对比学习，构建2D-3D稠密匹配。

- **摘要**：We present an approach to learn dense, continuous 2D-3D correspondence distributions over the surface of objects. We also present a new method for 6D pose estimation of rigid objects using the learnt distributions to sample, score and refine pose hypotheses. The correspondence distributions are learnt with a contrastive loss.

- **存疑**：
1. 三维模型只有有限的位姿出现在训练数据中吧，如何学到的这种2D像素到三维模型的"环形"点集的对应关系？！---> 猜：图片和点云的自相似性 + 数据增强。 
2. 点p和像素q构成正样本对，那点p的对称点，如环形上的点，和q是否也是正样本对？！---> 训练时是采样1024个正样本对，和1024个负样本对，概率上姑且认为不会同时出现“p和q是正例，p的对称点和q是负例”这种情况。
3. 如何refine不清楚；

- **算法流程**：四阶段方法：检测物体，从crop图中学习分布，从分布中采样构造初始pose，refinement。
    ![SurfEmb_vis](assets_pose/SurfEmb_vis.png)
    ![SurfEmb_result](assets_pose/SurfEmb_result.png)

</details>


<details>
<summary> <b> EPOS (CVPR 2020) </b> </summary>

- **摘要**：An object is represented by compact surface fragments which allow handling symmetries in a systematic manner. Correspondences between densely sampled pixels and the fragments are predicted using an encoder-decoder network. At each pixel, the network predicts: (i) the probability of each object’s presence, (ii) the probability of the fragments given the object’s presence, and (iii) the precise 3D location on each fragment. A data-dependent number of
corresponding 3D locations is selected per pixel, and poses of possibly multiple object instances are estimated using a robust and efficient variant of the PnP-RANSAC algorithm.

- **网络结构**
    ![EPOS_illu](assets_pose/EPOS_illu.png)

</details>


<details>
<summary> <b> DualPoseNet (ICCV 2021) - 冗余预测：R/t/s回归 + NOCS回归 </b> </summary>

- 核心：同时使用“直接pose回归”和“NOCS坐标预测”两种pose估计方案。

- 摘要：DualPoseNet stacks two parallel pose decoders on top of a shared pose encoder. The implicit and explicit decoders thus impose complementary supervision on the training of pose encoder. We construct the encoder
based on spherical convolutions, and design Spherical Fusion wherein for a better embedding of pose sensitive features from the appearance and shape observations.

- 网络结构
    ![DualPoseNet_archi](assets_pose/DualPoseNet_archi.png)

- Refinement：在测试阶段，共有3种方式可以得到pose参数：(1)直接从explicit decoder的结果中得到pose参数；(2)从implicit decoder中得到NOCS坐标预测，然后Umeyama算法求解pose参数；(3) refinement。这里介绍第三种refine的方式，它是利用2个decoder的预测结果的几何一致性作为Loss，即输入点云P经过显式预测的R|T|s变换后得到的标准坐标，要和隐式预测的标准坐标一致，在测试阶段，固定两个decoder网络参数，并单独优化encoder。
    ![DualPoseNet_refine](assets_pose/DualPoseNet_refine.png)

</details>


<details>
<summary> <b> KDFNet (IROS 2021) - 基于距离场的2D关键点投票 </b> </summary>

- 关键词：Model-based；RGB input；关键点距离场；PnP；
- 要解决：基于pixel-wise voting的方法是direction-based，即每个像素预测它到关键点的2D方向；该方法有一个前提假设，投票方向之间的夹角要足够大，因此该假设不适用于细长的物体，为此，本文提出KDF。
- 本文的3D关键点，follow PVNet是基于FPS采样得到！本文的voting，是先采样一堆像素(voters)，然后每3个为一组，两两组合可以预测3个关键点位置候选，这样重复N次，得3N个候选，然后基于RANSAC思路，对每个候选，让所有voters对它投票，最后取score最高的候选作为最终预测！关于metric，除了常规的ADD accuracy和ADD AUC，还有2D projection accuracy（以偏差5个像素为阈值）。
- 摘自引言：主要有2类方法来定位2D keypoints，包括heatmap-based 和 voting-based，且后者对于遮挡情况更加鲁棒！
- 摘要：We propose a novel continuous representation called Keypoint Distance Field
(KDF) for projected 2D keypoint locations. Formulated as a 2D array, each element of the KDF stores the 2D Euclidean distance between the corresponding image pixel and a specified
projected 2D keypoint. We use a fully convolutional neural network to regress the KDF for each keypoint.

- 网络结构
    ![KDFNet_vis](assets_pose/KDFNet_vis.png)
    ![KDFNet_archi](assets_pose/KDFNet_archi.png)

</details>


<details>
<summary> <b> TemporalFusion (IROS 2021) -- 时序特征融合！ </b> </summary>

- 自评：1.该工作是model-based而非类别级；2. 时序融合的方式还是太粗糙，直接concat，不过好歹避免了对齐问题（FaF中直接concat特征图会引入对不齐的问题）；3. 实验方面仅仅对比了DenseFusion。

- 摘要：we present an end-to-end model named TemporalFusion, which integrates the temporal motion information from RGB-D images for 6D object pose estimation. The core of proposed TemporalFusion model is to embed and fuse the temporal motion information from multi-frame RGBD sequences, which could handle heavy occlusion in robotic grasping tasks. Furthermore, the proposed deep model can also obtain stable pose sequences, which is essential for realtime robotic grasping tasks. We evaluated the proposed method in the YCB-Video dataset.

- 网络结构：
    - (1) 语义分割得目标mask，其rgb和depth分别由PSPNet和Pointnet提特征；不同于DenseFusion，作者提出基于采样的特征融合，对于第$i$帧，采样$N*\alpha_i$个点，其中$\sum_{i=1}^t \alpha_i = 1$，于是t帧数据融合总共就有N个点，特征维度256；图中展示的效果是，对于远离当前帧的早期帧，采样点可以相对少一些；
    - (2) 运动推理：基于Open3D的视觉里程计算出位姿变化（这部分待续）；
    - (3) 时序融合：基于全局池化和最大池化，获取全局特征；用两个3-layer的卷积网络，将运动推理模块预测的$R'$和$T'$分别转化为运动特征；再结合N个逐点特征（局部），全部堆叠起来得融合的时序特征；
    - (4) 考虑到不同特征对最终的pose估计贡献不同，采用CBAM注意力给特征的通道加权，然后接3个head分别预测$R$, $T$和置信度$c$，取置信度最大的$c$对应的预测作为最终结果。
    ![TemporalFuse_archi](assets_pose/TemporalFuse_archi.png)
    ![TemporalFuse_reg](assets_pose/TemporalFuse_reg.png)

</details>


<details>
<summary> <b> Morefusion (CVPR 2020)  </b> </summary>

- 处理known objects
- **摘要**：We present a system which can estimate the accurate poses of multiple **known objects** in contact and occlusion from real-time, embodied multi-view vision. Our approach makes 3D object pose proposals from single RGBD views, accumulates pose estimates and non-parametric occupancy information from multiple views as the camera moves, and performs joint optimization to estimate consistent, non-intersecting poses for multiple objects in contact. ...test on YCB-Video, and our own challenging Cluttered YCB-Video.

- **pipiline的四个阶段**：
    - object-level volumetric fusion: 用目标实例的mask处理RGB+depth，再结合相机位姿的tracking(基于ORB-SLAM2)，创建一个volumetric map(包括已知的目标和未知的目标)；
    - volumetric pose prediction： 利用volumetric map作为目标周围的信息，结合目标经mask后的特征grid，估计一个初始的位姿；
    - collision-based pose refinement：使用物体CAD模型上的采样点(经估计的pose转换)，和volumatric map上的occupied space进行碰撞检查，通过梯度下降联合优化多个目标的位姿；
    - CAD alignment：将多个相机坐标系下估计的目标物体pose，转换到统一的世界坐标系下，然后两两计算pose loss一并优化，使各视角下预测的pose是一致的。

- **网络结构**
    ![MoreFusion_archi](assets_pose/MoreFusion_archi.png)

- **子网络结构**
    ![MoreFusion_sub](assets_pose/MoreFusion_sub.png)

</details>


<details>
<summary> <b> Cosypose (ECCV 2020) - 物体级场景重建 </b> </summary>

- Cosy是consistency之意！处理包含known objects的scene，即物体的3D CAD模型已知；算法输入是多视图的img；第一阶段，先按单目的方式去检测物体并估计其pose；第二阶段，参见Fig.5，对任意2个view的img pair，挑2组object pair，并根据object pose得到这2个view的cam delta pose(基于RANSAC丢掉不靠谱的cam pose预测，也即丢掉了不靠谱的object pair：对应标签一致但不是同一个实例)；然后就可以构建graph，其中顶点是object，边连接构成pair的object，于是graph中的一个连通分量就对应了同一个object实例！第三阶段，构建object-level的BA优化问题，扫了一眼代码，应该是直接手写前向雅可比，及LM优化步骤的，未用到优化库。
- 优点：将单目物体pose估计，融合到多视图优化框架下！能同时得到object-level的scene重建，各物体的pose，以及各view的相机pose；缺点：自分析，根据第二阶段的相对相机pose的计算，CosyPose应该只能处理静态场景！

- **摘要**：We introduce an approach for recovering the 6D pose of multiple known objects in a scene captured by a set of input images with unknown camera viewpoints. (1) We present a **single-view single-object** 6D pose estimation method to generate pose hypotheses; (2) We **jointly estimate** camera viewpoints and 6D poses of all objects in a single consistent scene; (3) We develop a method for global scene refinement by solving an **object-level bundle adjustment** problem. ... test on YCB-Video and T-LESS datasets.

- **示意图**
    ![CosyPose_vis](assets_pose/CosyPose_vis.png)

- **算法流程**
    ![CosyPose_pipe](assets_pose/CosyPose_pipe.png)

- 第二阶段计算相机delta pose的图示
    ![CosyPose_Fig5](assets_pose/CosyPose_Fig5.png)

</details>


<details>
<summary> <b> Latentfusion (CVPR 2020) - 小样本：重建+渲染 </b> </summary>

- 处理unseen objects；render-and-compare；
- **摘要**：We present a network that reconstructs a latent 3D representation of an object using a small number of reference views at inference time. Our network is able to render the latent 3D representation from arbitrary views. Using this neural renderer, we directly optimize for pose given an input image. By training our network with a large number of 3D shapes for **reconstruction and rendering**, our network generalizes well to **unseen objects**. We present a new dataset for unseen object pose estimation–**MOPED**. ...test on MOPED as well as the ModelNet and LINEMOD datasets.

- **重建+渲染用于位姿估计**
    1. **重建**：这里是指在latent space中的重建！给定一些reference RGB-D图片，利用modeler构建latent object，其实就是先用2D UNet提特征，然后lift到3D grid中，再3D UNet继续提特征，简单理解，正常3D空间中的一个点只有3维坐标信息，现在扩展成了C维的特征向量！每个视图view下都能构建一个latent object，它们分别处于各自的cam坐标系下，可以转到obj坐标系下，这样就能整合成唯一的latent object，这里可以通过channel-wise的均值池化，论文中是采用RNN的方式融合！
    2. **渲染**：简单把它看作上述重建的逆过程，也是通过3D/2D的UNet进行特征处理，最后输出depth和mask图；因为rgb图中的高频信息不太容易由NN去学习，所以论文采用了**Image-based Rendering(IBR)**技术，单独得到rgb图。简单来说，query的像素点，根据cam内参及query和ref的pose，可以找到各个ref图像上匹配像素，取这些匹配像素的rgb进行blend就得到了query的rgb值！
    3. **位姿估计**：利用重建网络的modeler处理ref图，获取latent object，然后用渲染网络，在给定的init_pose下，渲染得到对应的depth和mask图，基于若干loss进行梯度反传，直接优化pose参数！
    ![LatentFusion_archi](assets_pose/LatentFusion_archi.png)

- **重建+渲染的网络结构**
    ![LatentFusion_archi2](assets_pose/LatentFusion_archi2.png)

</details>


<details>
<summary> <b> RLLG (CVPR 2020)  </b> </summary>

- **摘要**：We propose a learning-based method whose input is a collection of images of a target object, and whose output is the pose of the object in a novel view. At inference time, our method maps from the RoI features of the input image
to a dense collection of object-centric 3D coordinates, one per pixel. This dense 2D-3D mapping is then used to determine 6dof pose using standard PnP plus RANSAC. We seamlessly build our model upon Mask R-CNN. We contribute a new head – the object coordinate head – to the same backbone, whose output is the dense 3D coordinates of the object in object-centric frame. 

- **推理阶段可视化**
    ![RLLG_vis](assets_pose/RLLG_vis.png)

</details>


<details>
<summary> <b> PPR-Net (CVPR 2020)  </b> </summary>

- **摘要**：We propose a simple but novel Point-wise Pose Regression Network (PPR-Net). For each point in the point cloud, the network regresses a 6D pose of the object instance that the point belongs to. We argue that the regressed poses of points from the same object instance should be located closely in pose space. Thus, these points can be clustered into different instances and their corresponding objects’ 6D poses can be estimated simultaneously. It works well in real world robot
bin-picking tasks.

- **网络结构** 
    - (1) 输入点云，先Pointnet++提特征，后接4个分支；其中一个做语义分割，得到逐点的类别预测，将语义类别concat到原点云特征，得到所谓semantic-class-aware的特征；该组合特征输入到另外3个分支，分别回归逐点的center预测，逐点的旋转角预测，逐点的物体可见性预测（可见性衡量了该点所属物体被遮挡的程度）；
    - (2) 推理阶段：高于可见性阈值的点，才有voting的权利，先将这些点基于密度聚类（同一物体上的点，预测的center位置应该是聚在一起的），相当于在前面语义分割的基础上，再进行实例分割，然后每个实例的pose，就是它包含的有效点的voting的平均；
    - (3) 训练阶段：包含3个损失，一个是语义分割（即逐点的分类）损失，用的交叉熵；一个是逐点的可见性，这里gt是启发式得到的，用当前点所属物体包含的点数，除以场景中一个物体包含的最大点数，即近似了物体被遮挡程度；第三个是pose约束，用到了另一篇文章中定义的pose metric，直接在欧式空间中算distance！
    ![PPRNet_archi](assets_pose/PPRNet_archi.png)

</details>


<details>
<summary> <b> NeRF-Pose (arxiv 2022)  </b> </summary>

- **核心**：NeRF隐式重建和体渲染 + 基于NOCS的2D-3D匹配 + PnP；

- **摘要**：Precise annotation of 6D poses in real data is intricate, timeconsuming and not scalable, while synthetic data scales well but lacks realism. To avoid these problems, we present a weakly-supervised reconstruction-based pipeline, named NeRF-Pose, which needs only 2D object segmentation and known relative camera poses during training. Following the **first-reconstruct-then-regress** idea, we first reconstruct the objects from multiple views in the form of an implicit neural representation. Then, we train a pose regression network to predict pixel-wise 2D-3D correspondences between images and the reconstructed model...

- **网络结构**： **第一阶段**：将nerf中的采样点，转到obj坐标系，得到obj坐标系下的3D隐式表达（OBJ-NeRF），它用来生成gt_NOCS_map，监督pose回归网络的nocs预测；**第二阶段**：三个步骤，先检测obj bbox；再pose reg得到nocs和mask预测；最后PnP;
    ![NeRFPose_pipe](assets_pose/NeRFPose_pipe.png)

</details>


<details>
<summary> <b> VideoPose (arxiv 2021) </b> - 时序特征融合 </summary> 

- 输入是RGB video stream和3D CAD模型，核心想法是利用时序信息，手段是进行时序特征融合，整体创新性有限，论文细节不清(比如fig2的特征变换层)，暂略！
- **摘要**： Our proposed network takes a pre-trained 2D object detector as input, and aggregates visual features through a recurrent neural network to make predictions at each frame...

- **网络结构**
    ![VideoPose_fig1](assets_pose/VideoPose_fig1.png)
    ![VideoPose_fig2](assets_pose/VideoPose_fig2.png)

</details>


<details>
<summary> <b> SC6D (3DV 2022) 基于模板，位姿解耦 [I like]  </b> </summary>

- **摘要**： SC6D requires neither the 3D CAD model of the object nor any prior knowledge of the symmetries. The pose estimation is decomposed into three sub-tasks: a) object 3D rotation representation learning and matching; b) estimation of the 2D location of the object center; and c) scaleinvariant distance estimation (the translation along the zaxis) via classification...
- **关键点**： 无需3D model；旋转egocentric vs. allocentric；基于分类隐式处理对称歧义(类似EPOS/SurfEmbd)；旋转的估计采用模板+分类的思想，参考了Implict-PDF和3D-RCNN这两个工作！ 

- **网络结构**
    ![SC6D_pipe](assets_pose/SC6D_pipe.png)

</details>


<details>
<summary> <b> IST-Net (arxiv 2023) 免形状先验 [I like] </b> </summary>

- **摘要**：Our empirical study shows that the 3D
prior itself is not the credit to the high performance. The keypoint actually is the explicit deformation process, which aligns camera and world coordinates supervised by worldspace 3D models (also called canonical space). Inspired by these observation, we introduce a simple prior-free implicit space transformation network, namely IST-Net, to transform camera-space features to world-space counterparts and build correspondence between them in an implicit manner without relying on 3D priors. 
- **核心** 隐式特征变换，免显式的形状prior
- **网络结构**
    ![IST-Net_compare](assets_pose/IST-Net_compare.png)
    ![IST-Net_archi](assets_pose/IST-Net_archi.png)

</details>

<details>
<summary> <b> SLO-LocNet (VR2023) cam定位+obj位姿  </b> </summary>

- **摘要**：In this paper, we focus on simultaneous sceneindependent camera localization and category-level object pose estimation with a unified learning framework. The system consists of a localization branch called SLO-LocNet, a pose estimation branch called SLO-ObjNet, a feature fusion module for feature sharing between two tasks, and two decoders for creating coordinate maps.
- **核心** 同时估计相机和物体位姿，自然是基于**静态场景假设**；各种特征融合让两个任务互相促进；孪生网络架构，输入相邻两帧图像。

- **网络结构**
    ![SLO-LocNet_archi](assets_pose/SLO-LocNet_archi.png)
    ![SLO-LocNet_archi2](assets_pose/SLO-LocNet_archi2.png)

</details>


<details>
<summary> <b> StablePose (CVPR 2021) 几何稳定性分析 </b> </summary>

- **摘要**：We introduce the concept of geometric stability to the problem of 6D object pose estimation and propose to learn pose inference based on geometrically stable patches extracted from observed 3D point clouds. According to the
theory of geometric stability analysis, a minimal set of three planar/cylindrical patches are geometrically stable and determine the full 6DoFs of the object pose...It also performs well in category-level pose estimation.
- **核心**：提出稳定块组概念_同测实例和类别级

- **网络结构**
    ![StablePose_intro](assets_pose/StablePose_intro.png)
    ![StablePose_archi](assets_pose/StablePose_archi.png)

</details>


<details>
<summary> <b> RePoNet (NIPS 2022) 半监督  </b> </summary>

- **摘要**： In this paper, we collect **Wild6D**, a new unlabeled RGBD object video dataset with diverse instances and backgrounds...We propose a new model, called Rendering for Pose estimation network (RePoNet), that is jointly trained using the free groundtruths with the synthetic data, and a silhouette matching objective function on
the real-world data.
- **核心**：半监督；可微渲染；RGBD特征+NOCS预测位姿使端到端；

- **网络结构**
    ![RePoNet_archi](assets_pose/RePoNet_archi.png)

</details>


<details>
<summary> <b> Self-Pose (ICLR 2023) 自监督 </b> </summary>

- **摘要**：In this paper, we overcome this barrier by introducing a self-supervised learning
approach trained directly on large-scale real-world object videos for category-level
6D pose estimation in the wild. Our framework reconstructs the canonical 3D shape of an object category and learns dense correspondences between input images and the canonical shape via Categorical Surface Embedding (**CSE**, i.e. the per-vertex embedding). For training, we propose novel geometrical **cycle-consistency losses** which construct cycles across 2D-3D spaces, across different instances and different time steps. 
- **核心**：自监督；无标注；循环一致性损失；纹理迁移；可微渲染；

- **网络结构**
    ![Self-Pose_archi](assets_pose/Self-Pose_archi.png)

</details>


<details>
<summary> <b> GCASP (PMLR 2023) 语义基元 [关注]  </b> </summary>

- **摘要**：In this paper, we propose a
novel framework for category-level object shape and pose estimation from a single RGB-D image. To handle the intra-category variation, we adopt a semantic primitive representation that encodes diverse shapes into a unified latent space,
which is the key to establish reliable correspondences between observed point clouds and estimated shapes. Then, by using a **SIM(3)-invariant** shape descriptor, we gracefully decouple the shape and pose. 
- **动机**：The insight behind the semantic primitive representation is that although different instances in the same category have various shapes, they tend to have similar semantic parts, e.g., each camera instance has a lens, and each laptop has four corners. We use **sphere primitives** here and alpha = (c, r), where c is the 3D center of a sphere and r is the radius. (参考DualSDF)
- **核心**：生成式；位姿加形状

- **网络结构**
    ![GCASP_archi](assets_pose/GCASP_archi.png)
    ![GCASP_sim3](assets_pose/GCASP_sim3.png)

</details>


<details>
<summary> <b> Shape Prior (ECCV 2020) 形状先验 </b> </summary>

- **摘要**：To handle the intra-class shape variation, we propose a deep network to reconstruct the 3D object model by explicitly modeling the deformation from a pre-learned categorical shape prior. We design an autoencoder that trains on a collection of object models and compute the mean latent embedding for each category to learn the categorical shape priors.
- **类别形状先验** 在大量对齐的同类三维模型上，训练自动编码器网络，然后将它们的隐变量取均值，送入解码器，得到的重构点云就作为该类的形状先验！--> **IST-Net**诟病了这种依赖大量三维模型获取先验的方式，并用实验证明了形状先验是不必要的。

- **网络结构**
    ![ShapePrior_overview](assets_pose/ShapePrior_overview.png)
    ![ShapePrior_archi](assets_pose/ShapePrior_archi.png)

</details>


<details>
<summary> <b> SGPA (ICCV 2021) 自适应形状先验  </b> </summary>

- **摘要**：We take advantage of category prior to overcome the problem of intra-class variation by innovating a structure-guided prior adaptation scheme to accurately estimate 6D pose for individual objects. We propose to leverage their structure similarity to dynamically adapt the prior to the observed object

- **网络结构**
    ![SGPA_archi](assets_pose/SGPA_archi.png)

</details>


<details>
<summary> <b> CASS (CVPR 2020) 隐式标准空间 </b> </summary>

- **摘要**：To tackle intra-class shape variations, we learn canonical shape space (CASS), a unified representation for a large variety of instances of a certain object category. In particular, **CASS is modeled as the latent space** of a deep generative model of canonical
3D shapes with normalized pose. We train a variational auto-encoder (VAE) for generating 3D point clouds in the canonical space from an RGBD image.

- **流程说明** 
1. 浅蓝模块：基于VAE学习CASS，编码阶段，一个分支提取RGB-D嵌入特征，类似用DenseFusion结构；一个分支用PointNet，提取标准点云（三维模型采样500个点）的几何特征，仅训练时用到；然后用Batch Mixing的训练策略，使两个分支共享特征空间，RGB-D分支对应的位姿/视图无关的特征空间，即所谓CASS。注意，RGB-D分支的特征之所以视图无关，因为无论RGB-D中的物体位姿如何，其提取的特征，都要求对齐到canonical物体点云对应的特征空间。解码阶段，FoldingNet基于CASS编码，将椭球形warp到目标点云实现重构之目的，重构的点云也是canonical的，并且是metric size，即对应实际的物体大小，所以可以直接取最小bbox得到size。
2.浅绿模块：因为浅蓝模块中的PointNet分支，输入输出都是canonical的，所以它学的特征并非位姿无关，继而可以复用它，提取位姿相关的几何特征，再结合另一CNN提取位姿相关的RGB特征；
3.浅黄模块：将位姿相关的几何+RGB特征，堆叠上位姿无关的CASS编码，就可以“对比”回归出位姿R,t。
- **评述**：如IST-Net所言，同ShapePrior一样，依赖大量三维模型，比如Shapenet。

- **网络结构**
    ![CASS_archi](assets_pose/CASS_archi.png)

</details>


<details>
<summary> <b> Neural-Obj-Fitting (ECCV2020) 风格生成  </b> </summary>

- **摘要**： We integrate a neural synthesis module into an optimization based model fitting framework to simultaneously recover object pose, shape and appearance from a single RGB or RGB-D image. This module is implemented as a deep network that can generate images of objects with control over poses and variations in shapes and appearances. We use our 3D style-based image generation network as decoder and a standard
CNN as encoder.
- **评论** **Analysis-by-Synthesis**有点类似于**render and compare**，前者在本文中是基于深度生成模型（pose-conditioned appearance generation），后者则是利用图形学中的可微渲染，二者共同点则是让生成/渲染图片，跟观测图尽量一致，从而调整位姿；

- **网络结构**
1. 由于平移T和平面内旋转Rz可以由2D的相似变换得到，所以pose-ware的图像生成，分成2个步骤，先由Decoder_3D生成带平面外旋转的"图像"，再由Decoder_2D继续warp得到最终图像；（设相机的z轴沿着光轴，y轴朝上，则Rz对应in-plane旋转，Rx是俯仰elevation，Ry是偏航azimuth）
2. 训练数据获取：带位姿标签的物体不同视图的图像，由ShapeNet中的物体渲染而来，由于Rz和平移无须训练，只用生成Out-of-plane的物体图像。
3. 隐特征的维度取16，兼顾图像质量和位姿准确度；推理时，采样多个初始值，平行优化，取最好的结果。
4. 不太明白Fig.1中Training阶段，z为何示意Fixed，不同的输入，提取的隐特征应该不同呀，还是说Fixed指的是固定z的分布的均值和方差？暂略~
    ![Neural-Obj-fitting_overview](assets_pose/Neural-Obj-fitting_overview.png)
    ![Neural-Obj-fitting_generator](assets_pose/Neural-Obj-fitting_generator.png)

</details>


<details>
<summary> <b> CPS++ (arxiv 2020) 形状+位姿+自监督  </b> </summary>

- **摘要**：we propose a novel method for class-level monocular 6D pose estimation, coupled with metric shape retrieval. we additionally propose the idea of **synthetic-to-real** domain transfer for class-level 6D poses by means of **self-supervised learning**. We leverage recent
advances in **differentiable rendering** to self-supervise the model with unannotated real RGB-D data. 

- **网络结构**
1. Fig.2中展示的是**CPS**的架构，参考3D-RCNN，预测的是**allocentric**四元数旋转，要进一步转换为**egocentric**旋转；所谓ego/allo，是相对观察者，即相对相机来说的，若是以自我(cam)为中心的ego旋转，则单纯的平移会造成外观的变化，所以对于ROI Align的输入，预测allo旋转更合适，参见Fig.3；
2. 生成的点云表达的shape，可选地，能够继续转换为mesh，参见Fig.4；Fig.2中的Shape Encoding预测的是形状的偏移，要加上均值形状（该类的隐向量均值），然后送入解码器生成点云；
3. 损失函数是将生成的点云，按照预测的pose转换到cam坐标系下，与真值点云（从三维模型上均匀采样2048个点并经过真值pose转换得到）计算chamfer损失，参见Fig.5，这种把各个回归项整合到一起的约束损失，说是比单独约束每个回归项更好。此外还有形状的正则损失，和掩码的交叉熵损失。
4. 由于上述损失的计算依赖真值标注，较难获取，所以只在合成数据上训练，这就导致了domain gap的问题；为了缓解该问题，本文借鉴**Self6D**，提出自监督版的**CPS++**；
5. 参见Fig.6，自监督学习要额外引入Depth模态，一方面，结合预测的掩码，反投影得到观测点云；另一方面，将生成的点云转换为mesh（即CPS中无须转为mesh），然后借助可微渲染，得到物体的掩码和深度图（未采用color mesh，故没有渲染得color img），继而得到预测的点云；自监督损失包括两个点云的chamfer loss，两个mask的交叉熵loss；其中一个细节是，计算chamer loss时利用两个点云的均值，进行粗略对齐。
6. 小结一下，CPS++的完整损失，就是在合成数据上的监督损失（上述3中介绍），以及在真实的无标注的RGB-D数据上的自监督损失（上述5中介绍）。
    ![CPS++_overview](assets_pose/CPS++_overview.png)
    ![CPS++_ego_vs_allo](assets_pose/CPS++_ego_vs_allo.png)
    ![CPS++_mesh](assets_pose/CPS++_mesh.png)
    ![CPS++_loss](assets_pose/CPS++_loss.png)
    ![CPS++_self-super](assets_pose/CPS++_self-super.png)

</details>


<details>
<summary> <b> FS-Net (CVPR 2021) 观测点云的重建&分割 </b> </summary>

- **概述**：FS-Net借鉴了G2L-Net的想法，利用YOLOv3进行RGB 2D bbox检测（由于无须预测2D掩码，模型可以加速），基于**Depth模态进行点云分割和姿态估计**；该工作基于3D图卷积建立自编码器，通过**重建观测点云**（相比于CASS、Shape-Prior中重建完整形状，重建难度降低），学习**旋转感知的隐特征**（因为3D图卷积设计为对平移和尺度具有不变性），然后并行**回归两个旋转轴**，再由另一分支基于分割点云，PointNet回归出残差平移和残差尺度，该工作还设计了一种基于3D变形机制的在线数据增强策略，在GPU上的速度可达到20fps。 

- **网络结构** 
1. 细节存疑，暂略：Fig.2中预测残差输入的是分割点云，还是无须分割的点云；以及论文中说，借助3D分割结果，仅使用观察物体上的点对应的特征进行重构，但是Fig.2中看起来分割和重建是同步的，并非有先后。
2. Fig.5中是**基于cage的点云增强**；Fig.8中可视化了重建观测点云的结果；Fig.13中列出了TLess中的不同对称类型，论文对不同对称类型作了不同处理。
    ![FS-Net_archi](assets_pose/FS-Net_archi.png)
    ![FS-Net_aug](assets_pose/FS-Net_aug.png)
    ![FS-Net_vis](assets_pose/FS-Net_vis.png)
    ![FS-Net_sym](assets_pose/FS-Net_sym.png)

</details>


<details>
<summary> <b> Equi-Pose (NIPS 2021) 等变/不变性 </b> </summary>

- **摘要**：we propose for the first time a self-supervised learning framework to estimate **category-level 6D object pose from single 3D point clouds**. During training, our method assumes **no ground-truth pose annotations, no CAD models, and no multi-view supervision**. The key to our method is to disentangle shape and pose through an **invariant** shape reconstruction module and an **equivariant** pose estimation module, empowered by SE(3) equivariant point cloud networks.
- **核心**：NOCS是人为预定义标准坐标系，这里则希望由网络，基于不变性特征自己学出来！正因为标准空间是学出来的（不能直接取bbox得尺度大小），输出位姿不包含对尺度的估计！同时，基于等变性特征，估计位姿；然后基于重构损失，实现自监督训练！注意在评估的时候，要将学出的空间，和标注的空间进行对齐处理。

- **网络结构**
    ![Equi-Pose_archi](assets_pose/Equi-Pose_archi.png)
    ![Equi-Pose_vis](assets_pose/Equi-Pose_vis.png)

</details>


<details>
<summary> <b> SAR-Net (CVPR 2022) 三合一方案 </b> </summary>

- **摘要**：we rely on the shape information predominately from the depth (D) channel. The key idea is to explore the shape alignment of each instance against its corresponding category-level template shape, and the symmetric correspondence of each object category for estimating a coarse 3D object shape.
- **核心**：用到了模板先验、对称先验；整合了基于对应点、基于回归和基于投票的位姿估计方案。

- **网络结构**
1. Given the 3D template dataset – ShapeNetCore [4], we randomly select one template per category as the category-level template shape, which is normalized by scale, translation, and rotation as in [58]. We further sample the category-level template shape into a sparse 3D point cloud Kc ∈ R3×Nk by using Farthest Point Sampling (FPS).
2. We practivally obtain the ground-truth deformed template point cloud K by applying
actual object rotation to the category-level template point cloud Kc.
3. Remark. It is noteworthy that specific object instances are never fully symmetric due to shape variations. Thus, exploiting the underlying symmetry by point clouds of objects
is applicable to objects which have the global symmetric shapes but asymmetric local parts, as our framework does not demand the exact 3D shape recovery.
    ![SAR-Net_archi](assets_pose/SAR-Net_archi.png)
    ![SAR-Net_result](assets_pose/SAR-Net_result.png)

</details>


<details>
<summary> <b> MV6D (IROS 2022) 多视图(PVN3D+DenseFusion)  </b> </summary>

- **摘要**：We base our approach, **designed for very distinct views in heavily cluttered scenes**, on the **PVN3D** network that uses a single RGB-D image to predict keypoints of the target objects. We extend this approach by using a combined point cloud from **multiple views** and fusing the images from each view with a **DenseFusion** layer. In contrast to current multi-view pose detection networks such as CosyPose, our MV6D can learn the fusion of multiple perspectives in an **end-to-end** manner and does not require multiple prediction stages or subsequent fine tuning of the prediction. 

- **算法流程 & 网络结构**
1. **Point Cloud Fusion**: We first convert all depth images into point clouds and combine them to a single point cloud using the known camera poses. We further process only a random subset of points and attach the corresponding RGB value as well as the surface normal to each remaining point. 
2. **RGB Image Fusion**: For each RGB image, we independently extract pixel-wise visual features. Similar to DenseFusion, we concatenate to each point in the point cloud the PSPNet feature of the associated pixel from the associated view.
3. **Instance Semantic Segmentation**:  The semantic segmentation module predicts an object class for each point. The center offset module estimates the translation offset from the given point to the center of the object that it belongs to. Following [7], we apply mean shift clustering [58] to obtain the final object center predictions. These are then used to **further refine the segmentation map by rejecting points which are too far away from the object center**.
4. **3D Keypoint Detection**: In advance of the training, we select **eight target keypoints from the mesh** of each object using the farthest point sampling (FPS). All keypoint predictions belonging to an instance are clustered by mean shift clustering [58] in
order to obtain the final 3D keypoint predictions as in [7].
5. **Datasets**: We created three novel
photorealistic datasets （MV-YCB FixCam/WiggleCam/MovingCam） of diverse cluttered scenes with heavy occlusions. All of these datasets contain **RGB-D images from multiple very distinct perspectives**. Our datasets are composed of eleven non-symmetric objects
from the YCB object set. Using **Blender** with physics, we created cluttered scenes by
spawning the YCB objects above the ground in the center with a 3D normally distributed offset.

    ![MV6D_archi](assets_pose/MV6D_archi.png)
    ![MV6D_feat](assets_pose/MV6D_feat.png)

</details>


<details>
<summary> <b> SO-Pose (ICCV 2021) 引入自遮挡特征 </b> </summary>

- **摘要**：**While end-to-end methods have recently demonstrated promising results at high efficiency, they are still inferior when compared with elaborate PnP/RANSACbased approaches in terms of pose accuracy**. In this work, we address this shortcoming by means of a novel reasoning about self-occlusion, in order to establish a two-layer representation for 3D objects which considerably enhances the accuracy of end-to-end 6D pose estimation. Our framework, named **SO-Pose, takes a single RGB image as input and respectively generates 2D-3D correspondences as well as self-occlusion information** harnessing a shared encoder
and two separate decoders. **Both outputs are then fused to directly regress the 6DoF pose parameters**. 

- **网络结构**
1. **旋转表达**：We feed SO-Pose with a zoomed-in RGB image [20, 43] of size 256×256 as input and directly output 6D pose. **Similar to GDR-Net** [43], we parameterize the 3D rotation using its **allocentric** 6D representation. （补：**凡是基于ROI-Align Crop Img直接回归旋转的，都要沿用3D-RCNN中提到的allocentric旋转**，除了SO-Pose，还有比如SC6D，GDR-Net，Self6D）
2. **自遮挡**：参见Fig.3，不可见点被限定为OP与物体坐标系平面的交点，这些交点可以由P点坐标，以及真值R,t，解析得到，所以让网络预测这些交点的坐标，将其作为中间表达，辅助对pose的直接回归！注意这些交点少了一个自由度，所以Fig.2的(d)分支中，只须预测3个点的6维坐标map；而(e)分支中预测2D-3D对应关系，3个通道的真值估计就是三维模型按gt_pose投影得到。
3. **速度**：Given a 640×480 image from YCB-V with multiple objects, our method takes about **30ms to handle a single object and 50ms to process all objects** in the image on an Intel
3.30GHz CPU and a TITAN X (12G) GPU. This includes the additional **15ms for 2D localization using Yolov3**.
    ![SO-Pose_archi](assets_pose/SO-Pose_archi.png)
    ![SO-Pose_occu](assets_pose/SO-Pose_occu.png)
    ![SO-Pose_result](assets_pose/SO-Pose_result.png)
    ![SO-Pose_result2](assets_pose/SO-Pose_result2.png)
    ![SO-Pose_speed](assets_pose/SO-Pose_speed.png)

</details>


<details>
<summary> <b> InstancePose (ICCV-W 2021) 6D坐标图 </b> </summary>

- **摘要**：This study presents a method to estimate 6DoF object poses for **multi-instance** object detection that requires less time
and is accurate. The proposed method uses a deep neural network, which **outputs 4 types of feature maps**: the error object mask (dim=1), semantic object masks (dim=C+1), center vector maps (CVM, dim=2) and 6D coordinate maps (dim=6). These feature maps are combined in post processing to detect and estimate **multi-object 2D-3D correspondences** in parallel for PnP RANSAC estimation. The experiments show that the method can process input RGB images containing 7 different object categories/ instances at a speed of 25 fps.
- **一句话概括**：沿用PVNet思路区分实例，引入6D坐标图构建2D-3D对应点。
- **网络结构**
1. The output feature maps have dimensions of H×W×(1+C+1+2+6). each of the feature maps is trained using supervised ground truths that are
calculated from the corresponding 3D object models, except for the **error object mask** which shows the pixel-wise confidence in the quality of the predicted 6D coordinate maps.
2. The CVM is to allows pixels in each object category to vote for the object centers to which they belong. This is used to determine which pixels go with which objects.
3. Fig.4示意了6D coordinate maps：回忆NOCS Map的生成，是将三维模型按gt_pose投影所得，其实也就是只有靠近cam的一侧三维点可见；想象一个对称物体比如碗，沿着对称面砍掉一半后，仍按原来的gt_pose投影到图像，能够得到跟完整碗一样的物体剪影；但此时像素对应的三维点，都是远离cam的那一侧本来不可见的点，所以相当于增加了PnP的约束！
4. Non-Maximum Preservation：表述不够清晰，暂略~
    
    ![InstancePose_archi](assets_pose/InstancePose_archi.png)
    ![InstancePose_6Dcoord](assets_pose/InstancePose_6Dcoord.png)
    ![InstancePose_result](assets_pose/InstancePose_result.png)

</details>


<details>
<summary> <b> ROPE (WACV 2021) 多精度热力图 </b> </summary>

- **摘要**：we develop a novel occlude-and-blackout batch augmentation technique to learn occlusion-robust deep features, and a multiprecision supervision architecture to encourage holistic pose representation learning for accurate and coherent landmark predictions.

- **网络结构**
1. **Backbone**：Our 2D landmark prediction is based on the Mask R-CNN [16] framework. A basic improvement is substituting the original backbone network with HRNet [54, 60] to exploit its high-resolution feature maps.
2. **遮挡数据增强**：参见Fig.2左下输入，Inspired by the ideas of random erasing [65], hideand-seek [50], and batch augmentation [21], we develop a novel Occlude-and-blackout Batch Augmentation (OBA) to promote robust landmark prediction under occlusion. Similar to hide-and-seek, we divide the image region enveloped by the object bounding box into a grid of patches and replace each patch, under certain probability, with either noise or a random patch elsewhere from the same image. We then blackout everything outside of the object bounding box. $L_{JS}$ is the Jensen–Shannon divergence.
3. **多精度预测热力图关键点**：参见Fig.3，we propose a Multi-Precision Supervision (MPS) architecture: using three keypoint heads to predict groundtruth Gaussian heatmaps with different variance. We use σ equal to 8, 3 and 1.5 pixels respectively for the three keypoint heads, thus creating low, medium and high precision target heatmaps. In the testing phase, we only use the predicted heatmaps from the high-precision keypoint head to obtain the landmark coordinates.
4. **Landmark filtering**：A landmark prediction from the high-precision head will only be selected for the pose solver if it is verified by the corresponding medium-precision prediction, where ϵ is the verification threshold.
5. For each object model we apply the farthest point sampling (FPS) algorithm on the 3D point cloud and select 11 landmarks. 
    ![ROPE_archi](assets_pose/ROPE_archi.png)
    ![ROPE_MPS](assets_pose/ROPE_MPS.png)
    ![ROPE_result](assets_pose/ROPE_result.png)

</details>


<details>
<summary> <b> CheckerPose (arxiv 2023) 二值编码关键点坐标  </b> </summary>

- **摘要**：we propose a novel pose estimation algorithm, named CheckerPose due to the checkerboard-like binary pattern, which improves on three main aspects. 1) CheckerPose densely samples 3D keypoints from the surface of the 3D object and finds their 2D correspondences progressively in the image. 2) for our 3D-to-2D correspondence, we design a compact binary code representation for 2D image locations. 3) Thirdly, we adopt a graph neural network to explicitly model the interactions among the sampled 3D keypoints.

- **网络结构**
1. Compared with dense representations like heatmaps [48, 44] and vector-fields [49, 22], our novel representation needs only 2d+1 binary bits for each keypoint, thus greatly reduces the memory usage for dense keypoint localization. 
2. In addition, during inference, we can efficiently convert the binary codes to the 2D coordinates. Furthermore, our representation can be naturally predicted in a progressive way, which allows to gradually improve the localization via iterative refinements.
3. Compared with generating all bits at the network output layer, our progressive prediction enables image feature fusion at each refinement stage, as shown in Figure 2(b).
    ![CheckerPose_illu](assets_pose/CheckerPose_illu.png)
    ![CheckerPose_archi](assets_pose/CheckerPose_archi.png)
    ![CheckerPose_binary](assets_pose/CheckerPose_binary.png)
    ![CheckerPose_result](assets_pose/CheckerPose_result.png)
    ![CheckerPose_result2](assets_pose/CheckerPose_result2.png)

</details>


<details>
<summary> <b> OVE6D (CVPR 2022) 基于模板  </b> </summary>

- **摘要**：This paper proposes a universal framework, called OVE6D, for **model-based** 6D object pose estimation **from a single depth image and a target object mask**. Our model is trained using **purely synthetic data** rendered from ShapeNet, and, unlike most of the existing methods, it generalizes well on new real-world objects without any fine-tuning. We achieve this by **decomposing the 6D pose into viewpoint, inplane rotation** around the camera optical axis and translation, and introducing novel lightweight modules for estimating each component in a cascaded manner. The resulting
network contains **less than 4M parameters**.

- **网络结构**
1. 离线阶段构建模板库，按Fig.2所示，均匀采样视角viewpoint，然后渲染对应的深度图，提取对应的特征向量v，模板库则包括特征v、视角真值、三维模型及其id；如图Fig.3(B)所示，推理时可以挑选出top-K个视角作为候选；
2. 训练方式按照Fig.4(A)所示，V_gama是不同视角，V_theta是不同in-plane旋转，于是利用度量损失，要求特征v和v_theta，相比v和v_gama更相似，从而使特征向量v只关乎于视角，而无关in-plane旋转；
3. in-plane旋转按照Fig.4(B)回归得到，因为z和z_theta对应的输入只有in-plane的差异；
4. 因为构造了K个旋转量的候选，所以设置OCV验证模块进行打分排序；推理时，是让这K个旋转对应的K个视角下的深度图所对应的特征图，按照估计的in-plane旋转分别warp，然后和观测对应的特征图拼接送入OVE head，输出一致性得分；训练时，是按照Fig.4(C)，基于度量损失，让s_theta大于s_gama。
5. Location Refinement：对初始平移量的微调，基于了一个假设，不太理解，暂略~
    ![OVE6D_illu](assets_pose/OVE6D_illu.png)
    ![OVE6D_archi](assets_pose/OVE6D_archi.png)
    ![OVE6D_train](assets_pose/OVE6D_train.png)
    ![OVE6D_result](assets_pose/OVE6D_result.png)

</details>


<details>
<summary> <b> TexPose (CVPR 2023) 自监督,NeRF学纹理 </b> </summary>

- **摘要**：In this paper, we introduce neural texture learning for 6D object pose estimation **from synthetic data and a few unlabelled real images**. Our major contribution is a novel learning scheme which removes the drawbacks of previous works, namely the strong dependency on co-modalities or additional refinement. We 
decompose **self-supervision** for 6D object pose into **Texture learning and Pose learning**. We propose a surfel-conditioned adversarial training loss and a synthetic texture regularisation term to handle pose errors and segmentation imperfection.

- **区别**：Different from the previous attempts that heavily rely on render-and-compare for self-supervision [46,47], we instead propose to regard realistic textures of the objects as
an intermediate representation.

- **纹理**：We leverage neural radiance fields
(**NeRF**) [36] to embed the texture information due to its simplicity and capability of photorealistic view synthesis.

- **网络结构**
    ![TexPose_archi](assets_pose/TexPose_archi.png)
    ![TexPose_eq2](assets_pose/TexPose_eq2.png)
    ![TexPose_result](assets_pose/TexPose_result.png)

</details>


<details>
<summary> <b> GDR-Net (CVPR 2021) 基于匹配map回归位姿  </b> </summary>

- **摘要**：In this work, we perform an indepth investigation on both direct and indirect methods, and propose a simple yet effective Geometry-guided Direct Regression Network (GDR-Net) to learn the 6D pose in an **end-to-end** manner from dense **correspondence-base** intermediate geometric representations.

- **网络结构**
1. **Parameterization of 3D Rotation**: Common choices are unit quaternions [60,33,27], log quaternions [37], or Lie algebra-based vectors [11]. Nevertheless, it is well-known that all representations with four or fewer dimensions for 3D rotation have discontinuities in the Euclidean space. [65] proposed a novel continuous 6-dimensional representation for R in SO(3), which has proven promising [65, 25]. Specifically, the 6-dimensional representation
**R6d is defined as the first two columns of R**.
We propose to let the network predict the **allocentric representation** [24] of rotation Ra6d.
2. **Surface Region Attention Maps (MSRA)**: For each pixel we classify the corresponding regions,
thus the probabilities in the predicted MSRA implicitly represent the symmetry of an object. For instance, if a pixel is assigned to two potential fragments due to a plane of symmetry, Minimizing this assignment will return a probability of 0.5 for each fragment. Moreover, leveraging MSRA not only mitigates the influence of ambiguities but also acts as an auxiliary task on top of M3D (3x64x64 xyz_map).
    ![GDR-Net_illu](assets_pose/GDR-Net_illu.png)
    ![GDR-Net_archi](assets_pose/GDR-Net_archi.png)
    ![GDR-Net_result](assets_pose/GDR-Net_result.png)

</details>


<details>
<summary> <b> MegaPose (arxiv 2022) 渲染+比较 </b> </summary>

- **摘要**：We introduce MegaPose, a method to estimate the 6D pose of **novel objects** in a single RGB or RGB-D image, that is, objects unseen during training. At inference time, the method only assumes knowledge of (i) a **region of interest** displaying the object in the image and (ii) a **CAD model** of the observed object. The contributions are threefold. 1) We present a 6D pose refiner based on a **render & compare** strategy. 2) We leverage a network trained to classify whether the pose error between a synthetic rendering and an observed image of the same object can be corrected by the refiner. 3) We introduce a large scale synthetic dataset of photorealistic images. We train our approach on this large synthetic dataset and apply it **without retraining** to hundreds of novel objects in real images.

- **比较**：类别级方法受限于一个类别，而本文这类处理novel物体的方法，则受限于模型依赖，主观上二者地位一样，各有侧重。

- **网络结构**
1. **Components**: Similar to DeepIM [20] and CosyPose [4], our method consists of three components (1) object detection, (2) coarse pose estimation and (3) pose refinement.
2. **Inputs**: Our approach can accept either RGB or RGBD inputs, if depth is available the RGB and D images are concatenated before being passed into the network. 
3. **Coarse Estimator**: Since we are performing classification, our method can implicitly handle object symmetries, as multiple poses can be classified as correct.
4. **Refiner**: In order to generalize to novel objects, the network can infer the location of the anchor point O as the intersection point of camera rays that pass through the image center, see Figure 2(b).
5. **Training data**: All of of our methods are trained purely on synthetic data generated using BlenderProc [31]. We add data augmentation similar to CosyPose [4] to the RGB images which was shown to be a key to successful sim-to-real transfer. We also apply data augmentation to the depth images.
6. **Limitations**: 1) The most common failure mode is due to inaccurate initial pose estimates from the coarse model. 2) Another limitation is the runtime of our coarse model. We use M = 520 pose hypotheses per object which takes around 2.5 seconds to be rendered and evaluated by our coarse model. In a tracking scenario however, the coarse model is run just once at the initial frame and the object can be tracked using the refiner which runs at 20Hz.
    ![MegaPose_archi](assets_pose/MegaPose_archi.png)
    ![MegaPose_result](assets_pose/MegaPose_result.png)

</details>


<details>
<summary> <b> suo-slam (CVPR 2022) 关键点+g2o图优化   </b> </summary>

- **关键词**：依赖3D CAD模型；RGB输入；2D热力图关键点；3D人工标注关键点；同时优化相机和物体的pose。
- **隐含假设**：(1) 静态场景，即物体不动; (2) 场景中不含多实例，代码中是直接根据物体id进行前后帧的物体关联！
- **摘要**：We propose a **keypoint-based object-level SLAM** framework that can provide globally consistent 6DoF pose estimates for symmetric and asymmetric objects alike. To the best of our knowledge, our system is among **the first to utilize the camera pose information from SLAM to provide prior knowledge for tracking keypoints** on symmetric objects – ensuring that new measurements are consistent with the current 3D scene. Moreover, our semantic keypoint network is trained to predict the Gaussian covariance for the keypoints that captures the true error of the prediction, and thus is not only useful as a weight for the residuals in the system’s optimization problems, but also as a means to detect harmful statistical outliers without choosing a manual threshold...at a real-time speed...

- **算法框架**

    - **整体流程**：输入一帧RGB图片，分两路处理包含的物体：即先处理非对称物体，再处理对称物体，这么做的原因是，为了基于非对称物体估计当前帧的cam_pose，从而为后续处理对称物体时，构造prior。前端tracking看作是获取当前帧的cam_pose，后端优化看作是对obj_pose和cam_pose进行同步优化！
    - **符号约定**：本文中，将第一帧设定为world或称global坐标系(记为G)，obj_pose是T_O2G，cam_pose是T_G2C；
    - **估计obj_pose的方式**：基于bbox，将输入图片ROI_align到固定尺寸，和prior tensor(无prior时是0填充)堆叠，输入关键点网络，预测物体的2D关键点，然后结合3D model上标注的3D关键点，就可以PnP计算该obj相对cam的pose，T_O2C；对于第一帧的物体，T_O2C也是T_O2G。注意，场景中的物体pose会被维护记录下来，代码中是存于字典`self.obj_poses`中，对于第一次检测到的物体，若它被成功初始化，即估计出了T_O2G，则该pose会立马存入字典；对于在前序帧已经被初始化的物体，即字典中已有该物体的pose值，只有在当前帧估计的pose，比字典中存储的更准确时才会更新（比如取最近15帧统计重投影时的内点数来确定），对应代码中的re-init步骤！
    - **估计cam_pose的方式**：(1) 根据保存的前序帧估计的物体pose(T_O2G)，和当前帧基于PnP的物体pose(T_O2C)，基于RANSAC得当前帧cam_pose: T_G2C = T_O2C @ inv(T_O2G)；其中，RANSAC中要check的hypoth，看作是根据不同物体的"O"获得的T_G2C；(2) 如果估计失败，就构造物体的3D bbox(对应T_O2G的平移量)和2D bbox的center的匹配，然后PnP得cam_pose；（3）如果还失败，就用恒速模型！
    - **图优化**：(1) 构造object slam问题，传统slam中图优化的顶点包括cam位姿，和map_point的3D位置，这里的顶点包括cam位姿，和obj位姿！(2) 局部优化时，只优化当前帧的cam位姿；全局优化时，则连同obj位姿一起优化，所以edge对应有一元边和二元边两种情形！(3) 各3D关键点作为edge的参数传入，经过T_O2G和T_G2C和内参K转化为预测的图像坐标uv，对应的检测到的2D关键点作为edge的观测量obs，两者之差即为重投影误差error！(4) 主观上，物体级slam中的obj位姿顶点，好比是传统slam中若干相对固定的map_point的集合！因为对于某个物体，假设它有10个关键点被检测到，则该物体可以构造出10条二元边edge，每条edge一头连着cam位姿，一头连着obj位姿，可以发现这10条edge连接的顶点是共享的，只是传入的3D关键点参数和设置的观测值不同而已！所以，这10个关键点可以看作是传统slam中，位置相对固定的点集，优化时不能各自自由调整位置，而是始终约束具有固定的相对位置！
    - **图优化代码说明**：作者在g2o代码库中，新增了object_slam类型，定义了2种edge，实现上述(3)中的功能：
    `edge = g2o.EdgeSE3ProjectFromFixedObject(cam_k, model_pts[k], object_verts[obj_id])`, 和`edge = g2o.EdgeSE3ProjectFromObject(cam_k, model_pts[k])`；前者是一元边，用于局部优化当前帧，后者是二元边，用于全局优化！可见，相机内参和3D关键点是作为参数传入，对于一元边，物体位姿T_O2G也是作为参数传入，然后检测的2D关键点坐标充当edge的观测量：`edge.set_measurement(uv[k])`。另外，整个代码实现，加入了很多robust操作：比如check pose的有效性(不能让投影后的深度为"负"值)，比如根据edge.chi2()，将各关键点设置为inlier/outlier，从而动态调整参与图优化的edge集合！
    - **其它**：关于对预测结果添加置信度，可参考DenseFusion，GPV-Pose；关于不确定性建模，可参考suo-slam，CenterPoseTrack的Eq.(1)~(3)。
    ![suo-slam_pipe](assets_pose/suo-slam_pipe.png)

</details>


<details>
<summary> <b> ---  </b> </summary>

- **摘要**：

- **网络结构**
    ![](assets_pose/.png)
    ![](assets_pose/.png)

</details>


<details>
<summary> <b> ---  </b> </summary>

- **摘要**：

- **网络结构**
    ![](assets_pose/.png)
    ![](assets_pose/.png)

</details>


---
## Pose Tracking


<details>
<summary> <b> TP-AE (ICRA 2022) - GRU轨迹先验 + AE形状重建！ </b> </summary>

- Instance-level tracking; 考虑遮挡下的对称/低纹理物体的位姿估计；号称优于CosyPose, PoseRBPF；
- **摘要**：This paper focuses on the instance-level 6D pose tracking problem with a symmetric and textureless object under occlusion. The proposed TP-AE framework consists of a prediction step and a temporally primed pose estimation step. ... test on T-LESS dataset while running in real-time at 26 FPS.

- **网络结构**： 
    (1) 在每个time step，先验位姿估计模块，将历史位姿估计序列输入GRU-based网络，生成当前帧的位姿先验；
    (2) 预测的位姿先验，和当前帧的RGB-D数据，一并输入pose-image融合模块，生成RGB-Cloud pair，接着送入3个分支，分别预测物体旋转、平移和可见部分。
    (3) 注意只有训练阶段需要encoder和decoder一起学习latent code，推理阶段，不再需要decoder，因为如下图，是直接基于latent code去预测R/t；
    (4) 自：至少预测的平移量t的误差，来自2个方面，即GRU先验预测，和对$\Delta{T}$的预测。
    ![TPAE_archi](assets_pose/TPAE_archi.png)
    ![TPAE_fig4](assets_pose/TPAE_fig4.png)

</details>


<details>
<summary> <b> ...  </b> </summary>

- **摘要**：
- **网络结构**
    ![](assets_pose/.png)
    ![](assets_pose/.png)

</details>