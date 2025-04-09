

# OodDINO

OodDINO is a state-of-the-art framework for anomaly detection on road anomaly datasets. It achieves top performance on both the [RoadAnomaly Benchmark](https://paperswithcode.com/sota/anomaly-detection-on-road-anomaly) and the [SegmentMeIfYouCan Benchmark](https://segmentmeifyoucan.com/leaderboard).

## Framework 

Our framework is built on top of [MMDetection](https://github.com/open-mmlab/mmdetection), a powerful open-source object detection toolbox. Please follow the [installation instructions](https://mmdetection.readthedocs.io/en/latest/install.html) to set up the environment.

## Datasets Preparation

Before using the framework, you need to download and prepare the datasets. Here are the steps:

### Validation Set

Download the validation set from the following link:  
[Validation Dataset](https://drive.google.com/file/d/1IbD_zl5MecMCEj6ozGh40FE5u-MaxPLh/view?usp=sharing)

After downloading, the dataset structure should look like this:

-- val
-- road_anomaly ... -- segment_me ...



### Training Set

Download the training set from the following link:  
[Training Dataset](https://drive.google.com/file/d/1k25FpVP4pG3ER3eXsR-go_iprZMEdEae/view?usp=sharing)

The training dataset structure should look like this:

-- train_dataset -- offline_dataset ... -- offline_dataset_score ... -- offline_dataset_score_view ... -- ood.json

## Framework Details

The framework utilizes uncertainty fusion layers. These layers can be found in the `layers` directory of the codebase. Please refer to the code for detailed implementation and usage of these layers.

## Baseline

The baseline for our framework is provided by [RPL](https://github.com/yyliu01/RPL). Please follow the instructions below to install and set it up.


### Pre-trained Weights

You can download the pre-trained weights from the following link:  
[Pre-trained Weights](https://drive.google.com/file/d/1osPT__BIqCYrBT0F-Dmi2IfbUSmuUZPd/view?usp=drive_link)

Please place the downloaded weights in the appropriate directory as per the configuration files.

Replace the threshold-net section with the RPL training code as follows:

Locate the threshold-net section of your code and replace it with the relevant code from the RPL repository for training. You can follow the instructions in the RPL repository to adapt the training process.

you can perform inference using the following command:

1. Go to the RPL directory, Run the inference script:

   ```
   cd /rpl_corocl.code
   python test.py
   ```


Acknowledgements
This framework is based on MMDetection.

The OodDINO framework has been evaluated on the RoadAnomaly and SegmentMeIfYouCan benchmarks.

The baseline for this framework is provided by RPL.

