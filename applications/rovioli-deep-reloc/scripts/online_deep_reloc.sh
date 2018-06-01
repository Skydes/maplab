#!/usr/bin/env bash

LOCALIZATION_MAP=$1
ROSBAG=$2
RETRIEVAL_MODEL=$3
RETRIEVAL_INDEX=$4

NCAMERA_CALIBRATION="$ROVIO_CONFIG_DIR/ncamera-euroc.yaml"
IMU_PARAMETERS_MAPLAB="$ROVIO_CONFIG_DIR/imu-adis16488.yaml"
REST=$@

rosrun rovioli_deep_reloc rovioli \
  --alsologtostderr=1 \
  --v=2 \
  --ncamera_calibration=$NCAMERA_CALIBRATION  \
  --imu_parameters_maplab=$IMU_PARAMETERS_MAPLAB \
  --publish_debug_markers  \
  --vio_localization_vimap_folder=$LOCALIZATION_MAP \
  --lc_use_deep_retrieval=1 \
  --lc_deep_retrieval_model_path=$RETRIEVAL_MODEL \
  --lc_deep_retrieval_index_path=$RETRIEVAL_INDEX \
  --lc_do_covisibility_filtering=0 \
  --lc_num_neighbors=2 \
  --lc_use_better_descriptors=1 \
  --lc_use_lowe_ratio_test=1 \
  --lc_detector_engine="kd_tree" \
  --lc_knn_epsilon=3.0 \
  --datasource_type="rosbag" \
  --datasource_rosbag=$ROSBAG $REST


