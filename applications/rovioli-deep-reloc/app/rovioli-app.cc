#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vi-map/vi-map.h>
#include <vi-map/vi-map-serialization.h>
#include <maplab-common/sigint-breaker.h>
#include <maplab-common/threading-helpers.h>
#include <message-flow/message-dispatcher-fifo.h>
#include <message-flow/message-flow.h>
#include <aslam/common/timer.h>
#include <ros/ros.h>
#include <sensors/imu.h>
#include <sensors/sensor-factory.h>
#include <signal.h>

#include "rovioli-deep-reloc/rovioli-node.h"

DEFINE_string(
    vio_localization_vimap_folder, "",
    "Path to a full VI-map used for localization.");
DEFINE_string(
    ncamera_calibration, "ncamera.yaml", "Path to camera calibration yaml.");
DEFINE_string(
    imu_parameters_maplab, "imu-maplab.yaml",
    "Path to the imu configuration yaml for MAPLAB.");
DEFINE_string(
    external_imu_parameters_rovio, "",
    "Optional, path to the IMU configuration yaml for ROVIO. If none is "
    "provided the maplab values will be used for ROVIO as well.");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  ros::init(argc, argv, "rovioli");
  ros::NodeHandle nh;

  // Load map.
  CHECK(!FLAGS_vio_localization_vimap_folder.empty());
  std::unique_ptr<vi_map::VIMap> localization_map(new vi_map::VIMap);
  CHECK(vi_map::serialization::loadMapFromFolder(
          FLAGS_vio_localization_vimap_folder, localization_map.get()));

  // Load camera calibration and imu parameters.
  aslam::NCamera::Ptr camera_system =
      aslam::NCamera::loadFromYaml(FLAGS_ncamera_calibration);
  CHECK(camera_system) << "Could not load the camera calibration from: \'"
                       << FLAGS_ncamera_calibration << "\'";

  vi_map::Imu::UniquePtr maplab_imu_sensor =
      vi_map::createFromYaml<vi_map::Imu>(FLAGS_imu_parameters_maplab);
  CHECK(maplab_imu_sensor)
      << "Could not load IMU parameters for MAPLAB from: \'"
      << FLAGS_imu_parameters_maplab << "\'";
  CHECK(maplab_imu_sensor->getImuSigmas().isValid());

  // Optionally, load external values for the ROVIO sigmas; otherwise also use
  // the maplab values for ROVIO.
  vi_map::ImuSigmas rovio_imu_sigmas;
  if (!FLAGS_external_imu_parameters_rovio.empty()) {
    CHECK(rovio_imu_sigmas.loadFromYaml(FLAGS_external_imu_parameters_rovio))
        << "Could not load IMU parameters for ROVIO from: \'"
        << FLAGS_external_imu_parameters_rovio << "\'";
    CHECK(rovio_imu_sigmas.isValid());
  } else {
    rovio_imu_sigmas = maplab_imu_sensor->getImuSigmas();
  }

  // Construct the application.
  ros::AsyncSpinner ros_spinner(common::getNumHardwareThreads());
  std::unique_ptr<message_flow::MessageFlow> flow(
      message_flow::MessageFlow::create<message_flow::MessageDispatcherFifo>(
          common::getNumHardwareThreads()));

  rovioli_deep_reloc::RovioliNode rovio_localization_node(
      camera_system, std::move(maplab_imu_sensor), rovio_imu_sigmas,
      localization_map.get(), flow.get());

  // Start the pipeline. The ROS spinner will handle SIGINT for us and abort
  // the application on CTRL+C.
  ros_spinner.start();
  rovio_localization_node.start();

  std::atomic<bool>& end_of_days_signal_received =
      rovio_localization_node.isDataSourceExhausted();
  while (ros::ok() && !end_of_days_signal_received.load()) {
    VLOG_EVERY_N(1, 10) << "\n" << flow->printDeliveryQueueStatistics();
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  rovio_localization_node.shutdown();
  flow->shutdown();
  flow->waitUntilIdle();

  timing::Timing::Print(std::cout);

  return 0;
}
