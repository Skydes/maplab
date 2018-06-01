#include "rovioli-deep-reloc/rovioli-node.h"

#include <string>

#include <aslam/cameras/ncamera.h>
#include <vi-map/vi-map.h>
#include <message-flow/message-flow.h>
#include <message-flow/message-topic-registration.h>
#include <vio-common/vio-types.h>

#include <rovioli/datasource-flow.h>
#include <rovioli/imu-camera-synchronizer-flow.h>
#include <rovioli/feature-tracking-flow.h>

#include "rovioli-deep-reloc/data-publisher-flow.h"
#include "rovioli-deep-reloc/localizer-flow.h"

namespace rovioli_deep_reloc {
RovioliNode::RovioliNode(
    const aslam::NCamera::Ptr& camera_system,
    vi_map::Imu::UniquePtr maplab_imu_sensor,
    const vi_map::ImuSigmas& rovio_imu_sigmas,
    const vi_map::VIMap* const localization_map,
    message_flow::MessageFlow* flow)
    : is_datasource_exhausted_(false) {
  CHECK(camera_system);
  CHECK(maplab_imu_sensor);
  CHECK_NOTNULL(localization_map);
  CHECK_NOTNULL(flow);

  datasource_flow_.reset(
      new rovioli::DataSourceFlow(*camera_system, *maplab_imu_sensor));
  datasource_flow_->attachToMessageFlow(flow);

  constexpr bool kVisualizeLocalization = true;
  localizer_flow_.reset(
      new LocalizerFlow(*localization_map, kVisualizeLocalization));
  localizer_flow_->attachToMessageFlow(flow);

  // Launch the synchronizer after the localizer because creating the
  // localization database can take some time. This can cause the
  // synchronizer's detection of missing image or IMU measurements to fire
  // early.
  synchronizer_flow_.reset(
      new rovioli::ImuCameraSynchronizerFlow(camera_system));
  synchronizer_flow_->attachToMessageFlow(flow);

  tracker_flow_.reset(
      new rovioli::FeatureTrackingFlow(camera_system, *maplab_imu_sensor));
  tracker_flow_->attachToMessageFlow(flow);

  data_publisher_flow_.reset(new DataPublisherFlow);
  data_publisher_flow_->attachToMessageFlow(flow);

  // Subscribe to end of days signal from the datasource.
  datasource_flow_->registerEndOfDataCallback(
      [&]() { is_datasource_exhausted_.store(true); });
}

RovioliNode::~RovioliNode() {
  shutdown();
}

void RovioliNode::start() {
  CHECK(!is_datasource_exhausted_.load())
      << "Cannot start localization node after the "
      << "end-of-days signal was received!";
  datasource_flow_->startStreaming();
  VLOG(1) << "Starting data source...";
}

void RovioliNode::shutdown() {
  datasource_flow_->shutdown();
  VLOG(1) << "Closing data source...";
}

std::atomic<bool>& RovioliNode::isDataSourceExhausted() {
  return is_datasource_exhausted_;
}

}  // namespace rovioli_deep_reloc
