#ifndef ROVIOLI_DEEP_RELOC_ROVIOLI_NODE_H_
#define ROVIOLI_DEEP_RELOC_ROVIOLI_NODE_H_
#include <memory>

#include <atomic>
#include <string>

#include <message-flow/message-flow.h>
#include <sensors/imu.h>

#include <rovioli/datasource-flow.h>
#include <rovioli/imu-camera-synchronizer-flow.h>
#include <rovioli/feature-tracking-flow.h>

#include "rovioli-deep-reloc/data-publisher-flow.h"
#include "rovioli-deep-reloc/localizer-flow.h"

namespace rovioli_deep_reloc {
class RovioliNode final {
 public:
  RovioliNode(
      const aslam::NCamera::Ptr& camera_system,
      vi_map::Imu::UniquePtr maplab_imu_sensor,
      const vi_map::ImuSigmas& rovio_imu_sigmas,
      const vi_map::VIMap* const vi_map,
      message_flow::MessageFlow* flow);
  ~RovioliNode();

  void start();
  void shutdown();

  std::atomic<bool>& isDataSourceExhausted();

 private:
  std::unique_ptr<rovioli::DataSourceFlow> datasource_flow_;
  std::unique_ptr<LocalizerFlow> localizer_flow_;
  std::unique_ptr<rovioli::ImuCameraSynchronizerFlow> synchronizer_flow_;
  std::unique_ptr<rovioli::FeatureTrackingFlow> tracker_flow_;
  std::unique_ptr<DataPublisherFlow> data_publisher_flow_;

  // Set to true once the data-source has played back all its data. Will never
  // be true for infinite data-sources (live-data).
  std::atomic<bool> is_datasource_exhausted_;
};
}  // namespace rovioli_deep_reloc
#endif  // ROVIOLI_DEEP_RELOC_ROVIOLI_NODE_H_
