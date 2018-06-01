#ifndef ROVIOLI_DEEP_RELOC_LOCALIZER_FLOW_H_
#define ROVIOLI_DEEP_RELOC_LOCALIZER_FLOW_H_

#include <vi-map/vi-map.h>
#include <message-flow/message-flow.h>
#include <vio-common/vio-types.h>

#include "rovioli-deep-reloc/localizer.h"

namespace rovioli_deep_reloc {
class LocalizerFlow {
 public:
  explicit LocalizerFlow(
      const vi_map::VIMap& localization_map,
      const bool visualize_localization);

  void attachToMessageFlow(message_flow::MessageFlow* flow);

 private:
  void processTrackedNFrameAndImu(
      const vio::SynchronizedNFrameImu::ConstPtr& nframe_imu);

  Localizer localizer_;

  std::function<void(vio::LocalizationResult::ConstPtr)>
      publish_localization_result_;

  // All members below are used for throttling the localizations.
  const int64_t min_localization_timestamp_diff_ns_;
  int64_t previous_nframe_timestamp_ns_;
  mutable std::mutex m_previous_nframe_timestamp_ns_;
};
}  // namespace rovioli_deep_reloc
#endif  // ROVIOLI_DEEP_RELOC_LOCALIZER_FLOW_H_
