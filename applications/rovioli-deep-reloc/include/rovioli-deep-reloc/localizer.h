#ifndef ROVIOLI_DEEP_RELOC_LOCALIZER_H_
#define ROVIOLI_DEEP_RELOC_LOCALIZER_H_

#include <vi-map/vi-map.h>
#include <loop-closure-handler/loop-detector-node.h>
#include <maplab-common/macros.h>
#include <vio-common/vio-types.h>

namespace rovioli_deep_reloc {

class Localizer {
 public:
  typedef vio::LocalizationResult::LocalizationMode LocalizationMode;

  Localizer() = delete;
  MAPLAB_POINTER_TYPEDEFS(Localizer);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Localizer(
      const vi_map::VIMap& localization_map,
      const bool visualize_localization);

  LocalizationMode getCurrentLocalizationMode() const;

  bool localizeNFrame(
      const aslam::VisualNFrame::ConstPtr& nframe,
      vio::LocalizationResult* localization_result) const;

 private:
  bool localizeNFrameGlobal(
      const aslam::VisualNFrame::ConstPtr& nframe,
      vio::LocalizationResult* localization_result) const;
  bool localizeNFrameMapTracking(
      const aslam::VisualNFrame::ConstPtr& nframe,
      vio::LocalizationResult* localization_result) const;

  LocalizationMode current_localization_mode_;
  loop_detector_node::LoopDetectorNode::UniquePtr global_loop_detector_;

  const vi_map::VIMap& localization_map_;
};

}  // namespace rovioli

#endif  // ROVIOLI_DEEP_RELOC_LOCALIZER_H_
