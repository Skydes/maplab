#include "rovioli-deep-reloc/localizer.h"

#include <aslam/common/occupancy-grid.h>
#include <gflags/gflags.h>
#include <vi-map/vi-map.h>
#include <loop-closure-handler/loop-detector-node.h>
#include <vio-common/vio-types.h>

#include "rovioli-deep-reloc/localizer-helpers.h"

namespace rovioli_deep_reloc {
Localizer::Localizer(
    const vi_map::VIMap& localization_map,
    const bool visualize_localization)
    : localization_map_(localization_map) {
  current_localization_mode_ = Localizer::LocalizationMode::kGlobal;

  global_loop_detector_.reset(new loop_detector_node::LoopDetectorNode);

  CHECK(global_loop_detector_ != nullptr);
  if (visualize_localization) {
    global_loop_detector_->instantiateVisualizer();
  }

  LOG(INFO) << "Building localization database...";
  vi_map::LandmarkIdList all_landmark_ids_list;
  localization_map_.getAllIds(&all_landmark_ids_list);
  vi_map::LandmarkIdSet all_landmark_ids_set(
      all_landmark_ids_list.begin(), all_landmark_ids_list.end());
  global_loop_detector_->addLandmarkSetToDatabase(
      all_landmark_ids_set, localization_map_);
  LOG(INFO) << "Done.";
}

Localizer::LocalizationMode Localizer::getCurrentLocalizationMode() const {
  return current_localization_mode_;
}

bool Localizer::localizeNFrame(
    const aslam::VisualNFrame::ConstPtr& nframe,
    vio::LocalizationResult* localization_result) const {
  CHECK(nframe);
  CHECK_NOTNULL(localization_result);

  bool result = false;
  switch (current_localization_mode_) {
    case Localizer::LocalizationMode::kGlobal:
      result = localizeNFrameGlobal(nframe, localization_result);
      break;
    case Localizer::LocalizationMode::kMapTracking:
      result = localizeNFrameMapTracking(nframe, localization_result);
      break;
    default:
      LOG(FATAL) << "Unknown localization mode.";
      break;
  }

  //localization_result->summery_map_id = localization_map_.id();
  localization_result->timestamp_ns = nframe->getMinTimestampNanoseconds();
  localization_result->nframe_id = nframe->getId();
  localization_result->localization_type = current_localization_mode_;
  return result;
}

bool Localizer::localizeNFrameGlobal(
    const aslam::VisualNFrame::ConstPtr& nframe,
    vio::LocalizationResult* localization_result) const {
  CHECK_NOTNULL(localization_result);

  constexpr bool kSkipUntrackedKeypoints = false;
  unsigned int num_lc_matches;
  vi_map::VertexKeyPointToStructureMatchList inlier_structure_matches;
  constexpr pose_graph::VertexId* kVertexIdClosestToStructureMatches = nullptr;
  LOG(INFO) << "Querying loop closure for NFrame.";
  const bool success = global_loop_detector_->findNFrameInDatabase(
      *nframe, kSkipUntrackedKeypoints,
      const_cast<vi_map::VIMap*>(&localization_map_),  // not nice
      &localization_result->T_G_I_lc_pnp, &num_lc_matches,
      &inlier_structure_matches, kVertexIdClosestToStructureMatches);

  if (!success || inlier_structure_matches.empty()) {
    return false;
  }

  convertVertexKeyPointToStructureMatchListToLocalizationResult(
      localization_map_, *nframe, inlier_structure_matches,
      localization_result);

  return true;
}

bool Localizer::localizeNFrameMapTracking(
    const aslam::VisualNFrame::ConstPtr& /*nframe*/,
    vio::LocalizationResult* /*localization_result*/) const {
  LOG(FATAL) << "Not implemented yet.";
  return false;
}

}  // namespace rovioli_deep_reloc
