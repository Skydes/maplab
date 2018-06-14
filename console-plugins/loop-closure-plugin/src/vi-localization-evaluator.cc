#include "loop-closure-plugin/vi-localization-evaluator.h"

#include <localization-evaluator/localization-evaluator.h>
#include <localization-evaluator/mission-aligner.h>
#include <maplab-common/file-system-tools.h>
#include <vi-map/unique-id.h>
#include <vi-map/vi-map.h>

namespace loop_closure_plugin {

VILocalizationEvaluator::VILocalizationEvaluator(
    vi_map::VIMap* map, visualization::ViwlsGraphRvizPlotter* plotter)
    : map_(map), plotter_(plotter) {
  CHECK_NOTNULL(map_);
}

void VILocalizationEvaluator::alignMissionsForEvaluation(
    const vi_map::MissionId& query_mission_id) {
  vi_map::MissionIdList all_mission_ids;
  map_->getAllMissionIds(&all_mission_ids);

  vi_map::MissionIdSet db_mission_ids(
      all_mission_ids.begin(), all_mission_ids.end());
  CHECK_GT(db_mission_ids.count(query_mission_id), 0u);
  db_mission_ids.erase(query_mission_id);

  constexpr bool kAlignMapMissions = true;
  constexpr bool kOptimizeOnlyQueryMission = false;
  localization_evaluator::alignAndCooptimizeMissionsWithoutLandmarkMerge(
      query_mission_id, db_mission_ids, kAlignMapMissions,
      kOptimizeOnlyQueryMission, map_);
}

void VILocalizationEvaluator::evaluateLocalizationPerformance(
    const vi_map::MissionId& query_mission_id) {
  vi_map::MissionIdList all_mission_ids;
  map_->getAllMissionIds(&all_mission_ids);

  vi_map::MissionIdSet db_mission_ids(
      all_mission_ids.begin(), all_mission_ids.end());
  CHECK_GT(db_mission_ids.count(query_mission_id), 0u);
  db_mission_ids.erase(query_mission_id);

  // Collect all database landmarks.
  vi_map::LandmarkIdSet selected_landmarks;
  for (const vi_map::MissionId& mission_id : db_mission_ids) {
    vi_map::LandmarkIdList mission_landmarks;
    map_->getAllLandmarkIdsInMission(mission_id, &mission_landmarks);
    selected_landmarks.insert(
        mission_landmarks.begin(), mission_landmarks.end());
  }
  LOG(INFO) << "Will query against " << selected_landmarks.size()
            << " landmarks.";

  localization_evaluator::MissionEvaluationStats mission_statistics;
  localization_evaluator::LocalizationEvaluator benchmark(
      selected_landmarks, map_);
  LOG(INFO) << "Evaluating the localizations.";
  benchmark.evaluateMission(query_mission_id, &mission_statistics);

  if (mission_statistics.num_vertices > 0u) {
    LOG(INFO) << "Recall: "
            << static_cast<double>(mission_statistics.successful_localizations) /
                   mission_statistics.num_vertices
            << " (" << mission_statistics.successful_localizations << "/"
            << mission_statistics.num_vertices << ")";
    LOG(INFO) << "Wrong localizations: "
            << static_cast<double>(mission_statistics.bad_localization_p_G_I.size()) /
                   mission_statistics.num_vertices
            << " (" << mission_statistics.bad_localization_p_G_I.size() << "/"
            << mission_statistics.num_vertices << ")";

    if (plotter_) {
      plotter_->visualizeMap(*map_);
      constexpr size_t kMarkerId = 0u;

      visualization::SphereVector good_localization_spheres;
      for (const Eigen::Vector3d& p_G_I :
           mission_statistics.localization_p_G_I) {
        visualization::Sphere sphere;
        sphere.position = p_G_I;
        sphere.radius = 1.0;
        sphere.color = visualization::kCommonGreen;
        sphere.alpha = 0.8;
        good_localization_spheres.push_back(sphere);
      }
      visualization::publishSpheres(
          good_localization_spheres, kMarkerId, visualization::kDefaultMapFrame,
          "loc_eval", "successful_localizations");

      visualization::SphereVector bad_localization_spheres;
      for (const Eigen::Vector3d& p_G_I :
           mission_statistics.bad_localization_p_G_I) {
        visualization::Sphere sphere;
        sphere.position = p_G_I;
        sphere.radius = 1.0;
        sphere.color = visualization::kCommonRed;
        sphere.alpha = 0.8;
        bad_localization_spheres.push_back(sphere);
      }
      visualization::publishSpheres(
          bad_localization_spheres, kMarkerId, visualization::kDefaultMapFrame,
          "loc_eval", "wrong_localizations");
    }
  } else {
    LOG(WARNING) << "No vertices evaluated!";
  }
}

}  // namespace loop_closure_plugin
