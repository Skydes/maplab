#include "loop-closure-handler/visualization/loop-closure-visualizer.h"

#include <visualization/color-palette.h>
#include <visualization/common-rviz-visualization.h>
#include <visualization/viz-primitives.h>
#include <opencv2/opencv.hpp>

DEFINE_bool(
    lc_visualize_outliers, false,
    "If outlier matches should be published on the loop closure topic.");

namespace loop_closure_visualization {

LoopClosureVisualizer::LoopClosureVisualizer()
    : loop_closures_topic_("loop_closures"),
      landmark_pairs_topic_("landmark_pairs") {}

LoopClosureVisualizer::~LoopClosureVisualizer() {}

void LoopClosureVisualizer::visualizeMergedLandmarks(
    const loop_closure_handler::LoopClosureHandler::
        MergedLandmark3dPositionVector& landmark_pairs_merged) {
  visualization::LineSegmentVector landmark_pairs;
  for (const loop_closure_handler::LoopClosureHandler::Vector3dPair&
           landmark_positions_pair : landmark_pairs_merged) {
    const Eigen::Vector3d& p_G_landmark_1 = landmark_positions_pair.first;
    const Eigen::Vector3d& p_G_landmark_2 = landmark_positions_pair.second;

    visualization::LineSegment landmark_connection;
    landmark_connection.alpha = 1.0;
    landmark_connection.color.red = 255;
    landmark_connection.color.green = 0;
    landmark_connection.color.blue = 100;
    landmark_connection.scale = 0.02;
    landmark_connection.from = p_G_landmark_1;
    landmark_connection.to = p_G_landmark_2;

    landmark_pairs.push_back(landmark_connection);
  }

  constexpr size_t kMarkerId = 0u;
  visualization::publishLines(
      landmark_pairs, kMarkerId, visualization::kDefaultMapFrame,
      visualization::kDefaultNamespace, landmark_pairs_topic_);
}

void LoopClosureVisualizer::visualizeKeyframeToStructureMatches(
    const vi_map::LoopClosureConstraintVector& inlier_constraints,
    const vi_map::LoopClosureConstraintVector& all_constraints,
    const LandmarkToLandmarkMap& merged_landmark_map,
    const vi_map::VIMap& map) {
  visualization::LineSegmentVector matches;
  vi_map::LandmarkIdSet added_landmarks;
  for (const vi_map::LoopClosureConstraint& constraint : inlier_constraints) {
    visualization::Color color;
    visualization::GetRandomRGBColor(&color);
    addKeyframeToStructureMatchMarker(
        constraint, color, merged_landmark_map, map, &matches,
        &added_landmarks);
  }

  if (FLAGS_lc_visualize_outliers) {
    for (const vi_map::LoopClosureConstraint& constraint : all_constraints) {
      vi_map::LoopClosureConstraint constraint_without_inliers;
      constraint_without_inliers.query_vertex_id = constraint.query_vertex_id;
      for (const vi_map::VertexKeyPointToStructureMatch& match :
           constraint.structure_matches) {
        if (added_landmarks.count(match.landmark_result) == 0u) {
          // The landmark was not added before so it's not an inlier. We will
          // visualize it as an outlier.
          constraint_without_inliers.structure_matches.push_back(match);
        }
      }
      // Visualize outliers in a dark-grey color.
      addKeyframeToStructureMatchMarker(
          constraint_without_inliers, visualization::kCommonDarkGray,
          merged_landmark_map, map, &matches, &added_landmarks);
    }
  }

  constexpr size_t kInlierMarkerId = 0u;
  visualization::publishLines(
      matches, kInlierMarkerId, visualization::kDefaultMapFrame,
      visualization::kDefaultNamespace, loop_closures_topic_);
}

void LoopClosureVisualizer::addKeyframeToStructureMatchMarker(
    const vi_map::LoopClosureConstraint& constraint,
    const visualization::Color& color,
    const LandmarkToLandmarkMap& merged_landmark_map, const vi_map::VIMap& map,
    visualization::LineSegmentVector* matches,
    vi_map::LandmarkIdSet* added_landmarks) {
  CHECK_NOTNULL(matches);
  CHECK_NOTNULL(added_landmarks);

  CHECK(constraint.query_vertex_id.isValid());
  const Eigen::Vector3d vertex_p_G =
      map.getVertex_G_p_I(constraint.query_vertex_id);

  visualization::LineSegment line_segment;
  line_segment.from = vertex_p_G;
  line_segment.scale = 0.03;
  line_segment.alpha = 0.6;

  line_segment.color.red = color.red;
  line_segment.color.green = color.green;
  line_segment.color.blue = color.blue;

  for (const vi_map::VertexKeyPointToStructureMatch& match :
       constraint.structure_matches) {
    vi_map::LandmarkId landmark_id = match.landmark_result;
    added_landmarks->emplace(landmark_id);

    LandmarkToLandmarkMap::const_iterator it_landmark_already_changed;
    do {
      it_landmark_already_changed = merged_landmark_map.find(landmark_id);
      if (it_landmark_already_changed != merged_landmark_map.end()) {
        landmark_id = it_landmark_already_changed->second;
      }
    } while (it_landmark_already_changed != merged_landmark_map.end());

    CHECK(landmark_id.isValid());
    line_segment.to = map.getLandmark_G_p_fi(landmark_id);
    matches->push_back(line_segment);
  }
}

void LoopClosureVisualizer::visualizeKeyframeToStructureMatch(
    const vi_map::VertexKeyPointToStructureMatchList& structure_matches,
    const Eigen::Vector3d& query_position,
    const summary_map::LocalizationSummaryMap& localization_summary_map) {
  visualization::LineSegmentVector loop_closures;

  visualization::LineSegment line_segment;
  line_segment.from = query_position;
  line_segment.scale = 0.05;
  line_segment.color.red = 170;
  line_segment.color.green = 170;
  line_segment.color.blue = 220;
  line_segment.alpha = 0.4;

  Eigen::Matrix3Xd inlier_landmarks;
  inlier_landmarks.resize(Eigen::NoChange, structure_matches.size());

  int idx = 0;
  for (const vi_map::VertexKeyPointToStructureMatch& match :
       structure_matches) {
    const Eigen::Vector3d landmark_p_G =
        localization_summary_map.getGLandmarkPosition(match.landmark_result);
    line_segment.to = landmark_p_G;
    loop_closures.push_back(line_segment);

    CHECK_LT(idx, inlier_landmarks.cols());
    inlier_landmarks.col(idx) = landmark_p_G;

    ++idx;
  }

  constexpr double kInlierAlpha = 0.8;
  visualization::publish3DPointsAsPointCloud(
      inlier_landmarks, visualization::kCommonYellow, kInlierAlpha,
      visualization::kDefaultMapFrame, "loopclosure_inliers");

  constexpr size_t kMarkerId = 0u;
  visualization::publishLines(
      loop_closures, kMarkerId, visualization::kDefaultMapFrame,
      visualization::kDefaultNamespace, loop_closures_topic_);
}

void LoopClosureVisualizer::visualizeFullMapDatabase(
    const vi_map::MissionIdList& missions, const vi_map::VIMap& map) {
  vi_map_plotter_.publishVertices(map, missions);
  vi_map_plotter_.publishEdges(map, missions);
}

void LoopClosureVisualizer::visualizeKeyframeToStructureMatch(
    const vi_map::VertexKeyPointToStructureMatchList& structure_matches,
    const Eigen::Vector3d& query_position,
    const vi_map::VIMap* localization_map) {
  visualization::LineSegmentVector loop_closures;

  visualization::LineSegment line_segment;
  line_segment.from = query_position;
  line_segment.scale = 0.05;
  line_segment.color.red = 170;
  line_segment.color.green = 170;
  line_segment.color.blue = 220;
  line_segment.alpha = 0.4;

  Eigen::Matrix3Xd inlier_landmarks;
  inlier_landmarks.resize(Eigen::NoChange, structure_matches.size());

  int idx = 0;
  for (const vi_map::VertexKeyPointToStructureMatch& match :
       structure_matches) {
    const Eigen::Vector3d landmark_p_G =
        localization_map->getLandmark_G_p_fi(match.landmark_result);
    line_segment.to = landmark_p_G;
    loop_closures.push_back(line_segment);

    CHECK_LT(idx, inlier_landmarks.cols());
    inlier_landmarks.col(idx) = landmark_p_G;

    ++idx;
  }

  constexpr double kInlierAlpha = 0.8;
  visualization::publish3DPointsAsPointCloud(
      inlier_landmarks, visualization::kCommonYellow, kInlierAlpha,
      visualization::kDefaultMapFrame, "loopclosure_inliers");

  constexpr size_t kMarkerId = 0u;
  visualization::publishLines(
      loop_closures, kMarkerId, visualization::kDefaultMapFrame,
      visualization::kDefaultNamespace, loop_closures_topic_);
}

void LoopClosureVisualizer::visualizeSummaryMapDatabase(
    const summary_map::LocalizationSummaryMap& localization_summary_map) {
  constexpr double kAlpha = 0.7;
  visualization::publish3DPointsAsPointCloud(
      localization_summary_map.GLandmarkPosition().cast<double>(),
      visualization::kCommonDarkGray, kAlpha, visualization::kDefaultMapFrame,
      "loopclosure_database");
}

void LoopClosureVisualizer::visualizeDescriptorMatches(
    const vi_map::VertexKeyPointToStructureMatchList& structure_matches,
    const loop_closure::ProjectedImage::Ptr& projected_query_image,
    const aslam::VisualFrame::ConstPtr& query_frame,
    const vi_map::VIMap* map,
    const std::unordered_map<vi_map::VisualFrameIdentifier,
                             std::shared_ptr<loop_closure::ProjectedImage>>&
        index_frame_projected_image_map) {
  std::unordered_map<vi_map::VisualFrameIdentifier,
                     vi_map::VertexKeyPointToStructureMatchList>
      frame_id_to_matches;
  for (const vi_map::VertexKeyPointToStructureMatch& match :
       structure_matches) {
    frame_id_to_matches[match.frame_identifier_result].push_back(match);
  }
  vi_map::VisualFrameIdentifier max_matches_frame_id;
  size_t num_max_matches = 0;
  for (const auto& frame_id_matches_pair : frame_id_to_matches) {
    if (frame_id_matches_pair.second.size() > num_max_matches) {
      num_max_matches = frame_id_matches_pair.second.size();
      max_matches_frame_id = frame_id_matches_pair.first;
    }
  }
  const vi_map::VertexKeyPointToStructureMatchList& selected_matches =
      frame_id_to_matches[max_matches_frame_id];
  std::vector<cv::KeyPoint> cv_keypoints_1, cv_keypoints_2;
  std::vector<cv::DMatch> cv_matches;
  size_t cnt = 0;
  for (const vi_map::VertexKeyPointToStructureMatch& match :
       selected_matches) {
    const Eigen::Vector2d& kp1 = projected_query_image->measurements.col(
        match.keypoint_index_query);
    cv_keypoints_1.emplace_back(kp1(0), kp1(1), 1);
    const loop_closure::ProjectedImage& proj_image_result =
        *index_frame_projected_image_map.at(match.frame_identifier_result);
    bool found = false;
    for (size_t i = 0; i < proj_image_result.landmarks.size(); i++) {
      if (proj_image_result.landmarks[i] == match.landmark_result) {
        CHECK_LT(i, proj_image_result.measurements.cols());
        const Eigen::Vector2d& kp2 = proj_image_result.measurements.col(i);
        cv_keypoints_2.emplace_back(kp2(0), kp2(1), 1);
        found = true;
        break;
      }
    }
    CHECK(found);
    cv_matches.emplace_back(cnt, cnt, 1);
    ++cnt;
  }

  const cv::Mat& raw_image_1 = query_frame->getRawImage();
  cv::Mat raw_image_2;
  const vi_map::Vertex& max_matches_vertex = map->getVertex(
      max_matches_frame_id.vertex_id);
  map->getRawImage(
        max_matches_vertex, max_matches_frame_id.frame_index, &raw_image_2);
  cv::Mat patch_image;
  cv::drawMatches(raw_image_1, cv_keypoints_1, raw_image_2, cv_keypoints_2,
                  cv_matches, patch_image);
  const std::string topic = "lc/match_query_retrieved";
  visualization::RVizVisualizationSink::publish(topic, patch_image);
}

}  // namespace loop_closure_visualization
