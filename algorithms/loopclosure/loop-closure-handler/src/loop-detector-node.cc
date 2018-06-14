#include "loop-closure-handler/loop-detector-node.h"

#include <algorithm>
#include <mutex>
#include <sstream>  // NOLINT
#include <string>
#include <fstream>

#include <Eigen/Geometry>
#include <aslam/common/statistics/statistics.h>
#include <descriptor-projection/descriptor-projection.h>
#include <descriptor-projection/flags.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <localization-summary-map/localization-summary-map.h>
#include <loopclosure-common/types.h>
#include <maplab-common/accessors.h>
#include <maplab-common/geometry.h>
#include <maplab-common/multi-threaded-progress-bar.h>
#include <maplab-common/parallel-process.h>
#include <maplab-common/file-system-tools.h>
#include <maplab-common/progress-bar.h>
#include <matching-based-loopclosure/detector-settings.h>
#include <matching-based-loopclosure/loop-detector-interface.h>
#include <matching-based-loopclosure/matching-based-engine.h>
#include <matching-based-loopclosure/scoring.h>
#include <vi-map/landmark-quality-metrics.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <maplab-common/eigen-proto.h>
#include <deep-relocalization/place-retrieval.h>
#include <deep-relocalization/descriptor_index.pb.h>

#include "loop-closure-handler/loop-closure-handler.h"
#include "loop-closure-handler/visualization/loop-closure-visualizer.h"
#include "loop-closure-handler/debug_fusion.pb.h"  // DEBUG

DEFINE_bool(
    lc_filter_underconstrained_landmarks, true,
    "If underconstrained landmarks should be filtered for the "
    "loop-closure.");
DEFINE_bool(lc_use_random_pnp_seed, true, "Use random seed for pnp RANSAC.");

DEFINE_bool(lc_use_deep_retrieval, false, "");
DEFINE_string(lc_deep_retrieval_model_path, "", "");
DEFINE_string(lc_deep_retrieval_index_path, "", "");
DEFINE_uint64(lc_deep_retrieval_num_nn, 20, "");
DEFINE_double(lc_deep_retrieval_max_nn_distance, 2, "");
DEFINE_uint64(lc_deep_retrieval_expand_components, 0, "");
DEFINE_bool(lc_use_better_descriptors, false, "");
DEFINE_bool(lc_assume_perfect_retrieval, false, "");
DEFINE_bool(lc_detect_sift_scale, false, "");

#define ADD_DEBUG_STATS 1

namespace loop_detector_node {
LoopDetectorNode::LoopDetectorNode()
    : use_random_pnp_seed_(FLAGS_lc_use_random_pnp_seed),
      use_deep_retrieval_(FLAGS_lc_use_deep_retrieval),
      use_better_descriptors_(FLAGS_lc_use_better_descriptors) {
  matching_based_loopclosure::MatchingBasedEngineSettings
      matching_engine_settings;
  loop_detector_ =
      std::make_shared<matching_based_loopclosure::MatchingBasedLoopDetector>(
          matching_engine_settings);

  if (use_better_descriptors_) {
    better_descriptor_extractor_ = cv::xfeatures2d::SIFT::create();
    CHECK_EQ(
        static_cast<int>(matching_engine_settings.detector_engine_type),
        static_cast<int>(matching_based_loopclosure::MatchingBasedEngineSettings
          ::DetectorEngineType::kMatchingLDKdTree));
  }

  if (use_deep_retrieval_) {
    std::string retrieval_model_path = FLAGS_lc_deep_retrieval_model_path;
    std::string retrieval_index_path = FLAGS_lc_deep_retrieval_index_path;
    CHECK(!retrieval_model_path.empty());
    CHECK(!retrieval_index_path.empty());

    deep_retrieval_.reset(new PlaceRetrieval(retrieval_model_path));
    deep_relocalization::proto::DescriptorIndex proto_retrieval_index;
    std::fstream input(retrieval_index_path, std::ios::in | std::ios::binary);
    CHECK(proto_retrieval_index.ParseFromIstream(&input));
    deep_retrieval_->LoadIndex(proto_retrieval_index);
  }
}

const std::string LoopDetectorNode::serialization_filename_ =
    "loop_detector_node";

std::string LoopDetectorNode::printStatus() const {
  std::stringstream ss;
  ss << "Loop-detector status:" << std::endl;
  if (loop_detector_ != nullptr) {
    ss << "\tNum entries:" << loop_detector_->NumEntries() << std::endl;
    ss << "\tNum descriptors: " << loop_detector_->NumDescriptors()
       << std::endl;
  } else {
    ss << "\t NULL" << std::endl;
  }
  return ss.str();
}

bool LoopDetectorNode::convertFrameMatchesToConstraint(
    const loop_closure::FrameIdMatchesPair& query_frame_id_and_matches,
    vi_map::LoopClosureConstraint* constraint_ptr) const {
  CHECK_NOTNULL(constraint_ptr);

  const loop_closure::MatchVector& matches = query_frame_id_and_matches.second;
  if (matches.empty()) {
    return false;
  }

  vi_map::LoopClosureConstraint& constraint = *constraint_ptr;

  using vi_map::FrameKeyPointToStructureMatch;

  // Translate frame_ids to vertex id and frame index.
  constraint.structure_matches.clear();
  constraint.structure_matches.reserve(matches.size());
  constraint.query_vertex_id = query_frame_id_and_matches.first.vertex_id;
  for (const FrameKeyPointToStructureMatch& match : matches) {
    CHECK(match.isValid());
    vi_map::VertexKeyPointToStructureMatch structure_match;
    structure_match.landmark_result = match.landmark_result;
    structure_match.keypoint_index_query =
        match.keypoint_id_query.keypoint_index;
    structure_match.frame_identifier_result = match.keyframe_id_result;
    structure_match.frame_index_query =
        match.keypoint_id_query.frame_id.frame_index;
    constraint.structure_matches.push_back(structure_match);
  }
  return true;
}

void LoopDetectorNode::convertFrameToProjectedImage(
    const vi_map::VIMap& map, const vi_map::VisualFrameIdentifier& frame_id,
    const aslam::VisualFrame& frame,
    const vi_map::LandmarkIdList& observed_landmark_ids,
    const vi_map::MissionId& mission_id, const bool skip_invalid_landmark_ids,
    loop_closure::ProjectedImage* projected_image) const {
  CHECK_NOTNULL(projected_image);
  // We want to add all landmarks.
  vi_map::LandmarkIdSet landmarks_to_add(
      observed_landmark_ids.begin(), observed_landmark_ids.end());
  convertFrameToProjectedImageOnlyUsingProvidedLandmarkIds(
      map, frame_id, frame, observed_landmark_ids, mission_id,
      skip_invalid_landmark_ids, landmarks_to_add, projected_image);
}

void LoopDetectorNode::convertFrameToProjectedImageOnlyUsingProvidedLandmarkIds(
    const vi_map::VIMap& map, const vi_map::VisualFrameIdentifier& frame_id,
    const aslam::VisualFrame& frame,
    const vi_map::LandmarkIdList& observed_landmark_ids,
    const vi_map::MissionId& mission_id, const bool skip_invalid_landmark_ids,
    const vi_map::LandmarkIdSet& landmarks_to_add,
    loop_closure::ProjectedImage* projected_image) const {
  CHECK_NOTNULL(projected_image);
  projected_image->dataset_id = mission_id;
  projected_image->keyframe_id = frame_id;
  projected_image->timestamp_nanoseconds = frame.getTimestampNanoseconds();

  CHECK_EQ(
      static_cast<int>(observed_landmark_ids.size()),
      frame.getKeypointMeasurements().cols());
  CHECK_EQ(
      static_cast<int>(observed_landmark_ids.size()),
      frame.getDescriptors().cols());

  const Eigen::Matrix2Xd& original_measurements =
      frame.getKeypointMeasurements();
  const aslam::VisualFrame::DescriptorsT& original_descriptors =
      frame.getDescriptors();

  aslam::VisualFrame::DescriptorsT valid_descriptors(
      original_descriptors.rows(), original_descriptors.cols());
  Eigen::Matrix2Xd valid_measurements(2, original_measurements.cols());
  vi_map::LandmarkIdList valid_landmark_ids(original_measurements.cols());

  int num_valid_landmarks = 0;
  for (int i = 0; i < original_measurements.cols(); ++i) {
    const bool is_landmark_id_valid = observed_landmark_ids[i].isValid();
    const bool is_landmark_valid =
        !skip_invalid_landmark_ids || is_landmark_id_valid;

    const bool landmark_well_constrained =
        !skip_invalid_landmark_ids ||
        !FLAGS_lc_filter_underconstrained_landmarks ||
        (is_landmark_id_valid &&
         vi_map::isLandmarkWellConstrained(
             map, map.getLandmark(observed_landmark_ids[i])));

    if (skip_invalid_landmark_ids && is_landmark_id_valid) {
      CHECK(
          map.getLandmark(observed_landmark_ids[i]).getQuality() !=
          vi_map::Landmark::Quality::kUnknown)
          << "Please "
          << "retriangulate the landmarks before using the loop closure "
          << "engine.";
    }

    const bool is_landmark_in_set_to_add =
        landmarks_to_add.count(observed_landmark_ids[i]) > 0u;

    if (landmark_well_constrained && is_landmark_in_set_to_add &&
        is_landmark_valid) {
      valid_measurements.col(num_valid_landmarks) =
          original_measurements.col(i);
      valid_descriptors.col(num_valid_landmarks) = original_descriptors.col(i);
      valid_landmark_ids[num_valid_landmarks] = observed_landmark_ids[i];
      ++num_valid_landmarks;
    }
  }

  valid_measurements.conservativeResize(Eigen::NoChange, num_valid_landmarks);
  valid_descriptors.conservativeResize(Eigen::NoChange, num_valid_landmarks);
  valid_landmark_ids.resize(num_valid_landmarks);

  if (skip_invalid_landmark_ids) {
    statistics::StatsCollector stats_landmarks("LC num_landmarks insertion");
    stats_landmarks.AddSample(num_valid_landmarks);
  } else {
    statistics::StatsCollector stats_landmarks("LC num_landmarks query");
    stats_landmarks.AddSample(num_valid_landmarks);
  }

  projected_image->landmarks.swap(valid_landmark_ids);
  projected_image->measurements.swap(valid_measurements);
  loop_detector_->ProjectDescriptors(
      valid_descriptors, &projected_image->projected_descriptors);
}

void LoopDetectorNode::convertLocalizationFrameToProjectedImage(
    const aslam::VisualNFrame& nframe,
    const loop_closure::KeyframeId& keyframe_id,
    const bool skip_untracked_keypoints,
    const loop_closure::ProjectedImage::Ptr& projected_image,
    KeyframeToKeypointReindexMap* keyframe_to_keypoint_reindexing,
    vi_map::LandmarkIdList* observed_landmark_ids) const {
  CHECK(projected_image != nullptr);
  CHECK_NOTNULL(keyframe_to_keypoint_reindexing);
  CHECK_NOTNULL(observed_landmark_ids)->clear();

  const aslam::VisualFrame& frame = nframe.getFrame(keyframe_id.frame_index);
  if (skip_untracked_keypoints) {
    CHECK(frame.hasTrackIds()) << "Can only skip untracked keypoints if the "
                               << "track id channel is available.";
  }

  // Create some dummy ids for the localization frame that isn't part of the map
  // yet. This is required to use the same interfaces from the loop-closure
  // backend.
  projected_image->dataset_id = common::createRandomId<vi_map::MissionId>();
  projected_image->keyframe_id = keyframe_id;
  projected_image->timestamp_nanoseconds = frame.getTimestampNanoseconds();

  // Project the selected binary descriptors.
  const Eigen::Matrix2Xd& original_measurements =
      frame.getKeypointMeasurements();
  const aslam::VisualFrame::DescriptorsT& original_descriptors =
      frame.getDescriptors();
  CHECK_EQ(original_measurements.cols(), original_descriptors.cols());
  const Eigen::VectorXi* frame_trackids = nullptr;
  if (frame.hasTrackIds()) {
    frame_trackids = &frame.getTrackIds();
  }

  aslam::VisualFrame::DescriptorsT valid_descriptors(
      original_descriptors.rows(), original_descriptors.cols());
  Eigen::Matrix2Xd valid_measurements(2, original_measurements.cols());
  vi_map::LandmarkIdList valid_landmark_ids;
  valid_landmark_ids.reserve(original_measurements.cols());
  observed_landmark_ids->resize(original_measurements.cols());

  unsigned int num_valid_landmarks = 0;
  for (int i = 0; i < original_measurements.cols(); ++i) {
    if (skip_untracked_keypoints && (frame_trackids != nullptr) &&
        (*frame_trackids)(i) < 0) {
      continue;
    }

    valid_measurements.col(num_valid_landmarks) = original_measurements.col(i);
    valid_descriptors.col(num_valid_landmarks) = original_descriptors.col(i);
    const vi_map::LandmarkId random_landmark_id =
        common::createRandomId<vi_map::LandmarkId>();
    (*observed_landmark_ids)[i] = random_landmark_id;
    valid_landmark_ids.push_back(random_landmark_id);
    (*keyframe_to_keypoint_reindexing)[keyframe_id].emplace_back(i);
    ++num_valid_landmarks;
  }

  valid_measurements.conservativeResize(Eigen::NoChange, num_valid_landmarks);
  valid_descriptors.conservativeResize(Eigen::NoChange, num_valid_landmarks);
  valid_landmark_ids.shrink_to_fit();

  projected_image->landmarks.swap(valid_landmark_ids);
  projected_image->measurements.swap(valid_measurements);
  loop_detector_->ProjectDescriptors(
      valid_descriptors, &projected_image->projected_descriptors);
}

void LoopDetectorNode::addVertexToDatabase(
    const pose_graph::VertexId& vertex_id, const vi_map::VIMap& map) {
  CHECK(map.hasVertex(vertex_id));
  const vi_map::Vertex& vertex = map.getVertex(vertex_id);
  const unsigned int num_frames = vertex.numFrames();
  for (unsigned int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
    if (vertex.isVisualFrameSet(frame_idx) &&
        vertex.isVisualFrameValid(frame_idx)) {
      std::shared_ptr<loop_closure::ProjectedImage> projected_image =
          std::make_shared<loop_closure::ProjectedImage>();
      constexpr bool kSkipInvalidLandmarkIds = true;
      vi_map::LandmarkIdList landmark_ids;
      vertex.getFrameObservedLandmarkIds(frame_idx, &landmark_ids);
      VLOG(200) << "Frame " << frame_idx << " of vertex " << vertex_id
                << " with "
                << vertex.getVisualFrame(frame_idx).getDescriptors().cols()
                << " descriptors";
      convertFrameToProjectedImage(
          map, vi_map::VisualFrameIdentifier(vertex.id(), frame_idx),
          vertex.getVisualFrame(frame_idx), landmark_ids, vertex.getMissionId(),
          kSkipInvalidLandmarkIds, projected_image.get());

      loop_detector_->Insert(projected_image);
    }
  }
}

bool LoopDetectorNode::hasMissionInDatabase(
    const vi_map::MissionId& mission_id) const {
  return missions_in_database_.count(mission_id) > 0u;
}

void LoopDetectorNode::addMissionToDatabase(
    const vi_map::MissionId& mission_id, const vi_map::VIMap& map) {
  CHECK(map.hasMission(mission_id));
  missions_in_database_.emplace(mission_id);

  vi_map::LandmarkIdSet selected_landmarks;
  vi_map::LandmarkIdList mission_landmarks;
  map.getAllLandmarkIdsInMission(mission_id, &mission_landmarks);
  selected_landmarks.insert(mission_landmarks.begin(), mission_landmarks.end());
  addLandmarkSetToDatabase(selected_landmarks, map);

  /*
  VLOG(1) << "Getting vertices in mission " << mission_id;
  pose_graph::VertexIdList all_vertices;
  map.getAllVertexIdsInMissionAlongGraph(mission_id, &all_vertices);

  VLOG(1) << "Got vertices in mission " << mission_id;
  VLOG(1) << "Adding mission " << mission_id << " to database.";

  addVerticesToDatabase(all_vertices, map);
  */
}

void LoopDetectorNode::addVerticesToDatabase(
    const pose_graph::VertexIdList& vertex_ids, const vi_map::VIMap& map) {
  common::ProgressBar progress_bar(vertex_ids.size());

  for (const pose_graph::VertexId& vertex_id : vertex_ids) {
    progress_bar.increment();
    addVertexToDatabase(vertex_id, map);
  }
}

void LoopDetectorNode::addLocalizationSummaryMapToDatabase(
    const summary_map::LocalizationSummaryMap& localization_summary_map) {
  CHECK(
      summary_maps_in_database_.emplace(localization_summary_map.id()).second);

  pose_graph::VertexIdList observer_ids;
  localization_summary_map.getAllObserverIds(&observer_ids);

  const Eigen::Matrix3Xf& G_observer_positions =
      localization_summary_map.GObserverPosition();
  if (observer_ids.empty()) {
    if (G_observer_positions.cols() > 0) {
      // Vertex ids were not stored in the summary map. Generating random ones.
      observer_ids.resize(G_observer_positions.cols());
      for (pose_graph::VertexId& vertex_id : observer_ids) {
        common::generateId(&vertex_id);
      }
    } else {
      LOG(FATAL) << "No observers in the summary map found. Is it initialized?";
    }
  }

  std::vector<std::vector<int>> observer_observations;
  observer_observations.resize(observer_ids.size());

  // The index of the observer for every observation in the summary map.
  const Eigen::Matrix<unsigned int, Eigen::Dynamic, 1>& observer_indices =
      localization_summary_map.observerIndices();

  // Accumulate the observation indices per observer.
  for (int i = 0; i < observer_indices.rows(); ++i) {
    const int observer_index = observer_indices(i, 0);
    CHECK_LT(observer_index, static_cast<int>(observer_observations.size()));
    observer_observations[observer_index].push_back(i);
  }
  // Generate a random mission_id for this map.
  vi_map::MissionId mission_id;
  common::generateId(&mission_id);

  vi_map::LandmarkIdList observed_landmark_ids;
  localization_summary_map.getAllLandmarkIds(&observed_landmark_ids);

  const Eigen::MatrixXf& projected_descriptors =
      localization_summary_map.projectedDescriptors();

  const int descriptor_dimensionality = projected_descriptors.rows();

  const Eigen::Matrix<unsigned int, Eigen::Dynamic, 1>&
      observation_to_landmark_index =
          localization_summary_map.observationToLandmarkIndex();

  for (size_t observer_idx = 0; observer_idx < observer_observations.size();
       ++observer_idx) {
    std::shared_ptr<loop_closure::ProjectedImage> projected_image_ptr =
        std::make_shared<loop_closure::ProjectedImage>();
    loop_closure::ProjectedImage& projected_image = *projected_image_ptr;

    // Timestamp not relevant since this image will not collide with any
    // other image given its unique mission-id.
    projected_image.timestamp_nanoseconds = 0;
    projected_image.dataset_id = mission_id;
    constexpr unsigned int kFrameIndex = 0;
    projected_image.keyframe_id =
        vi_map::VisualFrameIdentifier(observer_ids[observer_idx], kFrameIndex);

    const std::vector<int>& observations = observer_observations[observer_idx];
    projected_image.landmarks.resize(observations.size());
    // Measurements have no meaning, so we add a zero block.
    projected_image.measurements.setZero(2, observations.size());
    projected_image.projected_descriptors.resize(
        descriptor_dimensionality, observations.size());

    for (size_t i = 0; i < observations.size(); ++i) {
      const int observation_index = observations[i];
      CHECK_LT(observation_index, projected_descriptors.cols());
      projected_image.projected_descriptors.col(i) =
          projected_descriptors.col(observation_index);

      CHECK_LT(observation_index, observation_to_landmark_index.rows());
      const size_t landmark_index =
          observation_to_landmark_index(observation_index, 0);
      CHECK_LT(landmark_index, observed_landmark_ids.size());
      projected_image.landmarks[i] = observed_landmark_ids[landmark_index];
    }
    loop_detector_->Insert(projected_image_ptr);
  }
}

void LoopDetectorNode::addLandmarkSetToDatabase(
    const vi_map::LandmarkIdSet& landmark_id_set,
    const vi_map::VIMap& map) {
  typedef std::unordered_map<vi_map::VisualFrameIdentifier,
                             vi_map::LandmarkIdSet>
      VisualFrameToGlobalLandmarkIdsMap;
  VisualFrameToGlobalLandmarkIdsMap visual_frame_to_global_landmarks_map;

  for (const vi_map::LandmarkId& landmark_id : landmark_id_set) {
    const vi_map::Landmark& landmark = map.getLandmark(landmark_id);
    landmark.forEachObservation(
        [&](const vi_map::KeypointIdentifier& observer_backlink) {
          visual_frame_to_global_landmarks_map[observer_backlink.frame_id]
              .emplace(landmark_id);
        });
  }

  for (const VisualFrameToGlobalLandmarkIdsMap::value_type&
           frameid_and_landmarks : visual_frame_to_global_landmarks_map) {
    const vi_map::VisualFrameIdentifier& frame_identifier =
        frameid_and_landmarks.first;
    const vi_map::Vertex& vertex = map.getVertex(frame_identifier.vertex_id);

    vi_map::LandmarkIdList landmark_ids;
    vertex.getFrameObservedLandmarkIds(
        frame_identifier.frame_index, &landmark_ids);

    std::shared_ptr<loop_closure::ProjectedImage> projected_image =
        std::make_shared<loop_closure::ProjectedImage>();
    constexpr bool kSkipInvalidLandmarkIds = true;
    convertFrameToProjectedImageOnlyUsingProvidedLandmarkIds(
        map, frame_identifier,
        vertex.getVisualFrame(frame_identifier.frame_index), landmark_ids,
        vertex.getMissionId(), kSkipInvalidLandmarkIds,
        frameid_and_landmarks.second, projected_image.get());

      if (use_better_descriptors_) {
        cv::Mat raw_image;
        if (!map.hasRawImage(vertex, frame_identifier.frame_index)) {
          statistics::StatsCollector noresource_counter(
              "No resource for indexing");
          noresource_counter.IncrementOne();
          LOG(WARNING) << "Vertex " << vertex.id()
            << " has no resource, can't index, skipping.";
          continue;
        }
        CHECK(map.getRawImage(
            vertex, frame_identifier.frame_index, &raw_image));
        addBetterDescriptorsToProjectedImage(raw_image, projected_image);
      }

    visual_frame_to_projected_image_map_.emplace(
        frame_identifier, projected_image);
    if (use_deep_retrieval_) {
      //visual_frame_to_projected_image_map_.emplace(
          //frame_identifier, projected_image);
    } else {
      loop_detector_->Insert(projected_image);
    }
  }
}

bool LoopDetectorNode::findNFrameInSummaryMapDatabase(
    const aslam::VisualNFrame& n_frame, const bool skip_untracked_keypoints,
    const summary_map::LocalizationSummaryMap& localization_summary_map,
    pose::Transformation* T_G_I, unsigned int* num_of_lc_matches,
    vi_map::VertexKeyPointToStructureMatchList* inlier_structure_matches)
    const {
  CHECK_NOTNULL(T_G_I);
  CHECK_NOTNULL(num_of_lc_matches);
  CHECK_NOTNULL(inlier_structure_matches);

  CHECK(!summary_maps_in_database_.empty())
      << "No summary maps were added "
      << "to the database. This method only operates on summary maps.";

  loop_closure::FrameToMatches frame_matches_list;

  std::vector<vi_map::LandmarkIdList> query_vertex_observed_landmark_ids;

  findNearestNeighborMatchesForNFrame(
      n_frame, skip_untracked_keypoints, &query_vertex_observed_landmark_ids,
      num_of_lc_matches, &frame_matches_list);

  timing::Timer timer_compute_relative("lc compute absolute transform");
  constexpr bool kMergeLandmarks = false;
  constexpr bool kAddLoopclosureEdges = false;
  loop_closure_handler::LoopClosureHandler handler(&localization_summary_map,
                                                   &landmark_id_old_to_new_);

  constexpr pose_graph::VertexId* kVertexIdClosestToStructureMatches = nullptr;
  const bool success = computeAbsoluteTransformFromFrameMatches(
      n_frame, query_vertex_observed_landmark_ids, frame_matches_list,
      kMergeLandmarks, kAddLoopclosureEdges, handler, T_G_I,
      inlier_structure_matches, kVertexIdClosestToStructureMatches);

  if (visualizer_ && success) {
    LOG(INFO) << "Successful localization.";
    visualizer_->visualizeSummaryMapDatabase(localization_summary_map);
    visualizer_->visualizeKeyframeToStructureMatch(
        *inlier_structure_matches, T_G_I->getPosition(),
        localization_summary_map);
  } else {
    LOG(INFO) << "Localization failed.";
  }

  return success;
}

bool LoopDetectorNode::findNFrameInDatabase(
    const aslam::VisualNFrame& n_frame, const bool skip_untracked_keypoints,
    vi_map::VIMap* map, pose::Transformation* T_G_I,
    unsigned int* num_of_lc_matches,
    vi_map::VertexKeyPointToStructureMatchList* inlier_structure_matches,
    pose_graph::VertexId* vertex_id_closest_to_structure_matches) const {
  CHECK_NOTNULL(map);
  CHECK_NOTNULL(T_G_I);
  CHECK_NOTNULL(num_of_lc_matches);
  CHECK_NOTNULL(inlier_structure_matches)->clear();
  // Note: vertex_id_closest_to_structure_matches is optional and may be NULL.

  constexpr bool kMergeLandmarks = false;
  constexpr bool kAddLoopclosureEdges = false;
  std::vector<vi_map::LandmarkIdList> query_vertex_observed_landmark_ids;
  bool success;

  if (use_deep_retrieval_) {
    CHECK(!skip_untracked_keypoints);  // not implemented

    const size_t num_frames = n_frame.getNumFrames();
    loop_closure::ProjectedImagePtrList projected_image_ptr_list;
    projected_image_ptr_list.reserve(num_frames);
    query_vertex_observed_landmark_ids.resize(num_frames);
    KeyframeToKeypointReindexMap keyframe_to_keypoint_reindexing;  // unused
    keyframe_to_keypoint_reindexing.reserve(num_frames);

    std::unordered_set<vi_map::VisualFrameIdentifier> all_retrieved_frames_set;

    const pose_graph::VertexId query_vertex_id(
        common::createRandomId<pose_graph::VertexId>());
    for (size_t frame_idx = 0u; frame_idx < num_frames; ++frame_idx) {
      if (!n_frame.isFrameSet(frame_idx) || !n_frame.isFrameValid(frame_idx)) {
        continue;
      }
      const aslam::VisualFrame::ConstPtr frame =
          n_frame.getFrameShared(frame_idx);
      CHECK(frame->hasKeypointMeasurements());
      CHECK(frame->hasRawImage());
      if (frame->getNumKeypointMeasurements() == 0u) {
        continue;
      }
      const cv::Mat& raw_image = frame->getRawImage();

      // Compute the projected image
      loop_closure::KeyframeId frame_id(query_vertex_id, frame_idx);
      projected_image_ptr_list.push_back(
          std::make_shared<loop_closure::ProjectedImage>());
      convertLocalizationFrameToProjectedImage(
          n_frame, frame_id, skip_untracked_keypoints,
          projected_image_ptr_list.back(), &keyframe_to_keypoint_reindexing,
          &(query_vertex_observed_landmark_ids)[frame_idx]);

      // Retrieve some prior frames
      vi_map::VisualFrameIdentifierList retrieved_frames;
      deep_retrieval_->RetrieveNearestNeighbors(
          raw_image, FLAGS_lc_deep_retrieval_num_nn,
          FLAGS_lc_deep_retrieval_max_nn_distance, &retrieved_frames);
      all_retrieved_frames_set.insert(
          retrieved_frames.begin(), retrieved_frames.end());

      // Optionally replace the projected descriptors by better descriptors
      if (use_better_descriptors_) {
        addBetterDescriptorsToProjectedImage(
            raw_image, projected_image_ptr_list.back());
      }
    }

    CHECK(all_retrieved_frames_set.size());
    vi_map::VisualFrameIdentifierList all_retrieved_frames_list(
        all_retrieved_frames_set.begin(), all_retrieved_frames_set.end());

    // dummy arguments
    vi_map::VertexKeyPointToStructureMatchList raw_structure_matches;
    loop_closure_handler::LoopClosureHandler::MergedLandmark3dPositionVector
        landmark_pairs_merged;
    double inlier_ratio;

    success = lcWithPrior(
        projected_image_ptr_list, n_frame, all_retrieved_frames_list,
        query_vertex_observed_landmark_ids, kMergeLandmarks,
        kAddLoopclosureEdges, map, T_G_I, num_of_lc_matches, &inlier_ratio,
        inlier_structure_matches, &raw_structure_matches,
        &landmark_pairs_merged);

    if (visualizer_ && success) {
      CHECK_EQ(projected_image_ptr_list.size(), 1);
      timing::Timer timer_viz_matches("lc visualize descriptor matches");
      visualizer_->visualizeDescriptorMatches(
          *inlier_structure_matches, projected_image_ptr_list[0],
          n_frame.getFrameShared(0), map, visual_frame_to_projected_image_map_);
      timer_viz_matches.Stop();
    }
  } else {
    loop_closure::FrameToMatches frame_matches_list;
    findNearestNeighborMatchesForNFrame(
        n_frame, skip_untracked_keypoints, &query_vertex_observed_landmark_ids,
        num_of_lc_matches, &frame_matches_list);

    timing::Timer timer_compute_relative("lc compute absolute transform");
    loop_closure_handler::LoopClosureHandler handler(
            map, &landmark_id_old_to_new_);
    success = computeAbsoluteTransformFromFrameMatches(
        n_frame, query_vertex_observed_landmark_ids, frame_matches_list,
        kMergeLandmarks, kAddLoopclosureEdges, handler, T_G_I,
        inlier_structure_matches, vertex_id_closest_to_structure_matches);
    timer_compute_relative.Stop();
  }

  if (visualizer_ && success) {
    LOG(INFO) << "Successful localization.";
    visualizer_->visualizeKeyframeToStructureMatch(
        *inlier_structure_matches, T_G_I->getPosition(), map);
  } else {
    LOG(WARNING) << "Localization failed: " << *num_of_lc_matches<< " matches";
  }

  return success;
}

void LoopDetectorNode::findNearestNeighborMatchesForNFrame(
    const aslam::VisualNFrame& n_frame, const bool skip_untracked_keypoints,
    std::vector<vi_map::LandmarkIdList>* query_vertex_observed_landmark_ids,
    unsigned int* num_of_lc_matches,
    loop_closure::FrameToMatches* frame_matches_list) const {
  CHECK_NOTNULL(query_vertex_observed_landmark_ids)->clear();
  CHECK_NOTNULL(num_of_lc_matches);
  CHECK_NOTNULL(frame_matches_list);

  *num_of_lc_matches = 0u;

  timing::Timer timer_preprocess("Loop Closure: preprocess frames");
  const size_t num_frames = n_frame.getNumFrames();
  loop_closure::ProjectedImagePtrList projected_image_ptr_list;
  projected_image_ptr_list.reserve(num_frames);
  query_vertex_observed_landmark_ids->resize(num_frames);
  std::vector<loop_closure::KeyframeId> frame_ids;
  frame_ids.reserve(num_frames);
  KeyframeToKeypointReindexMap keyframe_to_keypoint_reindexing;
  keyframe_to_keypoint_reindexing.reserve(num_frames);

  const pose_graph::VertexId query_vertex_id(
      common::createRandomId<pose_graph::VertexId>());
  for (size_t frame_idx = 0u; frame_idx < num_frames; ++frame_idx) {
    if (n_frame.isFrameSet(frame_idx) && n_frame.isFrameValid(frame_idx)) {
      const aslam::VisualFrame::ConstPtr frame =
          n_frame.getFrameShared(frame_idx);

      CHECK(frame->hasKeypointMeasurements());
      if (frame->getNumKeypointMeasurements() == 0u) {
        // Skip frame if zero measurements found.
        continue;
      }
      frame_ids.emplace_back(query_vertex_id, frame_idx);

      projected_image_ptr_list.push_back(
          std::make_shared<loop_closure::ProjectedImage>());
      convertLocalizationFrameToProjectedImage(
          n_frame, frame_ids.back(), skip_untracked_keypoints,
          projected_image_ptr_list.back(), &keyframe_to_keypoint_reindexing,
          &(*query_vertex_observed_landmark_ids)[frame_idx]);
    }
  }
  timer_preprocess.Stop();
  constexpr bool kParallelFindIfPossible = true;
  loop_detector_->Find(
      projected_image_ptr_list, kParallelFindIfPossible, frame_matches_list);

  // Correct the indices in case untracked keypoints were removed.
  // For the pose recovery with RANSAC, the keypoint indices of the frame are
  // decisive, not those stored in the projected image. Therefore, the
  // keypoint indices of the matches (inferred from the projected image) have to
  // be mapped back to the keypoint indices of the frame.
  if (skip_untracked_keypoints) {
    for (loop_closure::FrameToMatches::value_type& frame_matches :
         *frame_matches_list) {
      for (loop_closure::Match& match : frame_matches.second) {
        KeyframeToKeypointReindexMap::const_iterator iter_keyframe_supsampling =
            keyframe_to_keypoint_reindexing.find(
                match.keypoint_id_query.frame_id);
        CHECK(
            iter_keyframe_supsampling != keyframe_to_keypoint_reindexing.end());
        match.keypoint_id_query.keypoint_index =
            iter_keyframe_supsampling
                ->second[match.keypoint_id_query.keypoint_index];
      }
    }
  }

  *num_of_lc_matches = loop_closure::getNumberOfMatches(*frame_matches_list);
}

bool isGtMatch(
    const pose::Transformation& T_G_I,
    const pose::Transformation& gt_T_G_I,
    double dist_thresh = 3.) {
  double position_error = (gt_T_G_I.getPosition() - T_G_I.getPosition()).norm();
  double angle_error = std::acos(((
          gt_T_G_I.getRotationMatrix().inverse() * T_G_I.getRotationMatrix()
          ).trace() - 1.) / 2.);
  return position_error < dist_thresh && angle_error < (70. * 3.142 / 180);
}
bool isGtMatch(
    const vi_map::VIMap& map,
    const loop_closure::ProjectedImage& projected_image,
    const pose::Transformation& gt_T_G_I) {
  pose::Transformation T_G_I = map.getVertex_T_G_I(
      projected_image.keyframe_id.vertex_id);
  return isGtMatch(T_G_I, gt_T_G_I);
}

void LoopDetectorNode::serialize_debug(std::string file_path) {
  std::fstream output(file_path,
                      std::ios::out | std::ios::trunc | std::ios::binary);
  CHECK(debug_fusion_proto_.SerializeToOstream(&output));
}

bool LoopDetectorNode::lcWithPrior(
    const loop_closure::ProjectedImagePtrList& query_projected_image_ptr_list,
    const aslam::VisualNFrame& query_n_frame,
    const vi_map::VisualFrameIdentifierList& prior_frames_list,
    const std::vector<vi_map::LandmarkIdList>& query_vertex_landmark_ids,
    const bool merge_landmarks,
    const bool add_lc_edges,
    vi_map::VIMap* map, pose::Transformation* T_G_I,
    unsigned int* num_of_lc_matches, double* inlier_ratio,
    vi_map::VertexKeyPointToStructureMatchList* inlier_structure_matches,
    vi_map::VertexKeyPointToStructureMatchList* raw_structure_matches,
    loop_closure_handler::LoopClosureHandler::MergedLandmark3dPositionVector*
        landmark_pairs_merged,
    pose::Transformation* gt_T_G_I) const {
  CHECK_NOTNULL(map);
  CHECK_NOTNULL(T_G_I);
  CHECK_NOTNULL(num_of_lc_matches);
  CHECK_NOTNULL(inlier_ratio);
  CHECK_NOTNULL(inlier_structure_matches);
  CHECK_NOTNULL(raw_structure_matches);
  CHECK_NOTNULL(landmark_pairs_merged);

  typedef int ComponentId;
  constexpr ComponentId kInvalidComponentId = -1;
  typedef std::unordered_map<vi_map::VisualFrameIdentifier, ComponentId>
      FramesToComponents;
  typedef std::unordered_map<ComponentId,
                             std::unordered_set<vi_map::VisualFrameIdentifier>>
      Components;
  typedef std::unordered_map<vi_map::LandmarkId,
                             std::vector<vi_map::VisualFrameIdentifier>>
      LandmarkFrames;

  // Build the covisibility graph
  FramesToComponents frames_to_components;
  LandmarkFrames landmark_frames;
  for (const vi_map::VisualFrameIdentifier& frame_id : prior_frames_list) {
    const VisualFrameToProjectedImageMap::const_iterator projected_image_it =
        visual_frame_to_projected_image_map_.find(frame_id);
    CHECK(projected_image_it != visual_frame_to_projected_image_map_.end());
    for (const vi_map::LandmarkId& landmark_id :
         projected_image_it->second->landmarks) {
      landmark_frames[landmark_id].emplace_back(frame_id);
    }
    frames_to_components.emplace(frame_id, kInvalidComponentId);
  }

  // Expand the set of prior frames to neighboring frames in the graph
  for (size_t i = 0; i < FLAGS_lc_deep_retrieval_expand_components; i++) {
    std::unordered_set<vi_map::VisualFrameIdentifier> new_frames;
    for (const LandmarkFrames::value_type& lm_frames : landmark_frames) {
      const vi_map::LandmarkId& landmark_id = lm_frames.first;
      const vi_map::Landmark& landmark = map->getLandmark(landmark_id);
      for (const vi_map::KeypointIdentifier& kp_id :
           landmark.getObservations()) {
        const vi_map::VisualFrameIdentifier& obs_frame_id = kp_id.frame_id;
        if (frames_to_components.find(obs_frame_id)
            == frames_to_components.end()) {
          new_frames.emplace(obs_frame_id);
        }
      }
    }
    statistics::StatsCollector num_new_frames_expansion_stats(
        "lcWithPrior -- Number of frames added when expanding");
    num_new_frames_expansion_stats.AddSample(new_frames.size());

    for (const vi_map::VisualFrameIdentifier& frame_id : new_frames) {
      CHECK(frames_to_components.find(frame_id) == frames_to_components.end());
      const VisualFrameToProjectedImageMap::const_iterator projected_image_it =
          visual_frame_to_projected_image_map_.find(frame_id);
      CHECK(projected_image_it != visual_frame_to_projected_image_map_.end());
      for (const vi_map::LandmarkId& landmark_id :
           projected_image_it->second->landmarks) {
        if (landmark_frames.find(landmark_id) == landmark_frames.end()) {
          continue;
        }
        landmark_frames[landmark_id].emplace_back(frame_id);
      }
      CHECK(frames_to_components.emplace(frame_id, kInvalidComponentId).second);
    }
  }

  // Label the connected components
  timing::Timer timer_find_components(
          "lcWithPrior -- Find connected components");
  ComponentId count_component_index = 0;
  Components components;
  for (const FramesToComponents::value_type& frame_to_component :
       frames_to_components) {
    if (frame_to_component.second != kInvalidComponentId)
      continue;
    ComponentId component_id = count_component_index++;

    // Find the largest set of frames connected by landmark covisibility.
    std::queue<vi_map::VisualFrameIdentifier> exploration_queue;
    exploration_queue.push(frame_to_component.first);
    while (!exploration_queue.empty()) {
      const vi_map::VisualFrameIdentifier& exploration_frame =
          exploration_queue.front();

      const FramesToComponents::iterator exploration_frame_and_component =
          frames_to_components.find(exploration_frame);
      CHECK(exploration_frame_and_component != frames_to_components.end());

      if (exploration_frame_and_component->second == kInvalidComponentId) {
        // Not part of a connected component.
        exploration_frame_and_component->second = component_id;
        components[component_id].insert(exploration_frame);

        // Find other prior frames connected through the landmarks
        const loop_closure::ProjectedImage& projected_image =
            *visual_frame_to_projected_image_map_.at(exploration_frame);
        for (const vi_map::LandmarkId& landmark_id :
             projected_image.landmarks) {
          if (landmark_frames.find(landmark_id) == landmark_frames.end()) {
            continue;
          }
          for (const vi_map::VisualFrameIdentifier& connected_frame :
               landmark_frames[landmark_id]) {
            if (frames_to_components[connected_frame] == kInvalidComponentId) {
              exploration_queue.push(connected_frame);
            }
          }
        }
      }
      exploration_queue.pop();
    }
  }
  timer_find_components.Stop();
  statistics::StatsCollector num_comp_stats(
      "lcWithPrior -- Number of components");
  num_comp_stats.AddSample(components.size());

#ifdef ADD_DEBUG_STATS
  // Build some stats on the clusters localizability
  statistics::StatsCollector prior_list_has_any_gt_match_stats(
      "lcWithPrior -- Prior list has any gt match");
  statistics::StatsCollector num_comp_with_matches_stats_sup1(
      "lcWithPrior -- Number of components with gt match >1");
  statistics::StatsCollector size_comp_with_matches_stats(
      "lcWithPrior -- Average size of components with gt match");
  bool prior_list_has_any_gt_match = false;
  std::unordered_set<size_t> sizes_components_with_matches;
  ComponentId id_component_with_match = kInvalidComponentId;
  for (const vi_map::VisualFrameIdentifier& frame_id : prior_frames_list) {
    const loop_closure::ProjectedImage& proj_im =
        *visual_frame_to_projected_image_map_.at(frame_id);
    bool is_match = isGtMatch(*map, proj_im, *gt_T_G_I);
    prior_list_has_any_gt_match |= is_match;
    if (is_match) {
      id_component_with_match = frames_to_components[frame_id];
      sizes_components_with_matches.insert(
          components[id_component_with_match].size());
    }
  }
  prior_list_has_any_gt_match_stats.AddSample(prior_list_has_any_gt_match);
  if(prior_list_has_any_gt_match) {
    if (sizes_components_with_matches.size() > 1) {
      num_comp_with_matches_stats_sup1.AddSample(
          sizes_components_with_matches.size());
    }
    size_comp_with_matches_stats.AddSample(
        std::accumulate(sizes_components_with_matches.begin(),
                        sizes_components_with_matches.end(), 0.0)
        / static_cast<double>(sizes_components_with_matches.size()));
  }
  ComponentId selected_component_id = kInvalidComponentId;
  size_t num_processed_components = 0;
#endif

  // Sort components by decreasing size
  std::vector<ComponentId> component_ids;
  component_ids.reserve(components.size());
  for (const Components::value_type& component_to_frames : components) {
      component_ids.push_back(component_to_frames.first);
  }
  std::sort(component_ids.begin(), component_ids.end(),
            [&](const ComponentId& a, const ComponentId& b) -> bool
            { return components[a].size() > components[b].size(); });

  // Do PnP+RANSAC for every component until a pose if found
  unsigned int& num_matches = *num_of_lc_matches;
  bool ransac_ok = false;
  timing::Timer timer_pnp_all_components("lcWithPrior -- Compute Pnp+RANSAC");
  for (const ComponentId& component_id : component_ids) {
#ifdef ADD_DEBUG_STATS
    num_processed_components++;
#endif
    const std::unordered_set<vi_map::VisualFrameIdentifier>& component_frames =
        components[component_id];
    loop_closure::FrameToMatches frame_to_matches;
    // Lock the loop detector
    {
      std::unique_lock<std::mutex> lock_detector(loop_detector_mutex_);
      // Rebuild the index with the component frames
      loop_detector_->Clear();
      for (const vi_map::VisualFrameIdentifier& frame_id : component_frames) {
          loop_detector_->Insert(
            visual_frame_to_projected_image_map_.at(frame_id));
      }

      // Find descriptor matches
      constexpr bool kParallelFindIfPossible = true;
      loop_detector_->Find(
          query_projected_image_ptr_list, kParallelFindIfPossible,
          &frame_to_matches);
    }
    num_matches = loop_closure::getNumberOfMatches(frame_to_matches);
    if (num_matches == 0u) {
#ifdef ADD_DEBUG_STATS
      if (component_id == id_component_with_match) {
        statistics::StatsCollector size_comp_when_no_match_found_stats(
            "lcWithPrior -- Size gt comp when no match is found");
        size_comp_when_no_match_found_stats.AddSample(component_frames.size());
      }
#endif
      continue;
    }

    // Create constraint for PnP
    vi_map::LoopClosureConstraint constraint;
    for (const loop_closure::FrameIdMatchesPair& frame_matches_pair :
         frame_to_matches) {
      vi_map::LoopClosureConstraint tmp_constraint;
      const bool conversion_success =
          convertFrameMatchesToConstraint(frame_matches_pair, &tmp_constraint);
      if (!conversion_success) {
        continue;
      }
      constraint.query_vertex_id = tmp_constraint.query_vertex_id;
      constraint.structure_matches.insert(
          constraint.structure_matches.end(),
          tmp_constraint.structure_matches.begin(),
          tmp_constraint.structure_matches.end());
    }
    *raw_structure_matches = constraint.structure_matches;

    // Perform PnP+RANSAC
    int inlier_count = 0;
    pose_graph::VertexId vertex_id_closest_to_structure_matches;
    std::mutex map_mutex;
    loop_closure_handler::LoopClosureHandler handler(
        map, &landmark_id_old_to_new_);
    ransac_ok = handler.handleLoopClosure(
        query_n_frame, query_vertex_landmark_ids, constraint.query_vertex_id,
        constraint.structure_matches, merge_landmarks, add_lc_edges,
        &inlier_count, inlier_ratio, T_G_I, inlier_structure_matches,
        landmark_pairs_merged, &vertex_id_closest_to_structure_matches,
        &map_mutex, use_random_pnp_seed_);

#ifdef ADD_DEBUG_STATS
    if(ransac_ok || component_id == id_component_with_match) {
      proto::DebugFusion::QueryComponent* query_comp_proto =
          debug_fusion_proto_.add_query_components();
      if (ransac_ok) {
        selected_component_id = component_id;
        std::string status = isGtMatch(*T_G_I, *gt_T_G_I) ? "ok" : "wrong";
        query_comp_proto->set_status(status);
        statistics::StatsCollector size_comp_when_gt_comp_succeeds_stats(
            "lcWithPrior -- Size comp when gt comp succeeds");
        size_comp_when_gt_comp_succeeds_stats.AddSample(component_frames.size());
        statistics::StatsCollector num_matches_when_gt_comp_succeeds_stats(
            "lcWithPrior -- Num matches when gt comp succeeds");
        num_matches_when_gt_comp_succeeds_stats.AddSample(num_matches);
        statistics::StatsCollector num_inliers_when_gt_comp_succeeds_stats(
            "lcWithPrior -- Num inliers when gt comp succeeds");
        num_inliers_when_gt_comp_succeeds_stats.AddSample(inlier_count);
        statistics::StatsCollector inlier_ratio_when_gt_comp_succeeds_stats(
            "lcWithPrior -- Inlier ratio when gt comp succeeds");
        inlier_ratio_when_gt_comp_succeeds_stats.AddSample(*inlier_ratio);
      } else {
        query_comp_proto->set_status("fail");
        statistics::StatsCollector size_comp_when_gt_comp_fails_stats(
            "lcWithPrior -- Size comp when gt comp fails");
        size_comp_when_gt_comp_fails_stats.AddSample(component_frames.size());
        statistics::StatsCollector num_matches_when_gt_comp_fails_stats(
            "lcWithPrior -- Num matches when gt comp fails");
        num_matches_when_gt_comp_fails_stats.AddSample(num_matches);
        if(*inlier_ratio < 0.0) {
          statistics::StatsCollector num_inliers_when_gt_comp_fails_num_stats(
              "lcWithPrior -- Num inliers when gt comp fails bc of num");
          num_inliers_when_gt_comp_fails_num_stats.AddSample(inlier_count);
        } else if(*inlier_ratio > 0.0) {
          statistics::StatsCollector num_inliers_when_gt_comp_fails_ratio_stats(
              "lcWithPrior -- Num inliers when gt comp fails bc of ratio");
          num_inliers_when_gt_comp_fails_ratio_stats.AddSample(inlier_count);
          statistics::StatsCollector inlier_ratio_when_gt_comp_fails_ratio_stats(
              "lcWithPrior -- Inlier ratio when gt comp fails bc of ratio");
          inlier_ratio_when_gt_comp_fails_ratio_stats.AddSample(*inlier_ratio);
        }
      }
      query_projected_image_ptr_list[0]->keyframe_id.vertex_id.serialize(
          query_comp_proto->mutable_query_id());
      query_comp_proto->set_num_matches(num_matches);
      query_comp_proto->set_num_inliers(inlier_count);
      query_comp_proto->set_inlier_ratio(*inlier_ratio);
      for (const vi_map::VisualFrameIdentifier& frame_id : component_frames) {
        frame_id.vertex_id.serialize(query_comp_proto->add_retrieved_ids());
      }
      for (const loop_closure::Match& match :
           frame_to_matches[query_projected_image_ptr_list[0]->keyframe_id]) {
        proto::DebugFusion::QueryComponent::Match* match_proto =
          query_comp_proto->add_matches();
        CHECK_LT(match.keypoint_id_query.keypoint_index,
                 query_projected_image_ptr_list[0]->measurements.cols());
        common::eigen_proto::serialize(
            Eigen::MatrixXd(query_projected_image_ptr_list[0]->measurements.col(
                match.keypoint_id_query.keypoint_index)),
            match_proto->mutable_query_measurement());
        const loop_closure::ProjectedImage& proj_image_result =
          *visual_frame_to_projected_image_map_.at(match.keyframe_id_result);

        bool found = false;
        for (size_t i = 0; i < proj_image_result.landmarks.size(); i++) {
          if (proj_image_result.landmarks[i] == match.landmark_result) {
            CHECK_LT(i, proj_image_result.measurements.cols());
            common::eigen_proto::serialize(
                Eigen::MatrixXd(proj_image_result.measurements.col(i)),
                match_proto->mutable_db_measurement());
            found = true;
          }
        }
        CHECK(found);
        match.keyframe_id_result.vertex_id.serialize(
            match_proto->mutable_db_vertex_id());
      }
      common::eigen_proto::serialize(
          Eigen::MatrixXd(gt_T_G_I->getPosition()),
          query_comp_proto->mutable_gt_position());
      if (ransac_ok) {
        common::eigen_proto::serialize(
            Eigen::MatrixXd(T_G_I->getPosition()),
            query_comp_proto->mutable_pnp_position());
        query_comp_proto->set_num_evaluated_clusters(num_processed_components);
        query_comp_proto->set_total_num_clusters(components.size());
      }
    }
#endif

    if (ransac_ok) {
      break;
    }
  }
  timer_pnp_all_components.Stop();

#ifdef ADD_DEBUG_STATS
  // Stats again
  statistics::StatsCollector select_the_right_component_stats(
      "lcWithPrior -- Select the right component");
  statistics::StatsCollector size_comp_selected_right_stats(
      "lcWithPrior -- Size comp when selected right one");
  statistics::StatsCollector size_comp_selected_wrong_stats(
      "lcWithPrior -- Size comp when selected wrong one");
  statistics::StatsCollector num_fail_find_any_component_despite_match(
      "lcWithPrior -- Fail find any component despite gt match");
  statistics::StatsCollector size_comp_selected_stats(
      "lcWithPrior -- Size of the  selected component");
  statistics::StatsCollector num_processed_comp_stats(
      "lcWithPrior -- Number of components processed before success");
  bool found_one_component = selected_component_id != kInvalidComponentId;
  if (found_one_component) {
    num_processed_comp_stats.AddSample(num_processed_components);
  }
  if (prior_list_has_any_gt_match) {
    bool selected_right_comp = selected_component_id == id_component_with_match;
    select_the_right_component_stats.AddSample(selected_right_comp);
    num_fail_find_any_component_despite_match.AddSample(!found_one_component);
    if (found_one_component) {
      size_t size_comp = components[id_component_with_match].size();
      size_comp_selected_stats.AddSample(size_comp);
      if (selected_right_comp) {
        size_comp_selected_right_stats.AddSample(size_comp);
      } else {
        size_comp_selected_wrong_stats.AddSample(size_comp);
      }
    }
  }
#endif

  return ransac_ok;
}

void LoopDetectorNode::addBetterDescriptorsToProjectedImage(
    const cv::Mat& raw_image,
    const loop_closure::ProjectedImage::Ptr& projected_image) const {
  CHECK_NOTNULL(raw_image.data);
  CHECK(projected_image != nullptr);

  const size_t new_descriptor_size =
      better_descriptor_extractor_->descriptorSize();
  const size_t num_detections = projected_image->measurements.cols();
  const Eigen::MatrixXd& measurements = projected_image->measurements;
  Eigen::MatrixXf& descriptors = projected_image->projected_descriptors;
  descriptors.resize(new_descriptor_size, Eigen::NoChange);
  timing::Timer timer_compute_sift("lc -- Compute SIFT");

  constexpr float kKeypointScale = 5.6;  // heuristic: average over test data.
  std::vector<cv::KeyPoint> keypoints;
  for (size_t i = 0; i < num_detections; i++) {
    keypoints.emplace_back(
        measurements(0, i), measurements(1, i), kKeypointScale);
  }

  if (FLAGS_lc_detect_sift_scale) {
    timing::Timer timer_compute_sift_scale("lc -- Compute SIFT scale");
    cv::Ptr<cv::xfeatures2d::SIFT> scale_detector =
      cv::xfeatures2d::SIFT::create(1500, 3, 0.02, 20);
    std::vector<cv::KeyPoint> sift_keypoints;
    scale_detector->detect(raw_image, sift_keypoints);
    for (cv::KeyPoint& kpt : keypoints) {
      double distance_to_nearest = -1;
      size_t nearest_idx = 0, idx = 0;
      for (const cv::KeyPoint& sift_kpt : sift_keypoints) {
        double distance = cv::norm(sift_kpt.pt - kpt.pt);
        if (distance < distance_to_nearest || distance_to_nearest == -1) {
          distance_to_nearest = distance;
          nearest_idx = idx;
        }
        ++idx;
      }
      if (distance_to_nearest != -1) {
        const cv::KeyPoint& nearest_kpt = sift_keypoints[nearest_idx];
        if (distance_to_nearest <= 5) {
          kpt.size = nearest_kpt.size;
          kpt.octave = nearest_kpt.octave;
          //kpt.angle = nearest_kpt.angle;
          //kpt.response = nearest_kpt.response;
          statistics::StatsCollector stats_distance_to_nearest(
              "SIFT descriptors - dist to nearest");
          stats_distance_to_nearest.AddSample(distance_to_nearest);
          statistics::StatsCollector stats_keypoint_scales(
              "SIFT descriptors - selected keypoint scales");
          stats_keypoint_scales.AddSample(nearest_kpt.size);
        }
        statistics::StatsCollector stats_keypoint_scales(
            "SIFT descriptors - keypoint scales");
        stats_keypoint_scales.AddSample(nearest_kpt.size);
      }
    }
    statistics::StatsCollector stats_num_total_kpt(
        "SIFT descriptors - num initial keypoints");
    stats_num_total_kpt.AddSample(keypoints.size());
    statistics::StatsCollector stats_num_detected_kpt(
        "SIFT descriptors - num detected keypoints");
    stats_num_detected_kpt.AddSample(sift_keypoints.size());
  }

  // OpenCV doc mentions that SIFT can discard some keypoints or compute
  // duplicated descriptors. For now we simply check that it's not the case.
  // TODO: - remove the landmarks associated with rejected keypoints;
  //       - remove duplicates.
  std::vector<cv::KeyPoint> original_keypoints(keypoints);
  cv::Mat cv_descriptors;
  better_descriptor_extractor_->compute(raw_image, keypoints, cv_descriptors);
  CHECK_EQ(keypoints.size(), num_detections);
  for (size_t i = 0; i < original_keypoints.size(); i++) {
    CHECK_EQ(original_keypoints[i].pt, keypoints[i].pt);
  }

  cv::cv2eigen(cv_descriptors, descriptors);
  descriptors.transposeInPlace();
  descriptors /= 512.f;  // scale back to [0, 1]
  CHECK_EQ(descriptors.rows(), new_descriptor_size);
  CHECK_EQ(descriptors.cols(), num_detections);
}

bool LoopDetectorNode::findVertexInDatabase(
    const vi_map::Vertex& query_vertex, const bool merge_landmarks,
    const bool add_lc_edges, vi_map::VIMap* map, pose::Transformation* T_G_I,
    unsigned int* num_of_lc_matches,
    vi_map::LoopClosureConstraint* inlier_constraint,
    pose::Transformation* gt_T_G_I) const {
  CHECK_NOTNULL(map);
  CHECK_NOTNULL(T_G_I);
  CHECK_NOTNULL(num_of_lc_matches);
  CHECK_NOTNULL(inlier_constraint);

  if (!use_deep_retrieval_ && loop_detector_->NumEntries() == 0u) {
    return false;
  }

  const size_t num_frames = query_vertex.numFrames();
  loop_closure::ProjectedImagePtrList projected_image_ptr_list;
  projected_image_ptr_list.reserve(num_frames);

  std::unordered_set<vi_map::VisualFrameIdentifier> all_retrieved_frames_set;

  for (size_t frame_idx = 0u; frame_idx < num_frames; ++frame_idx) {
    if (query_vertex.isVisualFrameSet(frame_idx) &&
        query_vertex.isVisualFrameValid(frame_idx)) {
      vi_map::VisualFrameIdentifier query_frame_id(
          query_vertex.id(), frame_idx);

      std::vector<vi_map::LandmarkId> observed_landmark_ids;
      query_vertex.getFrameObservedLandmarkIds(
          frame_idx, &observed_landmark_ids);
      projected_image_ptr_list.push_back(
          std::make_shared<loop_closure::ProjectedImage>());
      constexpr bool kSkipInvalidLandmarkIds = false;
      convertFrameToProjectedImage(
          *map, query_frame_id, query_vertex.getVisualFrame(frame_idx),
          observed_landmark_ids, query_vertex.getMissionId(),
          kSkipInvalidLandmarkIds, projected_image_ptr_list.back().get());

      cv::Mat raw_image;
      if (use_deep_retrieval_ || use_better_descriptors_) {
        if (!map->hasRawImage(query_vertex, frame_idx)) {
          continue;
        }
        CHECK(map->getRawImage(query_vertex, frame_idx, &raw_image));
      }

      // Retrieve some prior frames
      if (use_deep_retrieval_) {
        vi_map::VisualFrameIdentifierList retrieved_frames;
        deep_retrieval_->RetrieveNearestNeighbors(
            raw_image, FLAGS_lc_deep_retrieval_num_nn,
            FLAGS_lc_deep_retrieval_max_nn_distance, &retrieved_frames);
        all_retrieved_frames_set.insert(
            retrieved_frames.begin(), retrieved_frames.end());
      }

      // Optionally replace the projected descriptors by better descriptors
      if (use_better_descriptors_) {
        addBetterDescriptorsToProjectedImage(
            raw_image, projected_image_ptr_list.back());
      }
    }
  }

  if (FLAGS_lc_assume_perfect_retrieval) {
    CHECK(use_deep_retrieval_);
    CHECK_NOTNULL(gt_T_G_I);
    all_retrieved_frames_set.clear();
    std::vector<std::pair<vi_map::VisualFrameIdentifier, double>>
      candidate_frames_error;
    for (const VisualFrameToProjectedImageMap::value_type& frame_proj_im :
         visual_frame_to_projected_image_map_) {
      const pose::Transformation& candidate_T_G_I = map->getVertex_T_G_I(
          frame_proj_im.first.vertex_id);
      if (isGtMatch(candidate_T_G_I, *gt_T_G_I, 30.)) {
        double pos_error = (candidate_T_G_I.getPosition()
                            - gt_T_G_I->getPosition()).norm();
        candidate_frames_error.push_back(
            std::make_pair(frame_proj_im.first, pos_error));
      }
    }
    std::sort(candidate_frames_error.begin(), candidate_frames_error.end(),
              [&](const std::pair<vi_map::VisualFrameIdentifier, double>& a,
                  const std::pair<vi_map::VisualFrameIdentifier, double>& b)
              -> bool { return a.second < b.second; });
    size_t cnt = 0;
    for (const std::pair<vi_map::VisualFrameIdentifier, double>& p :
         candidate_frames_error) {
      all_retrieved_frames_set.insert(p.first);
      if (++cnt >= FLAGS_lc_deep_retrieval_num_nn) {
        break;
      }
    }
    statistics::StatsCollector num_prior_frames(
        "Perfect retrieval - num prior frames");
    num_prior_frames.AddSample(all_retrieved_frames_set.size());
  }

  if (use_deep_retrieval_ && !all_retrieved_frames_set.size()) {
    statistics::StatsCollector noresource_counter("No resource for retrieval");
    noresource_counter.IncrementOne();
    LOG(WARNING) << "Vertex " << query_vertex.id() << " has no resource, skip";
    return false;
  }

  if (use_deep_retrieval_) {
    vi_map::VisualFrameIdentifierList all_retrieved_frames_list(
        all_retrieved_frames_set.begin(), all_retrieved_frames_set.end());
    std::vector<vi_map::LandmarkIdList> query_vertex_observed_landmark_ids;
    query_vertex.getAllObservedLandmarkIds(&query_vertex_observed_landmark_ids);
    inlier_constraint->query_vertex_id = query_vertex.id();

    vi_map::VertexKeyPointToStructureMatchList raw_structure_matches;
    loop_closure_handler::LoopClosureHandler::MergedLandmark3dPositionVector
        landmark_pairs_merged;
    double inlier_ratio;

    return lcWithPrior(
        projected_image_ptr_list, query_vertex.getVisualNFrame(),
        all_retrieved_frames_list, query_vertex_observed_landmark_ids,
        merge_landmarks, add_lc_edges, map, T_G_I, num_of_lc_matches,
        &inlier_ratio, &inlier_constraint->structure_matches,
        &raw_structure_matches, &landmark_pairs_merged,
        gt_T_G_I);
  }

  loop_closure::FrameToMatches frame_matches_list;
  constexpr bool kParallelFindIfPossible = true;
  loop_detector_->Find(
      projected_image_ptr_list, kParallelFindIfPossible, &frame_matches_list);
  *num_of_lc_matches = loop_closure::getNumberOfMatches(frame_matches_list);

  timing::Timer timer_compute_relative("lc compute absolute transform");
  pose_graph::VertexId vertex_id_closest_to_structure_matches;
  bool ransac_ok = computeAbsoluteTransformFromFrameMatches(
      frame_matches_list, merge_landmarks, add_lc_edges, map, T_G_I,
      inlier_constraint, &vertex_id_closest_to_structure_matches);
  timer_compute_relative.Stop();

#ifdef ADD_DEBUG_STATS
  // Debug
  proto::DebugFusion::QueryComponent* query_comp_proto =
    debug_fusion_proto_.add_query_components();
  if (ransac_ok) {
    query_comp_proto->set_status("ok");
  } else {
    query_comp_proto->set_status("fail");
  }
  CHECK_EQ(projected_image_ptr_list.size(), 1);
  projected_image_ptr_list[0]->keyframe_id.vertex_id.serialize(
      query_comp_proto->mutable_query_id());
  query_comp_proto->set_num_matches(*num_of_lc_matches);
  query_comp_proto->set_num_inliers(
      inlier_constraint->structure_matches.size());
  query_comp_proto->set_inlier_ratio(-1.);

  std::unordered_set<pose_graph::VertexId> matching_vertices;
  for (const loop_closure::Match& match :
       frame_matches_list[projected_image_ptr_list[0]->keyframe_id]) {
    proto::DebugFusion::QueryComponent::Match* match_proto =
      query_comp_proto->add_matches();
    CHECK_LT(match.keypoint_id_query.keypoint_index,
             projected_image_ptr_list[0]->measurements.cols());
    common::eigen_proto::serialize(
        Eigen::MatrixXd(projected_image_ptr_list[0]->measurements.col(
            match.keypoint_id_query.keypoint_index)),
        match_proto->mutable_query_measurement());

    //const aslam::VisualFrame& frame = query_vertex.getVisualFrame(
        //projected_image_ptr_list[0]->keyframe_id.frame_index);
    //Eigen::Vector2d projected_kp;
    //CHECK(frame.getRawCameraGeometry());
    //frame.toRawImageCoordinates(
        //Eigen::Vector2d(projected_image_ptr_list[0]->measurements.col(
            //match.keypoint_id_query.keypoint_index)),
        //&projected_kp);
    //common::eigen_proto::serialize(Eigen::MatrixXd(projected_kp),
                                   //match_proto->mutable_query_measurement());

    const loop_closure::ProjectedImage& proj_image_result =
      *visual_frame_to_projected_image_map_.at(match.keyframe_id_result);

    bool found = false;
    for (size_t i = 0; i < proj_image_result.landmarks.size(); i++) {
      if (proj_image_result.landmarks[i] == match.landmark_result) {
        CHECK_LT(i, proj_image_result.measurements.cols());
        common::eigen_proto::serialize(
            Eigen::MatrixXd(proj_image_result.measurements.col(i)),
            match_proto->mutable_db_measurement());
        found = true;
      }
    }
    CHECK(found);
    match.keyframe_id_result.vertex_id.serialize(
        match_proto->mutable_db_vertex_id());
    matching_vertices.insert(match.keyframe_id_result.vertex_id);
  }
  for (const pose_graph::VertexId& vid : matching_vertices) {
    vid.serialize(query_comp_proto->add_retrieved_ids());
  }

  common::eigen_proto::serialize(
      Eigen::MatrixXd(gt_T_G_I->getPosition()),
      query_comp_proto->mutable_gt_position());
  if (ransac_ok) {
    common::eigen_proto::serialize(
        Eigen::MatrixXd(T_G_I->getPosition()),
        query_comp_proto->mutable_pnp_position());
  }
#endif

  return ransac_ok;
}

bool LoopDetectorNode::computeAbsoluteTransformFromFrameMatches(
    const loop_closure::FrameToMatches& frame_to_matches,
    const bool merge_landmarks, const bool add_lc_edges, vi_map::VIMap* map,
    pose::Transformation* T_G_I,
    vi_map::LoopClosureConstraint* inlier_constraints,
    pose_graph::VertexId* vertex_id_closest_to_structure_matches) const {
  CHECK_NOTNULL(map);
  CHECK_NOTNULL(T_G_I);
  CHECK_NOTNULL(inlier_constraints);
  CHECK_NOTNULL(vertex_id_closest_to_structure_matches);

  const size_t num_matches = loop_closure::getNumberOfMatches(frame_to_matches);
  if (num_matches == 0u) {
    return false;
  }

  vi_map::LoopClosureConstraint constraint;
  for (const loop_closure::FrameIdMatchesPair& frame_matches_pair :
       frame_to_matches) {
    vi_map::LoopClosureConstraint tmp_constraint;
    const bool conversion_success =
        convertFrameMatchesToConstraint(frame_matches_pair, &tmp_constraint);
    if (!conversion_success) {
      continue;
    }
    constraint.query_vertex_id = tmp_constraint.query_vertex_id;
    constraint.structure_matches.insert(
        constraint.structure_matches.end(),
        tmp_constraint.structure_matches.begin(),
        tmp_constraint.structure_matches.end());
  }
  int num_inliers = 0;
  double inlier_ratio = 0.0;

  // The estimated transformation of this vertex to the map.
  pose::Transformation& T_G_I_ransac = *T_G_I;
  loop_closure_handler::LoopClosureHandler::MergedLandmark3dPositionVector
      landmark_pairs_merged;
  std::mutex map_mutex;
  bool ransac_ok = handleLoopClosures(
      constraint, merge_landmarks, add_lc_edges, &num_inliers, &inlier_ratio,
      map, &T_G_I_ransac, inlier_constraints, &landmark_pairs_merged,
      vertex_id_closest_to_structure_matches, &map_mutex);

  statistics::StatsCollector stats_ransac_inliers(
      "LC AbsolutePoseRansacInliers");
  stats_ransac_inliers.AddSample(num_inliers);
  statistics::StatsCollector stats_ransac_inlier_ratio(
      "LC AbsolutePoseRansacInlierRatio");
  stats_ransac_inlier_ratio.AddSample(num_inliers);

  return ransac_ok;
}

bool LoopDetectorNode::computeAbsoluteTransformFromFrameMatches(
    const aslam::VisualNFrame& query_vertex_n_frame,
    const std::vector<vi_map::LandmarkIdList>&
        query_vertex_observed_landmark_ids,
    const loop_closure::FrameToMatches& frame_to_matches,
    const bool merge_landmarks, const bool add_lc_edges,
    const loop_closure_handler::LoopClosureHandler& handler,
    pose::Transformation* T_G_I,
    vi_map::VertexKeyPointToStructureMatchList* inlier_structure_matches,
    pose_graph::VertexId* vertex_id_closest_to_structure_matches) const {
  CHECK_NOTNULL(T_G_I);
  CHECK_NOTNULL(inlier_structure_matches);
  // Note: vertex_id_closest_to_structure_matches is optional and may be NULL.

  const size_t num_matches = loop_closure::getNumberOfMatches(frame_to_matches);
  if (num_matches == 0u) {
    return false;
  }
  pose_graph::VertexId invalid_vertex_id;
  vi_map::LoopClosureConstraint constraint;
  constraint.query_vertex_id = invalid_vertex_id;
  for (const loop_closure::FrameIdMatchesPair& frame_matches_pair :
       frame_to_matches) {
    vi_map::LoopClosureConstraint tmp_constraint;
    const bool conversion_success =
        convertFrameMatchesToConstraint(frame_matches_pair, &tmp_constraint);
    if (!conversion_success) {
      continue;
    }
    constraint.query_vertex_id = tmp_constraint.query_vertex_id;
    constraint.structure_matches.insert(
        constraint.structure_matches.end(),
        tmp_constraint.structure_matches.begin(),
        tmp_constraint.structure_matches.end());
  }
  int num_inliers = 0;
  double inlier_ratio = 0.0;

  // The estimated transformation of this vertex to the map.
  pose::Transformation& T_G_I_ransac = *T_G_I;
  loop_closure_handler::LoopClosureHandler::MergedLandmark3dPositionVector
      landmark_pairs_merged;
  std::mutex map_mutex;

  bool ransac_ok = handler.handleLoopClosure(
      query_vertex_n_frame, query_vertex_observed_landmark_ids,
      invalid_vertex_id, constraint.structure_matches, merge_landmarks,
      add_lc_edges, &num_inliers, &inlier_ratio, &T_G_I_ransac,
      inlier_structure_matches, &landmark_pairs_merged,
      vertex_id_closest_to_structure_matches, &map_mutex);

  statistics::StatsCollector stats_ransac_inliers(
      "LC AbsolutePoseRansacInliers");
  stats_ransac_inliers.AddSample(num_inliers);
  statistics::StatsCollector stats_ransac_inlier_ratio(
      "LC AbsolutePoseRansacInlierRatio");
  stats_ransac_inlier_ratio.AddSample(num_inliers);

  return ransac_ok;
}

void LoopDetectorNode::queryVertexInDatabase(
    const pose_graph::VertexId& query_vertex_id, const bool merge_landmarks,
    const bool add_lc_edges, vi_map::VIMap* map,
    vi_map::LoopClosureConstraint* raw_constraint,
    vi_map::LoopClosureConstraint* inlier_constraint,
    std::vector<double>* inlier_ratios,
    aslam::TransformationVector* T_G_M2_vector,
    loop_closure_handler::LoopClosureHandler::MergedLandmark3dPositionVector*
        landmark_pairs_merged,
    std::mutex* map_mutex) const {
  CHECK_NOTNULL(map);
  CHECK_NOTNULL(raw_constraint);
  CHECK_NOTNULL(inlier_constraint);
  CHECK_NOTNULL(inlier_ratios);
  CHECK_NOTNULL(T_G_M2_vector);
  CHECK_NOTNULL(landmark_pairs_merged);
  CHECK_NOTNULL(map_mutex);
  CHECK(query_vertex_id.isValid());

  /*
  map_mutex->lock();
  const vi_map::Vertex& query_vertex = map->getVertex(query_vertex_id);
  const size_t num_frames = query_vertex.numFrames();
  loop_closure::ProjectedImagePtrList projected_image_ptr_list;
  projected_image_ptr_list.reserve(num_frames);

  for (size_t frame_idx = 0u; frame_idx < num_frames; ++frame_idx) {
    if (query_vertex.isVisualFrameSet(frame_idx) &&
        query_vertex.isVisualFrameValid(frame_idx)) {
      const aslam::VisualFrame& frame = query_vertex.getVisualFrame(frame_idx);
      CHECK(frame.hasKeypointMeasurements());
      if (frame.getNumKeypointMeasurements() == 0u) {
        // Skip frame if zero measurements found.
        continue;
      }

      std::vector<vi_map::LandmarkId> observed_landmark_ids;
      query_vertex.getFrameObservedLandmarkIds(frame_idx,
                                               &observed_landmark_ids);
      projected_image_ptr_list.push_back(
          std::make_shared<loop_closure::ProjectedImage>());
      const vi_map::VisualFrameIdentifier query_frame_id(
          query_vertex_id, frame_idx);
      constexpr bool kSkipInvalidLandmarkIds = false;
      convertFrameToProjectedImage(
          *map, query_frame_id, query_vertex.getVisualFrame(frame_idx),
          observed_landmark_ids, query_vertex.getMissionId(),
          kSkipInvalidLandmarkIds, projected_image_ptr_list.back().get());
    }
  }
  map_mutex->unlock();
  */

  const vi_map::Vertex& query_vertex = map->getVertex(query_vertex_id);
  const size_t num_frames = query_vertex.numFrames();
  loop_closure::ProjectedImagePtrList projected_image_ptr_list;
  projected_image_ptr_list.reserve(num_frames);
  std::unordered_set<vi_map::VisualFrameIdentifier> all_retrieved_frames_set;

  for (size_t frame_idx = 0u; frame_idx < num_frames; ++frame_idx) {
    if (query_vertex.isVisualFrameSet(frame_idx) &&
        query_vertex.isVisualFrameValid(frame_idx)) {
      vi_map::VisualFrameIdentifier query_frame_id(
          query_vertex.id(), frame_idx);

      std::vector<vi_map::LandmarkId> observed_landmark_ids;
      query_vertex.getFrameObservedLandmarkIds(
          frame_idx, &observed_landmark_ids);
      projected_image_ptr_list.push_back(
          std::make_shared<loop_closure::ProjectedImage>());
      constexpr bool kSkipInvalidLandmarkIds = false;
      convertFrameToProjectedImage(
          *map, query_frame_id, query_vertex.getVisualFrame(frame_idx),
          observed_landmark_ids, query_vertex.getMissionId(),
          kSkipInvalidLandmarkIds, projected_image_ptr_list.back().get());

      cv::Mat raw_image;
      if (use_deep_retrieval_ || use_better_descriptors_) {
        if (!map->hasRawImage(query_vertex, frame_idx)) {
          continue;
        }
        CHECK(map->getRawImage(query_vertex, frame_idx, &raw_image));
      }

      // Retrieve some prior frames
      if (use_deep_retrieval_) {
        vi_map::VisualFrameIdentifierList retrieved_frames;
        deep_retrieval_->RetrieveNearestNeighbors(
            raw_image, FLAGS_lc_deep_retrieval_num_nn,
            FLAGS_lc_deep_retrieval_max_nn_distance, &retrieved_frames);
        all_retrieved_frames_set.insert(
            retrieved_frames.begin(), retrieved_frames.end());
      }

      // Optionally replace the projected descriptors by better descriptors
      if (use_better_descriptors_) {
        addBetterDescriptorsToProjectedImage(
            raw_image, projected_image_ptr_list.back());
      }
    }
  }

  if (use_deep_retrieval_ && !all_retrieved_frames_set.size()) {
    statistics::StatsCollector noresource_counter("No resource for retrieval");
    noresource_counter.IncrementOne();
    LOG(WARNING) << "Vertex " << query_vertex.id() << " has no resource, skip";
    return;
  }

  if (use_deep_retrieval_) {
    vi_map::VisualFrameIdentifierList all_retrieved_frames_list(
        all_retrieved_frames_set.begin(), all_retrieved_frames_set.end());
    std::vector<vi_map::LandmarkIdList> query_vertex_observed_landmark_ids;
    query_vertex.getAllObservedLandmarkIds(&query_vertex_observed_landmark_ids);
    inlier_constraint->query_vertex_id = query_vertex.id();
    raw_constraint->query_vertex_id = query_vertex.id();

    unsigned num_of_lc_matches;
    double inlier_ratio = 0.0;
    pose::Transformation T_G_I_ransac;

    bool ransac_ok = lcWithPrior(
        projected_image_ptr_list, query_vertex.getVisualNFrame(),
        all_retrieved_frames_list, query_vertex_observed_landmark_ids,
        merge_landmarks, add_lc_edges, map, &T_G_I_ransac, &num_of_lc_matches,
        &inlier_ratio, &inlier_constraint->structure_matches,
        &raw_constraint->structure_matches, landmark_pairs_merged);

    if (ransac_ok && inlier_ratio != 0.0) {
      map_mutex->lock();
      const pose::Transformation& T_M_I = query_vertex.get_T_M_I();
      const pose::Transformation T_G_M2 = T_G_I_ransac * T_M_I.inverse();
      map_mutex->unlock();

      T_G_M2_vector->push_back(T_G_M2);
      inlier_ratios->push_back(inlier_ratio);
    }
    return;
  }


  loop_closure::FrameToMatches frame_matches;
  // Do not parallelize if the current function is running in multiple
  // threads to avoid decrease in performance.
  constexpr bool kParallelFindIfPossible = false;
  loop_detector_->Find(
      projected_image_ptr_list, kParallelFindIfPossible, &frame_matches);

  if (!frame_matches.empty()) {
    for (const loop_closure::FrameIdMatchesPair& id_and_matches :
         frame_matches) {
      vi_map::LoopClosureConstraint tmp_constraint;
      const bool conversion_success =
          convertFrameMatchesToConstraint(id_and_matches, &tmp_constraint);
      if (!conversion_success) {
        continue;
      }
      raw_constraint->query_vertex_id = tmp_constraint.query_vertex_id;
      raw_constraint->structure_matches.insert(
          raw_constraint->structure_matches.end(),
          tmp_constraint.structure_matches.begin(),
          tmp_constraint.structure_matches.end());
    }

    int num_inliers = 0;
    double inlier_ratio = 0.0;

    // The estimated transformation of this vertex to the map.
    pose::Transformation T_G_I_ransac;
    constexpr pose_graph::VertexId* kVertexIdClosestToStructureMatches =
        nullptr;
    bool ransac_ok = handleLoopClosures(
        *raw_constraint, merge_landmarks, add_lc_edges, &num_inliers,
        &inlier_ratio, map, &T_G_I_ransac, inlier_constraint,
        landmark_pairs_merged, kVertexIdClosestToStructureMatches, map_mutex);

    if (ransac_ok && inlier_ratio != 0.0) {
      map_mutex->lock();
      const pose::Transformation& T_M_I = query_vertex.get_T_M_I();
      const pose::Transformation T_G_M2 = T_G_I_ransac * T_M_I.inverse();
      map_mutex->unlock();

      T_G_M2_vector->push_back(T_G_M2);
      inlier_ratios->push_back(inlier_ratio);
    }
  }
}

void LoopDetectorNode::detectLoopClosuresMissionToDatabase(
    const MissionId& mission_id, const bool merge_landmarks,
    const bool add_lc_edges, int* num_vertex_candidate_links,
    double* summary_landmark_match_inlier_ratio, vi_map::VIMap* map,
    pose::Transformation* T_G_M_estimate,
    vi_map::LoopClosureConstraintVector* inlier_constraints) const {
  CHECK(map->hasMission(mission_id));
  pose_graph::VertexIdList vertices;
  map->getAllVertexIdsInMission(mission_id, &vertices);
  detectLoopClosuresVerticesToDatabase(
      vertices, merge_landmarks, add_lc_edges, num_vertex_candidate_links,
      summary_landmark_match_inlier_ratio, map, T_G_M_estimate,
      inlier_constraints);
}

void LoopDetectorNode::detectLoopClosuresVerticesToDatabase(
    const pose_graph::VertexIdList& vertices, const bool merge_landmarks,
    const bool add_lc_edges, int* num_vertex_candidate_links,
    double* summary_landmark_match_inlier_ratio, vi_map::VIMap* map,
    pose::Transformation* T_G_M_estimate,
    vi_map::LoopClosureConstraintVector* inlier_constraints) const {
  CHECK(!vertices.empty());
  CHECK_NOTNULL(num_vertex_candidate_links);
  CHECK_NOTNULL(summary_landmark_match_inlier_ratio);
  CHECK_NOTNULL(map);
  CHECK_NOTNULL(T_G_M_estimate)->setIdentity();
  CHECK_NOTNULL(inlier_constraints)->clear();

  *num_vertex_candidate_links = 0;
  *summary_landmark_match_inlier_ratio = 0.0;

  if (VLOG_IS_ON(1)) {
    std::ostringstream ss;
    for (const MissionId mission : missions_in_database_) {
      ss << mission << ", ";
    }
    VLOG(1) << "Searching for loop closures in missions " << ss.str();
  }

  std::vector<double> inlier_ratios;
  aslam::TransformationVector T_G_M_vector;

  std::mutex map_mutex;
  std::mutex output_mutex;

  loop_closure_handler::LoopClosureHandler::MergedLandmark3dPositionVector
      landmark_pairs_merged;
  vi_map::LoopClosureConstraintVector raw_constraints;

  // Then search for all in the database.
  common::MultiThreadedProgressBar progress_bar;

  std::function<void(const std::vector<size_t>&)> query_helper = [&](
      const std::vector<size_t>& range) {
    int num_processed = 0;
    progress_bar.setNumElements(range.size());
    for (const size_t job_index : range) {
      const pose_graph::VertexId& query_vertex_id = vertices[job_index];
      progress_bar.update(++num_processed);

      // Allocate local buffers to avoid locking.
      vi_map::LoopClosureConstraint raw_constraint_local;
      vi_map::LoopClosureConstraint inlier_constraint_local;
      using loop_closure_handler::LoopClosureHandler;
      LoopClosureHandler::MergedLandmark3dPositionVector
          landmark_pairs_merged_local;
      std::vector<double> inlier_ratios_local;
      aslam::TransformationVector T_G_M2_vector_local;

      // Lock the output buffers
      {
        std::unique_lock<std::mutex> lock_output(output_mutex);
        // Perform the actual query.
        queryVertexInDatabase(
            query_vertex_id, merge_landmarks, add_lc_edges, map,
            &raw_constraint_local, &inlier_constraint_local, &inlier_ratios_local,
            &T_G_M2_vector_local, &landmark_pairs_merged_local, &map_mutex);

        // Transfer results.
        if (raw_constraint_local.query_vertex_id.isValid()) {
          raw_constraints.push_back(raw_constraint_local);
        }
        if (inlier_constraint_local.query_vertex_id.isValid()) {
          inlier_constraints->push_back(inlier_constraint_local);
        }

        landmark_pairs_merged.insert(
            landmark_pairs_merged.end(), landmark_pairs_merged_local.begin(),
            landmark_pairs_merged_local.end());
        inlier_ratios.insert(
            inlier_ratios.end(), inlier_ratios_local.begin(),
            inlier_ratios_local.end());
        T_G_M_vector.insert(
            T_G_M_vector.end(), T_G_M2_vector_local.begin(),
            T_G_M2_vector_local.end());
      }
    }
  };

  constexpr bool kAlwaysParallelize = true;
  const size_t num_threads = common::getNumHardwareThreads();

  timing::Timer timing_mission_lc("lc query mission");
  common::ParallelProcess(
      vertices.size(), query_helper, kAlwaysParallelize, 1); //num_threads);
  timing_mission_lc.Stop();

  VLOG(1) << "Searched " << vertices.size() << " frames.";

  // If the plotter object was assigned.
  if (visualizer_) {
    vi_map::MissionIdList missions(
        missions_in_database_.begin(), missions_in_database_.end());
    vi_map::MissionIdSet query_mission_set;
    map->getMissionIds(vertices, &query_mission_set);
    missions.insert(
        missions.end(), query_mission_set.begin(), query_mission_set.end());

    visualizer_->visualizeKeyframeToStructureMatches(
        *inlier_constraints, raw_constraints, landmark_id_old_to_new_, *map);
    visualizer_->visualizeMergedLandmarks(landmark_pairs_merged);
    visualizer_->visualizeFullMapDatabase(missions, *map);
  }

  if (inlier_ratios.empty()) {
    LOG(WARNING) << "No loop found!";
    *summary_landmark_match_inlier_ratio = 0;
  } else {
    // Compute the median inlier ratio:
    // nth_element is not used on purpose because this function will be used
    // only in offline scenarios. Additionally, we only sort once.
    std::sort(inlier_ratios.begin(), inlier_ratios.end());
    *summary_landmark_match_inlier_ratio =
        inlier_ratios[inlier_ratios.size() / 2];

    LOG(INFO) << "Successfully loopclosed " << inlier_ratios.size()
              << " vertices. Merged " << landmark_pairs_merged.size()
              << " landmark pairs.";

    VLOG(1) << "Median inlier ratio: " << *summary_landmark_match_inlier_ratio;

    if (VLOG_IS_ON(2)) {
      std::stringstream inlier_ss;
      inlier_ss << "Inlier ratios: ";
      for (double val : inlier_ratios) {
        inlier_ss << val << " ";
      }
      VLOG(2) << inlier_ss.str();
    }

    // RANSAC and LSQ estimate of the mission baseframe transformation.
    constexpr int kNumRansacIterations = 2000;
    constexpr double kPositionErrorThresholdMeters = 2;
    constexpr double kOrientationErrorThresholdRadians = 0.174;  // ~10 deg.
    constexpr double kInlierRatioThreshold = 0.2;
    const int kNumInliersThreshold =
        T_G_M_vector.size() * kInlierRatioThreshold;
    aslam::Transformation T_G_M_LS;
    int num_inliers = 0;
    std::random_device device;
    const int ransac_seed = device();

    common::transformationRansac(
        T_G_M_vector, kNumRansacIterations, kOrientationErrorThresholdRadians,
        kPositionErrorThresholdMeters, ransac_seed, &T_G_M_LS, &num_inliers);
    if (num_inliers < kNumInliersThreshold) {
      VLOG(1) << "Found loops rejected by RANSAC! (Inliers " << num_inliers
              << "/" << T_G_M_vector.size() << ")";
      *summary_landmark_match_inlier_ratio = 0;
      *num_vertex_candidate_links = inlier_ratios.size();
      return;
    }
    const Eigen::Quaterniond& q_G_M_LS =
        T_G_M_LS.getRotation().toImplementation();

    // The datasets should be gravity-aligned so only yaw-axis rotation is
    // necessary to prealign them.
    Eigen::Vector3d rpy_G_M_LS =
        common::RotationMatrixToRollPitchYaw(q_G_M_LS.toRotationMatrix());
    rpy_G_M_LS(0) = 0.0;
    rpy_G_M_LS(1) = 0.0;
    Eigen::Quaterniond q_G_M_LS_yaw_only(
        common::RollPitchYawToRotationMatrix(rpy_G_M_LS));

    T_G_M_LS.getRotation().toImplementation() = q_G_M_LS_yaw_only;
    *T_G_M_estimate = T_G_M_LS;
  }
  *num_vertex_candidate_links = inlier_ratios.size();
}

void LoopDetectorNode::detectLoopClosuresAndMergeLandmarks(
    const MissionId& mission, vi_map::VIMap* map) {
  CHECK_NOTNULL(map);

  constexpr bool kMergeLandmarks = true;
  constexpr bool kAddLoopclosureEdges = false;
  int num_vertex_candidate_links;
  double summary_landmark_match_inlier_ratio;

  pose::Transformation T_G_M2;
  vi_map::LoopClosureConstraintVector inlier_constraints;
  detectLoopClosuresMissionToDatabase(
      mission, kMergeLandmarks, kAddLoopclosureEdges,
      &num_vertex_candidate_links, &summary_landmark_match_inlier_ratio, map,
      &T_G_M2, &inlier_constraints);

  VLOG(1) << "Handling loop closures done.";
}

bool LoopDetectorNode::handleLoopClosures(
    const vi_map::LoopClosureConstraint& constraint, const bool merge_landmarks,
    const bool add_lc_edges, int* num_inliers, double* inlier_ratio,
    vi_map::VIMap* map, pose::Transformation* T_G_I_ransac,
    vi_map::LoopClosureConstraint* inlier_constraints,
    loop_closure_handler::LoopClosureHandler::MergedLandmark3dPositionVector*
        landmark_pairs_merged,
    pose_graph::VertexId* vertex_id_closest_to_structure_matches,
    std::mutex* map_mutex) const {
  CHECK_NOTNULL(num_inliers);
  CHECK_NOTNULL(inlier_ratio);
  CHECK_NOTNULL(map);
  CHECK_NOTNULL(T_G_I_ransac);
  CHECK_NOTNULL(inlier_constraints);
  CHECK_NOTNULL(landmark_pairs_merged);
  CHECK_NOTNULL(map_mutex);
  // Note: vertex_id_closest_to_structure_matches is optional and may beb NULL.
  loop_closure_handler::LoopClosureHandler handler(
      map, &landmark_id_old_to_new_);
  return handler.handleLoopClosure(
      constraint, merge_landmarks, add_lc_edges, num_inliers, inlier_ratio,
      T_G_I_ransac, inlier_constraints, landmark_pairs_merged,
      vertex_id_closest_to_structure_matches, map_mutex, use_random_pnp_seed_);
}

void LoopDetectorNode::instantiateVisualizer() {
  visualizer_.reset(new loop_closure_visualization::LoopClosureVisualizer());
}

void LoopDetectorNode::clear() {
  loop_detector_->Clear();
}

void LoopDetectorNode::serialize(
    proto::LoopDetectorNode* proto_loop_detector_node) const {
  CHECK_NOTNULL(proto_loop_detector_node);

  for (const vi_map::MissionId& mission : missions_in_database_) {
    mission.serialize(
        CHECK_NOTNULL(proto_loop_detector_node->add_mission_ids()));
  }

  loop_detector_->serialize(
      proto_loop_detector_node->mutable_matching_based_loop_detector());
}

void LoopDetectorNode::deserialize(
    const proto::LoopDetectorNode& proto_loop_detector_node) {
  const int num_missions = proto_loop_detector_node.mission_ids_size();
  VLOG(1) << "Parsing loop detector with " << num_missions << " missions.";
  for (int idx = 0; idx < num_missions; ++idx) {
    vi_map::MissionId mission_id;
    mission_id.deserialize(proto_loop_detector_node.mission_ids(idx));
    CHECK(mission_id.isValid());
    missions_in_database_.insert(mission_id);
  }

  CHECK(loop_detector_);
  loop_detector_->deserialize(
      proto_loop_detector_node.matching_based_loop_detector());
}

const std::string& LoopDetectorNode::getDefaultSerializationFilename() {
  return serialization_filename_;
}

}  // namespace loop_detector_node
