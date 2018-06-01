#ifndef ROVIOLI_DEEP_RELOC_LOCALIZER_HELPERS_H_
#define ROVIOLI_DEEP_RELOC_LOCALIZER_HELPERS_H_

#include <localization-summary-map/localization-summary-map-queries.h>
#include <localization-summary-map/localization-summary-map.h>
#include <vi-map/vi-map.h>
#include <loop-closure-handler/loop-detector-node.h>
#include <maplab-common/macros.h>
#include <vio-common/vio-types.h>

namespace rovioli_deep_reloc {

void convertVertexKeyPointToStructureMatchListToLocalizationResult(
    const vi_map::VIMap& map,
    const aslam::VisualNFrame& query_nframe,
    const vi_map::VertexKeyPointToStructureMatchList& inlier_structure_matches,
    vio::LocalizationResult* localization_result);

void subselectStructureMatches(
    const summary_map::LocalizationSummaryMap& map,
    const summary_map::SummaryMapCachedLookups& map_cached_lookup,
    const aslam::VisualNFrame& nframe,
    size_t num_max_landmarks_to_keep_per_camera,
    vi_map::VertexKeyPointToStructureMatchList* structure_matches);

}  // namespace rovioli_deep_reloc

#endif  // ROVIOLI_DEEP_RELOC_LOCALIZER_HELPERS_H_
