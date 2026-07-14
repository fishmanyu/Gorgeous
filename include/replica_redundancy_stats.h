#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace diskann {

enum class ReplicaAdjacencySource : uint8_t {
  OWNER = 0,
  REPLICA = 1
};

struct ReplicaFirstSource {
  uint32_t page_id = 0;
  ReplicaAdjacencySource type = ReplicaAdjacencySource::OWNER;
};

struct ReplicaRedundancyCounters {
  uint64_t graph_page_ios = 0;
  uint64_t first_read_pages = 0;
  uint64_t repeated_page_reads = 0;

  uint64_t replica_attempts = 0;
  uint64_t replica_attempts_on_first_read_pages = 0;

  uint64_t unique_replica_adjacencies = 0;
  uint64_t all_replica_duplicates = 0;

  uint64_t cross_page_replica_duplicates = 0;
  uint64_t replica_to_replica_duplicates = 0;
  uint64_t owner_to_replica_duplicates = 0;

  uint64_t pages_with_any_replica_duplicate = 0;
  uint64_t fully_redundant_replica_pages = 0;
};

class ReplicaRedundancyTracker {
 public:
  bool record_page(uint32_t page_id, uint32_t owner_id,
                   const std::vector<uint32_t>& replica_ids) {
    counters_.graph_page_ios++;
    const bool first_page_read = read_pages_.insert(page_id).second;
    if (!first_page_read) {
      counters_.repeated_page_reads++;
      counters_.replica_attempts += replica_ids.size();
      return false;
    }

    counters_.first_read_pages++;
    first_sources_.emplace(
        owner_id, ReplicaFirstSource{page_id, ReplicaAdjacencySource::OWNER});

    uint64_t page_replica_count = 0;
    uint64_t page_duplicate_replica_count = 0;

    for (uint32_t replica_id : replica_ids) {
      counters_.replica_attempts++;
      counters_.replica_attempts_on_first_read_pages++;
      page_replica_count++;

      auto iter = first_sources_.find(replica_id);
      if (iter == first_sources_.end()) {
        first_sources_.emplace(
            replica_id,
            ReplicaFirstSource{page_id, ReplicaAdjacencySource::REPLICA});
        counters_.unique_replica_adjacencies++;
        continue;
      }

      counters_.all_replica_duplicates++;
      if (iter->second.page_id == page_id) {
        continue;
      }

      page_duplicate_replica_count++;
      counters_.cross_page_replica_duplicates++;
      if (iter->second.type == ReplicaAdjacencySource::REPLICA) {
        counters_.replica_to_replica_duplicates++;
      } else {
        counters_.owner_to_replica_duplicates++;
      }
    }

    if (page_duplicate_replica_count > 0) {
      counters_.pages_with_any_replica_duplicate++;
    }
    if (page_replica_count > 0 &&
        page_duplicate_replica_count == page_replica_count) {
      counters_.fully_redundant_replica_pages++;
    }
    return true;
  }

  const ReplicaRedundancyCounters& counters() const { return counters_; }

 private:
  ReplicaRedundancyCounters counters_;
  std::unordered_set<uint32_t> read_pages_;
  std::unordered_map<uint32_t, ReplicaFirstSource> first_sources_;
};

}  // namespace diskann
