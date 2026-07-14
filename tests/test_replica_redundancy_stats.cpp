#include <cassert>
#include <iostream>
#include <vector>

#include "replica_redundancy_stats.h"

int main() {
  diskann::ReplicaRedundancyTracker tracker;

  tracker.record_page(1, 1, std::vector<uint32_t>{3, 4});  // A: C, D
  tracker.record_page(2, 2, std::vector<uint32_t>{3, 4});  // B: C, D
  tracker.record_page(5, 5, std::vector<uint32_t>{4, 6});  // E: D, F

  const auto &after_first_reads = tracker.counters();
  assert(after_first_reads.replica_attempts_on_first_read_pages == 6);
  assert(after_first_reads.replica_to_replica_duplicates == 3);
  assert(after_first_reads.unique_replica_adjacencies == 3);
  assert(after_first_reads.first_read_pages == 3);
  assert(after_first_reads.repeated_page_reads == 0);

  tracker.record_page(1, 1, std::vector<uint32_t>{3, 4});  // repeated A

  const auto &after_repeat = tracker.counters();
  assert(after_repeat.repeated_page_reads == 1);
  assert(after_repeat.replica_attempts == 8);
  assert(after_repeat.replica_attempts_on_first_read_pages == 6);
  assert(after_repeat.replica_to_replica_duplicates == 3);

  std::cout << "replica redundancy tracker test passed\n";
  return 0;
}
