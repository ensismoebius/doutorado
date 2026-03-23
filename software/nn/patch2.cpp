--- a/software/nn/src/experiments/03/lib/src/experiment03.cpp
+++ b/software/nn/src/experiments/03/lib/src/experiment03.cpp
@@ -329,6 +329,9 @@
                 config_.prefetch_ram_cap_mb * std::size_t{1024 * 1024} //
             );
 
+            // Give the prefetcher a head start so the queue fills up!
+            std::this_thread::sleep_for(std::chrono::seconds(2));
+
             // Iterate over batches produced by the prefetcher.
             while (prefetcher_->hasNext()) [[likely]]
             {
