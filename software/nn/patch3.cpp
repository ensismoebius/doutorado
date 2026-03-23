--- a/software/nn/src/experiments/03/lib/src/experiment03.cpp
+++ b/software/nn/src/experiments/03/lib/src/experiment03.cpp
@@ -330,7 +330,7 @@
             );
 
             // Give the prefetcher a head start so the queue fills up!
             std::this_thread::sleep_for(std::chrono::seconds(2));
+            std::cout << "After sleep, prefetcher queue size: " << prefetcher_->diagnostics().push_successes << std::endl;
 
             // Iterate over batches produced by the prefetcher.
             while (prefetcher_->hasNext()) [[likely]]
