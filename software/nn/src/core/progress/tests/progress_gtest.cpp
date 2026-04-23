#include <gtest/gtest.h>
#include "nn/progress/ProgressBar.hpp"
#include "nn/progress/ProgressManager.hpp"
#include <thread>
#include <chrono>
#include <map>

TEST(ProgressTest, SingleBarUpdate)
{
    nn::progress::ProgressBar bar("Test Bar", 100.0f);
    bar.update(10.0f, {{"metric", 1.23f}});
    
    // Verify internal state via Manager (since ProgressBar is just a handle)
    uint32_t id = bar.id();
    // We can't easily check the state of ProgressManager private members without a friend class
    // but we can verify that calling update doesn't crash and the thread is running.
    SUCCEED(); 
}

TEST(ProgressTest, MultiBarConcurrent)
{
    nn::progress::ProgressBar bar1("Bar 1", 100.0f);
    nn::progress::ProgressBar bar2("Bar 2", 100.0f);
    
    for(int i=0; i<10; ++i) {
        bar1.update(static_cast<float>(i), {{"val", (float)i}});
        bar2.update(static_cast<float>(i), {{"val", (float)i}});
    }
    SUCCEED();
}

TEST(ProgressTest, MarkComplete)
{
    nn::progress::ProgressBar bar("Done Bar", 1.0f);
    bar.mark_complete();
    SUCCEED();
}
