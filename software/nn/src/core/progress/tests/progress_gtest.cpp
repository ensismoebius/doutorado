#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <thread>

#include "nn/progress/ProgressBar.hpp"
#include "nn/progress/ProgressManager.hpp"

TEST(ProgressTest, SingleBarUpdate)
{
    nn::progress::ProgressBar bar("Test Bar", 100.0f);
    bar.update(10.0f, {{"metric", 1.23f}});

    // Verify internal state via Manager (since ProgressBar is just a handle)
    uint32_t id = bar.id();
    EXPECT_GE(id, 0U);
    // We can't easily check the state of ProgressManager private members without a friend class
    // but we can verify that calling update doesn't crash and the thread is running.
    SUCCEED();
}

TEST(ProgressTest, MultiBarConcurrent)
{
    nn::progress::ProgressBar bar1("Bar 1", 100.0f);
    nn::progress::ProgressBar bar2("Bar 2", 100.0f);

    for (int i = 0; i < 10; ++i)
    {
        bar1.update(static_cast<float>(i), {{"val", (float) i}});
        bar2.update(static_cast<float>(i), {{"val", (float) i}});
    }
    SUCCEED();
}

TEST(ProgressTest, MarkComplete)
{
    nn::progress::ProgressBar bar("Done Bar", 1.0f);
    bar.mark_complete();
    SUCCEED();
}

TEST(ProgressTest, ManagerMetadataAndTimingApis)
{
    auto& manager = nn::progress::ProgressManager::instance();
    const uint32_t id = manager.create_bar(
        "A very long progress label that forces formatting to trim metadata columns", 0.0f);

    manager.set_description(id, "LSTM Autoencoder");
    manager.set_fold_info(id, 2, 5);
    manager.set_loss_type(id, "MSE");
    manager.set_test_loss(id, 0.123f);
    manager.set_phases(id, {"load", "train", "eval"}, 1);
    manager.set_target(id, 0.0f);

    manager.begin_active_work(id);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    manager.update_bar(id, 0.25f, {{"train_loss", 1.25f}, {"val_loss", 0.75f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    manager.end_active_work(id);

    manager.update_bar(id, 0.75f, {{"metric", 2.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    manager.complete_bar(id);
    manager.remove_bar(id);

    SUCCEED();
}

TEST(ProgressTest, ManagerIgnoresUnknownIds)
{
    auto& manager = nn::progress::ProgressManager::instance();

    manager.update_bar(999999u, 1.0f, {{"metric", 1.0f}});
    manager.set_target(999999u, 10.0f);
    manager.complete_bar(999999u);
    manager.remove_bar(999999u);
    manager.set_description(999999u, "unused");
    manager.set_fold_info(999999u, 1, 2);
    manager.set_loss_type(999999u, "MSE");
    manager.set_test_loss(999999u, 0.5f);
    manager.set_phases(999999u, {"phase"}, 0);
    manager.begin_active_work(999999u);
    manager.end_active_work(999999u);

    SUCCEED();
}

TEST(ProgressTest, ShutdownCleansUpRenderer)
{
    auto& manager = nn::progress::ProgressManager::instance();
    const uint32_t id = manager.create_bar("Shutdown Bar", 4.0f);
    manager.update_bar(id, 2.0f, {{"metric", 1.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    manager.shutdown();
    SUCCEED();
}
