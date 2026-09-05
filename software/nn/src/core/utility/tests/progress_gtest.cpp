/**
 * @file progress_gtest.cpp
 * @brief Characterization tests for printProgress().
 *
 * printProgress renders three stacked bars into a terminal. It had no tests at
 * all, which is why it could grow to 208 lines and cyclomatic complexity 32
 * without anyone noticing: there was nothing to break. These tests pin what it
 * actually prints today, so the function can be split up afterwards and any
 * change in the rendered text shows as a failing assertion rather than as a
 * progress bar that quietly reads 87% forever.
 *
 * Two things make this function awkward to test, and both are pinned here
 * rather than papered over:
 *  - it writes to a stream chosen at call time (the logger's console buffer if
 *    one is installed, otherwise std::cout), so the tests install their own;
 *  - it keeps STATIC state across calls (whether the reserved area was drawn,
 *    and a six-line log history). Assertions therefore key on the status text,
 *    never on the leading newlines or cursor-up escapes, which depend on
 *    whether this is the first call in the process.
 */

#include <gtest/gtest.h>

#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "logging/Logger.hpp"
#include "tensor/Tensor.hpp"
#include "utility/progress.hpp"

namespace
{

/// Redirects whatever stream printProgress picks into a string, for the
/// lifetime of the object. Installing the logger's console buffer is what
/// makes the choice deterministic instead of depending on std::cout's state.
class CapturedProgress
{
   public:
    CapturedProgress()
    {
        nn::logging::Logger::instance().set_console_rdbuf(buffer_.rdbuf());
    }
    ~CapturedProgress()
    {
        nn::logging::Logger::instance().set_console_rdbuf(nullptr);
    }

    CapturedProgress(const CapturedProgress&) = delete;
    auto operator=(const CapturedProgress&) -> CapturedProgress& = delete;

    [[nodiscard]] auto text() const -> std::string
    {
        return buffer_.str();
    }

    /// The line starting with `prefix` ("Fold:", "Epoch:", "Batch:"), stripped
    /// of the ANSI clear-line escape that precedes it.
    [[nodiscard]] auto line(const std::string& prefix) const -> std::string
    {
        std::istringstream stream(buffer_.str());
        std::string raw;
        while (std::getline(stream, raw))
        {
            const std::size_t at = raw.find(prefix);
            if (at != std::string::npos) return raw.substr(at);
        }
        return {};
    }

   private:
    std::ostringstream buffer_;
};

/// The characters between '[' and ']' on a status line.
auto bar_of(const std::string& status_line) -> std::string
{
    const std::size_t open = status_line.find('[');
    const std::size_t close = status_line.find(']', open);
    if (open == std::string::npos || close == std::string::npos) return {};
    return status_line.substr(open + 1, close - open - 1);
}

/// The integer immediately before the '%' on a status line.
auto percent_of(const std::string& status_line) -> int
{
    const std::size_t pct = status_line.find('%');
    if (pct == std::string::npos) return -1;
    std::size_t start = pct;
    while (start > 0 && (std::isdigit(static_cast<unsigned char>(status_line[start - 1])) != 0))
    {
        --start;
    }
    if (start == pct) return -1;
    return std::stoi(status_line.substr(start, pct - start));
}

constexpr double kNoLoss = std::numeric_limits<double>::quiet_NaN();

// ── Effective totals ────────────────────────────────────────────────────────

TEST(PrintProgress, CappedRunStillReachesOneHundredPercent)
{
    // 1000 samples at batch 10 is 100 batches, but the run is capped at 20.
    // Without the cap folded into the totals, a finished run would display 20%
    // and the bar would never fill — the whole reason "effective totals" exist.
    CapturedProgress captured;
    printProgress(/*dataset_total_samples=*/1000,
        /*batch_size=*/10,
        /*max_batches=*/20,
        /*seen_batches=*/20,
        /*processed_samples=*/200,
        /*done=*/true);

    EXPECT_EQ(percent_of(captured.line("Batch:")), 100) << captured.line("Batch:");
    EXPECT_NE(captured.line("Batch:").find("200/200s"), std::string::npos)
        << captured.line("Batch:");
    EXPECT_NE(captured.line("Batch:").find("20/20b"), std::string::npos) << captured.line("Batch:");
}

TEST(PrintProgress, UncappedRunUsesTheFullDatasetTotals)
{
    // max_batches above the dataset must not shrink the totals to it.
    CapturedProgress captured;
    printProgress(1000, 10, 100000, 50, 500, false);

    EXPECT_EQ(percent_of(captured.line("Batch:")), 50);
    EXPECT_NE(captured.line("Batch:").find("500/1000s"), std::string::npos)
        << captured.line("Batch:");
}

TEST(PrintProgress, ProgressBeyondTheTotalIsClampedNotOverflowed)
{
    // A loader that over-reports must not print 340% or a bar longer than the
    // field, which would corrupt the whole reserved area.
    CapturedProgress captured;
    printProgress(100, 10, 10, /*seen_batches=*/34, /*processed_samples=*/340, false);

    EXPECT_EQ(percent_of(captured.line("Batch:")), 100);
    EXPECT_EQ(bar_of(captured.line("Batch:")).size(), 40U);
}

TEST(PrintProgress, ZeroBatchSizeIsSilentlyTreatedAsOne)
{
    // Documenting current behaviour, NOT endorsing it: batch_size 0 is a
    // caller error and the project's no-fallback rule says it should throw.
    // Today it is quietly repaired to 1, so a misconfigured run prints a
    // plausible bar instead of stopping. Pinned so the repair cannot be
    // removed by accident, and so removing it deliberately is a visible change.
    CapturedProgress captured;
    EXPECT_NO_THROW(printProgress(10, /*batch_size=*/0, 10, 5, 5, false));
    EXPECT_NE(captured.line("Batch:").find("5/10s"), std::string::npos) << captured.line("Batch:");
}

TEST(PrintProgress, EmptyDatasetIsZeroPercentNotNaN)
{
    CapturedProgress captured;
    printProgress(0, 10, 10, 0, 0, false);

    EXPECT_EQ(percent_of(captured.line("Batch:")), 0);
    EXPECT_EQ(captured.text().find("nan"), std::string::npos) << captured.text();
}

// ── Bar geometry ────────────────────────────────────────────────────────────

TEST(PrintProgress, EveryBarIsExactlyFortyCharactersWide)
{
    // The reserved-area redraw overwrites fixed-width lines; a bar that changes
    // width leaves debris on screen. Checked at 0%, mid and 100%.
    for (const std::size_t seen : {std::size_t{0}, std::size_t{3}, std::size_t{10}})
    {
        CapturedProgress captured;
        printProgress(100, 10, 10, seen, seen * 10, false, 1, 3, 1, 4, seen, 10);

        EXPECT_EQ(bar_of(captured.line("Fold:")).size(), 40U) << "seen=" << seen;
        EXPECT_EQ(bar_of(captured.line("Epoch:")).size(), 40U) << "seen=" << seen;
        EXPECT_EQ(bar_of(captured.line("Batch:")).size(), 40U) << "seen=" << seen;
    }
}

TEST(PrintProgress, AnUnfinishedBarCarriesTheArrowHead)
{
    CapturedProgress captured;
    printProgress(100, 10, 10, 5, 50, false);

    const std::string bar = bar_of(captured.line("Batch:"));
    EXPECT_NE(bar.find('>'), std::string::npos) << bar;
    EXPECT_EQ(bar.find(">="), std::string::npos) << "arrow must be past the fill: " << bar;
}

TEST(PrintProgress, AFullBarHasNoArrowHead)
{
    CapturedProgress captured;
    printProgress(100, 10, 10, 10, 100, true);

    const std::string bar = bar_of(captured.line("Batch:"));
    EXPECT_EQ(bar.find('>'), std::string::npos) << bar;
    EXPECT_EQ(bar, std::string(40, '='));
}

// ── Nested fold / epoch / batch composition ─────────────────────────────────

TEST(PrintProgress, EpochRatioCountsCompletedEpochsNotTheCurrentOne)
{
    // Epoch 2 of 4 with no batches consumed yet is ONE finished epoch: 25%.
    // Off-by-one here is the classic silent bug — the bar would read 50% and
    // finish the run at 125% — so it is pinned explicitly.
    CapturedProgress captured;
    printProgress(100,
        10,
        10,
        0,
        0,
        false,
        /*current_fold=*/0,
        /*total_folds=*/0,
        /*current_epoch=*/2,
        /*total_epochs=*/4,
        /*epoch_seen_batches=*/0,
        /*epoch_total_batches=*/10);

    EXPECT_EQ(percent_of(captured.line("Epoch:")), 25) << captured.line("Epoch:");
}

TEST(PrintProgress, FoldRatioNestsTheEpochRatioInsideIt)
{
    // The nesting is two levels deep, so read it inside out:
    //   batches within the epoch: 5/10                       = 0.50
    //   epochs within the fold:   (2 - 1 + 0.50) / 2         = 0.75
    //   folds within the run:     (2 - 1 + 0.75) / 4         = 0.4375  -> 43%
    // Getting this wrong is invisible: the bar still moves, just at the wrong
    // speed, and only ever looks slightly off.
    CapturedProgress captured;
    printProgress(100,
        10,
        10,
        0,
        0,
        false,
        /*current_fold=*/2,
        /*total_folds=*/4,
        /*current_epoch=*/2,
        /*total_epochs=*/2,
        /*epoch_seen_batches=*/5,
        /*epoch_total_batches=*/10);

    EXPECT_EQ(percent_of(captured.line("Fold:")), 43) << captured.line("Fold:");
}

TEST(PrintProgress, CountersAppearOnlyWhenBothIndexAndTotalArePositive)
{
    CapturedProgress with_counters;
    printProgress(100, 10, 10, 1, 10, false, 2, 5, 3, 7, 1, 10);
    EXPECT_NE(with_counters.line("Fold:").find("(2/5)"), std::string::npos)
        << with_counters.line("Fold:");
    EXPECT_NE(with_counters.line("Epoch:").find("(3/7)"), std::string::npos)
        << with_counters.line("Epoch:");

    CapturedProgress without_counters;
    printProgress(100,
        10,
        10,
        1,
        10,
        false,
        /*current_fold=*/0,
        /*total_folds=*/5,
        /*current_epoch=*/3,
        /*total_epochs=*/0);
    EXPECT_EQ(without_counters.line("Fold:").find('('), std::string::npos)
        << without_counters.line("Fold:");
    EXPECT_EQ(without_counters.line("Epoch:").find('('), std::string::npos)
        << without_counters.line("Epoch:");
}

// ── Optional decorations ────────────────────────────────────────────────────

TEST(PrintProgress, AFiniteLossIsPrintedWithSixDecimals)
{
    CapturedProgress captured;
    printProgress(100, 10, 10, 5, 50, false, 0, 0, 0, 0, 0, 0, /*current_loss=*/0.5);

    EXPECT_NE(captured.line("Batch:").find("loss: 0.500000"), std::string::npos)
        << captured.line("Batch:");
}

TEST(PrintProgress, ANonFiniteLossIsOmittedRatherThanPrintedAsNan)
{
    // The default is NaN, meaning "no loss yet". Printing "loss: nan" every
    // step would train the reader to ignore a genuine NaN loss later.
    for (const double loss : {kNoLoss, std::numeric_limits<double>::infinity()})
    {
        CapturedProgress captured;
        printProgress(100, 10, 10, 5, 50, false, 0, 0, 0, 0, 0, 0, loss);
        EXPECT_EQ(captured.line("Batch:").find("loss:"), std::string::npos)
            << captured.line("Batch:");
    }
}

TEST(PrintProgress, ContextPrefixesTheFoldLineOnly)
{
    CapturedProgress captured;
    printProgress(100,
        10,
        10,
        5,
        50,
        false,
        0,
        0,
        0,
        0,
        0,
        0,
        kNoLoss,
        std::span<nn::Tensor*>{},
        /*context=*/"Subject 7");

    EXPECT_EQ(captured.line("Subject 7").rfind("Subject 7", 0), 0U);
    EXPECT_NE(captured.line("Subject 7").find("Fold:"), std::string::npos);
    EXPECT_EQ(captured.line("Epoch:").rfind("Subject 7", 0), std::string::npos);
}

TEST(PrintProgress, ParameterSummaryIsPrintedOnlyWhenDoneAndNonEmpty)
{
    nn::Tensor weights(2, 2);
    weights.at(0, 0) = 1.0F;
    weights.at(0, 1) = -2.0F;
    weights.at(1, 0) = 3.0F;
    weights.at(1, 1) = -4.0F;
    std::vector<nn::Tensor*> params{&weights};

    CapturedProgress finished;
    printProgress(100,
        10,
        10,
        10,
        100,
        /*done=*/true,
        0,
        0,
        0,
        0,
        0,
        0,
        kNoLoss,
        std::span<nn::Tensor*>{params});
    EXPECT_NE(finished.text().find("Final network parameters:"), std::string::npos);
    EXPECT_NE(finished.text().find("2x2"), std::string::npos) << finished.text();

    CapturedProgress unfinished;
    printProgress(100,
        10,
        10,
        5,
        50,
        /*done=*/false,
        0,
        0,
        0,
        0,
        0,
        0,
        kNoLoss,
        std::span<nn::Tensor*>{params});
    EXPECT_EQ(unfinished.text().find("Final network parameters:"), std::string::npos);

    CapturedProgress no_params;
    printProgress(100, 10, 10, 10, 100, /*done=*/true);
    EXPECT_EQ(no_params.text().find("Final network parameters:"), std::string::npos);
}

TEST(PrintProgress, ANullParameterPointerIsSkippedRatherThanDereferenced)
{
    nn::Tensor weights(1, 1);
    weights.at(0, 0) = 5.0F;
    std::vector<nn::Tensor*> params{nullptr, &weights};

    CapturedProgress captured;
    EXPECT_NO_THROW(printProgress(
        100, 10, 10, 10, 100, true, 0, 0, 0, 0, 0, 0, kNoLoss, std::span<nn::Tensor*>{params}));
    EXPECT_NE(captured.text().find("1x1"), std::string::npos) << captured.text();
}

} // namespace
