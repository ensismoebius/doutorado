#ifndef NN_LOGGER_HPP
#define NN_LOGGER_HPP

#include <atomic>
#include <chrono>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace nn::logging
{
enum class Level
{
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
};

class Logger
{
   public:
    static Logger& instance()
    {
        static Logger inst;
        return inst;
    }

    void set_level(Level l)
    {
        level_.store(static_cast<int>(l));
    }
    Level level() const
    {
        return static_cast<Level>(level_.load());
    }

    void log(Level l, const std::string& msg)
    {
        if (static_cast<int>(l) > level_.load()) return;
        std::lock_guard<std::mutex> g(mu_);
        auto now = std::chrono::system_clock::now();
        auto itt = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&itt);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%F %T") << " ";
        switch (l)
        {
            case Level::Error:
                oss << "ERROR: ";
                break;
            case Level::Warn:
                oss << "WARN:  ";
                break;
            case Level::Info:
                oss << "INFO:  ";
                break;
            case Level::Debug:
                oss << "DEBUG: ";
                break;
        }
        oss << msg << std::endl;
        const std::string formatted = oss.str();
        // Write to configured stream (file/err) and also keep a recent in-memory copy
        if (!suppress_console_output_)
        {
            output_stream_ << formatted;
            output_stream_.flush();
        }
        // store a trimmed copy without trailing newline for clean printing above progress
        std::string copy = formatted;
        if (!copy.empty() && copy.back() == '\n') copy.pop_back();
        recent_lines_.push_back(copy);
        if (recent_lines_.size() > recent_capacity_)
        {
            recent_lines_.pop_front();
        }
    }

    // Drain and return recent lines (clears the internal buffer). Thread-safe.
    std::vector<std::string> drain_recent_lines()
    {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<std::string> out;
        out.reserve(recent_lines_.size());
        for (auto& s : recent_lines_) out.push_back(s);
        recent_lines_.clear();
        return out;
    }

    // Non-destructive reader: return up to `n` most recent lines without
    // clearing the internal buffer. Thread-safe.
    std::vector<std::string> get_recent_lines(size_t n)
    {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<std::string> out;
        if (recent_lines_.empty() || n == 0) return out;
        size_t total = recent_lines_.size();
        size_t start = (n >= total) ? 0 : (total - n);
        out.reserve(std::min(n, total));
        for (size_t i = start; i < total; ++i) out.push_back(recent_lines_[i]);
        return out;
    }

    void set_output_stream(std::ostream& os)
    {
        std::lock_guard<std::mutex> g(mu_);
        output_stream_.rdbuf(os.rdbuf());
    }

    void set_suppress_console_output(bool v)
    {
        std::lock_guard<std::mutex> g(mu_);
        suppress_console_output_ = v;
    }

    // Store a pointer to the underlying console rdbuf (original stdout rdbuf)
    void set_console_rdbuf(std::streambuf* rb)
    {
        std::lock_guard<std::mutex> g(mu_);
        console_rdbuf_ = rb;
    }

    std::streambuf* get_console_rdbuf()
    {
        std::lock_guard<std::mutex> g(mu_);
        return console_rdbuf_;
    }

   private:
    Logger() : level_(static_cast<int>(Level::Info)), output_stream_(std::cerr.rdbuf()) {}
    std::atomic<int> level_;
    std::mutex mu_;
    std::ostream output_stream_;
    // Small in-memory ring buffer of recent formatted log lines (no trailing newline)
    std::deque<std::string> recent_lines_;
    size_t recent_capacity_ = 200;
    std::streambuf* console_rdbuf_ = nullptr;
    bool suppress_console_output_ = false;
};

inline void log(Level l, const std::string& msg)
{
    Logger::instance().log(l, msg);
}

} // namespace nn::logging

// Convenience macros
#define NN_LOG_ERROR(msg) ::nn::logging::log(::nn::logging::Level::Error, (msg))
#define NN_LOG_WARN(msg) ::nn::logging::log(::nn::logging::Level::Warn, (msg))
#define NN_LOG_INFO(msg) ::nn::logging::log(::nn::logging::Level::Info, (msg))
#define NN_LOG_DEBUG(msg) (void) 0

#endif // NN_LOGGER_HPP
