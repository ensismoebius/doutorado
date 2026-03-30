#ifndef NN_LOGGER_HPP
#define NN_LOGGER_HPP

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

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
        output_stream_ << oss.str();
        output_stream_.flush();
    }

    void set_output_stream(std::ostream& os)
    {
        std::lock_guard<std::mutex> g(mu_);
        output_stream_.rdbuf(os.rdbuf());
    }

   private:
    Logger() : level_(static_cast<int>(Level::Info)), output_stream_(std::cerr.rdbuf()) {}
    std::atomic<int> level_;
    std::mutex mu_;
    std::ostream output_stream_;
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
#define NN_LOG_DEBUG(msg) ::nn::logging::log(::nn::logging::Level::Debug, (msg))

#endif // NN_LOGGER_HPP
