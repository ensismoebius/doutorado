#pragma once

#include <iostream>
#include <memory>
#include <streambuf>
#include <string>

#include "nn/logging/Logger.hpp"

namespace nn::logging
{

// A small streambuf that forwards full lines to the Logger.
class StreamToLoggerBuf : public std::streambuf
{
   public:
    StreamToLoggerBuf(Level lev) : level_(lev) {}

   protected:
    int_type overflow(int_type ch) override
    {
        if (ch == traits_type::eof()) return ch;
        char c = static_cast<char>(ch);
        if (c == '\n')
        {
            Logger::instance().log(level_, buffer_);
            buffer_.clear();
        }
        else
        {
            buffer_.push_back(c);
        }
        return ch;
    }

    std::streamsize xsputn(const char* s, std::streamsize count) override
    {
        for (std::streamsize i = 0; i < count; ++i)
        {
            overflow(static_cast<unsigned char>(s[i]));
        }
        return count;
    }

   private:
    Level level_;
    std::string buffer_;
};

// RAII helper to redirect std::cout / std::cerr into the Logger.
class StreamRedirector
{
   public:
    StreamRedirector(bool redirect_stdout = true, bool redirect_stderr = true)
    {
        if (redirect_stdout)
        {
            out_buf_ = std::make_unique<StreamToLoggerBuf>(Level::Info);
            old_out_ = std::cout.rdbuf(out_buf_.get());
            // Tell Logger what the original console rdbuf was so callers can bypass
            Logger::instance().set_console_rdbuf(old_out_);
        }
        if (redirect_stderr)
        {
            err_buf_ = std::make_unique<StreamToLoggerBuf>(Level::Error);
            old_err_ = std::cerr.rdbuf(err_buf_.get());
            Logger::instance().set_console_rdbuf(old_err_);
        }
        // While redirecting, suppress immediate console writes from Logger so
        // we only show logs when the progress printer drains them.
        Logger::instance().set_suppress_console_output(true);
    }

    ~StreamRedirector()
    {
        if (old_out_) std::cout.rdbuf(old_out_);
        if (old_err_) std::cerr.rdbuf(old_err_);
        // restore Logger behavior before restoring streambufs
        Logger::instance().set_suppress_console_output(false);
    }

    // non-copyable
    StreamRedirector(const StreamRedirector&) = delete;
    StreamRedirector& operator=(const StreamRedirector&) = delete;

   private:
    std::streambuf* old_out_ = nullptr;
    std::streambuf* old_err_ = nullptr;
    std::unique_ptr<StreamToLoggerBuf> out_buf_;
    std::unique_ptr<StreamToLoggerBuf> err_buf_;
};

} // namespace nn::logging
