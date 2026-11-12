/**
 * @file src/core/data_loaders/10.1117/schema/SqliteDataset.cpp
 * @brief Implementation of the immutable read-only dataset open (see the header for why).
 */

#include "data_loaders/10.1117/schema/SqliteDataset.hpp"

#include <sqlite3.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <system_error>

namespace nn::dataLoaders
{

namespace
{

// A file name inside a "file:" URI: '%', '?' and '#' would start an escape, the query or the
// fragment, so they are percent-encoded; everything else is taken literally by SQLite.
auto uriPath(const std::string& path) -> std::string
{
    std::string out;
    out.reserve(path.size());
    for (const char c : path)
    {
        switch (c)
        {
            case '%':
                out += "%25";
                break;
            case '?':
                out += "%3f";
                break;
            case '#':
                out += "%23";
                break;
            default:
                out += c;
        }
    }
    return out;
}

} // namespace

auto openSqliteDatasetReadOnly(const std::filesystem::path& db_path) -> sqlite3*
{
    namespace fs = std::filesystem;
    const std::string name = db_path.string();

    std::error_code ec;
    if (!fs::is_regular_file(db_path, ec))
        throw std::runtime_error("SQLite dataset: " + name + " is not a regular file" +
                                 (ec ? " (" + ec.message() + ")" : std::string()));

    const fs::path wal(name + "-wal");
    if (fs::exists(wal, ec))
    {
        const std::uintmax_t bytes = fs::file_size(wal, ec);
        if (ec)
            throw std::runtime_error(
                "SQLite dataset: cannot read the size of " + wal.string() + ": " + ec.message());
        if (bytes > 0)
            throw std::runtime_error("SQLite dataset: " + wal.string() + " holds " +
                                     std::to_string(bytes) + " bytes of transactions not yet in " +
                                     name +
                                     "; opened immutable, SQLite would not read them. Checkpoint "
                                     "it first: sqlite3 '" +
                                     name + "' 'PRAGMA wal_checkpoint(TRUNCATE);'");
    }

    const std::string uri = "file:" + uriPath(name) + "?immutable=1";
    sqlite3* db = nullptr;
    const int rc =
        sqlite3_open_v2(uri.c_str(), &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, nullptr);
    if (rc != SQLITE_OK)
    {
        const std::string why = (db != nullptr) ? sqlite3_errmsg(db) : sqlite3_errstr(rc);
        sqlite3_close(db);
        throw std::runtime_error("SQLite dataset: cannot open " + name + ": " + why);
    }
    return db;
}

auto sqliteStepRow(sqlite3* db, sqlite3_stmt* stmt, const char* what) -> bool
{
    const int rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) return true;
    if (rc == SQLITE_DONE) return false;
    throw std::runtime_error(std::string("SQLite dataset: ") + what + " failed: " +
                             sqlite3_errmsg(db) + " (code " + std::to_string(rc) + ")");
}

} // namespace nn::dataLoaders
