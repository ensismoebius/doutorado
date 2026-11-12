/**
 * @file include/data_loaders/10.1117/schema/SqliteDataset.hpp
 * @brief Read-only access to the 10.1117 dataset's SQLite file, safe for many hosts at once.
 */
#pragma once

#include <filesystem>

struct sqlite3;
struct sqlite3_stmt;

namespace nn::dataLoaders
{

/**
 * Opens the dataset's SQLite file for reading, as an immutable database.
 *
 * Why not a plain SQLITE_OPEN_READONLY. The dataset file is in WAL journal mode. A WAL reader
 * shares a memory-mapped index (the "-shm" file) with every other reader, which works only when
 * all of them run on one machine ("WAL does not work over a network filesystem",
 * sqlite.org/wal.html). On GridUnesp, array tasks on different nodes read ~/database.sqlite over
 * NFS at the same time: on 2026-11-12, 6 of 16 tasks failed within seconds of each other on
 * "Failed to prepare subject query". `immutable=1` (sqlite.org/uri.html) skips locking, the
 * -shm index and change detection, so any number of readers on any number of hosts can use the
 * file without coordinating.
 *
 * What immutable costs, and how it is guarded:
 *  - the file must not change while it is open. The thesis never writes it, and
 *    gridunesp_deploy.sh refuses to replace it while a thesis-run job is running;
 *  - committed transactions still waiting in a "-wal" file would not be seen. A non-empty
 *    "<db>-wal" therefore throws (checkpoint it first), instead of being read around.
 *
 * Throws std::runtime_error naming the file and SQLite's own message on any failure; never
 * returns null. The caller owns the handle (sqlite3_close).
 */
auto openSqliteDatasetReadOnly(const std::filesystem::path& db_path) -> sqlite3*;

/**
 * Steps a statement expected to produce one more row: true on a row, false when the result
 * set is exhausted, and an exception for anything else (SQLITE_BUSY, I/O error, corruption).
 * A plain `sqlite3_step(stmt) == SQLITE_ROW` loop reads an error as "no more rows" and returns
 * a silently shortened result.
 */
auto sqliteStepRow(sqlite3* db, sqlite3_stmt* stmt, const char* what) -> bool;

} // namespace nn::dataLoaders
