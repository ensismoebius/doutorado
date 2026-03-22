#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nn::dataLoaders
{
class ShardReader
{
   public:
	struct ShardEntry
	{
		std::string file;
		std::size_t start = 0;
		std::size_t count = 0;
	};

	struct ShardIndex
	{
		std::vector<ShardEntry> audio;
		std::vector<ShardEntry> eeg;
	};

	struct Loc
	{
		std::string shard_file;
		std::size_t offset = 0;
	};

	explicit ShardReader(const std::string& index_path);
	~ShardReader();

	[[nodiscard]] auto locate_audio_row(std::size_t row) const -> std::optional<Loc>;
	[[nodiscard]] auto locate_eeg_row(std::size_t row) const -> std::optional<Loc>;

	auto read_row_from_shard(
		const std::string& shard_fullpath, const std::string& var_name, std::size_t offset)
		-> std::optional<std::vector<double>>;

	void preopen_shard(const std::string& shard_fullpath);
	void preopen_index_shards(const std::string& index_parent);

   private:
	static auto read_file_to_string(const std::string& path) -> std::string;
	void load_index();

	std::string index_path_;
	ShardIndex index_;

	static std::unordered_map<std::string, void*> open_shards_;
};
} // namespace nn::dataLoaders
