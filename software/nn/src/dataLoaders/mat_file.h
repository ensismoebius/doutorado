#pragma once

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace matio {

// Data types for MAT files
enum class DataType : uint32_t {
  MI_INT8 = 1,
  MI_UINT8 = 2,
  MI_INT16 = 3,
  MI_UINT16 = 4,
  MI_INT32 = 5,
  MI_UINT32 = 6,
  MI_SINGLE = 7,
  MI_DOUBLE = 9,
  MI_INT64 = 12,
  MI_UINT64 = 13,
  MI_MATRIX = 14,
  MI_COMPRESSED = 15,
  MI_UTF8 = 16,
  MI_UTF16 = 17,
  MI_UTF32 = 18
};

// Array types
enum class ArrayType : uint32_t {
  MX_CELL_CLASS = 1,
  MX_STRUCT_CLASS = 2,
  MX_OBJECT_CLASS = 3,
  MX_CHAR_CLASS = 4,
  MX_SPARSE_CLASS = 5,
  MX_DOUBLE_CLASS = 6,
  MX_SINGLE_CLASS = 7,
  MX_INT8_CLASS = 8,
  MX_UINT8_CLASS = 9,
  MX_INT16_CLASS = 10,
  MX_UINT16_CLASS = 11,
  MX_INT32_CLASS = 12,
  MX_UINT32_CLASS = 13,
  MX_INT64_CLASS = 14,
  MX_UINT64_CLASS = 15
};

// MAT file header (128 bytes)
struct MatHeader {
  char description[116];
  char subsystem_offset[8];
  char version[2];
  char endian[2];

  static constexpr std::string_view text = "MATLAB 5.0 MAT-file";

  MatHeader();
  [[nodiscard]] auto validate() const -> bool;
};

// Tag structure for data elements
struct DataTag {
  DataType data_type;
  uint32_t number_of_bytes;

  [[nodiscard]] auto is_small_data_element() const -> bool;
  [[nodiscard]] auto padding() const -> uint32_t;
};

// Array flags
struct ArrayFlags {
  ArrayType array_type;
  uint32_t flags;

  [[nodiscard]] auto is_complex() const -> bool;
  [[nodiscard]] auto is_global() const -> bool;
  [[nodiscard]] auto is_logical() const -> bool;
};

// Dimension array
using Dimensions = std::vector<int32_t>;

// Variable data type
using MatData =
    std::variant<std::vector<double>, std::vector<float>, std::vector<int8_t>,
                 std::vector<uint8_t>, std::vector<int16_t>,
                 std::vector<uint16_t>, std::vector<int32_t>,
                 std::vector<uint32_t>, std::vector<int64_t>,
                 std::vector<uint64_t>, std::string>;

// MAT variable
struct MatVar {
  std::string name;
  Dimensions dimensions;
  ArrayFlags flags;
  MatData data;

  template <typename T>
  [[nodiscard]] auto holds_type() const -> bool {
    return std::holds_alternative<std::vector<T>>(data);
  }

  template <typename T>
  const std::vector<T>& get_vector() const {
    return std::get<std::vector<T>>(data);
  }

  [[nodiscard]] auto num_elements() const -> size_t;
  [[nodiscard]] auto type_name() const -> std::string;
};

// Main MAT file class
class MatFile {
 private:
  std::string filename_;
  std::fstream file_;
  bool is_little_endian_;

  // Byte order swapping
  template <typename T>
  void swap_bytes(T& value) const
    requires std::is_arithmetic_v<T>;

  // Reading primitives
  template <typename T>
  auto read_primitive() -> T;

  // Writing primitives
  template <typename T>
  void write_primitive(T value);

  // Data type handling
  auto get_data_type_for(const MatData& data) const -> DataType;
  auto get_array_type_for(const MatData& data) const -> ArrayType;

  // Reading components
  auto read_tag() -> DataTag;
  auto read_string(uint32_t length) -> std::string;
  auto read_variable_name() -> std::string;
  auto read_dimensions_array() -> Dimensions;
  auto read_array_flags() -> ArrayFlags;
  auto read_numeric_data(DataType data_type, size_t num_elements) -> MatData;

  // Writing components
  void write_tag(DataType data_type, uint32_t num_bytes);
  void write_variable_name(const std::string& name);
  void write_dimensions_array(const Dimensions& dims);
  void write_array_flags(const ArrayFlags& flags);
  void write_numeric_data(const MatData& data);

 public:
  MatFile() = default;
  ~MatFile();

  // File operations
  auto open(const std::string& filename) -> bool;
  auto create(const std::string& filename) -> bool;
  void close();
  auto is_open() const -> bool;

  // Reading
  auto read_all_variables() -> std::unordered_map<std::string, MatVar>;
  auto read_variable(const std::string& name) -> std::optional<MatVar>;

  // Writing
  auto write_variable(const std::string& name, const MatData& data,
                      const Dimensions& dims = {1, 1},
                      ArrayFlags flags = {
                          .array_type = ArrayType::MX_DOUBLE_CLASS,
                          .flags = 0}) -> bool;

  // Convenience methods for common types
  auto write_double_matrix(const std::string& name,
                           const std::vector<double>& data,
                           const Dimensions& dims = {1, 1}) -> bool;
  auto write_single_matrix(const std::string& name,
                           const std::vector<float>& data,
                           const Dimensions& dims = {1, 1}) -> bool;
  auto write_int32_matrix(const std::string& name,
                          const std::vector<int32_t>& data,
                          const Dimensions& dims = {1, 1}) -> bool;
  auto write_string(const std::string& name, const std::string& str) -> bool;

  // Template convenience method
  template <typename T>
  auto write_matrix(const std::string& name, const std::vector<T>& data,
                    const Dimensions& dims = {1, 1}) -> bool {
    MatData mat_data = data;
    return write_variable(name, mat_data, dims);
  }
};

// Utility functions
namespace utils {
auto is_little_endian() -> bool;
auto array_type_to_string(ArrayType type) -> std::string;
auto data_type_to_string(DataType type) -> std::string;
}  // namespace utils

}  // namespace matio