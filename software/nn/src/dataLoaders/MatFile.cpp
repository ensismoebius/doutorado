#include "MatFile.h"

#include <zlib.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace matio {

// MatHeader implementation
MatHeader::MatHeader() {
  description.fill('\0');
  subsystem_offset.fill('\0');
  std::copy_n(text.data(), text.length(), description.begin());
  version[0] = 0x01;
  version[1] = 0x00;
  endian[0] = 'I';
  endian[1] = 'M';
}

auto MatHeader::validate() const -> bool {
  return std::string_view(description.data(), text.length()) == text;
}

// DataTag implementation
auto DataTag::is_small_data_element() const -> bool {
  return number_of_bytes <= 4;
}

auto DataTag::padding() const -> uint32_t {
  return (8 - (number_of_bytes % 8)) % 8;
}

// ArrayFlags implementation
auto ArrayFlags::is_complex() const -> bool { return (flags & 0x08) != 0U; }

auto ArrayFlags::is_global() const -> bool { return (flags & 0x04) != 0U; }

auto ArrayFlags::is_logical() const -> bool { return (flags & 0x02) != 0U; }

// MatVar implementation
auto MatVar::num_elements() const -> size_t {
  size_t count = 1;
  for (auto dim : dimensions) {
    count *= dim;
  }
  return count;
}

auto MatVar::type_name() const -> std::string {
  return std::visit(
      [](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::vector<double>>) {
          return "double";
        }

        if constexpr (std::is_same_v<T, std::vector<float>>) {
          return "single";
        }

        if constexpr (std::is_same_v<T, std::vector<int8_t>>) {
          return "int8";
        }

        if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
          return "uint8";
        }

        if constexpr (std::is_same_v<T, std::vector<int16_t>>) {
          return "int16";
        }

        if constexpr (std::is_same_v<T, std::vector<uint16_t>>) {
          return "uint16";
        }

        if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
          return "int32";
        }

        if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
          return "uint32";
        }

        if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          return "int64";
        }

        if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
          return "uint64";
        }

        if constexpr (std::is_same_v<T, std::string>) {
          return "char";
        }

        return "unknown";
      },
      data);
}

// MatFile implementation
template <typename T>
void MatFile::swap_bytes(T& value) const
  requires std::is_arithmetic_v<T>
{
  if (is_little_endian_ == utils::is_little_endian()) {
    return;
  }

  auto* bytes = reinterpret_cast<uint8_t*>(&value);
  for (size_t i = 0; i < sizeof(T) / 2; ++i) {
    std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
  }
}

// Ensure default initialization of endianness
MatFile::MatFile() : is_little_endian_(utils::is_little_endian()) {}

template <typename T>
T MatFile::read_primitive() {
  T value;
  file_.read(reinterpret_cast<char*>(&value), sizeof(T));
  swap_bytes(value);
  return value;
}

template <typename T>
void MatFile::write_primitive(T value) {
  swap_bytes(value);
  file_.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

auto MatFile::get_data_type_for(const MatData& data) -> DataType {
  return std::visit(
      [](auto&& arg) -> DataType {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::vector<double>>) {
          return DataType::MI_DOUBLE;
        }

        if constexpr (std::is_same_v<T, std::vector<float>>) {
          return DataType::MI_SINGLE;
        }

        if constexpr (std::is_same_v<T, std::vector<int8_t>>) {
          return DataType::MI_INT8;
        }

        if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
          return DataType::MI_UINT8;
        }

        if constexpr (std::is_same_v<T, std::vector<int16_t>>) {
          return DataType::MI_INT16;
        }

        if constexpr (std::is_same_v<T, std::vector<uint16_t>>) {
          return DataType::MI_UINT16;
        }

        if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
          return DataType::MI_INT32;
        }

        if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
          return DataType::MI_UINT32;
        }

        if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          return DataType::MI_INT64;
        }

        if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
          return DataType::MI_UINT64;
        }

        if constexpr (std::is_same_v<T, std::string>) {
          return DataType::MI_UTF8;
        }

        throw std::runtime_error("Unsupported data type");
      },
      data);
}

auto MatFile::get_array_type_for(const MatData& data) -> ArrayType {
  return std::visit(
      [](auto&& arg) -> ArrayType {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::vector<double>>) {
          return ArrayType::MX_DOUBLE_CLASS;
        }

        if constexpr (std::is_same_v<T, std::vector<float>>) {
          return ArrayType::MX_SINGLE_CLASS;
        }

        if constexpr (std::is_same_v<T, std::vector<int8_t>>) {
          return ArrayType::MX_INT8_CLASS;
        }

        if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
          return ArrayType::MX_UINT8_CLASS;
        }

        if constexpr (std::is_same_v<T, std::vector<int16_t>>) {
          return ArrayType::MX_INT16_CLASS;
        }

        if constexpr (std::is_same_v<T, std::vector<uint16_t>>) {
          return ArrayType::MX_UINT16_CLASS;
        }

        if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
          return ArrayType::MX_INT32_CLASS;
        }

        if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
          return ArrayType::MX_UINT32_CLASS;
        }

        if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          return ArrayType::MX_INT64_CLASS;
        }

        if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
          return ArrayType::MX_UINT64_CLASS;
        }

        if constexpr (std::is_same_v<T, std::string>) {
          return ArrayType::MX_CHAR_CLASS;
        }

        throw std::runtime_error("Unsupported array type");
      },
      data);
}

auto MatFile::read_tag() -> DataTag {
  DataTag tag;
  // Read first 4 bytes. This can be either a full tag (data_type uint32 +
  // num_bytes uint32) where we need to read another 4 bytes, or a small data
  // element where the single 4-byte word packs the data type (lower 16 bits)
  // and the number of bytes (upper 16 bits). Detect and handle both forms.
  uint32_t first = read_primitive<uint32_t>();

  uint16_t maybe_data_type = static_cast<uint16_t>(first & 0xFFFFU);
  uint16_t maybe_num_bytes = static_cast<uint16_t>((first >> 16) & 0xFFFFU);

  if (maybe_num_bytes != 0) {
    // Small data element format
    tag.data_type = static_cast<DataType>(maybe_data_type);
    tag.number_of_bytes = maybe_num_bytes;
  } else {
    // Full tag: first 4 bytes are data_type, next 4 bytes are number_of_bytes
    tag.data_type = static_cast<DataType>(first);
    tag.number_of_bytes = read_primitive<uint32_t>();
  }

  return tag;
}

auto MatFile::read_string(uint32_t length) -> std::string {
  std::string str(length, '\0');
  file_.read(str.data(), length);
  // Remove padding
  uint32_t padding = (8 - (length % 8)) % 8;
  file_.seekg(padding, std::ios::cur);
  return str;
}

auto MatFile::read_variable_name() -> std::string {
  auto tag = read_tag();
  if (tag.data_type != DataType::MI_INT8) {
    throw std::runtime_error("Invalid variable name data type");
  }
  return read_string(tag.number_of_bytes);
}

auto MatFile::read_dimensions_array() -> Dimensions {
  auto tag = read_tag();
  if (tag.data_type != DataType::MI_INT32) {
    throw std::runtime_error("Invalid dimensions data type");
  }

  size_t num_dims = tag.number_of_bytes / sizeof(int32_t);
  Dimensions dims(num_dims);
  for (size_t i = 0; i < num_dims; ++i) {
    dims[i] = read_primitive<int32_t>();
  }

  // Skip padding
  file_.seekg(tag.padding(), std::ios::cur);
  return dims;
}

auto MatFile::read_array_flags() -> ArrayFlags {
  auto tag = read_tag();
  if (tag.data_type != DataType::MI_UINT32) {
    throw std::runtime_error("Invalid array flags data type");
  }

  ArrayFlags flags;
  flags.array_type = static_cast<ArrayType>(read_primitive<uint32_t>());
  flags.flags = read_primitive<uint32_t>();

  // Skip padding
  file_.seekg(tag.padding(), std::ios::cur);

  return flags;
}

auto MatFile::read_numeric_data(DataType data_type, uint32_t num_bytes)
    -> MatData {
  if (data_type == DataType::MI_UTF8) {
    return read_string(num_bytes);
  }

  MatData result;
  size_t num_elements;

  switch (data_type) {
    case DataType::MI_DOUBLE: {
      num_elements = num_bytes / sizeof(double);
      std::vector<double> data(num_elements);
      for (size_t i = 0; i < num_elements; ++i) {
        data[i] = read_primitive<double>();
      }
      result = data;
      break;
    }
    case DataType::MI_SINGLE: {
      num_elements = num_bytes / sizeof(float);
      std::vector<float> data(num_elements);
      for (size_t i = 0; i < num_elements; ++i) {
        data[i] = read_primitive<float>();
      }
      result = data;
      break;
    }
    case DataType::MI_INT8: {
      num_elements = num_bytes / sizeof(int8_t);
      std::vector<int8_t> data(num_elements);
      file_.read(reinterpret_cast<char*>(data.data()),
                 static_cast<long>(num_elements));
      result = data;
      break;
    }
    case DataType::MI_UINT8: {
      num_elements = num_bytes / sizeof(uint8_t);
      std::vector<uint8_t> data(num_elements);
      file_.read(reinterpret_cast<char*>(data.data()),
                 static_cast<long>(num_elements));
      result = data;
      break;
    }
    case DataType::MI_INT16: {
      num_elements = num_bytes / sizeof(int16_t);
      std::vector<int16_t> data(num_elements);
      for (size_t i = 0; i < num_elements; ++i) {
        data[i] = read_primitive<int16_t>();
      }
      result = data;
      break;
    }
    case DataType::MI_UINT16: {
      num_elements = num_bytes / sizeof(uint16_t);
      std::vector<uint16_t> data(num_elements);
      for (size_t i = 0; i < num_elements; ++i) {
        data[i] = read_primitive<uint16_t>();
      }
      result = data;
      break;
    }
    case DataType::MI_INT32: {
      num_elements = num_bytes / sizeof(int32_t);
      std::vector<int32_t> data(num_elements);
      for (size_t i = 0; i < num_elements; ++i) {
        data[i] = read_primitive<int32_t>();
      }
      result = data;
      break;
    }
    case DataType::MI_UINT32: {
      num_elements = num_bytes / sizeof(uint32_t);
      std::vector<uint32_t> data(num_elements);
      for (size_t i = 0; i < num_elements; ++i) {
        data[i] = read_primitive<uint32_t>();
      }
      result = data;
      break;
    }
    default:
      throw std::runtime_error("Unsupported data type");
  }

  // Skip padding
  uint32_t padding = (8 - (num_bytes % 8)) % 8;
  if (padding > 0) {
    file_.seekg(padding, std::ios::cur);
  }

  return result;
}

void MatFile::write_tag(DataType data_type, uint32_t num_bytes) {
  write_primitive<uint32_t>(static_cast<uint32_t>(data_type));
  write_primitive<uint32_t>(num_bytes);
}

void MatFile::write_variable_name(const std::string& name) {
  write_tag(DataType::MI_INT8, static_cast<uint32_t>(name.length()));
  file_.write(name.data(), static_cast<long>(name.length()));

  // Add padding
  uint32_t padding = (8 - (name.length() % 8)) % 8;
  if (padding > 0) {
    std::array<char, 8> padding_bytes{};
    file_.write(padding_bytes.data(), padding);
  }
}

void MatFile::write_dimensions_array(const Dimensions& dims) {
  write_tag(DataType::MI_INT32,
            static_cast<uint32_t>(dims.size() * sizeof(int32_t)));
  for (auto dim : dims) {
    write_primitive<int32_t>(dim);
  }

  // Add padding
  auto total_bytes = static_cast<uint32_t>(dims.size() * sizeof(int32_t));
  uint32_t padding = (8 - (total_bytes % 8)) % 8;
  if (padding > 0) {
    std::array<char, 8> padding_bytes{};
    file_.write(padding_bytes.data(), padding);
  }
}

void MatFile::write_array_flags(const ArrayFlags& flags) {
  write_tag(DataType::MI_UINT32, 8);  // Always 8 bytes for flags
  write_primitive<uint32_t>(static_cast<uint32_t>(flags.array_type));
  write_primitive<uint32_t>(flags.flags);
}

void MatFile::write_numeric_data(const MatData& data) {
  std::visit(
      [this, &data](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, std::string>) {
          write_tag(DataType::MI_UTF8, static_cast<uint32_t>(arg.length()));
          file_.write(arg.data(), arg.length());

          uint32_t padding = (8 - (arg.length() % 8)) % 8;
          if (padding > 0) {
            std::array<char, 8> padding_bytes{};
            file_.write(padding_bytes.data(), padding);
          }
        } else {
          DataType data_type = get_data_type_for(data);
          auto num_bytes = static_cast<uint32_t>(
              arg.size() * sizeof(typename T::value_type));
          write_tag(data_type, num_bytes);

          for (const auto& value : arg) {
            write_primitive(value);
          }

          uint32_t padding = (8 - (num_bytes % 8)) % 8;
          if (padding > 0) {
            std::array<char, 8> padding_bytes{};
            file_.write(padding_bytes.data(), padding);
          }
        }
      },
      data);
}

MatFile::~MatFile() { close(); }

auto MatFile::open(const std::string& filename) -> bool {
  close();
  filename_ = filename;
  file_.open(filename, std::ios::binary | std::ios::in);

  if (!file_.is_open()) {
    return false;
  }

  // Read and validate header
  MatHeader header;
  file_.read(reinterpret_cast<char*>(&header), sizeof(MatHeader));

  (void)0;

  if (!header.validate()) {
    close();
    return false;
  }

  is_little_endian_ = (header.endian[0] == 'I' && header.endian[1] == 'M');
  return true;
}

bool MatFile::create(const std::string& filename) {
  close();
  filename_ = filename;
  file_.open(filename, std::ios::binary | std::ios::out | std::ios::trunc);

  if (!file_.is_open()) {
    return false;
  }

  // Write header
  MatHeader header;
  if (utils::is_little_endian()) {
    header.endian[0] = 'I';
    header.endian[1] = 'M';
  } else {
    header.endian[0] = 'M';
    header.endian[1] = 'I';
  }

  file_.write(reinterpret_cast<const char*>(&header), sizeof(MatHeader));
  return true;
}

void MatFile::close() {
  if (file_.is_open()) {
    file_.close();
  }
}

bool MatFile::is_open() const { return file_.is_open(); }

auto MatFile::read_all_variables() -> std::unordered_map<std::string, MatVar> {
  if (!file_.is_open()) {
    throw std::runtime_error("File is not open");
  }

  std::unordered_map<std::string, MatVar> variables;

  // No runtime debug traces in production build

  try {
    while (file_.peek() != EOF) {
      // Read variable
      auto tag = read_tag();

      if (tag.data_type == DataType::MI_COMPRESSED) {
        // Read compressed payload
        std::vector<uint8_t> comp(tag.number_of_bytes);
        file_.read(reinterpret_cast<char*>(comp.data()), tag.number_of_bytes);

        // Decompress using zlib inflate
        std::vector<uint8_t> out;
        z_stream strm{};
        strm.next_in = comp.data();
        strm.avail_in = static_cast<uInt>(comp.size());
        if (inflateInit(&strm) != Z_OK) {
          throw std::runtime_error("Failed to initialize zlib inflater");
        }

        const size_t CHUNK = 16384;
        std::vector<uint8_t> buffer(CHUNK);
        int ret = Z_OK;
        while (ret != Z_STREAM_END) {
          strm.next_out = buffer.data();
          strm.avail_out = static_cast<uInt>(buffer.size());
          ret = inflate(&strm, Z_NO_FLUSH);
          if (ret != Z_OK && ret != Z_STREAM_END && ret != Z_BUF_ERROR) {
            inflateEnd(&strm);
            throw std::runtime_error("zlib inflate failed");
          }
          size_t produced = buffer.size() - strm.avail_out;
          if (produced > 0) {
            out.insert(out.end(), buffer.data(), buffer.data() + produced);
          }
          if (ret == Z_BUF_ERROR && strm.avail_in == 0) break;
        }
        inflateEnd(&strm);

        // Parse decompressed buffer for embedded variables
        auto parse_from_buffer = [&](const std::vector<uint8_t>& buf) {
          size_t pos = 0;
          auto read_u32 = [&](uint32_t& dest) {
            if (pos + 4 > buf.size())
              throw std::runtime_error("Buffer underflow");
            std::memcpy(&dest, buf.data() + pos, 4);
            pos += 4;
            swap_bytes(dest);
          };

          auto read_tag_buf = [&]() -> DataTag {
            DataTag t;
            uint32_t first;
            read_u32(first);
            uint16_t maybe_data_type = static_cast<uint16_t>(first & 0xFFFFU);
            uint16_t maybe_num_bytes =
                static_cast<uint16_t>((first >> 16) & 0xFFFFU);
            if (maybe_num_bytes != 0) {
              t.data_type = static_cast<DataType>(maybe_data_type);
              t.number_of_bytes = maybe_num_bytes;
            } else {
              t.data_type = static_cast<DataType>(first);
              uint32_t nb;
              read_u32(nb);
              t.number_of_bytes = nb;
            }
            return t;
          };

          auto read_string_buf = [&](uint32_t length) {
            if (pos + length > buf.size())
              throw std::runtime_error("Buffer underflow");
            std::string s(reinterpret_cast<const char*>(buf.data() + pos),
                          length);
            pos += length;
            uint32_t padding = (8 - (length % 8)) % 8;
            pos += padding;
            return s;
          };

          auto read_dimensions_buf = [&]() {
            auto tag2 = read_tag_buf();
            if (tag2.data_type != DataType::MI_INT32)
              throw std::runtime_error("Invalid dims type");
            size_t num_dims = tag2.number_of_bytes / sizeof(int32_t);
            Dimensions dims(num_dims);
            for (size_t i = 0; i < num_dims; ++i) {
              int32_t v;
              if (pos + 4 > buf.size())
                throw std::runtime_error("Buffer underflow");
              std::memcpy(&v, buf.data() + pos, 4);
              pos += 4;
              swap_bytes(v);
              dims[i] = v;
            }
            pos += tag2.padding();
            return dims;
          };

          auto read_array_flags_buf = [&]() {
            auto tag2 = read_tag_buf();
            if (tag2.data_type != DataType::MI_UINT32)
              throw std::runtime_error("Invalid flags type");
            ArrayFlags flags;
            uint32_t atype;
            read_u32(atype);
            flags.array_type = static_cast<ArrayType>(atype);
            uint32_t f;
            read_u32(f);
            flags.flags = f;
            pos += tag2.padding();
            return flags;
          };

          auto read_numeric_buf = [&](DataType dtype,
                                      uint32_t num_bytes) -> MatData {
            if (dtype == DataType::MI_UTF8) {
              return read_string_buf(num_bytes);
            }
            MatData result;
            switch (dtype) {
              case DataType::MI_DOUBLE: {
                size_t n = num_bytes / sizeof(double);
                std::vector<double> v(n);
                for (size_t i = 0; i < n; ++i) {
                  double val;
                  if (pos + sizeof(double) > buf.size())
                    throw std::runtime_error("Buffer underflow");
                  std::memcpy(&val, buf.data() + pos, sizeof(double));
                  pos += sizeof(double);
                  swap_bytes(val);
                  v[i] = val;
                }
                result = std::move(v);
                break;
              }
              case DataType::MI_SINGLE: {
                size_t n = num_bytes / sizeof(float);
                std::vector<float> v(n);
                for (size_t i = 0; i < n; ++i) {
                  float val;
                  if (pos + sizeof(float) > buf.size())
                    throw std::runtime_error("Buffer underflow");
                  std::memcpy(&val, buf.data() + pos, sizeof(float));
                  pos += sizeof(float);
                  swap_bytes(val);
                  v[i] = val;
                }
                result = std::move(v);
                break;
              }
              case DataType::MI_INT8: {
                size_t n = num_bytes / sizeof(int8_t);
                std::vector<int8_t> v(n);
                if (pos + n > buf.size())
                  throw std::runtime_error("Buffer underflow");
                std::memcpy(v.data(), buf.data() + pos, n);
                pos += n;
                result = std::move(v);
                break;
              }
              case DataType::MI_UINT8: {
                size_t n = num_bytes / sizeof(uint8_t);
                std::vector<uint8_t> v(n);
                if (pos + n > buf.size())
                  throw std::runtime_error("Buffer underflow");
                std::memcpy(v.data(), buf.data() + pos, n);
                pos += n;
                result = std::move(v);
                break;
              }
              case DataType::MI_INT16: {
                size_t n = num_bytes / sizeof(int16_t);
                std::vector<int16_t> v(n);
                for (size_t i = 0; i < n; ++i) {
                  int16_t val;
                  if (pos + 2 > buf.size())
                    throw std::runtime_error("Buffer underflow");
                  std::memcpy(&val, buf.data() + pos, 2);
                  pos += 2;
                  swap_bytes(val);
                  v[i] = val;
                }
                result = std::move(v);
                break;
              }
              case DataType::MI_UINT16: {
                size_t n = num_bytes / sizeof(uint16_t);
                std::vector<uint16_t> v(n);
                for (size_t i = 0; i < n; ++i) {
                  uint16_t val;
                  if (pos + 2 > buf.size())
                    throw std::runtime_error("Buffer underflow");
                  std::memcpy(&val, buf.data() + pos, 2);
                  pos += 2;
                  swap_bytes(val);
                  v[i] = val;
                }
                result = std::move(v);
                break;
              }
              case DataType::MI_INT32: {
                size_t n = num_bytes / sizeof(int32_t);
                std::vector<int32_t> v(n);
                for (size_t i = 0; i < n; ++i) {
                  int32_t val;
                  if (pos + 4 > buf.size())
                    throw std::runtime_error("Buffer underflow");
                  std::memcpy(&val, buf.data() + pos, 4);
                  pos += 4;
                  swap_bytes(val);
                  v[i] = val;
                }
                result = std::move(v);
                break;
              }
              case DataType::MI_UINT32: {
                size_t n = num_bytes / sizeof(uint32_t);
                std::vector<uint32_t> v(n);
                for (size_t i = 0; i < n; ++i) {
                  uint32_t val;
                  if (pos + 4 > buf.size())
                    throw std::runtime_error("Buffer underflow");
                  std::memcpy(&val, buf.data() + pos, 4);
                  pos += 4;
                  swap_bytes(val);
                  v[i] = val;
                }
                result = std::move(v);
                break;
              }
              default:
                throw std::runtime_error(
                    "Unsupported data type in compressed payload");
            }

            uint32_t padding = (8 - (num_bytes % 8)) % 8;
            pos += padding;
            return result;
          };

          while (pos < buf.size()) {
            // Read a tag in the decompressed buffer
            DataTag t = read_tag_buf();
            if (t.data_type != DataType::MI_MATRIX) {
              // Skip unknown element
              pos += t.number_of_bytes + t.padding();
              continue;
            }

            MatVar var;
            var.flags = read_array_flags_buf();
            var.dimensions = read_dimensions_buf();
            var.name = [&]() {
              auto name_tag = read_tag_buf();
              if (name_tag.data_type != DataType::MI_INT8)
                throw std::runtime_error(
                    "Invalid name tag in compressed payload");
              return read_string_buf(name_tag.number_of_bytes);
            }();

            auto data_tag = read_tag_buf();
            var.data =
                read_numeric_buf(data_tag.data_type, data_tag.number_of_bytes);
            variables[var.name] = std::move(var);
          }
        };

        parse_from_buffer(out);
        continue;  // Continue reading next top-level element
      }

      if (tag.data_type != DataType::MI_MATRIX) {
        break;  // Not a variable, probably end of file
      }

      MatVar var;

      // Read array flags
      var.flags = read_array_flags();

      // Read dimensions
      var.dimensions = read_dimensions_array();

      // Read variable name
      var.name = read_variable_name();

      // Read data
      auto data_tag = read_tag();
      var.data =
          read_numeric_data(data_tag.data_type, data_tag.number_of_bytes);

      variables[var.name] = std::move(var);
    }
  } catch (const std::exception& e) {
    // End of file or error
    throw std::runtime_error(std::string("Error reading variables: ") +
                             e.what());
  }

  return variables;
}

auto MatFile::read_variable(const std::string& name) -> std::optional<MatVar> {
  if (!file_.is_open()) {
    return std::nullopt;
  }

  auto all_vars = read_all_variables();
  if (auto it = all_vars.find(name); it != all_vars.end()) {
    return it->second;
  }
  return std::nullopt;
}

auto MatFile::write_variable(const std::string& name, const MatData& data,
                             const Dimensions& dims, ArrayFlags flags) -> bool {
  if (!file_.is_open()) {
    return false;
  }

  try {
    // Write matrix tag
    write_tag(DataType::MI_MATRIX, 0);  // Size will be determined by content

    // Write array flags (use provided flags or auto-detect)
    if (flags.array_type == ArrayType::MX_DOUBLE_CLASS) {
      flags.array_type = get_array_type_for(data);
    }
    write_array_flags(flags);

    // Write dimensions
    write_dimensions_array(dims);

    // Write variable name
    write_variable_name(name);

    // Write data
    write_numeric_data(data);

    return true;
  } catch (...) {
    return false;
  }
}

auto MatFile::write_double_matrix(const std::string& name,
                                  const std::vector<double>& data,
                                  const Dimensions& dims) -> bool {
  return write_variable(name, data, dims, {ArrayType::MX_DOUBLE_CLASS, 0});
}

auto MatFile::write_single_matrix(const std::string& name,
                                  const std::vector<float>& data,
                                  const Dimensions& dims) -> bool {
  return write_variable(name, data, dims, {ArrayType::MX_SINGLE_CLASS, 0});
}

auto MatFile::write_int32_matrix(const std::string& name,
                                 const std::vector<int32_t>& data,
                                 const Dimensions& dims) -> bool {
  return write_variable(name, data, dims, {ArrayType::MX_INT32_CLASS, 0});
}

bool MatFile::write_string(const std::string& name, const std::string& str) {
  return write_variable(name,
                        str,
                        {1, static_cast<int32_t>(str.length())},
                        {ArrayType::MX_CHAR_CLASS, 0});
}

// Utility functions
auto utils::is_little_endian() -> bool {
  uint32_t test = 1;
  return *reinterpret_cast<uint8_t*>(&test) == 1;
}

auto utils::array_type_to_string(ArrayType type) -> std::string {
  switch (type) {
    case ArrayType::MX_CELL_CLASS:
      return "cell";
    case ArrayType::MX_STRUCT_CLASS:
      return "struct";
    case ArrayType::MX_OBJECT_CLASS:
      return "object";
    case ArrayType::MX_CHAR_CLASS:
      return "char";
    case ArrayType::MX_SPARSE_CLASS:
      return "sparse";
    case ArrayType::MX_DOUBLE_CLASS:
      return "double";
    case ArrayType::MX_SINGLE_CLASS:
      return "single";
    case ArrayType::MX_INT8_CLASS:
      return "int8";
    case ArrayType::MX_UINT8_CLASS:
      return "uint8";
    case ArrayType::MX_INT16_CLASS:
      return "int16";
    case ArrayType::MX_UINT16_CLASS:
      return "uint16";
    case ArrayType::MX_INT32_CLASS:
      return "int32";
    case ArrayType::MX_UINT32_CLASS:
      return "uint32";
    case ArrayType::MX_INT64_CLASS:
      return "int64";
    case ArrayType::MX_UINT64_CLASS:
      return "uint64";
    default:
      return "unknown";
  }
}

auto utils::data_type_to_string(DataType type) -> std::string {
  switch (type) {
    case DataType::MI_INT8:
      return "int8";
    case DataType::MI_UINT8:
      return "uint8";
    case DataType::MI_INT16:
      return "int16";
    case DataType::MI_UINT16:
      return "uint16";
    case DataType::MI_INT32:
      return "int32";
    case DataType::MI_UINT32:
      return "uint32";
    case DataType::MI_SINGLE:
      return "single";
    case DataType::MI_DOUBLE:
      return "double";
    case DataType::MI_INT64:
      return "int64";
    case DataType::MI_UINT64:
      return "uint64";
    case DataType::MI_UTF8:
      return "utf8";
    case DataType::MI_UTF16:
      return "utf16";
    case DataType::MI_UTF32:
      return "utf32";
    default:
      return "unknown";
  }
}

}  // namespace matio
