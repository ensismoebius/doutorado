# nn

## C++ STL APIs Used In This Project

This list is generated from direct std:: symbol usage across src/ and include/.

### Containers and Views

- std::array
- std::deque
- std::map
- std::optional
- std::pair
- std::set
- std::span
- std::string
- std::string_view
- std::tuple
- std::unordered_map
- std::unordered_set
- std::vector
- std::views

### Smart Pointers and Memory

- std::addressof
- std::dynamic_pointer_cast
- std::make_shared
- std::make_unique
- std::shared_ptr
- std::unique_ptr
- std::malloc
- std::free
- std::memcpy

### Algorithms and Numeric

- std::accumulate
- std::all_of
- std::any_of
- std::back_inserter
- std::clamp
- std::copy
- std::copy_if
- std::copy_n
- std::count
- std::distance
- std::equal
- std::fill
- std::find
- std::find_if
- std::for_each
- std::generate
- std::inner_product
- std::iota
- std::max
- std::max_element
- std::min
- std::min_element
- std::minmax_element
- std::move
- std::ranges
- std::reduce
- std::remove
- std::reverse
- std::shuffle
- std::sort
- std::swap
- std::transform
- std::transform_reduce
- std::unique
- std::upper_bound

### Math Functions and Constants

- std::abs
- std::acos
- std::bit_width
- std::ceil
- std::cos
- std::exp
- std::fabs
- std::floor
- std::isfinite
- std::isinf
- std::isnan
- std::llround
- std::log
- std::log1pf
- std::log2
- std::numbers
- std::pow
- std::round
- std::sin
- std::sqrt

### Strings, Streams, and Formatting

- std::cerr
- std::cout
- std::endl
- std::fixed
- std::flush
- std::fstream
- std::getline
- std::ifstream
- std::ios
- std::istream
- std::istreambuf_iterator
- std::left
- std::ofstream
- std::ostream
- std::ostringstream
- std::printf
- std::put_time
- std::right
- std::scientific
- std::setprecision
- std::setw
- std::smatch
- std::stod
- std::stof
- std::stoi
- std::stoul
- std::streambuf
- std::streamoff
- std::streamsize
- std::stringstream
- std::strtod
- std::to_string

### Filesystem and Regex

- std::filesystem
- std::regex
- std::regex_match

### Random

- std::discrete_distribution
- std::mt19937
- std::normal_distribution
- std::random_device
- std::uniform_int_distribution
- std::uniform_real_distribution

### Concurrency and Atomics

- std::atomic
- std::call_once
- std::condition_variable
- std::lock_guard
- std::memory_order_acquire
- std::memory_order_relaxed
- std::memory_order_release
- std::mutex
- std::once_flag
- std::this_thread
- std::thread
- std::unique_lock

### Types and Utilities

- std::chrono
- std::current_exception
- std::error_code
- std::exception
- std::exception_ptr
- std::exit
- std::function
- std::get
- std::hash
- std::initializer_list
- std::input_iterator_tag
- std::invalid_argument
- std::is_same_v
- std::nullopt
- std::numeric_limits
- std::out_of_range
- std::ptrdiff_t
- std::rethrow_exception
- std::runtime_error
- std::size_t
- std::tie
- std::tm
- std::uint8_t
- std::uint16_t
- std::uint32_t

### Character Classification and Conversion

- std::isalnum
- std::isspace
- std::tolower

### C stdio Wrappers Used With std:: Namespace

- std::fclose
- std::fopen
- std::fwrite

## Notes

- This is an API usage inventory, not a recommendation list.
- The list can be regenerated with:

```bash
rg -o --no-filename "std::[A-Za-z_][A-Za-z0-9_]*" src include | sort -u
```
--
```
for fold in K folds:
    reset model + optimizer        ← fresh weights each fold
    for epoch in training_epochs:
        TRAIN on train_trial_ids   ← backprop + optimizer step
        VALIDATE on val_trial_ids  ← forward only, no grad
        log "fold X/K  epoch Y/E  train loss: A  val loss: B"
    compute fold_mean_val = mean over epoch val losses
    log "fold X/K complete: mean val loss = Z"
grand mean_val_loss = mean over fold_mean_val_losses
log "K-Fold complete: grand mean val loss = Z"
```