/**
 * @file include/nn/dataLoaders/interfaces/IDatasetPrinter.hpp
 * @brief Common interface for dataset printing strategies.
 *
 * Supports polymorphic printing of different dataset types through
 * a unified `print()` method on each dataset implementation.
 */

#ifndef NN_DATALOADERS_IDATASETPRINTER_HPP
#define NN_DATALOADERS_IDATASETPRINTER_HPP

class Dataset;

/**
 * Abstract base class for dataset printer strategies.
 *
 * Implementations should provide type-safe printing for specific dataset types
 * by overriding the appropriate `print_*` methods.
 */
class IDatasetPrinter
{
   public:
    virtual ~IDatasetPrinter() = default;

    /**
     * Print a generic dataset. Called when no specific printer method exists
     * for the concrete dataset type.
     *
     * @param dataset The dataset to print.
     */
    virtual void print_generic(const Dataset& dataset) = 0;
};

#endif // NN_DATALOADERS_IDATASETPRINTER_HPP
