#ifndef NN_DATALOADERS_10_1117_INPUTMODECODEC_HPP
#define NN_DATALOADERS_10_1117_INPUTMODECODEC_HPP

#include <string>
#include <vector>

#include "nn/dataLoaders/10.1117/datasets/raw/Dataset101117.hpp"

/**
 * @brief Converts Protocol101117InputMode enum to canonical token.
 */
auto protocol101117InputModeToToken(Protocol101117InputMode mode) -> std::string;

/**
 * @brief Parses a token into Protocol101117InputMode (case-insensitive).
 */
auto parseProtocol101117InputModeToken(std::string token) -> Protocol101117InputMode;

/**
 * @brief Returns accepted CLI tokens for Protocol101117InputMode.
 */
auto supportedProtocol101117InputModeTokens() -> std::vector<std::string>;

#endif // NN_DATALOADERS_10_1117_INPUTMODECODEC_HPP
