/*
 * Helper routines for printing dataset summaries in experiments.
 */
#pragma once

#include <string>

class Protocol101117Dataset;

// Print a short summary of the dataset including subjects and per-subject sample counts.
void printDatasetSummary(const Protocol101117Dataset& dataset, const std::string& dataset_root);
