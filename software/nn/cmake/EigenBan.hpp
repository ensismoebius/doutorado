#pragma once
// Pre-included guard to prevent Eigen usage unless explicitly allowed.
// Define NN_ALLOW_EIGEN for translation units that are permitted to include Eigen
// (e.g., backend implementations only).

#ifndef NN_ALLOW_EIGEN

// Poison common Eigen identifiers so accidental usage fails compilation early.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC poison Eigen EIGEN_WORLD_VERSION EIGEN_MAJOR_VERSION EIGEN_CORE_H
#endif

// If Eigen somehow slips past the poison (e.g., via precompiled headers), stop the build.
#ifdef EIGEN_WORLD_VERSION
#error "Eigen is forbidden in this translation unit. Define NN_ALLOW_EIGEN explicitly to opt in."
#endif

#endif // NN_ALLOW_EIGEN
