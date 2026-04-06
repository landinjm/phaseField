#pragma once

#include <prismspf/config.h>

#include <libassert/assert.hpp>

PRISMS_PF_BEGIN_NAMESPACE

/**
 * @brief Wrappers for libassert DEBUG_ASSERT to the style that deal.II uses with Assert.
 */
#define Assert(...) DEBUG_ASSERT(__VA_ARGS__) // NOLINT

/**
 * @brief Wrappers for libassert ASSERT to the style that deal.II uses with AssertThrow.
 */
#define AssertThrow(...) ASSERT(__VA_ARGS__) // NOLINT

/**
 * @brief Wrapper for libassert PANIC.
 */
#define Panic(...) PANIC(__VA_ARGS__) // NOLINT

/**
 * @brief Wrapper for libassert UNREACHABLE
 */
#define Unreachable(...) UNREACHABLE(__VA_ARGS__) // NOLINT

PRISMS_PF_END_NAMESPACE
