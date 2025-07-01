// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE

/**
 * \brief Purely virtual class for each of the parameter handler subclasses.
 */
class ParameterBase
{
public:
  /**
   * \brief Constructor.
   */
  ParameterBase();

  /**
   * \brief Destructor.
   */
  virtual ~ParameterBase();

  /**
   * \brief Clear the parameters.
   */
  virtual void
  clear() = 0;

  /**
   * \brief Print the parameters to summary.log.
   */
  virtual void
  print() const = 0;

  /**
   * \brief Postprocess and validate parameters.
   */
  virtual void
  postproess_and_validate() = 0;

  /**
   * \brief Maximum number of subsections
   */
  static constexpr unsigned int max_subsections = 16;
};

PRISMS_PF_END_NAMESPACE
