// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <prismspf/config.h>

#include <string>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim>
class UserInputParameters;

/**
 * @brief Class that outputs a passed solution to vtu, vtk, or pvtu
 */
template <unsigned int dim, typename number>
class SolutionOutput
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  /**
   * @brief Constructor for a single field that must be output.
   */
  SolutionOutput(const VectorType               &solution,
                 const dealii::DoFHandler<dim>  &dof_handler,
                 const unsigned int             &degree,
                 const std::string              &name,
                 const UserInputParameters<dim> &user_inputs);

  /**
   * @brief Constructor for a multiple fields that must be output.
   */
  SolutionOutput(const std::map<unsigned int, VectorType *>         &solution_set,
                 const std::vector<const dealii::DoFHandler<dim> *> &dof_handlers,
                 const unsigned int                                 &degree,
                 const std::string                                  &name,
                 const UserInputParameters<dim>                     &user_inputs);
};

PRISMS_PF_END_NAMESPACE
