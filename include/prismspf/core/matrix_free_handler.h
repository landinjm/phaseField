// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim>
class UserInputParameters;

/**
 * @brief This class handlers the management and access of the matrix-free objects.
 */
template <unsigned int dim, typename number>
class MatrixfreeHandler
{
public:
  /**
   * @brief Constructor.
   */
  MatrixfreeHandler();

  /**
   * @brief Reinitialize the matrix-free object with the same quad rule.
   */
  void
  reinit(const dealii::Mapping<dim>              &mapping,
         const dealii::DoFHandler<dim>           &dof_handler,
         const dealii::AffineConstraints<number> &constraint,
         const dealii::Quadrature<1>             &quad);

  /**
   * @brief Reinitialize the matrix-free object with the same quad rule.
   */
  void
  reinit(const dealii::Mapping<dim>                                   &mapping,
         const std::vector<const dealii::DoFHandler<dim> *>           &dof_handler,
         const std::vector<const dealii::AffineConstraints<number> *> &constraint,
         const dealii::Quadrature<1>                                  &quad);

  /**
   * @brief Reinitialize the matrix-free object with the different quad rule.
   */
  void
  reinit(const dealii::Mapping<dim>                                   &mapping,
         const std::vector<const dealii::DoFHandler<dim> *>           &dof_handler,
         const std::vector<const dealii::AffineConstraints<number> *> &constraint,
         const std::vector<dealii::Quadrature<1>>                     &quad);

  /**
   * @brief Getter function for the matrix-free object (shared ptr).
   */
  [[nodiscard]] std::shared_ptr<
    dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>>
  get_matrix_free() const;

private:
  /**
   * @brief Matrix-free object that collects data to be used in cell loop operations.
   */
  std::shared_ptr<dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>>
    matrix_free_object;

  /**
   * @brief Additional data scheme
   */
  typename dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>::
    AdditionalData additional_data;

  /**
   * @brief Whether the matrix-free object has been initialized.
   */
  bool is_initialized = false;
};

PRISMS_PF_END_NAMESPACE
