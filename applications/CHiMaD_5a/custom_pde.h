// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <deal.II/base/config.h>
#include <deal.II/base/utilities.h>

#include <prismspf/core/pde_operator_base.h>

PRISMS_PF_BEGIN_NAMESPACE

constexpr unsigned int dim = 2;

template <unsigned int degree, typename number>
class CustomPDE : public PDEOperatorBase<dim, degree, number>
{
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad  = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorValue = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess  = dealii::Tensor<3, dim, ScalarValue>;
  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

  explicit CustomPDE(const UserInputParameters<dim> &_user_inputs,
                     PhaseFieldTools<dim>           &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools) {};

private:
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override {
  };

  void
  set_dirichlet([[maybe_unused]] const unsigned int       &index,
                [[maybe_unused]] const unsigned int       &boundary_id,
                [[maybe_unused]] const unsigned int       &component,
                [[maybe_unused]] const dealii::Point<dim> &point,
                [[maybe_unused]] number                   &scalar_value,
                [[maybe_unused]] number &vector_component_value) const override
  {
    if (boundary_id == 0 && component == 0 && index == 0)
      {
        vector_component_value = -0.001 * (point[1] - 3.0) * (point[1] - 3.0) + 0.009;
      }
  };

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solver_id) const override
  {
    if (solver_id == 1)
      {
        variable_list.set_value_term(0, rho * g);
        variable_list.set_value_term(1, number(0));
      }
  };

  void
  compute_lhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solver_id) const override
  {
    if (solver_id == 1)
      {
        VectorValue u     = variable_list.template get_value<Vector, LHS>(0);
        VectorGrad u_grad = variable_list.template get_symmetric_gradient<Vector, LHS>(0);
        ScalarValue u_div = variable_list.template get_divergence<Vector, LHS>(0);
        VectorValue u_lap = variable_list.template get_laplacian<Vector, LHS>(0);
        ScalarValue p     = variable_list.template get_value<Scalar, LHS>(1);
        ScalarGrad  p_grad = variable_list.template get_gradient<Scalar, LHS>(1);
        ScalarValue h      = variable_list.get_element_volume();

        VectorGrad p_identity;
        for (unsigned int i = 0; i < dim; ++i)
          {
            p_identity[i][i] = p;
          }

        ScalarGrad  residual = -mu * u_lap + p_grad - rho * g;
        ScalarValue tau      = stabilization_parameter(h);

        variable_list.set_gradient_term(0, mu * u_grad - p_identity);
        variable_list.set_value_term(1, u_div);
        variable_list.set_gradient_term(1, tau * residual);
      }
  };

  DEAL_II_ALWAYS_INLINE inline ScalarValue
  stabilization_parameter(ScalarValue &element_volume) const
  {
    using std::sqrt;

    constexpr number size_modifier = sqrt(number(4.0) / number(M_PI)) / number(degree);

    // Stabilization parameter
    ScalarValue h = sqrt(element_volume) * size_modifier;

    return (rho * h * h) / (number(12) * mu);
  }

  static constexpr number                         rho = number(100);
  static constexpr number                         mu  = number(1);
  static constexpr dealii::Tensor<1, dim, number> g {
    {number(0), number(0)}
  };
};

PRISMS_PF_END_NAMESPACE
