// ===========================================================================
// FUNCTION FOR INITIAL CONDITIONS
// ===========================================================================

#include <deal.II/base/utilities.h>

template <int dim, int degree>
void
customPDE<dim, degree>::setInitialCondition([[maybe_unused]] const Point<dim>  &p,
                                            [[maybe_unused]] const unsigned int index,
                                            [[maybe_unused]] double            &scalar_IC,
                                            [[maybe_unused]] Vector<double>    &vector_IC)
{
  if (index == 0 || index == 1)
    {
      for (unsigned int dimension = 0; dimension < dim; dimension++)
        {
          vector_IC[dimension] = 0.0;
        }
    }
  else
    {
      scalar_IC = 0.0;
    }
}

// ===========================================================================
// FUNCTION FOR NON-UNIFORM DIRICHLET BOUNDARY CONDITIONS
// ===========================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::setNonUniformDirichletBCs(
  [[maybe_unused]] const Point<dim>  &p,
  [[maybe_unused]] const unsigned int index,
  [[maybe_unused]] const unsigned int direction,
  [[maybe_unused]] const double       time,
  [[maybe_unused]] double            &scalar_BC,
  [[maybe_unused]] Vector<double>    &vector_BC)
{
  if ((index == 0 || index == 1) && direction == 0)
    {
      vector_BC[0] = 4.0 * u_max * p[1] * (userInputs.domain_size[1] - p[1]) /
                     userInputs.domain_size[1] / userInputs.domain_size[1];
    }
}