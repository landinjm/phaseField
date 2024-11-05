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
  if (index == 0 || index == 1 || index == 3)
    {
      vector_IC[0] = std::sin(p[0]) * std::cos(p[1]) * std::cos(p[2]);
      vector_IC[1] = -std::cos(p[0]) * std::sin(p[1]) * std::cos(p[2]);
      vector_IC[2] = 0.0;
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
{}
