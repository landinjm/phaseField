#include "matrixFreePDE.h"

using namespace dealii;

template <int dim, int degree>
class customPDE : public MatrixFreePDE<dim, degree>
{
public:
  // Constructor
  customPDE(userInputParameters<dim> _userInputs)
    : MatrixFreePDE<dim, degree>(_userInputs)
    , userInputs(_userInputs) {};

  // Function to set the initial conditions (in ICs_and_BCs.h)
  void
  setInitialCondition([[maybe_unused]] const Point<dim>  &p,
                      [[maybe_unused]] const unsigned int index,
                      [[maybe_unused]] double            &scalar_IC,
                      [[maybe_unused]] Vector<double>    &vector_IC) override;

  // Function to set the non-uniform Dirichlet boundary conditions (in
  // ICs_and_BCs.h)
  void
  setNonUniformDirichletBCs([[maybe_unused]] const Point<dim>  &p,
                            [[maybe_unused]] const unsigned int index,
                            [[maybe_unused]] const unsigned int direction,
                            [[maybe_unused]] const double       time,
                            [[maybe_unused]] double            &scalar_BC,
                            [[maybe_unused]] Vector<double>    &vector_BC) override;

private:
#include "typeDefs.h"

  const userInputParameters<dim> userInputs;

  // Function to set the RHS of the governing equations for explicit time
  // dependent equations (in equations.h)
  void
  explicitEquationRHS(
    [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>>
                                                        &variable_list,
    [[maybe_unused]] Point<dim, VectorizedArray<double>> q_point_loc) const override;

  // Function to set the RHS of the governing equations for all other equations
  // (in equations.h)
  void
  nonExplicitEquationRHS(
    [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>>
                                                        &variable_list,
    [[maybe_unused]] Point<dim, VectorizedArray<double>> q_point_loc) const override;

  // Function to set the LHS of the governing equations (in equations.h)
  void
  equationLHS(
    [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>>
                                                        &variable_list,
    [[maybe_unused]] Point<dim, VectorizedArray<double>> q_point_loc) const override;

// Function to set postprocessing expressions (in postprocess.h)
#ifdef POSTPROCESS_FILE_EXISTS
  void
  postProcessedFields(
    [[maybe_unused]] const variableContainer<dim, degree, VectorizedArray<double>>
      &variable_list,
    [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>>
                                                              &pp_variable_list,
    [[maybe_unused]] const Point<dim, VectorizedArray<double>> q_point_loc)
    const override;
#endif

// Function to set the nucleation probability (in nucleation.h)
#ifdef NUCLEATION_FILE_EXISTS
  double
  getNucleationProbability([[maybe_unused]] variableValueContainer variable_value,
                           [[maybe_unused]] double                 dV) const override;
#endif

  // ================================================================
  // Methods specific to this subclass
  // ================================================================

  void
  makeTriangulation(parallel::distributed::Triangulation<dim> &) const override;

  // ================================================================
  // Model constants specific to this subclass
  // ================================================================

  double D        = userInputs.get_model_constant_double("D");
  double W0       = userInputs.get_model_constant_double("W0");
  double delta    = userInputs.get_model_constant_double("delta");
  double epsilonM = userInputs.get_model_constant_double("epsilonM");
  double mult     = 6.0;

  // ================================================================
};

#include <deal.II/grid/grid_generator.h>

template <int dim, int degree>
void
customPDE<dim, degree>::makeTriangulation(
  parallel::distributed::Triangulation<dim> &tria) const
{
  // Generate hyper ball
  GridGenerator::hyper_ball(tria, Point<dim>(), userInputs.domain_size[0]);

  // Create spherical manifold
  const Point<dim> mesh_center;
  const double     core_radius  = userInputs.domain_size[0] / 5.0;
  const double     inner_radius = userInputs.domain_size[0] / 3.0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      if (mesh_center.distance(cell->center()) < 1e-5)
        {
          for (const auto v : cell->vertex_indices())
            cell->vertex(v) *= core_radius / mesh_center.distance(cell->vertex(v));
        }
    }

  for (const auto &cell : tria.active_cell_iterators())
    {
      if (mesh_center.distance(cell->center()) >= 1e-5)
        {
          cell->set_refine_flag();
        }
    }
  tria.execute_coarsening_and_refinement();
  for (const auto &cell : tria.active_cell_iterators())
    {
      for (const auto v : cell->vertex_indices())
        {
          const double dist = mesh_center.distance(cell->vertex(v));
          if (dist > core_radius * 1.0001 && dist < 0.9999)
            {
              cell->vertex(v) *= inner_radius / dist;
            }
        }
    }
  for (const auto &cell : tria.active_cell_iterators())
    {
      bool is_in_inner_circle = false;
      for (const auto v : cell->vertex_indices())
        if (mesh_center.distance(cell->vertex(v)) < inner_radius)
          {
            is_in_inner_circle = true;
            break;
          }
      if (is_in_inner_circle == false)
        {
          cell->set_all_manifold_ids(0);
        }
    }

  // Mark the boundaries
  for (const auto &cell : tria.active_cell_iterators())
    {
      // Mark all of the faces on the boundary with a boundary id of 0. This reduces the
      // complexity of the code at the cost of flexibility in boundary conditions. For the
      // benchmark case, we don't care about flexibility. If you plan to use this code to
      // create your own triangulation, modify this section accordingly.
      for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
           ++face_number)
        {
          const auto &face = cell->face(face_number);

          if (face->at_boundary())
            {
              face->set_boundary_id(0);
            }
        }
    }
}