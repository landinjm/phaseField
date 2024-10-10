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

  // Virtual method in MatrixFreePDE
  void
  create_triangulation(parallel::distributed::Triangulation<dim> &tria) const override;

  // ================================================================
  // Model constants specific to this subclass
  // ================================================================

  double McV     = userInputs.get_model_constant_double("McV");
  double KcV     = userInputs.get_model_constant_double("KcV");
  double rho     = userInputs.get_model_constant_double("rho");
  double c_alpha = userInputs.get_model_constant_double("c_alpha");
  double c_beta  = userInputs.get_model_constant_double("c_beta");
  double k       = userInputs.get_model_constant_double("k");
  double epsilon = userInputs.get_model_constant_double("epsilon");

  // ================================================================
};

#include <deal.II/grid/grid_generator.h>

template <int dim, int degree>
void
customPDE<dim, degree>::create_triangulation(
  parallel::distributed::Triangulation<dim> &tria) const
{
  parallel::distributed::Triangulation<dim> tria_box(MPI_COMM_WORLD),
    tria_semicircle(MPI_COMM_WORLD);
  if (dim == 3)
    {
      GridGenerator::subdivided_hyper_rectangle(tria_box,
                                                userInputs.subdivisions,
                                                Point<dim>(),
                                                Point<dim>(userInputs.domain_size[0],
                                                           userInputs.domain_size[1],
                                                           userInputs.domain_size[2]));
    }
  else if (dim == 2)
    {
      GridGenerator::subdivided_hyper_rectangle(tria_box,
                                                userInputs.subdivisions,
                                                Point<dim>(),
                                                Point<dim>(userInputs.domain_size[0],
                                                           userInputs.domain_size[1]));
    }
  else
    {
      GridGenerator::subdivided_hyper_rectangle(tria_box,
                                                userInputs.subdivisions,
                                                Point<dim>(),
                                                Point<dim>(userInputs.domain_size[0]));
    }

  GridGenerator::half_hyper_ball(tria_semicircle,
                                 Point<dim>(userInputs.domain_size[0],
                                            userInputs.domain_size[1] / 2.0),
                                 userInputs.domain_size[1] / 2.0);

  // Find the two non-corner vertices on the right side of the rectangular mesh
  Point<dim> pt1, pt2;
  typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell3 = tria_box.begin_active(),
    endc3 = tria_box.end();
  for (; cell3 != endc3; ++cell3)
    {
      for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
        {
          Point<dim> &v = cell3->vertex(i);
          if ((std::abs(v(0) - userInputs.domain_size[0]) < 1e-10) &&
              (v(1) > userInputs.domain_size[1] / 2.0) &&
              (v(1) < userInputs.domain_size[1] - 1.0e-10))
            {
              pt1 = v;
            }
          if ((std::abs(v(0) - userInputs.domain_size[0]) < 1e-10) &&
              (v(1) < userInputs.domain_size[1] / 2.0) && (v(1) > 1.0e-10))
            {
              pt2 = v;
            }
        }
    }
  // Move the vertices at the center of the half hyper ball so that they will
  // align with non-corner vertices on the right side of the rectangular mesh
  typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell2 = tria_semicircle.begin_active(),
    endc2 = tria_semicircle.end();
  for (; cell2 != endc2; ++cell2)
    {
      for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
        {
          Point<dim> &v = cell2->vertex(i);
          if ((std::abs(v(0) - userInputs.domain_size[0]) < 1e-10) &&
              (v(1) > userInputs.domain_size[1] / 2.0) &&
              (v(1) < userInputs.domain_size[1] - 1.0e-10))
            {
              v(1) = pt1(1);
            }
          if ((std::abs(v(0) - userInputs.domain_size[0]) < 1e-10) &&
              (v(1) < userInputs.domain_size[1] / 2.0) && (v(1) > 1.0e-10))
            {
              v(1) = pt2(1);
            }
        }
    }
  // Merge the rectangle and the semicircle
  GridGenerator::merge_triangulations(tria_box, tria_semicircle, tria);

  // Attach a spherical manifold to the semicircular part of the domain so that
  // it gets refined with rounded edges
  static const SphericalManifold<dim> boundary(
    Point<dim>(userInputs.domain_size[0], userInputs.domain_size[1] / 2.0));
  tria.set_manifold(8, boundary);

  typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();
  for (; cell != endc; ++cell)
    {
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          const Point<dim> face_center = cell->face(f)->center();
          if (face_center[0] > userInputs.domain_size[0] + 1.0e-10)
            {
              cell->face(f)->set_all_manifold_ids(8);
              if (face_center.distance(Point<dim>(userInputs.domain_size[0],
                                                  userInputs.domain_size[1] / 2.0)) >
                  0.2 * userInputs.domain_size[1])
                {
                  cell->set_all_manifold_ids(8);
                }
            }
        }
    }

  // Mark the boundaries
  for (const auto &cell4 : tria.active_cell_iterators())
    {
      // Mark all of the faces
      for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
           ++face_number)
        {
          if (cell4->face(face_number)->at_boundary())
            {
              for (unsigned int i = 0; i < dim; i++)
                {
                  if (i == 0)
                    {
                      if (std::fabs(cell4->face(face_number)->center()(i) - (0)) < 1e-12)
                        {
                          cell4->face(face_number)->set_boundary_id(2 * i);
                        }
                      else if (std::fabs(cell4->face(face_number)->center()(i) >
                                         (userInputs.domain_size[i])))
                        {
                          cell4->face(face_number)->set_boundary_id(2 * i + 1);
                        }
                    }
                  else
                    {
                      if (std::fabs(cell4->face(face_number)->center()(i) - (0)) < 1e-12)
                        {
                          cell4->face(face_number)->set_boundary_id(2 * i);
                        }
                      else if (std::fabs(cell4->face(face_number)->center()(i) -
                                         (userInputs.domain_size[i])) < 1e-12)
                        {
                          cell4->face(face_number)->set_boundary_id(2 * i + 1);
                        }
                    }
                }
            }
        }
    }
}
