#ifndef triangulation_handler_h
#define triangulation_handler_h

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <core/exceptions.h>
#include <core/user_inputs/user_input_parameters.h>
#include <fstream>

/**
 * \brief This class handlers the generation and manipulation of triangulations.
 */
template <int dim>
class triangulationHandler
{
public:
  using Triangulation =
    typename std::conditional<dim == 1,
                              dealii::Triangulation<dim>,
                              dealii::parallel::distributed::Triangulation<dim>>::type;

  /**
   * \brief Constructor.
   */
  triangulationHandler();

  /**
   * \brief Getter function for triangulation (constant reference).
   */
  [[nodiscard]] const Triangulation &
  get_triangulation() const;

  /**
   * \brief Return the global maximum level of the triangulation.
   */
  [[nodiscard]] uint
  get_n_global_levels() const;

  /**
   * \brief Generate mesh.
   */
  void
  generate_mesh(const userInputParameters<dim> &user_inputs);

  /**
   * \brief Export triangulation to vtk.
   */
  void
  export_triangulation_as_vtk(const std::string &filename) const;

private:
  /**
   * \brief Mark the domain ids on the triangulation to get the proper mapping of
   * specified boundary conditions.
   */
  void
  mark_boundaries(const userInputParameters<dim> &user_inputs) const;

  /**
   * \brief Mark certain faces of the triangulation periodic.
   */
  void
  mark_periodic(const userInputParameters<dim> &user_inputs);

  std::unique_ptr<Triangulation> triangulation;
};

template <int dim>
triangulationHandler<dim>::triangulationHandler()
{
  if constexpr (dim == 1)
    {
      triangulation = std::make_unique<dealii::Triangulation<dim>>(
        dealii::Triangulation<dim>::limit_level_difference_at_vertices);
    }
  else
    {
      triangulation = std::make_unique<dealii::parallel::distributed::Triangulation<dim>>(
        MPI_COMM_WORLD,
        dealii::Triangulation<dim>::limit_level_difference_at_vertices,
        dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
    }
}

template <int dim>
const typename triangulationHandler<dim>::Triangulation &
triangulationHandler<dim>::get_triangulation() const
{
  return *triangulation;
}

template <int dim>
uint
triangulationHandler<dim>::get_n_global_levels() const
{
  return triangulation->n_global_levels();
}

template <int dim>
void
triangulationHandler<dim>::generate_mesh(const userInputParameters<dim> &user_inputs)
{
  // Generate rectangle
  dealii::GridGenerator::subdivided_hyper_rectangle(
    *triangulation,
    user_inputs.spatial_discretization.subdivisions,
    dealii::Point<dim>(),
    dealii::Point<dim>(user_inputs.spatial_discretization.domain_size));

  // Mark boundaries. This is done before global refinement to reduce the number of cells
  // we have to loop through.
  mark_boundaries(user_inputs);

  // Mark periodicity
  mark_periodic(user_inputs);

  // Output triangulation to vtk if in debug mode
#ifdef DEBUG
  export_triangulation_as_vtk("triangulation");
#endif

  // Global refinement
  triangulation->refine_global(user_inputs.spatial_discretization.refine_factor);
}

template <int dim>
void
triangulationHandler<dim>::export_triangulation_as_vtk(const std::string &filename) const
{
  dealii::GridOut grid_out;
  std::ofstream   out(filename + ".vtk");
  grid_out.write_vtk(*triangulation, out);
  std::cout << "Triangulation written to " << filename << ".vtk\n";
}

template <int dim>
void
triangulationHandler<dim>::mark_boundaries(
  const userInputParameters<dim> &user_inputs) const
{
  double tolerance = 1e-12;

  // Loop through the cells
  for (const auto &cell : triangulation->active_cell_iterators())
    {
      // Mark the faces (faces_per_cell = 2*dim)
      for (uint face_number = 0; face_number < dealii::GeometryInfo<dim>::faces_per_cell;
           ++face_number)
        {
          // Direction for quad and hex cells
          uint direction = std::floor(face_number / 2);

          // Mark the boundary id for x=0, y=0, z=0
          if (std::fabs(cell->face(face_number)->center()(direction) - 0) < tolerance)
            {
              cell->face(face_number)->set_boundary_id(face_number);
            }
          // Mark the boundary id for x=max, y=max, z=max
          else if (std::fabs(
                     cell->face(face_number)->center()(direction) -
                     (user_inputs.spatial_discretization.domain_size[direction])) <
                   tolerance)
            {
              cell->face(face_number)->set_boundary_id(face_number);
            }
        }
    }
}

template <int dim>
void
triangulationHandler<dim>::mark_periodic(const userInputParameters<dim> &user_inputs)
{
  // Add periodicity in the triangulation where specified in the boundary conditions. Note
  // that if one field is periodic all others should be as well.
  for (const auto &[index, boundary_condition] :
       user_inputs.boundary_parameters.boundary_condition_list)
    {
      for (const auto &[component, condition] : boundary_condition)
        {
          for (const auto &[boundary_id, boundary_type] :
               condition.boundary_condition_map)
            {
              if (boundary_type == boundaryType::PERIODIC)
                {
                  // Skip boundary ids that are odd since those map to the even faces
                  if (boundary_id % 2 != 0)
                    {
                      continue;
                    }

                  // Create a vector of matched pairs that we fill and enforce upon the
                  // constaints
                  std::vector<dealii::GridTools::PeriodicFacePair<
                    typename Triangulation::cell_iterator>>
                    periodicity_vector;

                  // Determine the direction
                  const uint direction = std::floor(boundary_id / dim);

                  // Collect the matched pairs on the coarsest level of the mesh
                  dealii::GridTools::collect_periodic_faces(*triangulation,
                                                            boundary_id,
                                                            boundary_id + 1,
                                                            direction,
                                                            periodicity_vector);

                  // Set constraints
                  triangulation->add_periodicity(periodicity_vector);
                }
            }
        }
    }
}

#endif