#ifndef INCLUDE_DISCRETIZATION_H_
#define INCLUDE_DISCRETIZATION_H_

#include <deal.II/fe/fe_system.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/point.h>

#include "userInputParameters.h"

using namespace dealii;

/**
 * This class deals with adaptive refinement. Add more comments later
 */
template <int dim>
class discretization {
public:
    discretization(const userInputParameters<dim>& _userInputs);

    /*Parallel mesh object which holds information about the FE nodes, elements and parallel domain decomposition
     */
    parallel::distributed::Triangulation<dim> triangulation;

    /*A vector of finite element objects used in a model. For problems with only one primal field,
     *the size of this vector is one,otherwise the size is the number of primal fields in the problem.
     */
    std::vector<FESystem<dim>*> FESet;

    /**
     * Initializes the mesh, degrees of freedom, constraints and data structures using the user provided
     * inputs in the application parameters file.
     */
    void makeTriangulation(parallel::distributed::Triangulation<dim>&) const;

    /*Total degrees of freedom in a problem set.*/
    unsigned int totalDOFs;

private:
    userInputParameters<dim> userInputs;
};

template <int dim>
discretization<dim>::discretization(const userInputParameters<dim>& _userInputs)
    : userInputs(_userInputs)
    , triangulation(MPI_COMM_WORLD)
{
}

template <int dim>
void discretization<dim>::makeTriangulation(parallel::distributed::Triangulation<dim>& tria) const
{
    if (dim == 3) {
        GridGenerator::subdivided_hyper_rectangle(tria, userInputs.subdivisions, Point<dim>(), Point<dim>(userInputs.domain_size[0], userInputs.domain_size[1], userInputs.domain_size[2]));
    } else if (dim == 2) {
        GridGenerator::subdivided_hyper_rectangle(tria, userInputs.subdivisions, Point<dim>(), Point<dim>(userInputs.domain_size[0], userInputs.domain_size[1]));
    } else {
        GridGenerator::subdivided_hyper_rectangle(tria, userInputs.subdivisions, Point<dim>(), Point<dim>(userInputs.domain_size[0]));
    }
}

#endif
