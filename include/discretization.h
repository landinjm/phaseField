#ifndef INCLUDE_DISCRETIZATION_H_
#define INCLUDE_DISCRETIZATION_H_

#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "userInputParameters.h"

using namespace dealii;

/**
 * This class deals with the discretization. Add more comments later
 */
template <int dim>
class discretization {
public:
    discretization(const userInputParameters<dim>& _userInputs);

    /*Parallel mesh object which holds information about the FE nodes, elements and parallel domain decomposition.*/
    parallel::distributed::Triangulation<dim> triangulation;

    /*Initializes the mesh, degrees of freedom, constraints and data structures using the user provided inputs in the application parameters file.*/
    void makeTriangulation();

    /*A vector of finite element objects used in a model. For problems with only one primal field, the size of this vector is one, otherwise the size is the number of primal fields in the problem.*/
    std::vector<FESystem<dim>*> FESet;

    /*Initializes finite element object*/
    void makeFESystem(FESystem<dim>*, fieldType, int);

    /*Total degrees of freedom in a problem set.*/
    unsigned int totalDOFs;

    /*Placeholder for quadrature information. This would require a degree addition to the class template*/

    /*A vector of all the degree of freedom objects is the problem. A degree of freedom object handles the serial/parallel distribution of the degrees of freedom for all the primal fields in the problem.*/
    std::vector<const DoFHandler<dim>*> dofHandlersSet;

    /*A vector of the locally relevant degrees of freedom. Locally relevant degrees of freedom in a parallel implementation is a collection of the degrees of freedom owned by the current processor and the surrounding ghost nodes which are required for the field computations in this processor.*/
    std::vector<const IndexSet*> locally_relevant_dofsSet;

    /*Copies of dofHandlerSet elements, but stored as non-const.*/
    std::vector<DoFHandler<dim>*> dofHandlersSet_nonconst;

    /*Copies of locally_relevant_dofsSet elements, but stored as non-const.*/
    std::vector<IndexSet*> locally_relevant_dofsSet_nonconst;

    /*Object of class MatrixFree<dim>. This is primarily responsible for all the base matrix free functionality of this MatrixFreePDE<dim> class. Refer to deal.ii documentation of MatrixFree<dim> class for details.*/
    MatrixFree<dim, double> matrixFreeObject;

private:
    /*User inputs*/
    userInputParameters<dim> userInputs;

    /*Message stream*/
    ConditionalOStream pcout;

    /*Method to mark the boundary cells of the triangulation so that boundary conditions can be applied later.*/
    void markBoundaries(parallel::distributed::Triangulation<dim>&) const;
};

template <int dim>
discretization<dim>::discretization(const userInputParameters<dim>& _userInputs)
    : userInputs(_userInputs)
    , triangulation(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{
}

template <int dim>
void discretization<dim>::makeTriangulation()
{
    // Define the bounds of the triangulation domain
    Point<dim> Origin = Point<dim>();
    Point<dim> UpperBound;

    switch(dim) {
        case 3:
            UpperBound = Point<dim>(userInputs.domain_size[0], userInputs.domain_size[1], userInputs.domain_size[2]);
            break;
        case 2:
            UpperBound = Point<dim>(userInputs.domain_size[0], userInputs.domain_size[1]);
            break;
        case 1:
            UpperBound = Point<dim>(userInputs.domain_size[0]);
            break;
        default:
            std::cerr << "PRISMS-PF Error: Invalid number of dimensions" << std::endl;
            abort();
    }

    // Create a reference to the triangulation
    parallel::distributed::Triangulation<dim>& tria = triangulation;

    // Generate triangulation
    GridGenerator::subdivided_hyper_rectangle(tria, userInputs.subdivisions, Origin, UpperBound);

    // Mark boundaries for applying the boundary conditions
    markBoundaries(tria);
}

template <int dim>
void discretization<dim>::makeFESystem(FESystem<dim>* fe, fieldType field, int degree)
{
    switch(field) {
        case SCALAR:
            fe = new FESystem<dim>(FE_Q<dim>(QGaussLobatto<1>(degree + 1)), 1);
            break;
        case VECTOR:
            fe = new FESystem<dim>(FE_Q<dim>(QGaussLobatto<1>(degree + 1)), dim);
            break;
        default:
            pcout << std::endl << "matrixFreePDE.h: unknown field type" << std::endl;
            exit(-1);
    }

    FESet.push_back(fe);
}

template <int dim>
void discretization<dim>::markBoundaries(parallel::distributed::Triangulation<dim>& tria) const
{
    for (auto cell = tria.begin(); cell != tria.end(); ++cell) {
        // Mark all of the faces
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number) {
            for (unsigned int i = 0; i < dim; i++) {
                if (std::fabs(cell->face(face_number)->center()(i) - (0)) < 1e-12) {
                    cell->face(face_number)->set_boundary_id(2 * i);
                } else if (std::fabs(cell->face(face_number)->center()(i) - (userInputs.domain_size[i])) < 1e-12) {
                    cell->face(face_number)->set_boundary_id(2 * i + 1);
                }
            }
        }
    }
}

#endif
