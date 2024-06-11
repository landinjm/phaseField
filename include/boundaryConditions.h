#ifndef INCLUDE_BOUNDARYCONDITIONS_H_
#define INCLUDE_BOUNDARYCONDITIONS_H_

#include <deal.II/lac/affine_constraints.h>

#include "userInputParameters.h"

using namespace dealii;

/**
 * This class deals with the boundary conditions. Add more comments later
 */
template <int dim, int degree>
class boundaryConditions {
public:
    boundaryConditions(const userInputParameters<dim>& _userInputs);

    /*A vector of all the constraint sets in the problem. A constraint set is a map which holds the mapping between the degrees of freedom and the corresponding degree of freedom constraints.*/
    std::vector<const AffineConstraints<double>*> constraintsDirichletSet;

    /*Copies of constraintSet elements, but stored as non-const to enable application of constraints.*/
    std::vector<AffineConstraints<double>*> constraintsDirichletSet_nonconst;
    
private:
    /*User inputs*/
    userInputParameters<dim> userInputs;

};

template <int dim, int degree>
boundaryConditions<dim, degree>::boundaryConditions(const userInputParameters<dim>& _userInputs)
    : userInputs(_userInputs)
{
}

#endif
