#ifndef INCLUDE_PROBLEMFLAGS_H_
#define INCLUDE_PROBLEMFLAGS_H_


#include "userInputParameters.h"

using namespace dealii;

/**
 * This class deals with the various flags in the problem set. Add more comments later
 */
template <int dim>
class problemFlags {
public:
    problemFlags(const userInputParameters<dim>& _userInputs);

    /*Flag used to see if time stepping is neccessary*/
    bool isTimeDependentBVP;

    /*Flag used to mark problems with Elliptic fields*/
    bool isEllipticBVP; //Currently unused

    /*Flag used to mark problems with at least one explicit field*/
    bool hasExplicitEquation;

    /*Flag used to mark problems with at least one scalar explicit field*/
    bool hasScalarExplicitEquation;

    /*FLag sed to mark problems with at least one vector explicit field*/
    bool hasVectorExplicitEquation;

    /*Flag used to mark problems with at least one nonexplicit field*/
    bool hasNonExplicitEquation;

    /*Flag used to mark problems with at least one postprocessed field*/
    bool hasPostprocessedField;

    /*Flag used to mark problems with at least one scalar postprocessed field*/
    bool hasScalarPostprocessedField;

    /*Flag used to mark problems with at least one vector postprocessed field*/
    bool hasVectorPostprocessedField;

private:
    /*User inputs*/
    userInputParameters<dim> userInputs;
};

template <int dim>
problemFlags<dim>::problemFlags(const userInputParameters<dim>& _userInputs)
    : userInputs(_userInputs)
    , isTimeDependentBVP(false)
    , isEllipticBVP(false)
    , hasExplicitEquation(false)
    , hasScalarExplicitEquation(false)
    , hasVectorExplicitEquation(false)
    , hasNonExplicitEquation(false)
    , hasPostprocessedField(false)
    , hasScalarPostprocessedField(false)
    , hasVectorPostprocessedField(false)
{
}


#endif
