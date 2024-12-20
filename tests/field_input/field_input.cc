#include "catch.hpp"

#include <chrono>
#include <field_input/IntegrationTools/PField.hh>
#include <field_input/read_vtk.h>

TEST_CASE("Unstructured vtk File read-in")
{
  SECTION("2D")
  {
    auto start = std::chrono::high_resolution_clock::now();

    ReadVTK<2, double> test("field_input/allenCahn.vtk");

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();

    using ScalarField = PRISMS::PField<double *, double, 2>;
    using Body        = PRISMS::Body<double *, 2>;
    Body body;

    // REQUIRE_THROWS_AS(body.read_vtk("invalid"), std::runtime_error);

    body.read_vtk("field_input/allenCahn.vtk");

    ScalarField n_field  = body.find_scalar_field("n");
    ScalarField nx_field = body.find_scalar_field("mg_n");
    ScalarField test3    = body.find_scalar_field("f_tot");

    end = std::chrono::high_resolution_clock::now();

    elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    for (auto x : {0.0, 5.0, 10.0})
      {
        for (auto y : {0.0, 5.0, 10.0})
          {
            double point[2] = {x, y};
            // REQUIRE(n_field(point) == x);
            // REQUIRE(nx_field(point) == 1.0);
          }
      }

    REQUIRE_THROWS_AS(body.find_scalar_field("invalid"), std::invalid_argument);
  }

  SECTION("3D")
  {
    using ScalarField = PRISMS::PField<double *, double, 3>;
    using Body        = PRISMS::Body<double *, 3>;
    Body body;

    REQUIRE_THROWS_AS(body.read_vtk("invalid"), std::runtime_error);

    body.read_vtk("field_input/test_3D_1degree.vtk");

    ScalarField n_field  = body.find_scalar_field("n");
    ScalarField nx_field = body.find_scalar_field("|nx|");

    for (auto x : {0.0, 5.0, 10.0})
      {
        for (auto y : {0.0, 5.0, 10.0})
          {
            for (auto z : {0.0, 5.0, 10.0})
              {
                double point[3] = {x, y, z};
                REQUIRE(n_field(point) == x);
                REQUIRE(nx_field(point) == 1.0);
              }
          }
      }

    REQUIRE_THROWS_AS(body.find_scalar_field("invalid"), std::invalid_argument);
  }
}

TEST_CASE("Rectilinear vtk File read-in")
{}