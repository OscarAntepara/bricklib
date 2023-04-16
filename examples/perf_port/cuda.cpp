#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unistd.h>
#include "d3_stencils.h"
#include "brick.h"
#include "stencils/stencils.h"

namespace {
const char *const shortopt = "s:hb";
const char *help = "Running Stencil Performance Tests with %s\n\n"
                   "Program options\n"
                   "  Stencil type, just star or cube shapes\n"
                   "  -s: star or cube\n"
                   "Example usage:\n"
                   "  %s -s star\n"
                   "  %s -s cube\n"
                   "To run the tests with different domain size:\n"
                   "Go to macros_coeffs.h and change the domain size \n";
} // namespace

int main(int argc, char** argv) {

  int c;
  int sel = 0;
  std::string stencil_type;
  while ((c = getopt(argc, argv, shortopt)) != -1) {
    switch (c) {
    case 's':
      stencil_type = optarg;
      sel = sel != 0 ? -2 : 2;
      break;
    default:
      printf("Unknown options %c\n", c);
    case 'h':
      printf(help, "cuda", argv[0], argv[0]);
      sel = sel != 0 ? -2 : -1;
      exit(0);
    }
  }

  if (stencil_type=="star"){
    d3_stencils_star_cuda();
  }else if(stencil_type=="cube") {
#if defined (STENCIL_RADIUS_3) || (STENCIL_RADIUS_4)
    printf("Radius is not valid for this experiment \n");
    exit(0);
#endif
    d3_stencils_cube_cuda();
  }else{
    printf("No stencil shape selected. Please run executable with -h for more info \n");
  }
  return 0;
}
