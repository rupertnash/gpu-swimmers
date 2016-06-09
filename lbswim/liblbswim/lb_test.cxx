#include "Lattice.h"
#include "target/targetpp.h"
#include <chrono>
#include <iostream>
#include <unistd.h>

typedef std::chrono::high_resolution_clock Clock;

class LBTest {
public :
  LBTest(size_t box, size_t steps);
  ~LBTest();
  void Reset();
  int Run();
private:
  Lattice* lat;
  Shape shape;
  size_t nsteps;
};

TARGET_KERNEL_DECLARE(SetupK, 3, TARGET_DEFAULT_VVL, LDView);


LBTest::LBTest(size_t box, size_t steps) : shape(box, box, box), nsteps(steps) {
  lat = new Lattice(shape, 0.5, 0.5);
  Reset();
}

LBTest::~LBTest() {
  delete lat;
}

void LBTest::Reset() {
  SetupK k(shape);
  k(lat->data.Device());
  target::synchronize();
  lat->InitFromHydro();
  target::synchronize();
}
  
TARGET_KERNEL_DEFINE(SetupK, LDView data) {
  FOR_TLP(thr) {
    auto rho = thr.GetCurrentElements(data.rho);
    auto u = thr.GetCurrentElements(data.u);
    auto force = thr.GetCurrentElements(data.force);
    FOR_ILP(i) {
      rho[i][0] = 1.0;
      for (size_t d=0; d<3; d++) {
	u[i][d] = 0.0;
	force[i][d] = 0.0;
      }
    }
  }
}

int LBTest::Run() {
  // warm up
  lat->Step();
  
  auto t0 = Clock::now();
  for (auto i = 0; i<nsteps; ++i)
    lat->Step();
  
  auto t1 = Clock::now();
  auto dt = t1 - t0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
}

int main(int argc, char**argv) {
  size_t box = 32;
  size_t steps = 10;
  int opt;
  while ((opt = getopt(argc, argv, "b:s:")) != -1) {
    switch (opt) {
    case 'b':
      box = atoi(optarg);
      break;
    case 's':
      steps = atoi(optarg);
      break;
    default:
      std::cerr << "Usage: " << argv[0] << " [-b boxsize] [-s] nSteps" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  std::cout << "Box size = " << box << std::endl;
  std::cout << "Number of steps = " << steps << std::endl;
  LBTest t(100, 10);
  auto ms = t.Run();
  std::cout << "Run time / ms = " << ms << std::endl;
}
