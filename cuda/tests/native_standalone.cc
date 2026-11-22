#include <nanobind/nanobind.h>
void RegisterNativeCuda(nanobind::module_&);
NB_MODULE(_cuda_native,module){RegisterNativeCuda(module);}
