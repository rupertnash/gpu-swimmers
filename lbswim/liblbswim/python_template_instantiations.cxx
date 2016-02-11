#include "SharedItem.hpp"
#include "SharedArray.hpp"
#include "Lists.h"
#include "Fields.h"
#include "SwimmerArray.h"

template class SharedItem<ScalarList>;
template class SharedItem<VectorList>;

template class SharedItem<ScalarField>;
template class SharedItem<VectorField>;
template class SharedItem<DistField>;

template class SharedItem<RandList>;

template class SharedItem<CommonParams>;
template class SharedItem<LBParams>;
