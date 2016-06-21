#include "target/SharedItem.hpp"
#include "target/SharedNdArray.hpp"
#include "Lists.h"
#include "Fields.h"
#include "SwimmerArray.h"

template class target::SharedItem<ScalarList>;
template class target::SharedItem<VectorList>;

template class target::SharedItem<ScalarField>;
template class target::SharedItem<VectorField>;
template class target::SharedItem<DistField>;

template class target::SharedItem<RandList>;

template class target::SharedItem<CommonParams>;
template class target::SharedItem<LBParams>;
