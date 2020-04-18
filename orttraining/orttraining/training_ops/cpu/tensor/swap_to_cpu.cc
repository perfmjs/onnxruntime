// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/identity_op.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    SwapToCPU,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);
}
}