// RUN: triton-opt "--triton-to-linalg=global-kernel=false named-ops=True" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @tla_scalar_sqrt_f32
// CHECK-NOT: tensor<
// CHECK: math.sqrt %{{.*}} : f32
// CHECK-NOT: tensor<
// CHECK: return
module attributes {hacc.mode = "tla", hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tla_scalar_sqrt_f32(%x: f32) -> f32 {
    %r = math.sqrt %x : f32
    tt.return %r : f32
  }
}

// -----

// CHECK-LABEL: func.func @normal_scalar_sqrt_f32
// CHECK: tensor.insert
// CHECK: math.sqrt %{{.*}} : tensor<1xf32>
// CHECK: tensor.extract
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @normal_scalar_sqrt_f32(%x: f32) -> f32 {
    %r = math.sqrt %x : f32
    tt.return %r : f32
  }
}

// -----

// CHECK-LABEL: func.func @tla_scalar_sqrt_f16
// CHECK: tensor.insert
// CHECK: math.sqrt %{{.*}} : tensor<1xf16>
// CHECK: tensor.extract
// CHECK: return
module attributes {hacc.mode = "tla", hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @tla_scalar_sqrt_f16(%x: f16) -> f16 {
    %r = math.sqrt %x : f16
    tt.return %r : f16
  }
}
