module @jit_run_scan attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<801x801x801xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<801x800x801xf32> {tf.aliasing_output = 1 : i32}, %arg2: tensor<800x801x801xf32> {tf.aliasing_output = 2 : i32}, %arg3: tensor<800x800x801xf32> {tf.aliasing_output = 3 : i32}, %arg4: tensor<800x801x801xf32> {tf.aliasing_output = 4 : i32}, %arg5: tensor<801x800x801xf32> {tf.aliasing_output = 5 : i32}, %arg6: tensor<800x24x801xf32> {tf.aliasing_output = 6 : i32}, %arg7: tensor<24x800x801xf32> {tf.aliasing_output = 7 : i32}, %arg8: tensor<24x801x801xf32> {tf.aliasing_output = 8 : i32}, %arg9: tensor<800x801x24xf32> {tf.aliasing_output = 9 : i32}, %arg10: tensor<801x800x24xf32> {tf.aliasing_output = 10 : i32}, %arg11: tensor<801x24x801xf32> {tf.aliasing_output = 11 : i32}, %arg12: tensor<801x24x801xf32> {tf.aliasing_output = 12 : i32}, %arg13: tensor<24x801x801xf32> {tf.aliasing_output = 13 : i32}, %arg14: tensor<24x800x801xf32> {tf.aliasing_output = 14 : i32}, %arg15: tensor<801x800x24xf32> {tf.aliasing_output = 15 : i32}, %arg16: tensor<800x801x24xf32> {tf.aliasing_output = 16 : i32}, %arg17: tensor<800x24x801xf32> {tf.aliasing_output = 17 : i32}, %arg18: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 18 : i32}, %arg19: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 19 : i32}, %arg20: tensor<0xi32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 20 : i32}, %arg21: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 21 : i32}, %arg22: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 22 : i32}, %arg23: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 23 : i32}, %arg24: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 24 : i32}, %arg25: tensor<0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 25 : i32}, %arg26: tensor<0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 26 : i32}, %arg27: tensor<0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 27 : i32}, %arg28: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>, tf.aliasing_output = 28 : i32}, %arg29: tensor<i32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>, tf.aliasing_output = 29 : i32}, %arg30: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg31: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg32: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg33: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg34: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg35: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg36: tensor<f32>, %arg37: tensor<801x801x801xf32>, %arg38: tensor<f32>, %arg39: tensor<801x800x801xf32>, %arg40: tensor<f32>, %arg41: tensor<800x801x801xf32>) -> (tensor<801x801x801xf32> {jax.result_info = "result.ex"}, tensor<801x800x801xf32> {jax.result_info = "result.ey"}, tensor<800x801x801xf32> {jax.result_info = "result.ez"}, tensor<800x800x801xf32> {jax.result_info = "result.hx"}, tensor<800x801x801xf32> {jax.result_info = "result.hy"}, tensor<801x800x801xf32> {jax.result_info = "result.hz"}, tensor<800x24x801xf32> {jax.result_info = "result.cpml_psi_h_terms[0]"}, tensor<24x800x801xf32> {jax.result_info = "result.cpml_psi_h_terms[1]"}, tensor<24x801x801xf32> {jax.result_info = "result.cpml_psi_h_terms[2]"}, tensor<800x801x24xf32> {jax.result_info = "result.cpml_psi_h_terms[3]"}, tensor<801x800x24xf32> {jax.result_info = "result.cpml_psi_h_terms[4]"}, tensor<801x24x801xf32> {jax.result_info = "result.cpml_psi_h_terms[5]"}, tensor<801x24x801xf32> {jax.result_info = "result.cpml_psi_e_terms[0]"}, tensor<24x801x801xf32> {jax.result_info = "result.cpml_psi_e_terms[1]"}, tensor<24x800x801xf32> {jax.result_info = "result.cpml_psi_e_terms[2]"}, tensor<801x800x24xf32> {jax.result_info = "result.cpml_psi_e_terms[3]"}, tensor<800x801x24xf32> {jax.result_info = "result.cpml_psi_e_terms[4]"}, tensor<800x24x801xf32> {jax.result_info = "result.cpml_psi_e_terms[5]"}, tensor<0x0xf32> {jax.result_info = "result.powers"}, tensor<0x0xf32> {jax.result_info = "result.timestamps"}, tensor<0xi32> {jax.result_info = "result.counts"}, tensor<0x0xf32> {jax.result_info = "result.freq_flux_re"}, tensor<0x0xf32> {jax.result_info = "result.freq_flux_im"}, tensor<0x0xf32> {jax.result_info = "result.freq_phase_re"}, tensor<0x0xf32> {jax.result_info = "result.freq_phase_im"}, tensor<0xf32> {jax.result_info = "result.dft_vec_re"}, tensor<0xf32> {jax.result_info = "result.dft_vec_im"}, tensor<0xf32> {jax.result_info = "result.dft_weight_sum"}, tensor<f32> {jax.result_info = "result.t"}, tensor<i32> {jax.result_info = "result.current_step"}) {
    %cst = stablehlo.constant dense<1.46363323E-16> : tensor<f32>
    %c = stablehlo.constant dense_resource<__elided__> : tensor<6x3xi32>
    %cst_0 = stablehlo.constant dense<> : tensor<0xf32>
    %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_2 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_3 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_4 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_5 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_6 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_7 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_8 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_9 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_10 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_11 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_12 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_13 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_14 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_15 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_16 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_17 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_18 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %c_19 = stablehlo.constant dense_resource<__elided__> : tensor<6x5xi32>
    %c_20 = stablehlo.constant dense_resource<__elided__> : tensor<6x3xi32>
    %cst_21 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_22 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_23 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_24 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_25 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_26 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_27 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_28 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_29 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_30 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_31 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_32 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_33 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_34 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_35 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_36 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_37 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_38 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %c_39 = stablehlo.constant dense_resource<__elided__> : tensor<6x5xi32>
    %c_40 = stablehlo.constant dense<1> : tensor<ui32>
    %c_41 = stablehlo.constant dense<1> : tensor<ui32>
    %0 = stablehlo.partition_id : tensor<ui32>
    %1 = stablehlo.divide %0, %c_40 : tensor<ui32>
    %2 = stablehlo.remainder %1, %c_41 : tensor<ui32>
    %3 = stablehlo.convert %2 : (tensor<ui32>) -> tensor<i32>
    %c_42 = stablehlo.constant dense<801> : tensor<i32>
    %4 = stablehlo.multiply %3, %c_42 : tensor<i32>
    %5 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_43 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_43, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_44 = stablehlo.constant dense<800> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_44, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %9 = stablehlo.compare  LT, %5, %8,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %10 = stablehlo.and %7, %9 : tensor<800xi1>
    %11 = stablehlo.reshape %10 : (tensor<800xi1>) -> tensor<800x1x1xi1>
    %c_45 = stablehlo.constant dense<true> : tensor<i1>
    %12 = stablehlo.broadcast_in_dim %c_45, dims = [] : (tensor<i1>) -> tensor<800x1x1xi1>
    %13 = stablehlo.and %12, %11 : tensor<800x1x1xi1>
    %14 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_46 = stablehlo.constant dense<0> : tensor<i32>
    %15 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %16 = stablehlo.compare  GE, %14, %15,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_47 = stablehlo.constant dense<801> : tensor<i32>
    %17 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %18 = stablehlo.compare  LT, %14, %17,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %19 = stablehlo.and %16, %18 : tensor<801xi1>
    %20 = stablehlo.reshape %19 : (tensor<801xi1>) -> tensor<1x801x1xi1>
    %21 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<800x1x1xi1>) -> tensor<800x801x1xi1>
    %22 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<1x801x1xi1>) -> tensor<800x801x1xi1>
    %23 = stablehlo.and %21, %22 : tensor<800x801x1xi1>
    %24 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_48 = stablehlo.constant dense<12> : tensor<i32>
    %25 = stablehlo.broadcast_in_dim %c_48, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %26 = stablehlo.compare  LT, %24, %25,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_49 = stablehlo.constant dense<12> : tensor<i32>
    %27 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %28 = stablehlo.subtract %24, %27 : tensor<24xi32>
    %c_50 = stablehlo.constant dense<800> : tensor<i32>
    %29 = stablehlo.broadcast_in_dim %c_50, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %30 = stablehlo.add %28, %29 : tensor<24xi32>
    %c_51 = stablehlo.constant dense<12> : tensor<i32>
    %31 = stablehlo.broadcast_in_dim %c_51, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %32 = stablehlo.subtract %30, %31 : tensor<24xi32>
    %33 = call @_where(%26, %24, %32) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_52 = stablehlo.constant dense<0> : tensor<i32>
    %34 = stablehlo.broadcast_in_dim %c_52, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %35 = stablehlo.compare  GE, %33, %34,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_53 = stablehlo.constant dense<800> : tensor<i32>
    %36 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %37 = stablehlo.compare  LT, %33, %36,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %38 = stablehlo.and %35, %37 : tensor<24xi1>
    %39 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %40 = stablehlo.compare  GE, %33, %39,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_54 = stablehlo.constant dense<801> : tensor<i32>
    %41 = stablehlo.add %4, %c_54 : tensor<i32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %43 = stablehlo.compare  LT, %33, %42,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %44 = stablehlo.and %40, %43 : tensor<24xi1>
    %45 = stablehlo.and %38, %44 : tensor<24xi1>
    %46 = stablehlo.reshape %45 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %47 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2] : (tensor<800x801x1xi1>) -> tensor<800x801x24xi1>
    %48 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<800x801x24xi1>
    %49 = stablehlo.and %47, %48 : tensor<800x801x24xi1>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f32>) -> tensor<800x801x24xf32>
    %51 = call @_where_20(%49, %arg9, %50) : (tensor<800x801x24xi1>, tensor<800x801x24xf32>, tensor<800x801x24xf32>) -> tensor<800x801x24xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [1, 2, 3] : (tensor<800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %c_56 = stablehlo.constant dense<1> : tensor<ui32>
    %c_57 = stablehlo.constant dense<1> : tensor<ui32>
    %53 = stablehlo.partition_id : tensor<ui32>
    %54 = stablehlo.divide %53, %c_56 : tensor<ui32>
    %55 = stablehlo.remainder %54, %c_57 : tensor<ui32>
    %56 = stablehlo.convert %55 : (tensor<ui32>) -> tensor<i32>
    %c_58 = stablehlo.constant dense<801> : tensor<i32>
    %57 = stablehlo.multiply %56, %c_58 : tensor<i32>
    %58 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_59 = stablehlo.constant dense<0> : tensor<i32>
    %59 = stablehlo.broadcast_in_dim %c_59, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %60 = stablehlo.compare  GE, %58, %59,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_60 = stablehlo.constant dense<801> : tensor<i32>
    %61 = stablehlo.broadcast_in_dim %c_60, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %62 = stablehlo.compare  LT, %58, %61,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %63 = stablehlo.and %60, %62 : tensor<801xi1>
    %64 = stablehlo.reshape %63 : (tensor<801xi1>) -> tensor<801x1x1xi1>
    %c_61 = stablehlo.constant dense<true> : tensor<i1>
    %65 = stablehlo.broadcast_in_dim %c_61, dims = [] : (tensor<i1>) -> tensor<801x1x1xi1>
    %66 = stablehlo.and %65, %64 : tensor<801x1x1xi1>
    %67 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_62 = stablehlo.constant dense<0> : tensor<i32>
    %68 = stablehlo.broadcast_in_dim %c_62, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %69 = stablehlo.compare  GE, %67, %68,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_63 = stablehlo.constant dense<800> : tensor<i32>
    %70 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %71 = stablehlo.compare  LT, %67, %70,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %72 = stablehlo.and %69, %71 : tensor<800xi1>
    %73 = stablehlo.reshape %72 : (tensor<800xi1>) -> tensor<1x800x1xi1>
    %74 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2] : (tensor<801x1x1xi1>) -> tensor<801x800x1xi1>
    %75 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2] : (tensor<1x800x1xi1>) -> tensor<801x800x1xi1>
    %76 = stablehlo.and %74, %75 : tensor<801x800x1xi1>
    %77 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_64 = stablehlo.constant dense<12> : tensor<i32>
    %78 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %79 = stablehlo.compare  LT, %77, %78,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_65 = stablehlo.constant dense<12> : tensor<i32>
    %80 = stablehlo.broadcast_in_dim %c_65, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %81 = stablehlo.subtract %77, %80 : tensor<24xi32>
    %c_66 = stablehlo.constant dense<800> : tensor<i32>
    %82 = stablehlo.broadcast_in_dim %c_66, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %83 = stablehlo.add %81, %82 : tensor<24xi32>
    %c_67 = stablehlo.constant dense<12> : tensor<i32>
    %84 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %85 = stablehlo.subtract %83, %84 : tensor<24xi32>
    %86 = call @_where(%79, %77, %85) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_68 = stablehlo.constant dense<0> : tensor<i32>
    %87 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %88 = stablehlo.compare  GE, %86, %87,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_69 = stablehlo.constant dense<800> : tensor<i32>
    %89 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %90 = stablehlo.compare  LT, %86, %89,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %91 = stablehlo.and %88, %90 : tensor<24xi1>
    %92 = stablehlo.broadcast_in_dim %57, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %93 = stablehlo.compare  GE, %86, %92,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_70 = stablehlo.constant dense<801> : tensor<i32>
    %94 = stablehlo.add %57, %c_70 : tensor<i32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %96 = stablehlo.compare  LT, %86, %95,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %97 = stablehlo.and %93, %96 : tensor<24xi1>
    %98 = stablehlo.and %91, %97 : tensor<24xi1>
    %99 = stablehlo.reshape %98 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %100 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2] : (tensor<801x800x1xi1>) -> tensor<801x800x24xi1>
    %101 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<801x800x24xi1>
    %102 = stablehlo.and %100, %101 : tensor<801x800x24xi1>
    %cst_71 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %103 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<f32>) -> tensor<801x800x24xf32>
    %104 = call @_where_32(%102, %arg10, %103) : (tensor<801x800x24xi1>, tensor<801x800x24xf32>, tensor<801x800x24xf32>) -> tensor<801x800x24xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [1, 2, 3] : (tensor<801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %c_72 = stablehlo.constant dense<1> : tensor<ui32>
    %c_73 = stablehlo.constant dense<1> : tensor<ui32>
    %106 = stablehlo.partition_id : tensor<ui32>
    %107 = stablehlo.divide %106, %c_72 : tensor<ui32>
    %108 = stablehlo.remainder %107, %c_73 : tensor<ui32>
    %109 = stablehlo.convert %108 : (tensor<ui32>) -> tensor<i32>
    %c_74 = stablehlo.constant dense<801> : tensor<i32>
    %110 = stablehlo.multiply %109, %c_74 : tensor<i32>
    %111 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_75 = stablehlo.constant dense<0> : tensor<i32>
    %112 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %113 = stablehlo.compare  GE, %111, %112,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_76 = stablehlo.constant dense<801> : tensor<i32>
    %114 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %115 = stablehlo.compare  LT, %111, %114,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %116 = stablehlo.and %113, %115 : tensor<801xi1>
    %117 = stablehlo.reshape %116 : (tensor<801xi1>) -> tensor<801x1x1xi1>
    %c_77 = stablehlo.constant dense<true> : tensor<i1>
    %118 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i1>) -> tensor<801x1x1xi1>
    %119 = stablehlo.and %118, %117 : tensor<801x1x1xi1>
    %120 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_78 = stablehlo.constant dense<0> : tensor<i32>
    %121 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %122 = stablehlo.compare  GE, %120, %121,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_79 = stablehlo.constant dense<800> : tensor<i32>
    %123 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %124 = stablehlo.compare  LT, %120, %123,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %125 = stablehlo.and %122, %124 : tensor<800xi1>
    %126 = stablehlo.reshape %125 : (tensor<800xi1>) -> tensor<1x800x1xi1>
    %127 = stablehlo.broadcast_in_dim %119, dims = [0, 1, 2] : (tensor<801x1x1xi1>) -> tensor<801x800x1xi1>
    %128 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2] : (tensor<1x800x1xi1>) -> tensor<801x800x1xi1>
    %129 = stablehlo.and %127, %128 : tensor<801x800x1xi1>
    %130 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_80 = stablehlo.constant dense<12> : tensor<i32>
    %131 = stablehlo.broadcast_in_dim %c_80, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %132 = stablehlo.compare  LT, %130, %131,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_81 = stablehlo.constant dense<12> : tensor<i32>
    %133 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %134 = stablehlo.subtract %130, %133 : tensor<24xi32>
    %c_82 = stablehlo.constant dense<801> : tensor<i32>
    %135 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %136 = stablehlo.add %134, %135 : tensor<24xi32>
    %c_83 = stablehlo.constant dense<12> : tensor<i32>
    %137 = stablehlo.broadcast_in_dim %c_83, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %138 = stablehlo.subtract %136, %137 : tensor<24xi32>
    %139 = call @_where(%132, %130, %138) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_84 = stablehlo.constant dense<0> : tensor<i32>
    %140 = stablehlo.broadcast_in_dim %c_84, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %141 = stablehlo.compare  GE, %139, %140,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_85 = stablehlo.constant dense<801> : tensor<i32>
    %142 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %143 = stablehlo.compare  LT, %139, %142,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %144 = stablehlo.and %141, %143 : tensor<24xi1>
    %145 = stablehlo.broadcast_in_dim %110, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %146 = stablehlo.compare  GE, %139, %145,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_86 = stablehlo.constant dense<801> : tensor<i32>
    %147 = stablehlo.add %110, %c_86 : tensor<i32>
    %148 = stablehlo.broadcast_in_dim %147, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %149 = stablehlo.compare  LT, %139, %148,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %150 = stablehlo.and %146, %149 : tensor<24xi1>
    %151 = stablehlo.and %144, %150 : tensor<24xi1>
    %152 = stablehlo.reshape %151 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %153 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2] : (tensor<801x800x1xi1>) -> tensor<801x800x24xi1>
    %154 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<801x800x24xi1>
    %155 = stablehlo.and %153, %154 : tensor<801x800x24xi1>
    %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %156 = stablehlo.broadcast_in_dim %cst_87, dims = [] : (tensor<f32>) -> tensor<801x800x24xf32>
    %157 = call @_where_32(%155, %arg15, %156) : (tensor<801x800x24xi1>, tensor<801x800x24xf32>, tensor<801x800x24xf32>) -> tensor<801x800x24xf32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [1, 2, 3] : (tensor<801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %c_88 = stablehlo.constant dense<1> : tensor<ui32>
    %c_89 = stablehlo.constant dense<1> : tensor<ui32>
    %159 = stablehlo.partition_id : tensor<ui32>
    %160 = stablehlo.divide %159, %c_88 : tensor<ui32>
    %161 = stablehlo.remainder %160, %c_89 : tensor<ui32>
    %162 = stablehlo.convert %161 : (tensor<ui32>) -> tensor<i32>
    %c_90 = stablehlo.constant dense<801> : tensor<i32>
    %163 = stablehlo.multiply %162, %c_90 : tensor<i32>
    %164 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_91 = stablehlo.constant dense<0> : tensor<i32>
    %165 = stablehlo.broadcast_in_dim %c_91, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %166 = stablehlo.compare  GE, %164, %165,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_92 = stablehlo.constant dense<800> : tensor<i32>
    %167 = stablehlo.broadcast_in_dim %c_92, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %168 = stablehlo.compare  LT, %164, %167,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %169 = stablehlo.and %166, %168 : tensor<800xi1>
    %170 = stablehlo.reshape %169 : (tensor<800xi1>) -> tensor<800x1x1xi1>
    %c_93 = stablehlo.constant dense<true> : tensor<i1>
    %171 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i1>) -> tensor<800x1x1xi1>
    %172 = stablehlo.and %171, %170 : tensor<800x1x1xi1>
    %173 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_94 = stablehlo.constant dense<0> : tensor<i32>
    %174 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %175 = stablehlo.compare  GE, %173, %174,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_95 = stablehlo.constant dense<801> : tensor<i32>
    %176 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %177 = stablehlo.compare  LT, %173, %176,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %178 = stablehlo.and %175, %177 : tensor<801xi1>
    %179 = stablehlo.reshape %178 : (tensor<801xi1>) -> tensor<1x801x1xi1>
    %180 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2] : (tensor<800x1x1xi1>) -> tensor<800x801x1xi1>
    %181 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2] : (tensor<1x801x1xi1>) -> tensor<800x801x1xi1>
    %182 = stablehlo.and %180, %181 : tensor<800x801x1xi1>
    %183 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_96 = stablehlo.constant dense<12> : tensor<i32>
    %184 = stablehlo.broadcast_in_dim %c_96, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %185 = stablehlo.compare  LT, %183, %184,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_97 = stablehlo.constant dense<12> : tensor<i32>
    %186 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %187 = stablehlo.subtract %183, %186 : tensor<24xi32>
    %c_98 = stablehlo.constant dense<801> : tensor<i32>
    %188 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %189 = stablehlo.add %187, %188 : tensor<24xi32>
    %c_99 = stablehlo.constant dense<12> : tensor<i32>
    %190 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %191 = stablehlo.subtract %189, %190 : tensor<24xi32>
    %192 = call @_where(%185, %183, %191) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_100 = stablehlo.constant dense<0> : tensor<i32>
    %193 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %194 = stablehlo.compare  GE, %192, %193,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_101 = stablehlo.constant dense<801> : tensor<i32>
    %195 = stablehlo.broadcast_in_dim %c_101, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %196 = stablehlo.compare  LT, %192, %195,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %197 = stablehlo.and %194, %196 : tensor<24xi1>
    %198 = stablehlo.broadcast_in_dim %163, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %199 = stablehlo.compare  GE, %192, %198,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_102 = stablehlo.constant dense<801> : tensor<i32>
    %200 = stablehlo.add %163, %c_102 : tensor<i32>
    %201 = stablehlo.broadcast_in_dim %200, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %202 = stablehlo.compare  LT, %192, %201,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %203 = stablehlo.and %199, %202 : tensor<24xi1>
    %204 = stablehlo.and %197, %203 : tensor<24xi1>
    %205 = stablehlo.reshape %204 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %206 = stablehlo.broadcast_in_dim %182, dims = [0, 1, 2] : (tensor<800x801x1xi1>) -> tensor<800x801x24xi1>
    %207 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<800x801x24xi1>
    %208 = stablehlo.and %206, %207 : tensor<800x801x24xi1>
    %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %209 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<800x801x24xf32>
    %210 = call @_where_20(%208, %arg16, %209) : (tensor<800x801x24xi1>, tensor<800x801x24xf32>, tensor<800x801x24xf32>) -> tensor<800x801x24xf32>
    %211 = stablehlo.broadcast_in_dim %210, dims = [1, 2, 3] : (tensor<800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %212 = stablehlo.iota dim = 0 : tensor<32xi32>
    %c_104 = stablehlo.constant dense<0> : tensor<i32>
    %213:77 = stablehlo.while(%iterArg = %212, %iterArg_105 = %cst, %iterArg_106 = %arg28, %iterArg_107 = %c, %iterArg_108 = %arg30, %iterArg_109 = %arg32, %iterArg_110 = %arg34, %iterArg_111 = %arg31, %iterArg_112 = %arg33, %iterArg_113 = %arg35, %iterArg_114 = %cst_0, %iterArg_115 = %cst_1, %iterArg_116 = %cst_2, %iterArg_117 = %cst_3, %iterArg_118 = %cst_4, %iterArg_119 = %cst_5, %iterArg_120 = %cst_6, %iterArg_121 = %cst_7, %iterArg_122 = %cst_8, %iterArg_123 = %cst_9, %iterArg_124 = %cst_10, %iterArg_125 = %cst_11, %iterArg_126 = %cst_12, %iterArg_127 = %cst_13, %iterArg_128 = %cst_14, %iterArg_129 = %cst_15, %iterArg_130 = %cst_16, %iterArg_131 = %cst_17, %iterArg_132 = %cst_18, %iterArg_133 = %c_19, %iterArg_134 = %c_20, %iterArg_135 = %arg36, %iterArg_136 = %arg38, %iterArg_137 = %arg40, %iterArg_138 = %arg37, %iterArg_139 = %arg39, %iterArg_140 = %arg41, %iterArg_141 = %cst_21, %iterArg_142 = %cst_22, %iterArg_143 = %cst_23, %iterArg_144 = %cst_24, %iterArg_145 = %cst_25, %iterArg_146 = %cst_26, %iterArg_147 = %cst_27, %iterArg_148 = %cst_28, %iterArg_149 = %cst_29, %iterArg_150 = %cst_30, %iterArg_151 = %cst_31, %iterArg_152 = %cst_32, %iterArg_153 = %cst_33, %iterArg_154 = %cst_34, %iterArg_155 = %cst_35, %iterArg_156 = %cst_36, %iterArg_157 = %cst_37, %iterArg_158 = %cst_38, %iterArg_159 = %c_39, %iterArg_160 = %c_104, %iterArg_161 = %arg0, %iterArg_162 = %arg1, %iterArg_163 = %arg2, %iterArg_164 = %arg3, %iterArg_165 = %arg4, %iterArg_166 = %arg5, %iterArg_167 = %arg6, %iterArg_168 = %arg7, %iterArg_169 = %arg8, %iterArg_170 = %52, %iterArg_171 = %105, %iterArg_172 = %arg11, %iterArg_173 = %arg12, %iterArg_174 = %arg13, %iterArg_175 = %arg14, %iterArg_176 = %158, %iterArg_177 = %211, %iterArg_178 = %arg17, %iterArg_179 = %arg28, %iterArg_180 = %arg29) : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<i32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>
    cond {
      %c_181 = stablehlo.constant dense<32> : tensor<i32>
      %226 = stablehlo.compare  LT, %iterArg_160, %c_181,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %226 : tensor<i1>
    } do {
      %226 = stablehlo.dynamic_slice %iterArg, %iterArg_160, sizes = [1] : (tensor<32xi32>, tensor<i32>) -> tensor<1xi32>
      %227 = stablehlo.reshape %226 : (tensor<1xi32>) -> tensor<i32>
      %228:20 = func.call @closed_call(%iterArg_105, %iterArg_106, %iterArg_107, %iterArg_108, %iterArg_109, %iterArg_110, %iterArg_111, %iterArg_112, %iterArg_113, %iterArg_114, %iterArg_115, %iterArg_116, %iterArg_117, %iterArg_118, %iterArg_119, %iterArg_120, %iterArg_121, %iterArg_122, %iterArg_123, %iterArg_124, %iterArg_125, %iterArg_126, %iterArg_127, %iterArg_128, %iterArg_129, %iterArg_130, %iterArg_131, %iterArg_132, %iterArg_133, %iterArg_134, %iterArg_135, %iterArg_136, %iterArg_137, %iterArg_138, %iterArg_139, %iterArg_140, %iterArg_141, %iterArg_142, %iterArg_143, %iterArg_144, %iterArg_145, %iterArg_146, %iterArg_147, %iterArg_148, %iterArg_149, %iterArg_150, %iterArg_151, %iterArg_152, %iterArg_153, %iterArg_154, %iterArg_155, %iterArg_156, %iterArg_157, %iterArg_158, %iterArg_159, %iterArg_161, %iterArg_162, %iterArg_163, %iterArg_164, %iterArg_165, %iterArg_166, %iterArg_167, %iterArg_168, %iterArg_169, %iterArg_170, %iterArg_171, %iterArg_172, %iterArg_173, %iterArg_174, %iterArg_175, %iterArg_176, %iterArg_177, %iterArg_178, %iterArg_179, %iterArg_180, %227) : (tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>, tensor<i32>) -> (tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>)
      %c_181 = stablehlo.constant dense<1> : tensor<i32>
      %229 = stablehlo.add %iterArg_160, %c_181 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_105, %iterArg_106, %iterArg_107, %iterArg_108, %iterArg_109, %iterArg_110, %iterArg_111, %iterArg_112, %iterArg_113, %iterArg_114, %iterArg_115, %iterArg_116, %iterArg_117, %iterArg_118, %iterArg_119, %iterArg_120, %iterArg_121, %iterArg_122, %iterArg_123, %iterArg_124, %iterArg_125, %iterArg_126, %iterArg_127, %iterArg_128, %iterArg_129, %iterArg_130, %iterArg_131, %iterArg_132, %iterArg_133, %iterArg_134, %iterArg_135, %iterArg_136, %iterArg_137, %iterArg_138, %iterArg_139, %iterArg_140, %iterArg_141, %iterArg_142, %iterArg_143, %iterArg_144, %iterArg_145, %iterArg_146, %iterArg_147, %iterArg_148, %iterArg_149, %iterArg_150, %iterArg_151, %iterArg_152, %iterArg_153, %iterArg_154, %iterArg_155, %iterArg_156, %iterArg_157, %iterArg_158, %iterArg_159, %229, %228#0, %228#1, %228#2, %228#3, %228#4, %228#5, %228#6, %228#7, %228#8, %228#9, %228#10, %228#11, %228#12, %228#13, %228#14, %228#15, %228#16, %228#17, %228#18, %228#19 : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<i32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>
    }
    %214 = stablehlo.slice %213#66 [0:1, 0:800, 0:801, 0:24] : (tensor<1x800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %215 = stablehlo.reshape %214 : (tensor<1x800x801x24xf32>) -> tensor<800x801x24xf32>
    %216 = "stablehlo.all_reduce"(%215) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %226 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %226 : tensor<f32>
    }) : (tensor<800x801x24xf32>) -> tensor<800x801x24xf32>
    %217 = stablehlo.slice %213#67 [0:1, 0:801, 0:800, 0:24] : (tensor<1x801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %218 = stablehlo.reshape %217 : (tensor<1x801x800x24xf32>) -> tensor<801x800x24xf32>
    %219 = "stablehlo.all_reduce"(%218) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %226 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %226 : tensor<f32>
    }) : (tensor<801x800x24xf32>) -> tensor<801x800x24xf32>
    %220 = stablehlo.slice %213#72 [0:1, 0:801, 0:800, 0:24] : (tensor<1x801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %221 = stablehlo.reshape %220 : (tensor<1x801x800x24xf32>) -> tensor<801x800x24xf32>
    %222 = "stablehlo.all_reduce"(%221) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %226 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %226 : tensor<f32>
    }) : (tensor<801x800x24xf32>) -> tensor<801x800x24xf32>
    %223 = stablehlo.slice %213#73 [0:1, 0:800, 0:801, 0:24] : (tensor<1x800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %224 = stablehlo.reshape %223 : (tensor<1x800x801x24xf32>) -> tensor<800x801x24xf32>
    %225 = "stablehlo.all_reduce"(%224) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %226 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %226 : tensor<f32>
    }) : (tensor<800x801x24xf32>) -> tensor<800x801x24xf32>
    return %213#57, %213#58, %213#59, %213#60, %213#61, %213#62, %213#63, %213#64, %213#65, %216, %219, %213#68, %213#69, %213#70, %213#71, %222, %225, %213#74, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %213#75, %213#76 : tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<800x801x24xf32>, tensor<801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<801x800x24xf32>, tensor<800x801x24xf32>, tensor<800x24x801xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0xi32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @_where(%arg0: tensor<24xi1>, %arg1: tensor<24xi32>, %arg2: tensor<24xi32>) -> tensor<24xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24xi1>, tensor<24xi32>
    return %0 : tensor<24xi32>
  }
  func.func private @_where_20(%arg0: tensor<800x801x24xi1>, %arg1: tensor<800x801x24xf32>, %arg2: tensor<800x801x24xf32>) -> tensor<800x801x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<800x801x24xi1>, tensor<800x801x24xf32>
    return %0 : tensor<800x801x24xf32>
  }
  func.func private @_where_32(%arg0: tensor<801x800x24xi1>, %arg1: tensor<801x800x24xf32>, %arg2: tensor<801x800x24xf32>) -> tensor<801x800x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<801x800x24xi1>, tensor<801x800x24xf32>
    return %0 : tensor<801x800x24xf32>
  }
  func.func private @closed_call(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<6x3xi32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>, %arg9: tensor<0xf32>, %arg10: tensor<1x24x1xf32>, %arg11: tensor<1x24x1xf32>, %arg12: tensor<1x24x1xf32>, %arg13: tensor<24x1x1xf32>, %arg14: tensor<24x1x1xf32>, %arg15: tensor<24x1x1xf32>, %arg16: tensor<24x1x1xf32>, %arg17: tensor<24x1x1xf32>, %arg18: tensor<24x1x1xf32>, %arg19: tensor<1x1x24xf32>, %arg20: tensor<1x1x24xf32>, %arg21: tensor<1x1x24xf32>, %arg22: tensor<1x1x24xf32>, %arg23: tensor<1x1x24xf32>, %arg24: tensor<1x1x24xf32>, %arg25: tensor<1x24x1xf32>, %arg26: tensor<1x24x1xf32>, %arg27: tensor<1x24x1xf32>, %arg28: tensor<6x5xi32>, %arg29: tensor<6x3xi32>, %arg30: tensor<f32>, %arg31: tensor<f32>, %arg32: tensor<f32>, %arg33: tensor<801x801x801xf32>, %arg34: tensor<801x800x801xf32>, %arg35: tensor<800x801x801xf32>, %arg36: tensor<1x24x1xf32>, %arg37: tensor<1x24x1xf32>, %arg38: tensor<1x24x1xf32>, %arg39: tensor<24x1x1xf32>, %arg40: tensor<24x1x1xf32>, %arg41: tensor<24x1x1xf32>, %arg42: tensor<24x1x1xf32>, %arg43: tensor<24x1x1xf32>, %arg44: tensor<24x1x1xf32>, %arg45: tensor<1x1x24xf32>, %arg46: tensor<1x1x24xf32>, %arg47: tensor<1x1x24xf32>, %arg48: tensor<1x1x24xf32>, %arg49: tensor<1x1x24xf32>, %arg50: tensor<1x1x24xf32>, %arg51: tensor<1x24x1xf32>, %arg52: tensor<1x24x1xf32>, %arg53: tensor<1x24x1xf32>, %arg54: tensor<6x5xi32>, %arg55: tensor<801x801x801xf32>, %arg56: tensor<801x800x801xf32>, %arg57: tensor<800x801x801xf32>, %arg58: tensor<800x800x801xf32>, %arg59: tensor<800x801x801xf32>, %arg60: tensor<801x800x801xf32>, %arg61: tensor<800x24x801xf32>, %arg62: tensor<24x800x801xf32>, %arg63: tensor<24x801x801xf32>, %arg64: tensor<1x800x801x24xf32>, %arg65: tensor<1x801x800x24xf32>, %arg66: tensor<801x24x801xf32>, %arg67: tensor<801x24x801xf32>, %arg68: tensor<24x801x801xf32>, %arg69: tensor<24x800x801xf32>, %arg70: tensor<1x801x800x24xf32>, %arg71: tensor<1x800x801x24xf32>, %arg72: tensor<800x24x801xf32>, %arg73: tensor<f32>, %arg74: tensor<i32>, %arg75: tensor<i32>) -> (tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg75, %c : tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<f32>
    %3 = stablehlo.add %arg1, %2 : tensor<f32>
    %c_0 = stablehlo.constant dense<1> : tensor<ui32>
    %c_1 = stablehlo.constant dense<1> : tensor<ui32>
    %4 = stablehlo.partition_id : tensor<ui32>
    %5 = stablehlo.divide %4, %c_0 : tensor<ui32>
    %6 = stablehlo.remainder %5, %c_1 : tensor<ui32>
    %7 = stablehlo.convert %6 : (tensor<ui32>) -> tensor<i32>
    %c_2 = stablehlo.constant dense<801> : tensor<i32>
    %8 = stablehlo.multiply %7, %c_2 : tensor<i32>
    %c_3 = stablehlo.constant dense<2> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.concatenate %9, %10, %11, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [1] : (tensor<3xi32>) -> tensor<1x3xi32>
    %14 = stablehlo.concatenate %13, %arg2, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
    %15 = stablehlo.slice %arg64 [0:1, 0:800, 0:801, 0:24] : (tensor<1x800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %16 = stablehlo.reshape %15 : (tensor<1x800x801x24xf32>) -> tensor<800x801x24xf32>
    %17 = stablehlo.slice %arg65 [0:1, 0:801, 0:800, 0:24] : (tensor<1x801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %18 = stablehlo.reshape %17 : (tensor<1x801x800x24xf32>) -> tensor<801x800x24xf32>
    %19 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_5 = stablehlo.constant dense<0> : tensor<i32>
    %20 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %21 = stablehlo.compare  GE, %19, %20,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_6 = stablehlo.constant dense<800> : tensor<i32>
    %22 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %23 = stablehlo.compare  LT, %19, %22,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %24 = stablehlo.and %21, %23 : tensor<800xi1>
    %25 = stablehlo.reshape %24 : (tensor<800xi1>) -> tensor<800x1x1xi1>
    %c_7 = stablehlo.constant dense<true> : tensor<i1>
    %26 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i1>) -> tensor<800x1x1xi1>
    %27 = stablehlo.and %26, %25 : tensor<800x1x1xi1>
    %28 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_8 = stablehlo.constant dense<12> : tensor<i32>
    %29 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %30 = stablehlo.compare  LT, %28, %29,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_9 = stablehlo.constant dense<12> : tensor<i32>
    %31 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %32 = stablehlo.subtract %28, %31 : tensor<24xi32>
    %c_10 = stablehlo.constant dense<800> : tensor<i32>
    %33 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %34 = stablehlo.add %32, %33 : tensor<24xi32>
    %c_11 = stablehlo.constant dense<12> : tensor<i32>
    %35 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %36 = stablehlo.subtract %34, %35 : tensor<24xi32>
    %37 = call @_where(%30, %28, %36) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_12 = stablehlo.constant dense<0> : tensor<i32>
    %38 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %39 = stablehlo.compare  GE, %37, %38,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_13 = stablehlo.constant dense<800> : tensor<i32>
    %40 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %41 = stablehlo.compare  LT, %37, %40,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %42 = stablehlo.and %39, %41 : tensor<24xi1>
    %43 = stablehlo.reshape %42 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %44 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2] : (tensor<800x1x1xi1>) -> tensor<800x24x1xi1>
    %45 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<800x24x1xi1>
    %46 = stablehlo.and %44, %45 : tensor<800x24x1xi1>
    %47 = stablehlo.iota dim = 0 : tensor<801xi32>
    %48 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %49 = stablehlo.add %47, %48 : tensor<801xi32>
    %c_14 = stablehlo.constant dense<0> : tensor<i32>
    %50 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %51 = stablehlo.compare  GE, %49, %50,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_15 = stablehlo.constant dense<801> : tensor<i32>
    %52 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %53 = stablehlo.compare  LT, %49, %52,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %54 = stablehlo.and %51, %53 : tensor<801xi1>
    %55 = stablehlo.reshape %54 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %56 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<800x24x1xi1>) -> tensor<800x24x801xi1>
    %57 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<800x24x801xi1>
    %58 = stablehlo.and %56, %57 : tensor<800x24x801xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x24x801xf32>
    %60 = call @_where_65(%58, %arg61, %59) : (tensor<800x24x801xi1>, tensor<800x24x801xf32>, tensor<800x24x801xf32>) -> tensor<800x24x801xf32>
    %61 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_16 = stablehlo.constant dense<12> : tensor<i32>
    %62 = stablehlo.broadcast_in_dim %c_16, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %63 = stablehlo.compare  LT, %61, %62,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_17 = stablehlo.constant dense<12> : tensor<i32>
    %64 = stablehlo.broadcast_in_dim %c_17, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %65 = stablehlo.subtract %61, %64 : tensor<24xi32>
    %c_18 = stablehlo.constant dense<800> : tensor<i32>
    %66 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %67 = stablehlo.add %65, %66 : tensor<24xi32>
    %c_19 = stablehlo.constant dense<12> : tensor<i32>
    %68 = stablehlo.broadcast_in_dim %c_19, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %69 = stablehlo.subtract %67, %68 : tensor<24xi32>
    %70 = call @_where(%63, %61, %69) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_20 = stablehlo.constant dense<0> : tensor<i32>
    %71 = stablehlo.broadcast_in_dim %c_20, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %72 = stablehlo.compare  GE, %70, %71,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_21 = stablehlo.constant dense<800> : tensor<i32>
    %73 = stablehlo.broadcast_in_dim %c_21, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %74 = stablehlo.compare  LT, %70, %73,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %75 = stablehlo.and %72, %74 : tensor<24xi1>
    %76 = stablehlo.reshape %75 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_22 = stablehlo.constant dense<true> : tensor<i1>
    %77 = stablehlo.broadcast_in_dim %c_22, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %78 = stablehlo.and %77, %76 : tensor<24x1x1xi1>
    %79 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_23 = stablehlo.constant dense<0> : tensor<i32>
    %80 = stablehlo.broadcast_in_dim %c_23, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %81 = stablehlo.compare  GE, %79, %80,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_24 = stablehlo.constant dense<800> : tensor<i32>
    %82 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %83 = stablehlo.compare  LT, %79, %82,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %84 = stablehlo.and %81, %83 : tensor<800xi1>
    %85 = stablehlo.reshape %84 : (tensor<800xi1>) -> tensor<1x800x1xi1>
    %86 = stablehlo.broadcast_in_dim %78, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x800x1xi1>
    %87 = stablehlo.broadcast_in_dim %85, dims = [0, 1, 2] : (tensor<1x800x1xi1>) -> tensor<24x800x1xi1>
    %88 = stablehlo.and %86, %87 : tensor<24x800x1xi1>
    %89 = stablehlo.iota dim = 0 : tensor<801xi32>
    %90 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %91 = stablehlo.add %89, %90 : tensor<801xi32>
    %c_25 = stablehlo.constant dense<0> : tensor<i32>
    %92 = stablehlo.broadcast_in_dim %c_25, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %93 = stablehlo.compare  GE, %91, %92,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_26 = stablehlo.constant dense<801> : tensor<i32>
    %94 = stablehlo.broadcast_in_dim %c_26, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %95 = stablehlo.compare  LT, %91, %94,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %96 = stablehlo.and %93, %95 : tensor<801xi1>
    %97 = stablehlo.reshape %96 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %98 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2] : (tensor<24x800x1xi1>) -> tensor<24x800x801xi1>
    %99 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<24x800x801xi1>
    %100 = stablehlo.and %98, %99 : tensor<24x800x801xi1>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %101 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<24x800x801xf32>
    %102 = call @_where_74(%100, %arg62, %101) : (tensor<24x800x801xi1>, tensor<24x800x801xf32>, tensor<24x800x801xf32>) -> tensor<24x800x801xf32>
    %103 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_28 = stablehlo.constant dense<12> : tensor<i32>
    %104 = stablehlo.broadcast_in_dim %c_28, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %105 = stablehlo.compare  LT, %103, %104,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_29 = stablehlo.constant dense<12> : tensor<i32>
    %106 = stablehlo.broadcast_in_dim %c_29, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %107 = stablehlo.subtract %103, %106 : tensor<24xi32>
    %c_30 = stablehlo.constant dense<800> : tensor<i32>
    %108 = stablehlo.broadcast_in_dim %c_30, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %109 = stablehlo.add %107, %108 : tensor<24xi32>
    %c_31 = stablehlo.constant dense<12> : tensor<i32>
    %110 = stablehlo.broadcast_in_dim %c_31, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %111 = stablehlo.subtract %109, %110 : tensor<24xi32>
    %112 = call @_where(%105, %103, %111) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_32 = stablehlo.constant dense<0> : tensor<i32>
    %113 = stablehlo.broadcast_in_dim %c_32, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %114 = stablehlo.compare  GE, %112, %113,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_33 = stablehlo.constant dense<800> : tensor<i32>
    %115 = stablehlo.broadcast_in_dim %c_33, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %116 = stablehlo.compare  LT, %112, %115,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %117 = stablehlo.and %114, %116 : tensor<24xi1>
    %118 = stablehlo.reshape %117 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_34 = stablehlo.constant dense<true> : tensor<i1>
    %119 = stablehlo.broadcast_in_dim %c_34, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %120 = stablehlo.and %119, %118 : tensor<24x1x1xi1>
    %121 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_35 = stablehlo.constant dense<0> : tensor<i32>
    %122 = stablehlo.broadcast_in_dim %c_35, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %123 = stablehlo.compare  GE, %121, %122,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_36 = stablehlo.constant dense<801> : tensor<i32>
    %124 = stablehlo.broadcast_in_dim %c_36, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %125 = stablehlo.compare  LT, %121, %124,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %126 = stablehlo.and %123, %125 : tensor<801xi1>
    %127 = stablehlo.reshape %126 : (tensor<801xi1>) -> tensor<1x801x1xi1>
    %128 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x801x1xi1>
    %129 = stablehlo.broadcast_in_dim %127, dims = [0, 1, 2] : (tensor<1x801x1xi1>) -> tensor<24x801x1xi1>
    %130 = stablehlo.and %128, %129 : tensor<24x801x1xi1>
    %131 = stablehlo.iota dim = 0 : tensor<801xi32>
    %132 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %133 = stablehlo.add %131, %132 : tensor<801xi32>
    %c_37 = stablehlo.constant dense<0> : tensor<i32>
    %134 = stablehlo.broadcast_in_dim %c_37, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %135 = stablehlo.compare  GE, %133, %134,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_38 = stablehlo.constant dense<800> : tensor<i32>
    %136 = stablehlo.broadcast_in_dim %c_38, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %137 = stablehlo.compare  LT, %133, %136,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %138 = stablehlo.and %135, %137 : tensor<801xi1>
    %139 = stablehlo.reshape %138 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %140 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<24x801x1xi1>) -> tensor<24x801x801xi1>
    %141 = stablehlo.broadcast_in_dim %139, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<24x801x801xi1>
    %142 = stablehlo.and %140, %141 : tensor<24x801x801xi1>
    %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %143 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<24x801x801xf32>
    %144 = call @_where_81(%142, %arg63, %143) : (tensor<24x801x801xi1>, tensor<24x801x801xf32>, tensor<24x801x801xf32>) -> tensor<24x801x801xf32>
    %145 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_40 = stablehlo.constant dense<0> : tensor<i32>
    %146 = stablehlo.broadcast_in_dim %c_40, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %147 = stablehlo.compare  GE, %145, %146,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_41 = stablehlo.constant dense<800> : tensor<i32>
    %148 = stablehlo.broadcast_in_dim %c_41, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %149 = stablehlo.compare  LT, %145, %148,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %150 = stablehlo.and %147, %149 : tensor<800xi1>
    %151 = stablehlo.reshape %150 : (tensor<800xi1>) -> tensor<800x1x1xi1>
    %c_42 = stablehlo.constant dense<true> : tensor<i1>
    %152 = stablehlo.broadcast_in_dim %c_42, dims = [] : (tensor<i1>) -> tensor<800x1x1xi1>
    %153 = stablehlo.and %152, %151 : tensor<800x1x1xi1>
    %154 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_43 = stablehlo.constant dense<0> : tensor<i32>
    %155 = stablehlo.broadcast_in_dim %c_43, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %156 = stablehlo.compare  GE, %154, %155,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_44 = stablehlo.constant dense<801> : tensor<i32>
    %157 = stablehlo.broadcast_in_dim %c_44, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %158 = stablehlo.compare  LT, %154, %157,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %159 = stablehlo.and %156, %158 : tensor<801xi1>
    %160 = stablehlo.reshape %159 : (tensor<801xi1>) -> tensor<1x801x1xi1>
    %161 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2] : (tensor<800x1x1xi1>) -> tensor<800x801x1xi1>
    %162 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2] : (tensor<1x801x1xi1>) -> tensor<800x801x1xi1>
    %163 = stablehlo.and %161, %162 : tensor<800x801x1xi1>
    %164 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_45 = stablehlo.constant dense<12> : tensor<i32>
    %165 = stablehlo.broadcast_in_dim %c_45, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %166 = stablehlo.compare  LT, %164, %165,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_46 = stablehlo.constant dense<12> : tensor<i32>
    %167 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %168 = stablehlo.subtract %164, %167 : tensor<24xi32>
    %c_47 = stablehlo.constant dense<800> : tensor<i32>
    %169 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %170 = stablehlo.add %168, %169 : tensor<24xi32>
    %c_48 = stablehlo.constant dense<12> : tensor<i32>
    %171 = stablehlo.broadcast_in_dim %c_48, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %172 = stablehlo.subtract %170, %171 : tensor<24xi32>
    %173 = call @_where(%166, %164, %172) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_49 = stablehlo.constant dense<0> : tensor<i32>
    %174 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %175 = stablehlo.compare  GE, %173, %174,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_50 = stablehlo.constant dense<800> : tensor<i32>
    %176 = stablehlo.broadcast_in_dim %c_50, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %177 = stablehlo.compare  LT, %173, %176,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %178 = stablehlo.and %175, %177 : tensor<24xi1>
    %179 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %180 = stablehlo.compare  GE, %173, %179,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_51 = stablehlo.constant dense<801> : tensor<i32>
    %181 = stablehlo.add %8, %c_51 : tensor<i32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %183 = stablehlo.compare  LT, %173, %182,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %184 = stablehlo.and %180, %183 : tensor<24xi1>
    %185 = stablehlo.and %178, %184 : tensor<24xi1>
    %186 = stablehlo.reshape %185 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %187 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2] : (tensor<800x801x1xi1>) -> tensor<800x801x24xi1>
    %188 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<800x801x24xi1>
    %189 = stablehlo.and %187, %188 : tensor<800x801x24xi1>
    %cst_52 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %190 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<800x801x24xf32>
    %191 = call @_where_83(%189, %16, %190) : (tensor<800x801x24xi1>, tensor<800x801x24xf32>, tensor<800x801x24xf32>) -> tensor<800x801x24xf32>
    %192 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_53 = stablehlo.constant dense<0> : tensor<i32>
    %193 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %194 = stablehlo.compare  GE, %192, %193,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_54 = stablehlo.constant dense<801> : tensor<i32>
    %195 = stablehlo.broadcast_in_dim %c_54, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %196 = stablehlo.compare  LT, %192, %195,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %197 = stablehlo.and %194, %196 : tensor<801xi1>
    %198 = stablehlo.reshape %197 : (tensor<801xi1>) -> tensor<801x1x1xi1>
    %c_55 = stablehlo.constant dense<true> : tensor<i1>
    %199 = stablehlo.broadcast_in_dim %c_55, dims = [] : (tensor<i1>) -> tensor<801x1x1xi1>
    %200 = stablehlo.and %199, %198 : tensor<801x1x1xi1>
    %201 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_56 = stablehlo.constant dense<0> : tensor<i32>
    %202 = stablehlo.broadcast_in_dim %c_56, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %203 = stablehlo.compare  GE, %201, %202,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_57 = stablehlo.constant dense<800> : tensor<i32>
    %204 = stablehlo.broadcast_in_dim %c_57, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %205 = stablehlo.compare  LT, %201, %204,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %206 = stablehlo.and %203, %205 : tensor<800xi1>
    %207 = stablehlo.reshape %206 : (tensor<800xi1>) -> tensor<1x800x1xi1>
    %208 = stablehlo.broadcast_in_dim %200, dims = [0, 1, 2] : (tensor<801x1x1xi1>) -> tensor<801x800x1xi1>
    %209 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2] : (tensor<1x800x1xi1>) -> tensor<801x800x1xi1>
    %210 = stablehlo.and %208, %209 : tensor<801x800x1xi1>
    %211 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_58 = stablehlo.constant dense<12> : tensor<i32>
    %212 = stablehlo.broadcast_in_dim %c_58, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %213 = stablehlo.compare  LT, %211, %212,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_59 = stablehlo.constant dense<12> : tensor<i32>
    %214 = stablehlo.broadcast_in_dim %c_59, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %215 = stablehlo.subtract %211, %214 : tensor<24xi32>
    %c_60 = stablehlo.constant dense<800> : tensor<i32>
    %216 = stablehlo.broadcast_in_dim %c_60, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %217 = stablehlo.add %215, %216 : tensor<24xi32>
    %c_61 = stablehlo.constant dense<12> : tensor<i32>
    %218 = stablehlo.broadcast_in_dim %c_61, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %219 = stablehlo.subtract %217, %218 : tensor<24xi32>
    %220 = call @_where(%213, %211, %219) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_62 = stablehlo.constant dense<0> : tensor<i32>
    %221 = stablehlo.broadcast_in_dim %c_62, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %222 = stablehlo.compare  GE, %220, %221,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_63 = stablehlo.constant dense<800> : tensor<i32>
    %223 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %224 = stablehlo.compare  LT, %220, %223,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %225 = stablehlo.and %222, %224 : tensor<24xi1>
    %226 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %227 = stablehlo.compare  GE, %220, %226,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_64 = stablehlo.constant dense<801> : tensor<i32>
    %228 = stablehlo.add %8, %c_64 : tensor<i32>
    %229 = stablehlo.broadcast_in_dim %228, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %230 = stablehlo.compare  LT, %220, %229,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %231 = stablehlo.and %227, %230 : tensor<24xi1>
    %232 = stablehlo.and %225, %231 : tensor<24xi1>
    %233 = stablehlo.reshape %232 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %234 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<801x800x1xi1>) -> tensor<801x800x24xi1>
    %235 = stablehlo.broadcast_in_dim %233, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<801x800x24xi1>
    %236 = stablehlo.and %234, %235 : tensor<801x800x24xi1>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %237 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f32>) -> tensor<801x800x24xf32>
    %238 = call @_where_84(%236, %18, %237) : (tensor<801x800x24xi1>, tensor<801x800x24xf32>, tensor<801x800x24xf32>) -> tensor<801x800x24xf32>
    %239 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_66 = stablehlo.constant dense<0> : tensor<i32>
    %240 = stablehlo.broadcast_in_dim %c_66, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %241 = stablehlo.compare  GE, %239, %240,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_67 = stablehlo.constant dense<801> : tensor<i32>
    %242 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %243 = stablehlo.compare  LT, %239, %242,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %244 = stablehlo.and %241, %243 : tensor<801xi1>
    %245 = stablehlo.reshape %244 : (tensor<801xi1>) -> tensor<801x1x1xi1>
    %c_68 = stablehlo.constant dense<true> : tensor<i1>
    %246 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i1>) -> tensor<801x1x1xi1>
    %247 = stablehlo.and %246, %245 : tensor<801x1x1xi1>
    %248 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_69 = stablehlo.constant dense<12> : tensor<i32>
    %249 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %250 = stablehlo.compare  LT, %248, %249,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_70 = stablehlo.constant dense<12> : tensor<i32>
    %251 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %252 = stablehlo.subtract %248, %251 : tensor<24xi32>
    %c_71 = stablehlo.constant dense<800> : tensor<i32>
    %253 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %254 = stablehlo.add %252, %253 : tensor<24xi32>
    %c_72 = stablehlo.constant dense<12> : tensor<i32>
    %255 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %256 = stablehlo.subtract %254, %255 : tensor<24xi32>
    %257 = call @_where(%250, %248, %256) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_73 = stablehlo.constant dense<0> : tensor<i32>
    %258 = stablehlo.broadcast_in_dim %c_73, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %259 = stablehlo.compare  GE, %257, %258,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_74 = stablehlo.constant dense<800> : tensor<i32>
    %260 = stablehlo.broadcast_in_dim %c_74, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %261 = stablehlo.compare  LT, %257, %260,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %262 = stablehlo.and %259, %261 : tensor<24xi1>
    %263 = stablehlo.reshape %262 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %264 = stablehlo.broadcast_in_dim %247, dims = [0, 1, 2] : (tensor<801x1x1xi1>) -> tensor<801x24x1xi1>
    %265 = stablehlo.broadcast_in_dim %263, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<801x24x1xi1>
    %266 = stablehlo.and %264, %265 : tensor<801x24x1xi1>
    %267 = stablehlo.iota dim = 0 : tensor<801xi32>
    %268 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %269 = stablehlo.add %267, %268 : tensor<801xi32>
    %c_75 = stablehlo.constant dense<0> : tensor<i32>
    %270 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %271 = stablehlo.compare  GE, %269, %270,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_76 = stablehlo.constant dense<800> : tensor<i32>
    %272 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %273 = stablehlo.compare  LT, %269, %272,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %274 = stablehlo.and %271, %273 : tensor<801xi1>
    %275 = stablehlo.reshape %274 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %276 = stablehlo.broadcast_in_dim %266, dims = [0, 1, 2] : (tensor<801x24x1xi1>) -> tensor<801x24x801xi1>
    %277 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<801x24x801xi1>
    %278 = stablehlo.and %276, %277 : tensor<801x24x801xi1>
    %cst_77 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %279 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f32>) -> tensor<801x24x801xf32>
    %280 = call @_where_90(%278, %arg66, %279) : (tensor<801x24x801xi1>, tensor<801x24x801xf32>, tensor<801x24x801xf32>) -> tensor<801x24x801xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %281 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f32>) -> tensor<801x801x1xf32>
    %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %282 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f32>) -> tensor<801x801x1xf32>
    %283 = stablehlo.slice %arg56 [0:801, 0:800, 0:1] : (tensor<801x800x801xf32>) -> tensor<801x800x1xf32>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %284 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<801x800x1xf32>
    %285 = "stablehlo.collective_permute"(%283) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<801x800x1xf32>) -> tensor<801x800x1xf32>
    %286 = stablehlo.slice %arg57 [0:800, 0:801, 0:1] : (tensor<800x801x801xf32>) -> tensor<800x801x1xf32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %287 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<800x801x1xf32>
    %288 = "stablehlo.collective_permute"(%286) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<800x801x1xf32>) -> tensor<800x801x1xf32>
    %289:9 = stablehlo.custom_call @beamz_cuda_sharded(%arg58, %arg59, %arg60, %arg55, %arg56, %arg57, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg28, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %60, %102, %144, %191, %238, %280, %arg9, %arg9, %arg9, %14, %281, %282, %284, %285, %287, %288) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 0 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<800x801x24xf32>, tensor<801x800x24xf32>, tensor<801x24x801xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<801x801x1xf32>, tensor<801x801x1xf32>, tensor<801x800x1xf32>, tensor<801x800x1xf32>, tensor<800x801x1xf32>, tensor<800x801x1xf32>) -> (tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<800x801x24xf32>, tensor<801x800x24xf32>, tensor<801x24x801xf32>)
    %290 = stablehlo.broadcast_in_dim %289#6, dims = [1, 2, 3] : (tensor<800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %291 = stablehlo.broadcast_in_dim %289#7, dims = [1, 2, 3] : (tensor<801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %c_82 = stablehlo.constant dense<1> : tensor<ui32>
    %c_83 = stablehlo.constant dense<1> : tensor<ui32>
    %292 = stablehlo.partition_id : tensor<ui32>
    %293 = stablehlo.divide %292, %c_82 : tensor<ui32>
    %294 = stablehlo.remainder %293, %c_83 : tensor<ui32>
    %295 = stablehlo.convert %294 : (tensor<ui32>) -> tensor<i32>
    %c_84 = stablehlo.constant dense<801> : tensor<i32>
    %296 = stablehlo.multiply %295, %c_84 : tensor<i32>
    %c_85 = stablehlo.constant dense<2> : tensor<i32>
    %297 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %298 = stablehlo.broadcast_in_dim %296, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_86 = stablehlo.constant dense<0> : tensor<i32>
    %299 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %300 = stablehlo.concatenate %297, %298, %299, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %301 = stablehlo.broadcast_in_dim %300, dims = [1] : (tensor<3xi32>) -> tensor<1x3xi32>
    %302 = stablehlo.concatenate %301, %arg29, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
    %303 = stablehlo.slice %arg70 [0:1, 0:801, 0:800, 0:24] : (tensor<1x801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %304 = stablehlo.reshape %303 : (tensor<1x801x800x24xf32>) -> tensor<801x800x24xf32>
    %305 = stablehlo.slice %arg71 [0:1, 0:800, 0:801, 0:24] : (tensor<1x800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %306 = stablehlo.reshape %305 : (tensor<1x800x801x24xf32>) -> tensor<800x801x24xf32>
    %307 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_87 = stablehlo.constant dense<0> : tensor<i32>
    %308 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %309 = stablehlo.compare  GE, %307, %308,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_88 = stablehlo.constant dense<801> : tensor<i32>
    %310 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %311 = stablehlo.compare  LT, %307, %310,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %312 = stablehlo.and %309, %311 : tensor<801xi1>
    %313 = stablehlo.reshape %312 : (tensor<801xi1>) -> tensor<801x1x1xi1>
    %c_89 = stablehlo.constant dense<true> : tensor<i1>
    %314 = stablehlo.broadcast_in_dim %c_89, dims = [] : (tensor<i1>) -> tensor<801x1x1xi1>
    %315 = stablehlo.and %314, %313 : tensor<801x1x1xi1>
    %316 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_90 = stablehlo.constant dense<12> : tensor<i32>
    %317 = stablehlo.broadcast_in_dim %c_90, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %318 = stablehlo.compare  LT, %316, %317,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_91 = stablehlo.constant dense<12> : tensor<i32>
    %319 = stablehlo.broadcast_in_dim %c_91, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %320 = stablehlo.subtract %316, %319 : tensor<24xi32>
    %c_92 = stablehlo.constant dense<801> : tensor<i32>
    %321 = stablehlo.broadcast_in_dim %c_92, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %322 = stablehlo.add %320, %321 : tensor<24xi32>
    %c_93 = stablehlo.constant dense<12> : tensor<i32>
    %323 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %324 = stablehlo.subtract %322, %323 : tensor<24xi32>
    %325 = call @_where(%318, %316, %324) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_94 = stablehlo.constant dense<0> : tensor<i32>
    %326 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %327 = stablehlo.compare  GE, %325, %326,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_95 = stablehlo.constant dense<801> : tensor<i32>
    %328 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %329 = stablehlo.compare  LT, %325, %328,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %330 = stablehlo.and %327, %329 : tensor<24xi1>
    %331 = stablehlo.reshape %330 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %332 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<801x1x1xi1>) -> tensor<801x24x1xi1>
    %333 = stablehlo.broadcast_in_dim %331, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<801x24x1xi1>
    %334 = stablehlo.and %332, %333 : tensor<801x24x1xi1>
    %335 = stablehlo.iota dim = 0 : tensor<801xi32>
    %336 = stablehlo.broadcast_in_dim %296, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %337 = stablehlo.add %335, %336 : tensor<801xi32>
    %c_96 = stablehlo.constant dense<0> : tensor<i32>
    %338 = stablehlo.broadcast_in_dim %c_96, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %339 = stablehlo.compare  GE, %337, %338,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_97 = stablehlo.constant dense<800> : tensor<i32>
    %340 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %341 = stablehlo.compare  LT, %337, %340,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %342 = stablehlo.and %339, %341 : tensor<801xi1>
    %343 = stablehlo.reshape %342 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %344 = stablehlo.broadcast_in_dim %334, dims = [0, 1, 2] : (tensor<801x24x1xi1>) -> tensor<801x24x801xi1>
    %345 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<801x24x801xi1>
    %346 = stablehlo.and %344, %345 : tensor<801x24x801xi1>
    %cst_98 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %347 = stablehlo.broadcast_in_dim %cst_98, dims = [] : (tensor<f32>) -> tensor<801x24x801xf32>
    %348 = call @_where_90(%346, %arg67, %347) : (tensor<801x24x801xi1>, tensor<801x24x801xf32>, tensor<801x24x801xf32>) -> tensor<801x24x801xf32>
    %349 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_99 = stablehlo.constant dense<12> : tensor<i32>
    %350 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %351 = stablehlo.compare  LT, %349, %350,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_100 = stablehlo.constant dense<12> : tensor<i32>
    %352 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %353 = stablehlo.subtract %349, %352 : tensor<24xi32>
    %c_101 = stablehlo.constant dense<801> : tensor<i32>
    %354 = stablehlo.broadcast_in_dim %c_101, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %355 = stablehlo.add %353, %354 : tensor<24xi32>
    %c_102 = stablehlo.constant dense<12> : tensor<i32>
    %356 = stablehlo.broadcast_in_dim %c_102, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %357 = stablehlo.subtract %355, %356 : tensor<24xi32>
    %358 = call @_where(%351, %349, %357) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_103 = stablehlo.constant dense<0> : tensor<i32>
    %359 = stablehlo.broadcast_in_dim %c_103, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %360 = stablehlo.compare  GE, %358, %359,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_104 = stablehlo.constant dense<801> : tensor<i32>
    %361 = stablehlo.broadcast_in_dim %c_104, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %362 = stablehlo.compare  LT, %358, %361,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %363 = stablehlo.and %360, %362 : tensor<24xi1>
    %364 = stablehlo.reshape %363 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_105 = stablehlo.constant dense<true> : tensor<i1>
    %365 = stablehlo.broadcast_in_dim %c_105, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %366 = stablehlo.and %365, %364 : tensor<24x1x1xi1>
    %367 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_106 = stablehlo.constant dense<0> : tensor<i32>
    %368 = stablehlo.broadcast_in_dim %c_106, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %369 = stablehlo.compare  GE, %367, %368,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_107 = stablehlo.constant dense<801> : tensor<i32>
    %370 = stablehlo.broadcast_in_dim %c_107, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %371 = stablehlo.compare  LT, %367, %370,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %372 = stablehlo.and %369, %371 : tensor<801xi1>
    %373 = stablehlo.reshape %372 : (tensor<801xi1>) -> tensor<1x801x1xi1>
    %374 = stablehlo.broadcast_in_dim %366, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x801x1xi1>
    %375 = stablehlo.broadcast_in_dim %373, dims = [0, 1, 2] : (tensor<1x801x1xi1>) -> tensor<24x801x1xi1>
    %376 = stablehlo.and %374, %375 : tensor<24x801x1xi1>
    %377 = stablehlo.iota dim = 0 : tensor<801xi32>
    %378 = stablehlo.broadcast_in_dim %296, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %379 = stablehlo.add %377, %378 : tensor<801xi32>
    %c_108 = stablehlo.constant dense<0> : tensor<i32>
    %380 = stablehlo.broadcast_in_dim %c_108, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %381 = stablehlo.compare  GE, %379, %380,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_109 = stablehlo.constant dense<800> : tensor<i32>
    %382 = stablehlo.broadcast_in_dim %c_109, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %383 = stablehlo.compare  LT, %379, %382,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %384 = stablehlo.and %381, %383 : tensor<801xi1>
    %385 = stablehlo.reshape %384 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %386 = stablehlo.broadcast_in_dim %376, dims = [0, 1, 2] : (tensor<24x801x1xi1>) -> tensor<24x801x801xi1>
    %387 = stablehlo.broadcast_in_dim %385, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<24x801x801xi1>
    %388 = stablehlo.and %386, %387 : tensor<24x801x801xi1>
    %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %389 = stablehlo.broadcast_in_dim %cst_110, dims = [] : (tensor<f32>) -> tensor<24x801x801xf32>
    %390 = call @_where_81(%388, %arg68, %389) : (tensor<24x801x801xi1>, tensor<24x801x801xf32>, tensor<24x801x801xf32>) -> tensor<24x801x801xf32>
    %391 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_111 = stablehlo.constant dense<12> : tensor<i32>
    %392 = stablehlo.broadcast_in_dim %c_111, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %393 = stablehlo.compare  LT, %391, %392,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_112 = stablehlo.constant dense<12> : tensor<i32>
    %394 = stablehlo.broadcast_in_dim %c_112, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %395 = stablehlo.subtract %391, %394 : tensor<24xi32>
    %c_113 = stablehlo.constant dense<801> : tensor<i32>
    %396 = stablehlo.broadcast_in_dim %c_113, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %397 = stablehlo.add %395, %396 : tensor<24xi32>
    %c_114 = stablehlo.constant dense<12> : tensor<i32>
    %398 = stablehlo.broadcast_in_dim %c_114, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %399 = stablehlo.subtract %397, %398 : tensor<24xi32>
    %400 = call @_where(%393, %391, %399) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_115 = stablehlo.constant dense<0> : tensor<i32>
    %401 = stablehlo.broadcast_in_dim %c_115, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %402 = stablehlo.compare  GE, %400, %401,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_116 = stablehlo.constant dense<801> : tensor<i32>
    %403 = stablehlo.broadcast_in_dim %c_116, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %404 = stablehlo.compare  LT, %400, %403,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %405 = stablehlo.and %402, %404 : tensor<24xi1>
    %406 = stablehlo.reshape %405 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_117 = stablehlo.constant dense<true> : tensor<i1>
    %407 = stablehlo.broadcast_in_dim %c_117, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %408 = stablehlo.and %407, %406 : tensor<24x1x1xi1>
    %409 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_118 = stablehlo.constant dense<0> : tensor<i32>
    %410 = stablehlo.broadcast_in_dim %c_118, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %411 = stablehlo.compare  GE, %409, %410,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_119 = stablehlo.constant dense<800> : tensor<i32>
    %412 = stablehlo.broadcast_in_dim %c_119, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %413 = stablehlo.compare  LT, %409, %412,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %414 = stablehlo.and %411, %413 : tensor<800xi1>
    %415 = stablehlo.reshape %414 : (tensor<800xi1>) -> tensor<1x800x1xi1>
    %416 = stablehlo.broadcast_in_dim %408, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x800x1xi1>
    %417 = stablehlo.broadcast_in_dim %415, dims = [0, 1, 2] : (tensor<1x800x1xi1>) -> tensor<24x800x1xi1>
    %418 = stablehlo.and %416, %417 : tensor<24x800x1xi1>
    %419 = stablehlo.iota dim = 0 : tensor<801xi32>
    %420 = stablehlo.broadcast_in_dim %296, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %421 = stablehlo.add %419, %420 : tensor<801xi32>
    %c_120 = stablehlo.constant dense<0> : tensor<i32>
    %422 = stablehlo.broadcast_in_dim %c_120, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %423 = stablehlo.compare  GE, %421, %422,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_121 = stablehlo.constant dense<801> : tensor<i32>
    %424 = stablehlo.broadcast_in_dim %c_121, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %425 = stablehlo.compare  LT, %421, %424,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %426 = stablehlo.and %423, %425 : tensor<801xi1>
    %427 = stablehlo.reshape %426 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %428 = stablehlo.broadcast_in_dim %418, dims = [0, 1, 2] : (tensor<24x800x1xi1>) -> tensor<24x800x801xi1>
    %429 = stablehlo.broadcast_in_dim %427, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<24x800x801xi1>
    %430 = stablehlo.and %428, %429 : tensor<24x800x801xi1>
    %cst_122 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %431 = stablehlo.broadcast_in_dim %cst_122, dims = [] : (tensor<f32>) -> tensor<24x800x801xf32>
    %432 = call @_where_74(%430, %arg69, %431) : (tensor<24x800x801xi1>, tensor<24x800x801xf32>, tensor<24x800x801xf32>) -> tensor<24x800x801xf32>
    %433 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_123 = stablehlo.constant dense<0> : tensor<i32>
    %434 = stablehlo.broadcast_in_dim %c_123, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %435 = stablehlo.compare  GE, %433, %434,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_124 = stablehlo.constant dense<801> : tensor<i32>
    %436 = stablehlo.broadcast_in_dim %c_124, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %437 = stablehlo.compare  LT, %433, %436,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %438 = stablehlo.and %435, %437 : tensor<801xi1>
    %439 = stablehlo.reshape %438 : (tensor<801xi1>) -> tensor<801x1x1xi1>
    %c_125 = stablehlo.constant dense<true> : tensor<i1>
    %440 = stablehlo.broadcast_in_dim %c_125, dims = [] : (tensor<i1>) -> tensor<801x1x1xi1>
    %441 = stablehlo.and %440, %439 : tensor<801x1x1xi1>
    %442 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_126 = stablehlo.constant dense<0> : tensor<i32>
    %443 = stablehlo.broadcast_in_dim %c_126, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %444 = stablehlo.compare  GE, %442, %443,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_127 = stablehlo.constant dense<800> : tensor<i32>
    %445 = stablehlo.broadcast_in_dim %c_127, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %446 = stablehlo.compare  LT, %442, %445,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %447 = stablehlo.and %444, %446 : tensor<800xi1>
    %448 = stablehlo.reshape %447 : (tensor<800xi1>) -> tensor<1x800x1xi1>
    %449 = stablehlo.broadcast_in_dim %441, dims = [0, 1, 2] : (tensor<801x1x1xi1>) -> tensor<801x800x1xi1>
    %450 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2] : (tensor<1x800x1xi1>) -> tensor<801x800x1xi1>
    %451 = stablehlo.and %449, %450 : tensor<801x800x1xi1>
    %452 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_128 = stablehlo.constant dense<12> : tensor<i32>
    %453 = stablehlo.broadcast_in_dim %c_128, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %454 = stablehlo.compare  LT, %452, %453,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_129 = stablehlo.constant dense<12> : tensor<i32>
    %455 = stablehlo.broadcast_in_dim %c_129, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %456 = stablehlo.subtract %452, %455 : tensor<24xi32>
    %c_130 = stablehlo.constant dense<801> : tensor<i32>
    %457 = stablehlo.broadcast_in_dim %c_130, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %458 = stablehlo.add %456, %457 : tensor<24xi32>
    %c_131 = stablehlo.constant dense<12> : tensor<i32>
    %459 = stablehlo.broadcast_in_dim %c_131, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %460 = stablehlo.subtract %458, %459 : tensor<24xi32>
    %461 = call @_where(%454, %452, %460) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_132 = stablehlo.constant dense<0> : tensor<i32>
    %462 = stablehlo.broadcast_in_dim %c_132, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %463 = stablehlo.compare  GE, %461, %462,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_133 = stablehlo.constant dense<801> : tensor<i32>
    %464 = stablehlo.broadcast_in_dim %c_133, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %465 = stablehlo.compare  LT, %461, %464,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %466 = stablehlo.and %463, %465 : tensor<24xi1>
    %467 = stablehlo.broadcast_in_dim %296, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %468 = stablehlo.compare  GE, %461, %467,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_134 = stablehlo.constant dense<801> : tensor<i32>
    %469 = stablehlo.add %296, %c_134 : tensor<i32>
    %470 = stablehlo.broadcast_in_dim %469, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %471 = stablehlo.compare  LT, %461, %470,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %472 = stablehlo.and %468, %471 : tensor<24xi1>
    %473 = stablehlo.and %466, %472 : tensor<24xi1>
    %474 = stablehlo.reshape %473 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %475 = stablehlo.broadcast_in_dim %451, dims = [0, 1, 2] : (tensor<801x800x1xi1>) -> tensor<801x800x24xi1>
    %476 = stablehlo.broadcast_in_dim %474, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<801x800x24xi1>
    %477 = stablehlo.and %475, %476 : tensor<801x800x24xi1>
    %cst_135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %478 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<f32>) -> tensor<801x800x24xf32>
    %479 = call @_where_84(%477, %304, %478) : (tensor<801x800x24xi1>, tensor<801x800x24xf32>, tensor<801x800x24xf32>) -> tensor<801x800x24xf32>
    %480 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_136 = stablehlo.constant dense<0> : tensor<i32>
    %481 = stablehlo.broadcast_in_dim %c_136, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %482 = stablehlo.compare  GE, %480, %481,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_137 = stablehlo.constant dense<800> : tensor<i32>
    %483 = stablehlo.broadcast_in_dim %c_137, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %484 = stablehlo.compare  LT, %480, %483,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %485 = stablehlo.and %482, %484 : tensor<800xi1>
    %486 = stablehlo.reshape %485 : (tensor<800xi1>) -> tensor<800x1x1xi1>
    %c_138 = stablehlo.constant dense<true> : tensor<i1>
    %487 = stablehlo.broadcast_in_dim %c_138, dims = [] : (tensor<i1>) -> tensor<800x1x1xi1>
    %488 = stablehlo.and %487, %486 : tensor<800x1x1xi1>
    %489 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_139 = stablehlo.constant dense<0> : tensor<i32>
    %490 = stablehlo.broadcast_in_dim %c_139, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %491 = stablehlo.compare  GE, %489, %490,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_140 = stablehlo.constant dense<801> : tensor<i32>
    %492 = stablehlo.broadcast_in_dim %c_140, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %493 = stablehlo.compare  LT, %489, %492,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %494 = stablehlo.and %491, %493 : tensor<801xi1>
    %495 = stablehlo.reshape %494 : (tensor<801xi1>) -> tensor<1x801x1xi1>
    %496 = stablehlo.broadcast_in_dim %488, dims = [0, 1, 2] : (tensor<800x1x1xi1>) -> tensor<800x801x1xi1>
    %497 = stablehlo.broadcast_in_dim %495, dims = [0, 1, 2] : (tensor<1x801x1xi1>) -> tensor<800x801x1xi1>
    %498 = stablehlo.and %496, %497 : tensor<800x801x1xi1>
    %499 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_141 = stablehlo.constant dense<12> : tensor<i32>
    %500 = stablehlo.broadcast_in_dim %c_141, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %501 = stablehlo.compare  LT, %499, %500,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_142 = stablehlo.constant dense<12> : tensor<i32>
    %502 = stablehlo.broadcast_in_dim %c_142, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %503 = stablehlo.subtract %499, %502 : tensor<24xi32>
    %c_143 = stablehlo.constant dense<801> : tensor<i32>
    %504 = stablehlo.broadcast_in_dim %c_143, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %505 = stablehlo.add %503, %504 : tensor<24xi32>
    %c_144 = stablehlo.constant dense<12> : tensor<i32>
    %506 = stablehlo.broadcast_in_dim %c_144, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %507 = stablehlo.subtract %505, %506 : tensor<24xi32>
    %508 = call @_where(%501, %499, %507) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_145 = stablehlo.constant dense<0> : tensor<i32>
    %509 = stablehlo.broadcast_in_dim %c_145, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %510 = stablehlo.compare  GE, %508, %509,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_146 = stablehlo.constant dense<801> : tensor<i32>
    %511 = stablehlo.broadcast_in_dim %c_146, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %512 = stablehlo.compare  LT, %508, %511,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %513 = stablehlo.and %510, %512 : tensor<24xi1>
    %514 = stablehlo.broadcast_in_dim %296, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %515 = stablehlo.compare  GE, %508, %514,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_147 = stablehlo.constant dense<801> : tensor<i32>
    %516 = stablehlo.add %296, %c_147 : tensor<i32>
    %517 = stablehlo.broadcast_in_dim %516, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %518 = stablehlo.compare  LT, %508, %517,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %519 = stablehlo.and %515, %518 : tensor<24xi1>
    %520 = stablehlo.and %513, %519 : tensor<24xi1>
    %521 = stablehlo.reshape %520 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %522 = stablehlo.broadcast_in_dim %498, dims = [0, 1, 2] : (tensor<800x801x1xi1>) -> tensor<800x801x24xi1>
    %523 = stablehlo.broadcast_in_dim %521, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<800x801x24xi1>
    %524 = stablehlo.and %522, %523 : tensor<800x801x24xi1>
    %cst_148 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %525 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<f32>) -> tensor<800x801x24xf32>
    %526 = call @_where_83(%524, %306, %525) : (tensor<800x801x24xi1>, tensor<800x801x24xf32>, tensor<800x801x24xf32>) -> tensor<800x801x24xf32>
    %527 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_149 = stablehlo.constant dense<0> : tensor<i32>
    %528 = stablehlo.broadcast_in_dim %c_149, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %529 = stablehlo.compare  GE, %527, %528,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_150 = stablehlo.constant dense<800> : tensor<i32>
    %530 = stablehlo.broadcast_in_dim %c_150, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %531 = stablehlo.compare  LT, %527, %530,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %532 = stablehlo.and %529, %531 : tensor<800xi1>
    %533 = stablehlo.reshape %532 : (tensor<800xi1>) -> tensor<800x1x1xi1>
    %c_151 = stablehlo.constant dense<true> : tensor<i1>
    %534 = stablehlo.broadcast_in_dim %c_151, dims = [] : (tensor<i1>) -> tensor<800x1x1xi1>
    %535 = stablehlo.and %534, %533 : tensor<800x1x1xi1>
    %536 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_152 = stablehlo.constant dense<12> : tensor<i32>
    %537 = stablehlo.broadcast_in_dim %c_152, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %538 = stablehlo.compare  LT, %536, %537,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_153 = stablehlo.constant dense<12> : tensor<i32>
    %539 = stablehlo.broadcast_in_dim %c_153, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %540 = stablehlo.subtract %536, %539 : tensor<24xi32>
    %c_154 = stablehlo.constant dense<801> : tensor<i32>
    %541 = stablehlo.broadcast_in_dim %c_154, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %542 = stablehlo.add %540, %541 : tensor<24xi32>
    %c_155 = stablehlo.constant dense<12> : tensor<i32>
    %543 = stablehlo.broadcast_in_dim %c_155, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %544 = stablehlo.subtract %542, %543 : tensor<24xi32>
    %545 = call @_where(%538, %536, %544) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_156 = stablehlo.constant dense<0> : tensor<i32>
    %546 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %547 = stablehlo.compare  GE, %545, %546,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_157 = stablehlo.constant dense<801> : tensor<i32>
    %548 = stablehlo.broadcast_in_dim %c_157, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %549 = stablehlo.compare  LT, %545, %548,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %550 = stablehlo.and %547, %549 : tensor<24xi1>
    %551 = stablehlo.reshape %550 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %552 = stablehlo.broadcast_in_dim %535, dims = [0, 1, 2] : (tensor<800x1x1xi1>) -> tensor<800x24x1xi1>
    %553 = stablehlo.broadcast_in_dim %551, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<800x24x1xi1>
    %554 = stablehlo.and %552, %553 : tensor<800x24x1xi1>
    %555 = stablehlo.iota dim = 0 : tensor<801xi32>
    %556 = stablehlo.broadcast_in_dim %296, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %557 = stablehlo.add %555, %556 : tensor<801xi32>
    %c_158 = stablehlo.constant dense<0> : tensor<i32>
    %558 = stablehlo.broadcast_in_dim %c_158, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %559 = stablehlo.compare  GE, %557, %558,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_159 = stablehlo.constant dense<801> : tensor<i32>
    %560 = stablehlo.broadcast_in_dim %c_159, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %561 = stablehlo.compare  LT, %557, %560,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %562 = stablehlo.and %559, %561 : tensor<801xi1>
    %563 = stablehlo.reshape %562 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %564 = stablehlo.broadcast_in_dim %554, dims = [0, 1, 2] : (tensor<800x24x1xi1>) -> tensor<800x24x801xi1>
    %565 = stablehlo.broadcast_in_dim %563, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<800x24x801xi1>
    %566 = stablehlo.and %564, %565 : tensor<800x24x801xi1>
    %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %567 = stablehlo.broadcast_in_dim %cst_160, dims = [] : (tensor<f32>) -> tensor<800x24x801xf32>
    %568 = call @_where_65(%566, %arg72, %567) : (tensor<800x24x801xi1>, tensor<800x24x801xf32>, tensor<800x24x801xf32>) -> tensor<800x24x801xf32>
    %cst_161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %569 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f32>) -> tensor<800x800x1xf32>
    %cst_162 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %570 = stablehlo.broadcast_in_dim %cst_162, dims = [] : (tensor<f32>) -> tensor<800x800x1xf32>
    %571 = stablehlo.slice %289#1 [0:800, 0:801, 800:801] : (tensor<800x801x801xf32>) -> tensor<800x801x1xf32>
    %572 = "stablehlo.collective_permute"(%571) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<800x801x1xf32>) -> tensor<800x801x1xf32>
    %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %573 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<f32>) -> tensor<800x801x1xf32>
    %574 = stablehlo.slice %289#2 [0:801, 0:800, 800:801] : (tensor<801x800x801xf32>) -> tensor<801x800x1xf32>
    %575 = "stablehlo.collective_permute"(%574) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<801x800x1xf32>) -> tensor<801x800x1xf32>
    %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %576 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f32>) -> tensor<801x800x1xf32>
    %577:9 = stablehlo.custom_call @beamz_cuda_sharded(%arg55, %arg56, %arg57, %289#0, %289#1, %289#2, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg54, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %348, %390, %432, %479, %526, %568, %arg9, %arg9, %arg9, %302, %569, %570, %572, %573, %575, %576) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 1 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<801x800x24xf32>, tensor<800x801x24xf32>, tensor<800x24x801xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<800x800x1xf32>, tensor<800x800x1xf32>, tensor<800x801x1xf32>, tensor<800x801x1xf32>, tensor<801x800x1xf32>, tensor<801x800x1xf32>) -> (tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<801x800x24xf32>, tensor<800x801x24xf32>, tensor<800x24x801xf32>)
    %578 = stablehlo.broadcast_in_dim %577#6, dims = [1, 2, 3] : (tensor<801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %579 = stablehlo.broadcast_in_dim %577#7, dims = [1, 2, 3] : (tensor<800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %c_165 = stablehlo.constant dense<1> : tensor<i32>
    %580 = stablehlo.add %arg74, %c_165 : tensor<i32>
    return %577#0, %577#1, %577#2, %289#0, %289#1, %289#2, %289#3, %289#4, %289#5, %290, %291, %289#8, %577#3, %577#4, %577#5, %578, %579, %577#8, %3, %580 : tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @_where_65(%arg0: tensor<800x24x801xi1>, %arg1: tensor<800x24x801xf32>, %arg2: tensor<800x24x801xf32>) -> tensor<800x24x801xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<800x24x801xi1>, tensor<800x24x801xf32>
    return %0 : tensor<800x24x801xf32>
  }
  func.func private @_where_74(%arg0: tensor<24x800x801xi1>, %arg1: tensor<24x800x801xf32>, %arg2: tensor<24x800x801xf32>) -> tensor<24x800x801xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x800x801xi1>, tensor<24x800x801xf32>
    return %0 : tensor<24x800x801xf32>
  }
  func.func private @_where_81(%arg0: tensor<24x801x801xi1>, %arg1: tensor<24x801x801xf32>, %arg2: tensor<24x801x801xf32>) -> tensor<24x801x801xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x801x801xi1>, tensor<24x801x801xf32>
    return %0 : tensor<24x801x801xf32>
  }
  func.func private @_where_83(%arg0: tensor<800x801x24xi1>, %arg1: tensor<800x801x24xf32>, %arg2: tensor<800x801x24xf32>) -> tensor<800x801x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<800x801x24xi1>, tensor<800x801x24xf32>
    return %0 : tensor<800x801x24xf32>
  }
  func.func private @_where_84(%arg0: tensor<801x800x24xi1>, %arg1: tensor<801x800x24xf32>, %arg2: tensor<801x800x24xf32>) -> tensor<801x800x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<801x800x24xi1>, tensor<801x800x24xf32>
    return %0 : tensor<801x800x24xf32>
  }
  func.func private @_where_90(%arg0: tensor<801x24x801xi1>, %arg1: tensor<801x24x801xf32>, %arg2: tensor<801x24x801xf32>) -> tensor<801x24x801xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<801x24x801xi1>, tensor<801x24x801xf32>
    return %0 : tensor<801x24x801xf32>
  }
}
