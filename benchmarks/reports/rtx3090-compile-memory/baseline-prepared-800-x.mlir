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
    %c_19 = stablehlo.constant dense<[1, 2, 0]> : tensor<3xi32>
    %c_20 = stablehlo.constant dense<[2, 3, 1, 5, 6, 4]> : tensor<6xi32>
    %c_21 = stablehlo.constant dense<[2, 0, 1]> : tensor<3xi32>
    %c_22 = stablehlo.constant dense_resource<__elided__> : tensor<6x5xi32>
    %c_23 = stablehlo.constant dense_resource<__elided__> : tensor<6x3xi32>
    %cst_24 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_25 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_26 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_27 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_28 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_29 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_30 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_31 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_32 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_33 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_34 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_35 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_36 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_37 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_38 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_39 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_40 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_41 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %c_42 = stablehlo.constant dense<[1, 2, 0]> : tensor<3xi32>
    %c_43 = stablehlo.constant dense<[2, 3, 1, 5, 6, 4]> : tensor<6xi32>
    %c_44 = stablehlo.constant dense<[2, 0, 1]> : tensor<3xi32>
    %c_45 = stablehlo.constant dense_resource<__elided__> : tensor<6x5xi32>
    %c_46 = stablehlo.constant dense<1> : tensor<ui32>
    %c_47 = stablehlo.constant dense<1> : tensor<ui32>
    %0 = stablehlo.partition_id : tensor<ui32>
    %1 = stablehlo.divide %0, %c_46 : tensor<ui32>
    %2 = stablehlo.remainder %1, %c_47 : tensor<ui32>
    %3 = stablehlo.convert %2 : (tensor<ui32>) -> tensor<i32>
    %c_48 = stablehlo.constant dense<801> : tensor<i32>
    %4 = stablehlo.multiply %3, %c_48 : tensor<i32>
    %5 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_49 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_50 = stablehlo.constant dense<800> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_50, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %9 = stablehlo.compare  LT, %5, %8,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %10 = stablehlo.and %7, %9 : tensor<800xi1>
    %11 = stablehlo.reshape %10 : (tensor<800xi1>) -> tensor<800x1x1xi1>
    %c_51 = stablehlo.constant dense<true> : tensor<i1>
    %12 = stablehlo.broadcast_in_dim %c_51, dims = [] : (tensor<i1>) -> tensor<800x1x1xi1>
    %13 = stablehlo.and %12, %11 : tensor<800x1x1xi1>
    %14 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_52 = stablehlo.constant dense<0> : tensor<i32>
    %15 = stablehlo.broadcast_in_dim %c_52, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %16 = stablehlo.compare  GE, %14, %15,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_53 = stablehlo.constant dense<801> : tensor<i32>
    %17 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %18 = stablehlo.compare  LT, %14, %17,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %19 = stablehlo.and %16, %18 : tensor<801xi1>
    %20 = stablehlo.reshape %19 : (tensor<801xi1>) -> tensor<1x801x1xi1>
    %21 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<800x1x1xi1>) -> tensor<800x801x1xi1>
    %22 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<1x801x1xi1>) -> tensor<800x801x1xi1>
    %23 = stablehlo.and %21, %22 : tensor<800x801x1xi1>
    %24 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_54 = stablehlo.constant dense<12> : tensor<i32>
    %25 = stablehlo.broadcast_in_dim %c_54, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %26 = stablehlo.compare  LT, %24, %25,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_55 = stablehlo.constant dense<12> : tensor<i32>
    %27 = stablehlo.broadcast_in_dim %c_55, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %28 = stablehlo.subtract %24, %27 : tensor<24xi32>
    %c_56 = stablehlo.constant dense<800> : tensor<i32>
    %29 = stablehlo.broadcast_in_dim %c_56, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %30 = stablehlo.add %28, %29 : tensor<24xi32>
    %c_57 = stablehlo.constant dense<12> : tensor<i32>
    %31 = stablehlo.broadcast_in_dim %c_57, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %32 = stablehlo.subtract %30, %31 : tensor<24xi32>
    %33 = call @_where(%26, %24, %32) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_58 = stablehlo.constant dense<0> : tensor<i32>
    %34 = stablehlo.broadcast_in_dim %c_58, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %35 = stablehlo.compare  GE, %33, %34,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_59 = stablehlo.constant dense<800> : tensor<i32>
    %36 = stablehlo.broadcast_in_dim %c_59, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %37 = stablehlo.compare  LT, %33, %36,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %38 = stablehlo.and %35, %37 : tensor<24xi1>
    %39 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %40 = stablehlo.compare  GE, %33, %39,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_60 = stablehlo.constant dense<801> : tensor<i32>
    %41 = stablehlo.add %4, %c_60 : tensor<i32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %43 = stablehlo.compare  LT, %33, %42,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %44 = stablehlo.and %40, %43 : tensor<24xi1>
    %45 = stablehlo.and %38, %44 : tensor<24xi1>
    %46 = stablehlo.reshape %45 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %47 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2] : (tensor<800x801x1xi1>) -> tensor<800x801x24xi1>
    %48 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<800x801x24xi1>
    %49 = stablehlo.and %47, %48 : tensor<800x801x24xi1>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_61, dims = [] : (tensor<f32>) -> tensor<800x801x24xf32>
    %51 = call @_where_20(%49, %arg9, %50) : (tensor<800x801x24xi1>, tensor<800x801x24xf32>, tensor<800x801x24xf32>) -> tensor<800x801x24xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [1, 2, 3] : (tensor<800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %c_62 = stablehlo.constant dense<1> : tensor<ui32>
    %c_63 = stablehlo.constant dense<1> : tensor<ui32>
    %53 = stablehlo.partition_id : tensor<ui32>
    %54 = stablehlo.divide %53, %c_62 : tensor<ui32>
    %55 = stablehlo.remainder %54, %c_63 : tensor<ui32>
    %56 = stablehlo.convert %55 : (tensor<ui32>) -> tensor<i32>
    %c_64 = stablehlo.constant dense<801> : tensor<i32>
    %57 = stablehlo.multiply %56, %c_64 : tensor<i32>
    %58 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_65 = stablehlo.constant dense<0> : tensor<i32>
    %59 = stablehlo.broadcast_in_dim %c_65, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %60 = stablehlo.compare  GE, %58, %59,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_66 = stablehlo.constant dense<801> : tensor<i32>
    %61 = stablehlo.broadcast_in_dim %c_66, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %62 = stablehlo.compare  LT, %58, %61,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %63 = stablehlo.and %60, %62 : tensor<801xi1>
    %64 = stablehlo.reshape %63 : (tensor<801xi1>) -> tensor<801x1x1xi1>
    %c_67 = stablehlo.constant dense<true> : tensor<i1>
    %65 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i1>) -> tensor<801x1x1xi1>
    %66 = stablehlo.and %65, %64 : tensor<801x1x1xi1>
    %67 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_68 = stablehlo.constant dense<0> : tensor<i32>
    %68 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %69 = stablehlo.compare  GE, %67, %68,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_69 = stablehlo.constant dense<800> : tensor<i32>
    %70 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %71 = stablehlo.compare  LT, %67, %70,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %72 = stablehlo.and %69, %71 : tensor<800xi1>
    %73 = stablehlo.reshape %72 : (tensor<800xi1>) -> tensor<1x800x1xi1>
    %74 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2] : (tensor<801x1x1xi1>) -> tensor<801x800x1xi1>
    %75 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2] : (tensor<1x800x1xi1>) -> tensor<801x800x1xi1>
    %76 = stablehlo.and %74, %75 : tensor<801x800x1xi1>
    %77 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_70 = stablehlo.constant dense<12> : tensor<i32>
    %78 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %79 = stablehlo.compare  LT, %77, %78,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_71 = stablehlo.constant dense<12> : tensor<i32>
    %80 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %81 = stablehlo.subtract %77, %80 : tensor<24xi32>
    %c_72 = stablehlo.constant dense<800> : tensor<i32>
    %82 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %83 = stablehlo.add %81, %82 : tensor<24xi32>
    %c_73 = stablehlo.constant dense<12> : tensor<i32>
    %84 = stablehlo.broadcast_in_dim %c_73, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %85 = stablehlo.subtract %83, %84 : tensor<24xi32>
    %86 = call @_where(%79, %77, %85) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_74 = stablehlo.constant dense<0> : tensor<i32>
    %87 = stablehlo.broadcast_in_dim %c_74, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %88 = stablehlo.compare  GE, %86, %87,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_75 = stablehlo.constant dense<800> : tensor<i32>
    %89 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %90 = stablehlo.compare  LT, %86, %89,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %91 = stablehlo.and %88, %90 : tensor<24xi1>
    %92 = stablehlo.broadcast_in_dim %57, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %93 = stablehlo.compare  GE, %86, %92,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_76 = stablehlo.constant dense<801> : tensor<i32>
    %94 = stablehlo.add %57, %c_76 : tensor<i32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %96 = stablehlo.compare  LT, %86, %95,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %97 = stablehlo.and %93, %96 : tensor<24xi1>
    %98 = stablehlo.and %91, %97 : tensor<24xi1>
    %99 = stablehlo.reshape %98 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %100 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2] : (tensor<801x800x1xi1>) -> tensor<801x800x24xi1>
    %101 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<801x800x24xi1>
    %102 = stablehlo.and %100, %101 : tensor<801x800x24xi1>
    %cst_77 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %103 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f32>) -> tensor<801x800x24xf32>
    %104 = call @_where_32(%102, %arg10, %103) : (tensor<801x800x24xi1>, tensor<801x800x24xf32>, tensor<801x800x24xf32>) -> tensor<801x800x24xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [1, 2, 3] : (tensor<801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %c_78 = stablehlo.constant dense<1> : tensor<ui32>
    %c_79 = stablehlo.constant dense<1> : tensor<ui32>
    %106 = stablehlo.partition_id : tensor<ui32>
    %107 = stablehlo.divide %106, %c_78 : tensor<ui32>
    %108 = stablehlo.remainder %107, %c_79 : tensor<ui32>
    %109 = stablehlo.convert %108 : (tensor<ui32>) -> tensor<i32>
    %c_80 = stablehlo.constant dense<801> : tensor<i32>
    %110 = stablehlo.multiply %109, %c_80 : tensor<i32>
    %111 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_81 = stablehlo.constant dense<0> : tensor<i32>
    %112 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %113 = stablehlo.compare  GE, %111, %112,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_82 = stablehlo.constant dense<801> : tensor<i32>
    %114 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %115 = stablehlo.compare  LT, %111, %114,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %116 = stablehlo.and %113, %115 : tensor<801xi1>
    %117 = stablehlo.reshape %116 : (tensor<801xi1>) -> tensor<801x1x1xi1>
    %c_83 = stablehlo.constant dense<true> : tensor<i1>
    %118 = stablehlo.broadcast_in_dim %c_83, dims = [] : (tensor<i1>) -> tensor<801x1x1xi1>
    %119 = stablehlo.and %118, %117 : tensor<801x1x1xi1>
    %120 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_84 = stablehlo.constant dense<0> : tensor<i32>
    %121 = stablehlo.broadcast_in_dim %c_84, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %122 = stablehlo.compare  GE, %120, %121,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_85 = stablehlo.constant dense<800> : tensor<i32>
    %123 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %124 = stablehlo.compare  LT, %120, %123,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %125 = stablehlo.and %122, %124 : tensor<800xi1>
    %126 = stablehlo.reshape %125 : (tensor<800xi1>) -> tensor<1x800x1xi1>
    %127 = stablehlo.broadcast_in_dim %119, dims = [0, 1, 2] : (tensor<801x1x1xi1>) -> tensor<801x800x1xi1>
    %128 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2] : (tensor<1x800x1xi1>) -> tensor<801x800x1xi1>
    %129 = stablehlo.and %127, %128 : tensor<801x800x1xi1>
    %130 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_86 = stablehlo.constant dense<12> : tensor<i32>
    %131 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %132 = stablehlo.compare  LT, %130, %131,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_87 = stablehlo.constant dense<12> : tensor<i32>
    %133 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %134 = stablehlo.subtract %130, %133 : tensor<24xi32>
    %c_88 = stablehlo.constant dense<801> : tensor<i32>
    %135 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %136 = stablehlo.add %134, %135 : tensor<24xi32>
    %c_89 = stablehlo.constant dense<12> : tensor<i32>
    %137 = stablehlo.broadcast_in_dim %c_89, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %138 = stablehlo.subtract %136, %137 : tensor<24xi32>
    %139 = call @_where(%132, %130, %138) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_90 = stablehlo.constant dense<0> : tensor<i32>
    %140 = stablehlo.broadcast_in_dim %c_90, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %141 = stablehlo.compare  GE, %139, %140,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_91 = stablehlo.constant dense<801> : tensor<i32>
    %142 = stablehlo.broadcast_in_dim %c_91, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %143 = stablehlo.compare  LT, %139, %142,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %144 = stablehlo.and %141, %143 : tensor<24xi1>
    %145 = stablehlo.broadcast_in_dim %110, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %146 = stablehlo.compare  GE, %139, %145,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_92 = stablehlo.constant dense<801> : tensor<i32>
    %147 = stablehlo.add %110, %c_92 : tensor<i32>
    %148 = stablehlo.broadcast_in_dim %147, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %149 = stablehlo.compare  LT, %139, %148,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %150 = stablehlo.and %146, %149 : tensor<24xi1>
    %151 = stablehlo.and %144, %150 : tensor<24xi1>
    %152 = stablehlo.reshape %151 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %153 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2] : (tensor<801x800x1xi1>) -> tensor<801x800x24xi1>
    %154 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<801x800x24xi1>
    %155 = stablehlo.and %153, %154 : tensor<801x800x24xi1>
    %cst_93 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %156 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f32>) -> tensor<801x800x24xf32>
    %157 = call @_where_32(%155, %arg15, %156) : (tensor<801x800x24xi1>, tensor<801x800x24xf32>, tensor<801x800x24xf32>) -> tensor<801x800x24xf32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [1, 2, 3] : (tensor<801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %c_94 = stablehlo.constant dense<1> : tensor<ui32>
    %c_95 = stablehlo.constant dense<1> : tensor<ui32>
    %159 = stablehlo.partition_id : tensor<ui32>
    %160 = stablehlo.divide %159, %c_94 : tensor<ui32>
    %161 = stablehlo.remainder %160, %c_95 : tensor<ui32>
    %162 = stablehlo.convert %161 : (tensor<ui32>) -> tensor<i32>
    %c_96 = stablehlo.constant dense<801> : tensor<i32>
    %163 = stablehlo.multiply %162, %c_96 : tensor<i32>
    %164 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_97 = stablehlo.constant dense<0> : tensor<i32>
    %165 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %166 = stablehlo.compare  GE, %164, %165,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_98 = stablehlo.constant dense<800> : tensor<i32>
    %167 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %168 = stablehlo.compare  LT, %164, %167,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %169 = stablehlo.and %166, %168 : tensor<800xi1>
    %170 = stablehlo.reshape %169 : (tensor<800xi1>) -> tensor<800x1x1xi1>
    %c_99 = stablehlo.constant dense<true> : tensor<i1>
    %171 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i1>) -> tensor<800x1x1xi1>
    %172 = stablehlo.and %171, %170 : tensor<800x1x1xi1>
    %173 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_100 = stablehlo.constant dense<0> : tensor<i32>
    %174 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %175 = stablehlo.compare  GE, %173, %174,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_101 = stablehlo.constant dense<801> : tensor<i32>
    %176 = stablehlo.broadcast_in_dim %c_101, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %177 = stablehlo.compare  LT, %173, %176,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %178 = stablehlo.and %175, %177 : tensor<801xi1>
    %179 = stablehlo.reshape %178 : (tensor<801xi1>) -> tensor<1x801x1xi1>
    %180 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2] : (tensor<800x1x1xi1>) -> tensor<800x801x1xi1>
    %181 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2] : (tensor<1x801x1xi1>) -> tensor<800x801x1xi1>
    %182 = stablehlo.and %180, %181 : tensor<800x801x1xi1>
    %183 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_102 = stablehlo.constant dense<12> : tensor<i32>
    %184 = stablehlo.broadcast_in_dim %c_102, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %185 = stablehlo.compare  LT, %183, %184,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_103 = stablehlo.constant dense<12> : tensor<i32>
    %186 = stablehlo.broadcast_in_dim %c_103, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %187 = stablehlo.subtract %183, %186 : tensor<24xi32>
    %c_104 = stablehlo.constant dense<801> : tensor<i32>
    %188 = stablehlo.broadcast_in_dim %c_104, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %189 = stablehlo.add %187, %188 : tensor<24xi32>
    %c_105 = stablehlo.constant dense<12> : tensor<i32>
    %190 = stablehlo.broadcast_in_dim %c_105, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %191 = stablehlo.subtract %189, %190 : tensor<24xi32>
    %192 = call @_where(%185, %183, %191) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_106 = stablehlo.constant dense<0> : tensor<i32>
    %193 = stablehlo.broadcast_in_dim %c_106, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %194 = stablehlo.compare  GE, %192, %193,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_107 = stablehlo.constant dense<801> : tensor<i32>
    %195 = stablehlo.broadcast_in_dim %c_107, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %196 = stablehlo.compare  LT, %192, %195,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %197 = stablehlo.and %194, %196 : tensor<24xi1>
    %198 = stablehlo.broadcast_in_dim %163, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %199 = stablehlo.compare  GE, %192, %198,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_108 = stablehlo.constant dense<801> : tensor<i32>
    %200 = stablehlo.add %163, %c_108 : tensor<i32>
    %201 = stablehlo.broadcast_in_dim %200, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %202 = stablehlo.compare  LT, %192, %201,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %203 = stablehlo.and %199, %202 : tensor<24xi1>
    %204 = stablehlo.and %197, %203 : tensor<24xi1>
    %205 = stablehlo.reshape %204 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %206 = stablehlo.broadcast_in_dim %182, dims = [0, 1, 2] : (tensor<800x801x1xi1>) -> tensor<800x801x24xi1>
    %207 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<800x801x24xi1>
    %208 = stablehlo.and %206, %207 : tensor<800x801x24xi1>
    %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %209 = stablehlo.broadcast_in_dim %cst_109, dims = [] : (tensor<f32>) -> tensor<800x801x24xf32>
    %210 = call @_where_20(%208, %arg16, %209) : (tensor<800x801x24xi1>, tensor<800x801x24xf32>, tensor<800x801x24xf32>) -> tensor<800x801x24xf32>
    %211 = stablehlo.broadcast_in_dim %210, dims = [1, 2, 3] : (tensor<800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %212 = stablehlo.iota dim = 0 : tensor<32xi32>
    %c_110 = stablehlo.constant dense<0> : tensor<i32>
    %213:83 = stablehlo.while(%iterArg = %212, %iterArg_111 = %cst, %iterArg_112 = %arg28, %iterArg_113 = %c, %iterArg_114 = %arg30, %iterArg_115 = %arg32, %iterArg_116 = %arg34, %iterArg_117 = %arg31, %iterArg_118 = %arg33, %iterArg_119 = %arg35, %iterArg_120 = %cst_0, %iterArg_121 = %cst_1, %iterArg_122 = %cst_2, %iterArg_123 = %cst_3, %iterArg_124 = %cst_4, %iterArg_125 = %cst_5, %iterArg_126 = %cst_6, %iterArg_127 = %cst_7, %iterArg_128 = %cst_8, %iterArg_129 = %cst_9, %iterArg_130 = %cst_10, %iterArg_131 = %cst_11, %iterArg_132 = %cst_12, %iterArg_133 = %cst_13, %iterArg_134 = %cst_14, %iterArg_135 = %cst_15, %iterArg_136 = %cst_16, %iterArg_137 = %cst_17, %iterArg_138 = %cst_18, %iterArg_139 = %c_19, %iterArg_140 = %c_20, %iterArg_141 = %c_21, %iterArg_142 = %c_22, %iterArg_143 = %c_23, %iterArg_144 = %arg36, %iterArg_145 = %arg38, %iterArg_146 = %arg40, %iterArg_147 = %arg37, %iterArg_148 = %arg39, %iterArg_149 = %arg41, %iterArg_150 = %cst_24, %iterArg_151 = %cst_25, %iterArg_152 = %cst_26, %iterArg_153 = %cst_27, %iterArg_154 = %cst_28, %iterArg_155 = %cst_29, %iterArg_156 = %cst_30, %iterArg_157 = %cst_31, %iterArg_158 = %cst_32, %iterArg_159 = %cst_33, %iterArg_160 = %cst_34, %iterArg_161 = %cst_35, %iterArg_162 = %cst_36, %iterArg_163 = %cst_37, %iterArg_164 = %cst_38, %iterArg_165 = %cst_39, %iterArg_166 = %cst_40, %iterArg_167 = %cst_41, %iterArg_168 = %c_42, %iterArg_169 = %c_43, %iterArg_170 = %c_44, %iterArg_171 = %c_45, %iterArg_172 = %c_110, %iterArg_173 = %arg0, %iterArg_174 = %arg1, %iterArg_175 = %arg2, %iterArg_176 = %arg3, %iterArg_177 = %arg4, %iterArg_178 = %arg5, %iterArg_179 = %arg6, %iterArg_180 = %arg7, %iterArg_181 = %arg8, %iterArg_182 = %52, %iterArg_183 = %105, %iterArg_184 = %arg11, %iterArg_185 = %arg12, %iterArg_186 = %arg13, %iterArg_187 = %arg14, %iterArg_188 = %158, %iterArg_189 = %211, %iterArg_190 = %arg17, %iterArg_191 = %arg28, %iterArg_192 = %arg29) : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<i32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>
    cond {
      %c_193 = stablehlo.constant dense<32> : tensor<i32>
      %226 = stablehlo.compare  LT, %iterArg_172, %c_193,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %226 : tensor<i1>
    } do {
      %226 = stablehlo.dynamic_slice %iterArg, %iterArg_172, sizes = [1] : (tensor<32xi32>, tensor<i32>) -> tensor<1xi32>
      %227 = stablehlo.reshape %226 : (tensor<1xi32>) -> tensor<i32>
      %228:20 = func.call @closed_call(%iterArg_111, %iterArg_112, %iterArg_113, %iterArg_114, %iterArg_115, %iterArg_116, %iterArg_117, %iterArg_118, %iterArg_119, %iterArg_120, %iterArg_121, %iterArg_122, %iterArg_123, %iterArg_124, %iterArg_125, %iterArg_126, %iterArg_127, %iterArg_128, %iterArg_129, %iterArg_130, %iterArg_131, %iterArg_132, %iterArg_133, %iterArg_134, %iterArg_135, %iterArg_136, %iterArg_137, %iterArg_138, %iterArg_139, %iterArg_140, %iterArg_141, %iterArg_142, %iterArg_143, %iterArg_144, %iterArg_145, %iterArg_146, %iterArg_147, %iterArg_148, %iterArg_149, %iterArg_150, %iterArg_151, %iterArg_152, %iterArg_153, %iterArg_154, %iterArg_155, %iterArg_156, %iterArg_157, %iterArg_158, %iterArg_159, %iterArg_160, %iterArg_161, %iterArg_162, %iterArg_163, %iterArg_164, %iterArg_165, %iterArg_166, %iterArg_167, %iterArg_168, %iterArg_169, %iterArg_170, %iterArg_171, %iterArg_173, %iterArg_174, %iterArg_175, %iterArg_176, %iterArg_177, %iterArg_178, %iterArg_179, %iterArg_180, %iterArg_181, %iterArg_182, %iterArg_183, %iterArg_184, %iterArg_185, %iterArg_186, %iterArg_187, %iterArg_188, %iterArg_189, %iterArg_190, %iterArg_191, %iterArg_192, %227) : (tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>, tensor<i32>) -> (tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>)
      %c_193 = stablehlo.constant dense<1> : tensor<i32>
      %229 = stablehlo.add %iterArg_172, %c_193 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_111, %iterArg_112, %iterArg_113, %iterArg_114, %iterArg_115, %iterArg_116, %iterArg_117, %iterArg_118, %iterArg_119, %iterArg_120, %iterArg_121, %iterArg_122, %iterArg_123, %iterArg_124, %iterArg_125, %iterArg_126, %iterArg_127, %iterArg_128, %iterArg_129, %iterArg_130, %iterArg_131, %iterArg_132, %iterArg_133, %iterArg_134, %iterArg_135, %iterArg_136, %iterArg_137, %iterArg_138, %iterArg_139, %iterArg_140, %iterArg_141, %iterArg_142, %iterArg_143, %iterArg_144, %iterArg_145, %iterArg_146, %iterArg_147, %iterArg_148, %iterArg_149, %iterArg_150, %iterArg_151, %iterArg_152, %iterArg_153, %iterArg_154, %iterArg_155, %iterArg_156, %iterArg_157, %iterArg_158, %iterArg_159, %iterArg_160, %iterArg_161, %iterArg_162, %iterArg_163, %iterArg_164, %iterArg_165, %iterArg_166, %iterArg_167, %iterArg_168, %iterArg_169, %iterArg_170, %iterArg_171, %229, %228#0, %228#1, %228#2, %228#3, %228#4, %228#5, %228#6, %228#7, %228#8, %228#9, %228#10, %228#11, %228#12, %228#13, %228#14, %228#15, %228#16, %228#17, %228#18, %228#19 : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<i32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>
    }
    %214 = stablehlo.slice %213#72 [0:1, 0:800, 0:801, 0:24] : (tensor<1x800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %215 = stablehlo.reshape %214 : (tensor<1x800x801x24xf32>) -> tensor<800x801x24xf32>
    %216 = "stablehlo.all_reduce"(%215) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %226 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %226 : tensor<f32>
    }) : (tensor<800x801x24xf32>) -> tensor<800x801x24xf32>
    %217 = stablehlo.slice %213#73 [0:1, 0:801, 0:800, 0:24] : (tensor<1x801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %218 = stablehlo.reshape %217 : (tensor<1x801x800x24xf32>) -> tensor<801x800x24xf32>
    %219 = "stablehlo.all_reduce"(%218) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %226 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %226 : tensor<f32>
    }) : (tensor<801x800x24xf32>) -> tensor<801x800x24xf32>
    %220 = stablehlo.slice %213#78 [0:1, 0:801, 0:800, 0:24] : (tensor<1x801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %221 = stablehlo.reshape %220 : (tensor<1x801x800x24xf32>) -> tensor<801x800x24xf32>
    %222 = "stablehlo.all_reduce"(%221) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %226 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %226 : tensor<f32>
    }) : (tensor<801x800x24xf32>) -> tensor<801x800x24xf32>
    %223 = stablehlo.slice %213#79 [0:1, 0:800, 0:801, 0:24] : (tensor<1x800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %224 = stablehlo.reshape %223 : (tensor<1x800x801x24xf32>) -> tensor<800x801x24xf32>
    %225 = "stablehlo.all_reduce"(%224) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %226 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %226 : tensor<f32>
    }) : (tensor<800x801x24xf32>) -> tensor<800x801x24xf32>
    return %213#63, %213#64, %213#65, %213#66, %213#67, %213#68, %213#69, %213#70, %213#71, %216, %219, %213#74, %213#75, %213#76, %213#77, %222, %225, %213#80, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %213#81, %213#82 : tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<800x801x24xf32>, tensor<801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<801x800x24xf32>, tensor<800x801x24xf32>, tensor<800x24x801xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0xi32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<f32>, tensor<i32>
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
  func.func private @closed_call(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<6x3xi32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>, %arg9: tensor<0xf32>, %arg10: tensor<1x24x1xf32>, %arg11: tensor<1x24x1xf32>, %arg12: tensor<1x24x1xf32>, %arg13: tensor<24x1x1xf32>, %arg14: tensor<24x1x1xf32>, %arg15: tensor<24x1x1xf32>, %arg16: tensor<24x1x1xf32>, %arg17: tensor<24x1x1xf32>, %arg18: tensor<24x1x1xf32>, %arg19: tensor<1x1x24xf32>, %arg20: tensor<1x1x24xf32>, %arg21: tensor<1x1x24xf32>, %arg22: tensor<1x1x24xf32>, %arg23: tensor<1x1x24xf32>, %arg24: tensor<1x1x24xf32>, %arg25: tensor<1x24x1xf32>, %arg26: tensor<1x24x1xf32>, %arg27: tensor<1x24x1xf32>, %arg28: tensor<3xi32>, %arg29: tensor<6xi32>, %arg30: tensor<3xi32>, %arg31: tensor<6x5xi32>, %arg32: tensor<6x3xi32>, %arg33: tensor<f32>, %arg34: tensor<f32>, %arg35: tensor<f32>, %arg36: tensor<801x801x801xf32>, %arg37: tensor<801x800x801xf32>, %arg38: tensor<800x801x801xf32>, %arg39: tensor<1x24x1xf32>, %arg40: tensor<1x24x1xf32>, %arg41: tensor<1x24x1xf32>, %arg42: tensor<24x1x1xf32>, %arg43: tensor<24x1x1xf32>, %arg44: tensor<24x1x1xf32>, %arg45: tensor<24x1x1xf32>, %arg46: tensor<24x1x1xf32>, %arg47: tensor<24x1x1xf32>, %arg48: tensor<1x1x24xf32>, %arg49: tensor<1x1x24xf32>, %arg50: tensor<1x1x24xf32>, %arg51: tensor<1x1x24xf32>, %arg52: tensor<1x1x24xf32>, %arg53: tensor<1x1x24xf32>, %arg54: tensor<1x24x1xf32>, %arg55: tensor<1x24x1xf32>, %arg56: tensor<1x24x1xf32>, %arg57: tensor<3xi32>, %arg58: tensor<6xi32>, %arg59: tensor<3xi32>, %arg60: tensor<6x5xi32>, %arg61: tensor<801x801x801xf32>, %arg62: tensor<801x800x801xf32>, %arg63: tensor<800x801x801xf32>, %arg64: tensor<800x800x801xf32>, %arg65: tensor<800x801x801xf32>, %arg66: tensor<801x800x801xf32>, %arg67: tensor<800x24x801xf32>, %arg68: tensor<24x800x801xf32>, %arg69: tensor<24x801x801xf32>, %arg70: tensor<1x800x801x24xf32>, %arg71: tensor<1x801x800x24xf32>, %arg72: tensor<801x24x801xf32>, %arg73: tensor<801x24x801xf32>, %arg74: tensor<24x801x801xf32>, %arg75: tensor<24x800x801xf32>, %arg76: tensor<1x801x800x24xf32>, %arg77: tensor<1x800x801x24xf32>, %arg78: tensor<800x24x801xf32>, %arg79: tensor<f32>, %arg80: tensor<i32>, %arg81: tensor<i32>) -> (tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg81, %c : tensor<i32>
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
    %15 = stablehlo.slice %arg70 [0:1, 0:800, 0:801, 0:24] : (tensor<1x800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %16 = stablehlo.reshape %15 : (tensor<1x800x801x24xf32>) -> tensor<800x801x24xf32>
    %17 = stablehlo.slice %arg71 [0:1, 0:801, 0:800, 0:24] : (tensor<1x801x800x24xf32>) -> tensor<1x801x800x24xf32>
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
    %60 = call @_where_65(%58, %arg67, %59) : (tensor<800x24x801xi1>, tensor<800x24x801xf32>, tensor<800x24x801xf32>) -> tensor<800x24x801xf32>
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
    %102 = call @_where_74(%100, %arg68, %101) : (tensor<24x800x801xi1>, tensor<24x800x801xf32>, tensor<24x800x801xf32>) -> tensor<24x800x801xf32>
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
    %144 = call @_where_81(%142, %arg69, %143) : (tensor<24x801x801xi1>, tensor<24x801x801xf32>, tensor<24x801x801xf32>) -> tensor<24x801x801xf32>
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
    %280 = call @_where_90(%278, %arg72, %279) : (tensor<801x24x801xi1>, tensor<801x24x801xf32>, tensor<801x24x801xf32>) -> tensor<801x24x801xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %281 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f32>) -> tensor<801x801x1xf32>
    %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %282 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f32>) -> tensor<801x801x1xf32>
    %283 = stablehlo.slice %arg62 [0:801, 0:800, 0:1] : (tensor<801x800x801xf32>) -> tensor<801x800x1xf32>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %284 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<801x800x1xf32>
    %285 = "stablehlo.collective_permute"(%283) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<801x800x1xf32>) -> tensor<801x800x1xf32>
    %286 = stablehlo.slice %arg63 [0:800, 0:801, 0:1] : (tensor<800x801x801xf32>) -> tensor<800x801x1xf32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %287 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<800x801x1xf32>
    %288 = "stablehlo.collective_permute"(%286) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<800x801x1xf32>) -> tensor<800x801x1xf32>
    %289 = stablehlo.transpose %arg16, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %290 = stablehlo.transpose %arg17, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %291 = stablehlo.transpose %arg18, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %292 = stablehlo.transpose %arg19, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %293 = stablehlo.transpose %arg20, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %294 = stablehlo.transpose %arg21, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %295 = stablehlo.transpose %arg22, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %296 = stablehlo.transpose %arg23, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %297 = stablehlo.transpose %arg24, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %298 = stablehlo.transpose %arg25, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %299 = stablehlo.transpose %arg26, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %300 = stablehlo.transpose %arg27, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %301 = stablehlo.transpose %arg10, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %302 = stablehlo.transpose %arg11, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %303 = stablehlo.transpose %arg12, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %304 = stablehlo.transpose %arg13, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %305 = stablehlo.transpose %arg14, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %306 = stablehlo.transpose %arg15, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %307 = stablehlo.slice %14 [0:1, 0:1] : (tensor<7x3xi32>) -> tensor<1x1xi32>
    %308 = stablehlo.reshape %307 : (tensor<1x1xi32>) -> tensor<i32>
    %c_82 = stablehlo.constant dense<0> : tensor<i32>
    %309 = stablehlo.compare  LT, %308, %c_82,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_83 = stablehlo.constant dense<3> : tensor<i32>
    %310 = stablehlo.add %308, %c_83 : tensor<i32>
    %311 = stablehlo.select %309, %310, %308 : tensor<i1>, tensor<i32>
    %312 = stablehlo.dynamic_slice %arg28, %311, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
    %313 = stablehlo.reshape %312 : (tensor<1xi32>) -> tensor<i32>
    %314 = stablehlo.slice %14 [0:1, 0:3] : (tensor<7x3xi32>) -> tensor<1x3xi32>
    %c_84 = stablehlo.constant dense<0> : tensor<i32>
    %315 = stablehlo.broadcast_in_dim %c_84, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_85 = stablehlo.constant dense<0> : tensor<i32>
    %316 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %317 = stablehlo.concatenate %315, %316, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %318 = "stablehlo.scatter"(%314, %317, %313) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg82: tensor<i32>, %arg83: tensor<i32>):
      stablehlo.return %arg83 : tensor<i32>
    }) : (tensor<1x3xi32>, tensor<2xi32>, tensor<i32>) -> tensor<1x3xi32>
    %c_86 = stablehlo.constant dense<0> : tensor<i32>
    %319 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<6xi32>
    %320 = stablehlo.compare  LT, %arg29, %319,  SIGNED : (tensor<6xi32>, tensor<6xi32>) -> tensor<6xi1>
    %c_87 = stablehlo.constant dense<7> : tensor<i32>
    %321 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<6xi32>
    %322 = stablehlo.add %arg29, %321 : tensor<6xi32>
    %323 = stablehlo.select %320, %322, %arg29 : tensor<6xi1>, tensor<6xi32>
    %324 = stablehlo.broadcast_in_dim %323, dims = [0] : (tensor<6xi32>) -> tensor<6x1xi32>
    %325 = "stablehlo.gather"(%14, %324) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<7x3xi32>, tensor<6x1xi32>) -> tensor<6x3xi32>
    %c_88 = stablehlo.constant dense<0> : tensor<i32>
    %326 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %327 = stablehlo.compare  LT, %arg30, %326,  SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
    %c_89 = stablehlo.constant dense<3> : tensor<i32>
    %328 = stablehlo.broadcast_in_dim %c_89, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %329 = stablehlo.add %arg30, %328 : tensor<3xi32>
    %330 = stablehlo.select %327, %329, %arg30 : tensor<3xi1>, tensor<3xi32>
    %331 = stablehlo.broadcast_in_dim %330, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %332 = "stablehlo.gather"(%325, %331) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x3xi32>, tensor<3x1xi32>) -> tensor<6x3xi32>
    %333 = stablehlo.concatenate %318, %332, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
    %334 = stablehlo.transpose %284, dims = [2, 0, 1] : (tensor<801x800x1xf32>) -> tensor<1x801x800xf32>
    %335 = stablehlo.transpose %285, dims = [2, 0, 1] : (tensor<801x800x1xf32>) -> tensor<1x801x800xf32>
    %336 = stablehlo.transpose %287, dims = [2, 0, 1] : (tensor<800x801x1xf32>) -> tensor<1x800x801xf32>
    %337 = stablehlo.transpose %288, dims = [2, 0, 1] : (tensor<800x801x1xf32>) -> tensor<1x800x801xf32>
    %338 = stablehlo.transpose %281, dims = [2, 0, 1] : (tensor<801x801x1xf32>) -> tensor<1x801x801xf32>
    %339 = stablehlo.transpose %282, dims = [2, 0, 1] : (tensor<801x801x1xf32>) -> tensor<1x801x801xf32>
    %340 = stablehlo.transpose %arg65, dims = [2, 0, 1] : (tensor<800x801x801xf32>) -> tensor<801x800x801xf32>
    %341 = stablehlo.transpose %arg66, dims = [2, 0, 1] : (tensor<801x800x801xf32>) -> tensor<801x801x800xf32>
    %342 = stablehlo.transpose %arg64, dims = [2, 0, 1] : (tensor<800x800x801xf32>) -> tensor<801x800x800xf32>
    %343 = stablehlo.transpose %arg62, dims = [2, 0, 1] : (tensor<801x800x801xf32>) -> tensor<801x801x800xf32>
    %344 = stablehlo.transpose %arg63, dims = [2, 0, 1] : (tensor<800x801x801xf32>) -> tensor<801x800x801xf32>
    %345 = stablehlo.transpose %arg61, dims = [2, 0, 1] : (tensor<801x801x801xf32>) -> tensor<801x801x801xf32>
    %346 = stablehlo.transpose %144, dims = [2, 0, 1] : (tensor<24x801x801xf32>) -> tensor<801x24x801xf32>
    %347 = stablehlo.transpose %191, dims = [2, 0, 1] : (tensor<800x801x24xf32>) -> tensor<24x800x801xf32>
    %348 = stablehlo.transpose %238, dims = [2, 0, 1] : (tensor<801x800x24xf32>) -> tensor<24x801x800xf32>
    %349 = stablehlo.transpose %280, dims = [2, 0, 1] : (tensor<801x24x801xf32>) -> tensor<801x801x24xf32>
    %350 = stablehlo.transpose %60, dims = [2, 0, 1] : (tensor<800x24x801xf32>) -> tensor<801x800x24xf32>
    %351 = stablehlo.transpose %102, dims = [2, 0, 1] : (tensor<24x800x801xf32>) -> tensor<801x24x800xf32>
    %352:9 = stablehlo.custom_call @beamz_cuda_sharded(%340, %341, %342, %343, %344, %345, %arg4, %arg5, %arg3, %arg7, %arg8, %arg6, %arg31, %289, %290, %291, %292, %293, %294, %295, %296, %297, %298, %299, %300, %301, %302, %303, %304, %305, %306, %346, %347, %348, %349, %350, %351, %arg9, %arg9, %arg9, %333, %334, %335, %336, %337, %338, %339) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 0 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<801x800x801xf32>, tensor<801x801x800xf32>, tensor<801x800x800xf32>, tensor<801x801x800xf32>, tensor<801x800x801xf32>, tensor<801x801x801xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<801x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x800xf32>, tensor<801x801x24xf32>, tensor<801x800x24xf32>, tensor<801x24x800xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<1x801x800xf32>, tensor<1x801x800xf32>, tensor<1x800x801xf32>, tensor<1x800x801xf32>, tensor<1x801x801xf32>, tensor<1x801x801xf32>) -> (tensor<801x800x801xf32>, tensor<801x801x800xf32>, tensor<801x800x800xf32>, tensor<801x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x800xf32>, tensor<801x801x24xf32>, tensor<801x800x24xf32>, tensor<801x24x800xf32>)
    %353 = stablehlo.transpose %352#2, dims = [1, 2, 0] : (tensor<801x800x800xf32>) -> tensor<800x800x801xf32>
    %354 = stablehlo.transpose %352#0, dims = [1, 2, 0] : (tensor<801x800x801xf32>) -> tensor<800x801x801xf32>
    %355 = stablehlo.transpose %352#1, dims = [1, 2, 0] : (tensor<801x801x800xf32>) -> tensor<801x800x801xf32>
    %356 = stablehlo.transpose %352#7, dims = [1, 2, 0] : (tensor<801x800x24xf32>) -> tensor<800x24x801xf32>
    %357 = stablehlo.transpose %352#8, dims = [1, 2, 0] : (tensor<801x24x800xf32>) -> tensor<24x800x801xf32>
    %358 = stablehlo.transpose %352#3, dims = [1, 2, 0] : (tensor<801x24x801xf32>) -> tensor<24x801x801xf32>
    %359 = stablehlo.transpose %352#4, dims = [1, 2, 0] : (tensor<24x800x801xf32>) -> tensor<800x801x24xf32>
    %360 = stablehlo.transpose %352#5, dims = [1, 2, 0] : (tensor<24x801x800xf32>) -> tensor<801x800x24xf32>
    %361 = stablehlo.transpose %352#6, dims = [1, 2, 0] : (tensor<801x801x24xf32>) -> tensor<801x24x801xf32>
    %362 = stablehlo.broadcast_in_dim %359, dims = [1, 2, 3] : (tensor<800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %363 = stablehlo.broadcast_in_dim %360, dims = [1, 2, 3] : (tensor<801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %c_90 = stablehlo.constant dense<1> : tensor<ui32>
    %c_91 = stablehlo.constant dense<1> : tensor<ui32>
    %364 = stablehlo.partition_id : tensor<ui32>
    %365 = stablehlo.divide %364, %c_90 : tensor<ui32>
    %366 = stablehlo.remainder %365, %c_91 : tensor<ui32>
    %367 = stablehlo.convert %366 : (tensor<ui32>) -> tensor<i32>
    %c_92 = stablehlo.constant dense<801> : tensor<i32>
    %368 = stablehlo.multiply %367, %c_92 : tensor<i32>
    %c_93 = stablehlo.constant dense<2> : tensor<i32>
    %369 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %370 = stablehlo.broadcast_in_dim %368, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_94 = stablehlo.constant dense<0> : tensor<i32>
    %371 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %372 = stablehlo.concatenate %369, %370, %371, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %373 = stablehlo.broadcast_in_dim %372, dims = [1] : (tensor<3xi32>) -> tensor<1x3xi32>
    %374 = stablehlo.concatenate %373, %arg32, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
    %375 = stablehlo.slice %arg76 [0:1, 0:801, 0:800, 0:24] : (tensor<1x801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %376 = stablehlo.reshape %375 : (tensor<1x801x800x24xf32>) -> tensor<801x800x24xf32>
    %377 = stablehlo.slice %arg77 [0:1, 0:800, 0:801, 0:24] : (tensor<1x800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %378 = stablehlo.reshape %377 : (tensor<1x800x801x24xf32>) -> tensor<800x801x24xf32>
    %379 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_95 = stablehlo.constant dense<0> : tensor<i32>
    %380 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %381 = stablehlo.compare  GE, %379, %380,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_96 = stablehlo.constant dense<801> : tensor<i32>
    %382 = stablehlo.broadcast_in_dim %c_96, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %383 = stablehlo.compare  LT, %379, %382,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %384 = stablehlo.and %381, %383 : tensor<801xi1>
    %385 = stablehlo.reshape %384 : (tensor<801xi1>) -> tensor<801x1x1xi1>
    %c_97 = stablehlo.constant dense<true> : tensor<i1>
    %386 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i1>) -> tensor<801x1x1xi1>
    %387 = stablehlo.and %386, %385 : tensor<801x1x1xi1>
    %388 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_98 = stablehlo.constant dense<12> : tensor<i32>
    %389 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %390 = stablehlo.compare  LT, %388, %389,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_99 = stablehlo.constant dense<12> : tensor<i32>
    %391 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %392 = stablehlo.subtract %388, %391 : tensor<24xi32>
    %c_100 = stablehlo.constant dense<801> : tensor<i32>
    %393 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %394 = stablehlo.add %392, %393 : tensor<24xi32>
    %c_101 = stablehlo.constant dense<12> : tensor<i32>
    %395 = stablehlo.broadcast_in_dim %c_101, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %396 = stablehlo.subtract %394, %395 : tensor<24xi32>
    %397 = call @_where(%390, %388, %396) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_102 = stablehlo.constant dense<0> : tensor<i32>
    %398 = stablehlo.broadcast_in_dim %c_102, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %399 = stablehlo.compare  GE, %397, %398,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_103 = stablehlo.constant dense<801> : tensor<i32>
    %400 = stablehlo.broadcast_in_dim %c_103, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %401 = stablehlo.compare  LT, %397, %400,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %402 = stablehlo.and %399, %401 : tensor<24xi1>
    %403 = stablehlo.reshape %402 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %404 = stablehlo.broadcast_in_dim %387, dims = [0, 1, 2] : (tensor<801x1x1xi1>) -> tensor<801x24x1xi1>
    %405 = stablehlo.broadcast_in_dim %403, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<801x24x1xi1>
    %406 = stablehlo.and %404, %405 : tensor<801x24x1xi1>
    %407 = stablehlo.iota dim = 0 : tensor<801xi32>
    %408 = stablehlo.broadcast_in_dim %368, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %409 = stablehlo.add %407, %408 : tensor<801xi32>
    %c_104 = stablehlo.constant dense<0> : tensor<i32>
    %410 = stablehlo.broadcast_in_dim %c_104, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %411 = stablehlo.compare  GE, %409, %410,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_105 = stablehlo.constant dense<800> : tensor<i32>
    %412 = stablehlo.broadcast_in_dim %c_105, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %413 = stablehlo.compare  LT, %409, %412,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %414 = stablehlo.and %411, %413 : tensor<801xi1>
    %415 = stablehlo.reshape %414 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %416 = stablehlo.broadcast_in_dim %406, dims = [0, 1, 2] : (tensor<801x24x1xi1>) -> tensor<801x24x801xi1>
    %417 = stablehlo.broadcast_in_dim %415, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<801x24x801xi1>
    %418 = stablehlo.and %416, %417 : tensor<801x24x801xi1>
    %cst_106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %419 = stablehlo.broadcast_in_dim %cst_106, dims = [] : (tensor<f32>) -> tensor<801x24x801xf32>
    %420 = call @_where_90(%418, %arg73, %419) : (tensor<801x24x801xi1>, tensor<801x24x801xf32>, tensor<801x24x801xf32>) -> tensor<801x24x801xf32>
    %421 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_107 = stablehlo.constant dense<12> : tensor<i32>
    %422 = stablehlo.broadcast_in_dim %c_107, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %423 = stablehlo.compare  LT, %421, %422,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_108 = stablehlo.constant dense<12> : tensor<i32>
    %424 = stablehlo.broadcast_in_dim %c_108, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %425 = stablehlo.subtract %421, %424 : tensor<24xi32>
    %c_109 = stablehlo.constant dense<801> : tensor<i32>
    %426 = stablehlo.broadcast_in_dim %c_109, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %427 = stablehlo.add %425, %426 : tensor<24xi32>
    %c_110 = stablehlo.constant dense<12> : tensor<i32>
    %428 = stablehlo.broadcast_in_dim %c_110, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %429 = stablehlo.subtract %427, %428 : tensor<24xi32>
    %430 = call @_where(%423, %421, %429) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_111 = stablehlo.constant dense<0> : tensor<i32>
    %431 = stablehlo.broadcast_in_dim %c_111, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %432 = stablehlo.compare  GE, %430, %431,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_112 = stablehlo.constant dense<801> : tensor<i32>
    %433 = stablehlo.broadcast_in_dim %c_112, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %434 = stablehlo.compare  LT, %430, %433,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %435 = stablehlo.and %432, %434 : tensor<24xi1>
    %436 = stablehlo.reshape %435 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_113 = stablehlo.constant dense<true> : tensor<i1>
    %437 = stablehlo.broadcast_in_dim %c_113, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %438 = stablehlo.and %437, %436 : tensor<24x1x1xi1>
    %439 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_114 = stablehlo.constant dense<0> : tensor<i32>
    %440 = stablehlo.broadcast_in_dim %c_114, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %441 = stablehlo.compare  GE, %439, %440,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_115 = stablehlo.constant dense<801> : tensor<i32>
    %442 = stablehlo.broadcast_in_dim %c_115, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %443 = stablehlo.compare  LT, %439, %442,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %444 = stablehlo.and %441, %443 : tensor<801xi1>
    %445 = stablehlo.reshape %444 : (tensor<801xi1>) -> tensor<1x801x1xi1>
    %446 = stablehlo.broadcast_in_dim %438, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x801x1xi1>
    %447 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2] : (tensor<1x801x1xi1>) -> tensor<24x801x1xi1>
    %448 = stablehlo.and %446, %447 : tensor<24x801x1xi1>
    %449 = stablehlo.iota dim = 0 : tensor<801xi32>
    %450 = stablehlo.broadcast_in_dim %368, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %451 = stablehlo.add %449, %450 : tensor<801xi32>
    %c_116 = stablehlo.constant dense<0> : tensor<i32>
    %452 = stablehlo.broadcast_in_dim %c_116, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %453 = stablehlo.compare  GE, %451, %452,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_117 = stablehlo.constant dense<800> : tensor<i32>
    %454 = stablehlo.broadcast_in_dim %c_117, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %455 = stablehlo.compare  LT, %451, %454,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %456 = stablehlo.and %453, %455 : tensor<801xi1>
    %457 = stablehlo.reshape %456 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %458 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2] : (tensor<24x801x1xi1>) -> tensor<24x801x801xi1>
    %459 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<24x801x801xi1>
    %460 = stablehlo.and %458, %459 : tensor<24x801x801xi1>
    %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %461 = stablehlo.broadcast_in_dim %cst_118, dims = [] : (tensor<f32>) -> tensor<24x801x801xf32>
    %462 = call @_where_81(%460, %arg74, %461) : (tensor<24x801x801xi1>, tensor<24x801x801xf32>, tensor<24x801x801xf32>) -> tensor<24x801x801xf32>
    %463 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_119 = stablehlo.constant dense<12> : tensor<i32>
    %464 = stablehlo.broadcast_in_dim %c_119, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %465 = stablehlo.compare  LT, %463, %464,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_120 = stablehlo.constant dense<12> : tensor<i32>
    %466 = stablehlo.broadcast_in_dim %c_120, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %467 = stablehlo.subtract %463, %466 : tensor<24xi32>
    %c_121 = stablehlo.constant dense<801> : tensor<i32>
    %468 = stablehlo.broadcast_in_dim %c_121, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %469 = stablehlo.add %467, %468 : tensor<24xi32>
    %c_122 = stablehlo.constant dense<12> : tensor<i32>
    %470 = stablehlo.broadcast_in_dim %c_122, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %471 = stablehlo.subtract %469, %470 : tensor<24xi32>
    %472 = call @_where(%465, %463, %471) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_123 = stablehlo.constant dense<0> : tensor<i32>
    %473 = stablehlo.broadcast_in_dim %c_123, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %474 = stablehlo.compare  GE, %472, %473,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_124 = stablehlo.constant dense<801> : tensor<i32>
    %475 = stablehlo.broadcast_in_dim %c_124, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %476 = stablehlo.compare  LT, %472, %475,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %477 = stablehlo.and %474, %476 : tensor<24xi1>
    %478 = stablehlo.reshape %477 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_125 = stablehlo.constant dense<true> : tensor<i1>
    %479 = stablehlo.broadcast_in_dim %c_125, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %480 = stablehlo.and %479, %478 : tensor<24x1x1xi1>
    %481 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_126 = stablehlo.constant dense<0> : tensor<i32>
    %482 = stablehlo.broadcast_in_dim %c_126, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %483 = stablehlo.compare  GE, %481, %482,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_127 = stablehlo.constant dense<800> : tensor<i32>
    %484 = stablehlo.broadcast_in_dim %c_127, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %485 = stablehlo.compare  LT, %481, %484,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %486 = stablehlo.and %483, %485 : tensor<800xi1>
    %487 = stablehlo.reshape %486 : (tensor<800xi1>) -> tensor<1x800x1xi1>
    %488 = stablehlo.broadcast_in_dim %480, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x800x1xi1>
    %489 = stablehlo.broadcast_in_dim %487, dims = [0, 1, 2] : (tensor<1x800x1xi1>) -> tensor<24x800x1xi1>
    %490 = stablehlo.and %488, %489 : tensor<24x800x1xi1>
    %491 = stablehlo.iota dim = 0 : tensor<801xi32>
    %492 = stablehlo.broadcast_in_dim %368, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %493 = stablehlo.add %491, %492 : tensor<801xi32>
    %c_128 = stablehlo.constant dense<0> : tensor<i32>
    %494 = stablehlo.broadcast_in_dim %c_128, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %495 = stablehlo.compare  GE, %493, %494,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_129 = stablehlo.constant dense<801> : tensor<i32>
    %496 = stablehlo.broadcast_in_dim %c_129, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %497 = stablehlo.compare  LT, %493, %496,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %498 = stablehlo.and %495, %497 : tensor<801xi1>
    %499 = stablehlo.reshape %498 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %500 = stablehlo.broadcast_in_dim %490, dims = [0, 1, 2] : (tensor<24x800x1xi1>) -> tensor<24x800x801xi1>
    %501 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<24x800x801xi1>
    %502 = stablehlo.and %500, %501 : tensor<24x800x801xi1>
    %cst_130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %503 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<f32>) -> tensor<24x800x801xf32>
    %504 = call @_where_74(%502, %arg75, %503) : (tensor<24x800x801xi1>, tensor<24x800x801xf32>, tensor<24x800x801xf32>) -> tensor<24x800x801xf32>
    %505 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_131 = stablehlo.constant dense<0> : tensor<i32>
    %506 = stablehlo.broadcast_in_dim %c_131, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %507 = stablehlo.compare  GE, %505, %506,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_132 = stablehlo.constant dense<801> : tensor<i32>
    %508 = stablehlo.broadcast_in_dim %c_132, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %509 = stablehlo.compare  LT, %505, %508,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %510 = stablehlo.and %507, %509 : tensor<801xi1>
    %511 = stablehlo.reshape %510 : (tensor<801xi1>) -> tensor<801x1x1xi1>
    %c_133 = stablehlo.constant dense<true> : tensor<i1>
    %512 = stablehlo.broadcast_in_dim %c_133, dims = [] : (tensor<i1>) -> tensor<801x1x1xi1>
    %513 = stablehlo.and %512, %511 : tensor<801x1x1xi1>
    %514 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_134 = stablehlo.constant dense<0> : tensor<i32>
    %515 = stablehlo.broadcast_in_dim %c_134, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %516 = stablehlo.compare  GE, %514, %515,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_135 = stablehlo.constant dense<800> : tensor<i32>
    %517 = stablehlo.broadcast_in_dim %c_135, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %518 = stablehlo.compare  LT, %514, %517,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %519 = stablehlo.and %516, %518 : tensor<800xi1>
    %520 = stablehlo.reshape %519 : (tensor<800xi1>) -> tensor<1x800x1xi1>
    %521 = stablehlo.broadcast_in_dim %513, dims = [0, 1, 2] : (tensor<801x1x1xi1>) -> tensor<801x800x1xi1>
    %522 = stablehlo.broadcast_in_dim %520, dims = [0, 1, 2] : (tensor<1x800x1xi1>) -> tensor<801x800x1xi1>
    %523 = stablehlo.and %521, %522 : tensor<801x800x1xi1>
    %524 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_136 = stablehlo.constant dense<12> : tensor<i32>
    %525 = stablehlo.broadcast_in_dim %c_136, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %526 = stablehlo.compare  LT, %524, %525,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_137 = stablehlo.constant dense<12> : tensor<i32>
    %527 = stablehlo.broadcast_in_dim %c_137, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %528 = stablehlo.subtract %524, %527 : tensor<24xi32>
    %c_138 = stablehlo.constant dense<801> : tensor<i32>
    %529 = stablehlo.broadcast_in_dim %c_138, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %530 = stablehlo.add %528, %529 : tensor<24xi32>
    %c_139 = stablehlo.constant dense<12> : tensor<i32>
    %531 = stablehlo.broadcast_in_dim %c_139, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %532 = stablehlo.subtract %530, %531 : tensor<24xi32>
    %533 = call @_where(%526, %524, %532) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_140 = stablehlo.constant dense<0> : tensor<i32>
    %534 = stablehlo.broadcast_in_dim %c_140, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %535 = stablehlo.compare  GE, %533, %534,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_141 = stablehlo.constant dense<801> : tensor<i32>
    %536 = stablehlo.broadcast_in_dim %c_141, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %537 = stablehlo.compare  LT, %533, %536,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %538 = stablehlo.and %535, %537 : tensor<24xi1>
    %539 = stablehlo.broadcast_in_dim %368, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %540 = stablehlo.compare  GE, %533, %539,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_142 = stablehlo.constant dense<801> : tensor<i32>
    %541 = stablehlo.add %368, %c_142 : tensor<i32>
    %542 = stablehlo.broadcast_in_dim %541, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %543 = stablehlo.compare  LT, %533, %542,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %544 = stablehlo.and %540, %543 : tensor<24xi1>
    %545 = stablehlo.and %538, %544 : tensor<24xi1>
    %546 = stablehlo.reshape %545 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %547 = stablehlo.broadcast_in_dim %523, dims = [0, 1, 2] : (tensor<801x800x1xi1>) -> tensor<801x800x24xi1>
    %548 = stablehlo.broadcast_in_dim %546, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<801x800x24xi1>
    %549 = stablehlo.and %547, %548 : tensor<801x800x24xi1>
    %cst_143 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %550 = stablehlo.broadcast_in_dim %cst_143, dims = [] : (tensor<f32>) -> tensor<801x800x24xf32>
    %551 = call @_where_84(%549, %376, %550) : (tensor<801x800x24xi1>, tensor<801x800x24xf32>, tensor<801x800x24xf32>) -> tensor<801x800x24xf32>
    %552 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_144 = stablehlo.constant dense<0> : tensor<i32>
    %553 = stablehlo.broadcast_in_dim %c_144, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %554 = stablehlo.compare  GE, %552, %553,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_145 = stablehlo.constant dense<800> : tensor<i32>
    %555 = stablehlo.broadcast_in_dim %c_145, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %556 = stablehlo.compare  LT, %552, %555,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %557 = stablehlo.and %554, %556 : tensor<800xi1>
    %558 = stablehlo.reshape %557 : (tensor<800xi1>) -> tensor<800x1x1xi1>
    %c_146 = stablehlo.constant dense<true> : tensor<i1>
    %559 = stablehlo.broadcast_in_dim %c_146, dims = [] : (tensor<i1>) -> tensor<800x1x1xi1>
    %560 = stablehlo.and %559, %558 : tensor<800x1x1xi1>
    %561 = stablehlo.iota dim = 0 : tensor<801xi32>
    %c_147 = stablehlo.constant dense<0> : tensor<i32>
    %562 = stablehlo.broadcast_in_dim %c_147, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %563 = stablehlo.compare  GE, %561, %562,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_148 = stablehlo.constant dense<801> : tensor<i32>
    %564 = stablehlo.broadcast_in_dim %c_148, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %565 = stablehlo.compare  LT, %561, %564,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %566 = stablehlo.and %563, %565 : tensor<801xi1>
    %567 = stablehlo.reshape %566 : (tensor<801xi1>) -> tensor<1x801x1xi1>
    %568 = stablehlo.broadcast_in_dim %560, dims = [0, 1, 2] : (tensor<800x1x1xi1>) -> tensor<800x801x1xi1>
    %569 = stablehlo.broadcast_in_dim %567, dims = [0, 1, 2] : (tensor<1x801x1xi1>) -> tensor<800x801x1xi1>
    %570 = stablehlo.and %568, %569 : tensor<800x801x1xi1>
    %571 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_149 = stablehlo.constant dense<12> : tensor<i32>
    %572 = stablehlo.broadcast_in_dim %c_149, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %573 = stablehlo.compare  LT, %571, %572,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_150 = stablehlo.constant dense<12> : tensor<i32>
    %574 = stablehlo.broadcast_in_dim %c_150, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %575 = stablehlo.subtract %571, %574 : tensor<24xi32>
    %c_151 = stablehlo.constant dense<801> : tensor<i32>
    %576 = stablehlo.broadcast_in_dim %c_151, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %577 = stablehlo.add %575, %576 : tensor<24xi32>
    %c_152 = stablehlo.constant dense<12> : tensor<i32>
    %578 = stablehlo.broadcast_in_dim %c_152, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %579 = stablehlo.subtract %577, %578 : tensor<24xi32>
    %580 = call @_where(%573, %571, %579) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_153 = stablehlo.constant dense<0> : tensor<i32>
    %581 = stablehlo.broadcast_in_dim %c_153, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %582 = stablehlo.compare  GE, %580, %581,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_154 = stablehlo.constant dense<801> : tensor<i32>
    %583 = stablehlo.broadcast_in_dim %c_154, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %584 = stablehlo.compare  LT, %580, %583,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %585 = stablehlo.and %582, %584 : tensor<24xi1>
    %586 = stablehlo.broadcast_in_dim %368, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %587 = stablehlo.compare  GE, %580, %586,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_155 = stablehlo.constant dense<801> : tensor<i32>
    %588 = stablehlo.add %368, %c_155 : tensor<i32>
    %589 = stablehlo.broadcast_in_dim %588, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %590 = stablehlo.compare  LT, %580, %589,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %591 = stablehlo.and %587, %590 : tensor<24xi1>
    %592 = stablehlo.and %585, %591 : tensor<24xi1>
    %593 = stablehlo.reshape %592 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %594 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2] : (tensor<800x801x1xi1>) -> tensor<800x801x24xi1>
    %595 = stablehlo.broadcast_in_dim %593, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<800x801x24xi1>
    %596 = stablehlo.and %594, %595 : tensor<800x801x24xi1>
    %cst_156 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %597 = stablehlo.broadcast_in_dim %cst_156, dims = [] : (tensor<f32>) -> tensor<800x801x24xf32>
    %598 = call @_where_83(%596, %378, %597) : (tensor<800x801x24xi1>, tensor<800x801x24xf32>, tensor<800x801x24xf32>) -> tensor<800x801x24xf32>
    %599 = stablehlo.iota dim = 0 : tensor<800xi32>
    %c_157 = stablehlo.constant dense<0> : tensor<i32>
    %600 = stablehlo.broadcast_in_dim %c_157, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %601 = stablehlo.compare  GE, %599, %600,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %c_158 = stablehlo.constant dense<800> : tensor<i32>
    %602 = stablehlo.broadcast_in_dim %c_158, dims = [] : (tensor<i32>) -> tensor<800xi32>
    %603 = stablehlo.compare  LT, %599, %602,  SIGNED : (tensor<800xi32>, tensor<800xi32>) -> tensor<800xi1>
    %604 = stablehlo.and %601, %603 : tensor<800xi1>
    %605 = stablehlo.reshape %604 : (tensor<800xi1>) -> tensor<800x1x1xi1>
    %c_159 = stablehlo.constant dense<true> : tensor<i1>
    %606 = stablehlo.broadcast_in_dim %c_159, dims = [] : (tensor<i1>) -> tensor<800x1x1xi1>
    %607 = stablehlo.and %606, %605 : tensor<800x1x1xi1>
    %608 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_160 = stablehlo.constant dense<12> : tensor<i32>
    %609 = stablehlo.broadcast_in_dim %c_160, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %610 = stablehlo.compare  LT, %608, %609,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_161 = stablehlo.constant dense<12> : tensor<i32>
    %611 = stablehlo.broadcast_in_dim %c_161, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %612 = stablehlo.subtract %608, %611 : tensor<24xi32>
    %c_162 = stablehlo.constant dense<801> : tensor<i32>
    %613 = stablehlo.broadcast_in_dim %c_162, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %614 = stablehlo.add %612, %613 : tensor<24xi32>
    %c_163 = stablehlo.constant dense<12> : tensor<i32>
    %615 = stablehlo.broadcast_in_dim %c_163, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %616 = stablehlo.subtract %614, %615 : tensor<24xi32>
    %617 = call @_where(%610, %608, %616) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_164 = stablehlo.constant dense<0> : tensor<i32>
    %618 = stablehlo.broadcast_in_dim %c_164, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %619 = stablehlo.compare  GE, %617, %618,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_165 = stablehlo.constant dense<801> : tensor<i32>
    %620 = stablehlo.broadcast_in_dim %c_165, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %621 = stablehlo.compare  LT, %617, %620,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %622 = stablehlo.and %619, %621 : tensor<24xi1>
    %623 = stablehlo.reshape %622 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %624 = stablehlo.broadcast_in_dim %607, dims = [0, 1, 2] : (tensor<800x1x1xi1>) -> tensor<800x24x1xi1>
    %625 = stablehlo.broadcast_in_dim %623, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<800x24x1xi1>
    %626 = stablehlo.and %624, %625 : tensor<800x24x1xi1>
    %627 = stablehlo.iota dim = 0 : tensor<801xi32>
    %628 = stablehlo.broadcast_in_dim %368, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %629 = stablehlo.add %627, %628 : tensor<801xi32>
    %c_166 = stablehlo.constant dense<0> : tensor<i32>
    %630 = stablehlo.broadcast_in_dim %c_166, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %631 = stablehlo.compare  GE, %629, %630,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %c_167 = stablehlo.constant dense<801> : tensor<i32>
    %632 = stablehlo.broadcast_in_dim %c_167, dims = [] : (tensor<i32>) -> tensor<801xi32>
    %633 = stablehlo.compare  LT, %629, %632,  SIGNED : (tensor<801xi32>, tensor<801xi32>) -> tensor<801xi1>
    %634 = stablehlo.and %631, %633 : tensor<801xi1>
    %635 = stablehlo.reshape %634 : (tensor<801xi1>) -> tensor<1x1x801xi1>
    %636 = stablehlo.broadcast_in_dim %626, dims = [0, 1, 2] : (tensor<800x24x1xi1>) -> tensor<800x24x801xi1>
    %637 = stablehlo.broadcast_in_dim %635, dims = [0, 1, 2] : (tensor<1x1x801xi1>) -> tensor<800x24x801xi1>
    %638 = stablehlo.and %636, %637 : tensor<800x24x801xi1>
    %cst_168 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %639 = stablehlo.broadcast_in_dim %cst_168, dims = [] : (tensor<f32>) -> tensor<800x24x801xf32>
    %640 = call @_where_65(%638, %arg78, %639) : (tensor<800x24x801xi1>, tensor<800x24x801xf32>, tensor<800x24x801xf32>) -> tensor<800x24x801xf32>
    %cst_169 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %641 = stablehlo.broadcast_in_dim %cst_169, dims = [] : (tensor<f32>) -> tensor<800x800x1xf32>
    %cst_170 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %642 = stablehlo.broadcast_in_dim %cst_170, dims = [] : (tensor<f32>) -> tensor<800x800x1xf32>
    %643 = stablehlo.slice %354 [0:800, 0:801, 800:801] : (tensor<800x801x801xf32>) -> tensor<800x801x1xf32>
    %644 = "stablehlo.collective_permute"(%643) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<800x801x1xf32>) -> tensor<800x801x1xf32>
    %cst_171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %645 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<f32>) -> tensor<800x801x1xf32>
    %646 = stablehlo.slice %355 [0:801, 0:800, 800:801] : (tensor<801x800x801xf32>) -> tensor<801x800x1xf32>
    %647 = "stablehlo.collective_permute"(%646) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<801x800x1xf32>) -> tensor<801x800x1xf32>
    %cst_172 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %648 = stablehlo.broadcast_in_dim %cst_172, dims = [] : (tensor<f32>) -> tensor<801x800x1xf32>
    %649 = stablehlo.transpose %arg45, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %650 = stablehlo.transpose %arg46, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %651 = stablehlo.transpose %arg47, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %652 = stablehlo.transpose %arg48, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %653 = stablehlo.transpose %arg49, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %654 = stablehlo.transpose %arg50, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %655 = stablehlo.transpose %arg51, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %656 = stablehlo.transpose %arg52, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %657 = stablehlo.transpose %arg53, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %658 = stablehlo.transpose %arg54, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %659 = stablehlo.transpose %arg55, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %660 = stablehlo.transpose %arg56, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %661 = stablehlo.transpose %arg39, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %662 = stablehlo.transpose %arg40, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %663 = stablehlo.transpose %arg41, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %664 = stablehlo.transpose %arg42, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %665 = stablehlo.transpose %arg43, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %666 = stablehlo.transpose %arg44, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %667 = stablehlo.slice %374 [0:1, 0:1] : (tensor<7x3xi32>) -> tensor<1x1xi32>
    %668 = stablehlo.reshape %667 : (tensor<1x1xi32>) -> tensor<i32>
    %c_173 = stablehlo.constant dense<0> : tensor<i32>
    %669 = stablehlo.compare  LT, %668, %c_173,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_174 = stablehlo.constant dense<3> : tensor<i32>
    %670 = stablehlo.add %668, %c_174 : tensor<i32>
    %671 = stablehlo.select %669, %670, %668 : tensor<i1>, tensor<i32>
    %672 = stablehlo.dynamic_slice %arg57, %671, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
    %673 = stablehlo.reshape %672 : (tensor<1xi32>) -> tensor<i32>
    %674 = stablehlo.slice %374 [0:1, 0:3] : (tensor<7x3xi32>) -> tensor<1x3xi32>
    %c_175 = stablehlo.constant dense<0> : tensor<i32>
    %675 = stablehlo.broadcast_in_dim %c_175, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_176 = stablehlo.constant dense<0> : tensor<i32>
    %676 = stablehlo.broadcast_in_dim %c_176, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %677 = stablehlo.concatenate %675, %676, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %678 = "stablehlo.scatter"(%674, %677, %673) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg82: tensor<i32>, %arg83: tensor<i32>):
      stablehlo.return %arg83 : tensor<i32>
    }) : (tensor<1x3xi32>, tensor<2xi32>, tensor<i32>) -> tensor<1x3xi32>
    %c_177 = stablehlo.constant dense<0> : tensor<i32>
    %679 = stablehlo.broadcast_in_dim %c_177, dims = [] : (tensor<i32>) -> tensor<6xi32>
    %680 = stablehlo.compare  LT, %arg58, %679,  SIGNED : (tensor<6xi32>, tensor<6xi32>) -> tensor<6xi1>
    %c_178 = stablehlo.constant dense<7> : tensor<i32>
    %681 = stablehlo.broadcast_in_dim %c_178, dims = [] : (tensor<i32>) -> tensor<6xi32>
    %682 = stablehlo.add %arg58, %681 : tensor<6xi32>
    %683 = stablehlo.select %680, %682, %arg58 : tensor<6xi1>, tensor<6xi32>
    %684 = stablehlo.broadcast_in_dim %683, dims = [0] : (tensor<6xi32>) -> tensor<6x1xi32>
    %685 = "stablehlo.gather"(%374, %684) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<7x3xi32>, tensor<6x1xi32>) -> tensor<6x3xi32>
    %c_179 = stablehlo.constant dense<0> : tensor<i32>
    %686 = stablehlo.broadcast_in_dim %c_179, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %687 = stablehlo.compare  LT, %arg59, %686,  SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
    %c_180 = stablehlo.constant dense<3> : tensor<i32>
    %688 = stablehlo.broadcast_in_dim %c_180, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %689 = stablehlo.add %arg59, %688 : tensor<3xi32>
    %690 = stablehlo.select %687, %689, %arg59 : tensor<3xi1>, tensor<3xi32>
    %691 = stablehlo.broadcast_in_dim %690, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %692 = "stablehlo.gather"(%685, %691) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x3xi32>, tensor<3x1xi32>) -> tensor<6x3xi32>
    %693 = stablehlo.concatenate %678, %692, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
    %694 = stablehlo.transpose %644, dims = [2, 0, 1] : (tensor<800x801x1xf32>) -> tensor<1x800x801xf32>
    %695 = stablehlo.transpose %645, dims = [2, 0, 1] : (tensor<800x801x1xf32>) -> tensor<1x800x801xf32>
    %696 = stablehlo.transpose %647, dims = [2, 0, 1] : (tensor<801x800x1xf32>) -> tensor<1x801x800xf32>
    %697 = stablehlo.transpose %648, dims = [2, 0, 1] : (tensor<801x800x1xf32>) -> tensor<1x801x800xf32>
    %698 = stablehlo.transpose %641, dims = [2, 0, 1] : (tensor<800x800x1xf32>) -> tensor<1x800x800xf32>
    %699 = stablehlo.transpose %642, dims = [2, 0, 1] : (tensor<800x800x1xf32>) -> tensor<1x800x800xf32>
    %700 = stablehlo.transpose %arg62, dims = [2, 0, 1] : (tensor<801x800x801xf32>) -> tensor<801x801x800xf32>
    %701 = stablehlo.transpose %arg63, dims = [2, 0, 1] : (tensor<800x801x801xf32>) -> tensor<801x800x801xf32>
    %702 = stablehlo.transpose %arg61, dims = [2, 0, 1] : (tensor<801x801x801xf32>) -> tensor<801x801x801xf32>
    %703 = stablehlo.transpose %354, dims = [2, 0, 1] : (tensor<800x801x801xf32>) -> tensor<801x800x801xf32>
    %704 = stablehlo.transpose %355, dims = [2, 0, 1] : (tensor<801x800x801xf32>) -> tensor<801x801x800xf32>
    %705 = stablehlo.transpose %353, dims = [2, 0, 1] : (tensor<800x800x801xf32>) -> tensor<801x800x800xf32>
    %706 = stablehlo.transpose %arg37, dims = [2, 0, 1] : (tensor<801x800x801xf32>) -> tensor<801x801x800xf32>
    %707 = stablehlo.transpose %arg38, dims = [2, 0, 1] : (tensor<800x801x801xf32>) -> tensor<801x800x801xf32>
    %708 = stablehlo.transpose %arg36, dims = [2, 0, 1] : (tensor<801x801x801xf32>) -> tensor<801x801x801xf32>
    %709 = stablehlo.transpose %504, dims = [2, 0, 1] : (tensor<24x800x801xf32>) -> tensor<801x24x800xf32>
    %710 = stablehlo.transpose %551, dims = [2, 0, 1] : (tensor<801x800x24xf32>) -> tensor<24x801x800xf32>
    %711 = stablehlo.transpose %598, dims = [2, 0, 1] : (tensor<800x801x24xf32>) -> tensor<24x800x801xf32>
    %712 = stablehlo.transpose %640, dims = [2, 0, 1] : (tensor<800x24x801xf32>) -> tensor<801x800x24xf32>
    %713 = stablehlo.transpose %420, dims = [2, 0, 1] : (tensor<801x24x801xf32>) -> tensor<801x801x24xf32>
    %714 = stablehlo.transpose %462, dims = [2, 0, 1] : (tensor<24x801x801xf32>) -> tensor<801x24x801xf32>
    %715:9 = stablehlo.custom_call @beamz_cuda_sharded(%700, %701, %702, %703, %704, %705, %arg34, %arg35, %arg33, %706, %707, %708, %arg60, %649, %650, %651, %652, %653, %654, %655, %656, %657, %658, %659, %660, %661, %662, %663, %664, %665, %666, %709, %710, %711, %712, %713, %714, %arg9, %arg9, %arg9, %693, %694, %695, %696, %697, %698, %699) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 1 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<801x801x800xf32>, tensor<801x800x801xf32>, tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<801x801x800xf32>, tensor<801x800x800xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<801x801x800xf32>, tensor<801x800x801xf32>, tensor<801x801x801xf32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<801x24x800xf32>, tensor<24x801x800xf32>, tensor<24x800x801xf32>, tensor<801x800x24xf32>, tensor<801x801x24xf32>, tensor<801x24x801xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<1x800x801xf32>, tensor<1x800x801xf32>, tensor<1x801x800xf32>, tensor<1x801x800xf32>, tensor<1x800x800xf32>, tensor<1x800x800xf32>) -> (tensor<801x801x800xf32>, tensor<801x800x801xf32>, tensor<801x801x801xf32>, tensor<801x24x800xf32>, tensor<24x801x800xf32>, tensor<24x800x801xf32>, tensor<801x800x24xf32>, tensor<801x801x24xf32>, tensor<801x24x801xf32>)
    %716 = stablehlo.transpose %715#2, dims = [1, 2, 0] : (tensor<801x801x801xf32>) -> tensor<801x801x801xf32>
    %717 = stablehlo.transpose %715#0, dims = [1, 2, 0] : (tensor<801x801x800xf32>) -> tensor<801x800x801xf32>
    %718 = stablehlo.transpose %715#1, dims = [1, 2, 0] : (tensor<801x800x801xf32>) -> tensor<800x801x801xf32>
    %719 = stablehlo.transpose %715#7, dims = [1, 2, 0] : (tensor<801x801x24xf32>) -> tensor<801x24x801xf32>
    %720 = stablehlo.transpose %715#8, dims = [1, 2, 0] : (tensor<801x24x801xf32>) -> tensor<24x801x801xf32>
    %721 = stablehlo.transpose %715#3, dims = [1, 2, 0] : (tensor<801x24x800xf32>) -> tensor<24x800x801xf32>
    %722 = stablehlo.transpose %715#4, dims = [1, 2, 0] : (tensor<24x801x800xf32>) -> tensor<801x800x24xf32>
    %723 = stablehlo.transpose %715#5, dims = [1, 2, 0] : (tensor<24x800x801xf32>) -> tensor<800x801x24xf32>
    %724 = stablehlo.transpose %715#6, dims = [1, 2, 0] : (tensor<801x800x24xf32>) -> tensor<800x24x801xf32>
    %725 = stablehlo.broadcast_in_dim %722, dims = [1, 2, 3] : (tensor<801x800x24xf32>) -> tensor<1x801x800x24xf32>
    %726 = stablehlo.broadcast_in_dim %723, dims = [1, 2, 3] : (tensor<800x801x24xf32>) -> tensor<1x800x801x24xf32>
    %c_181 = stablehlo.constant dense<1> : tensor<i32>
    %727 = stablehlo.add %arg80, %c_181 : tensor<i32>
    return %716, %717, %718, %353, %354, %355, %356, %357, %358, %362, %363, %361, %719, %720, %721, %725, %726, %724, %3, %727 : tensor<801x801x801xf32>, tensor<801x800x801xf32>, tensor<800x801x801xf32>, tensor<800x800x801xf32>, tensor<800x801x801xf32>, tensor<801x800x801xf32>, tensor<800x24x801xf32>, tensor<24x800x801xf32>, tensor<24x801x801xf32>, tensor<1x800x801x24xf32>, tensor<1x801x800x24xf32>, tensor<801x24x801xf32>, tensor<801x24x801xf32>, tensor<24x801x801xf32>, tensor<24x800x801xf32>, tensor<1x801x800x24xf32>, tensor<1x800x801x24xf32>, tensor<800x24x801xf32>, tensor<f32>, tensor<i32>
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
