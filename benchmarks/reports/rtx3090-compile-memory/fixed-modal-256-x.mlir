module @jit_run_scan attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<257x257x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 0 : i32}, %arg1: tensor<257x256x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 1 : i32}, %arg2: tensor<256x257x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 2 : i32}, %arg3: tensor<256x256x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 3 : i32}, %arg4: tensor<256x257x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 4 : i32}, %arg5: tensor<257x256x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 5 : i32}, %arg6: tensor<256x24x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 6 : i32}, %arg7: tensor<24x256x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 7 : i32}, %arg8: tensor<24x257x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 8 : i32}, %arg9: tensor<256x257x24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 9 : i32}, %arg10: tensor<257x256x24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 10 : i32}, %arg11: tensor<257x24x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 11 : i32}, %arg12: tensor<257x24x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 12 : i32}, %arg13: tensor<24x257x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 13 : i32}, %arg14: tensor<24x256x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 14 : i32}, %arg15: tensor<257x256x24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 15 : i32}, %arg16: tensor<256x257x24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 16 : i32}, %arg17: tensor<256x24x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 17 : i32}, %arg18: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 18 : i32}, %arg19: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 19 : i32}, %arg20: tensor<2xi32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 20 : i32}, %arg21: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 21 : i32}, %arg22: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 22 : i32}, %arg23: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 23 : i32}, %arg24: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 24 : i32}, %arg25: tensor<7200xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 25 : i32}, %arg26: tensor<7200xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 26 : i32}, %arg27: tensor<6xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 27 : i32}, %arg28: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>, tf.aliasing_output = 28 : i32}, %arg29: tensor<i32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>, tf.aliasing_output = 29 : i32}, %arg30: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg31: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg32: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg33: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg34: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg35: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg36: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg37: tensor<257x257x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg38: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg39: tensor<257x256x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg40: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg41: tensor<256x257x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}) -> (tensor<257x257x257xf32> {jax.result_info = "result.ex"}, tensor<257x256x257xf32> {jax.result_info = "result.ey"}, tensor<256x257x257xf32> {jax.result_info = "result.ez"}, tensor<256x256x257xf32> {jax.result_info = "result.hx"}, tensor<256x257x257xf32> {jax.result_info = "result.hy"}, tensor<257x256x257xf32> {jax.result_info = "result.hz"}, tensor<256x24x257xf32> {jax.result_info = "result.cpml_psi_h_terms[0]"}, tensor<24x256x257xf32> {jax.result_info = "result.cpml_psi_h_terms[1]"}, tensor<24x257x257xf32> {jax.result_info = "result.cpml_psi_h_terms[2]"}, tensor<256x257x24xf32> {jax.result_info = "result.cpml_psi_h_terms[3]"}, tensor<257x256x24xf32> {jax.result_info = "result.cpml_psi_h_terms[4]"}, tensor<257x24x257xf32> {jax.result_info = "result.cpml_psi_h_terms[5]"}, tensor<257x24x257xf32> {jax.result_info = "result.cpml_psi_e_terms[0]"}, tensor<24x257x257xf32> {jax.result_info = "result.cpml_psi_e_terms[1]"}, tensor<24x256x257xf32> {jax.result_info = "result.cpml_psi_e_terms[2]"}, tensor<257x256x24xf32> {jax.result_info = "result.cpml_psi_e_terms[3]"}, tensor<256x257x24xf32> {jax.result_info = "result.cpml_psi_e_terms[4]"}, tensor<256x24x257xf32> {jax.result_info = "result.cpml_psi_e_terms[5]"}, tensor<2x1xf32> {jax.result_info = "result.powers"}, tensor<2x1xf32> {jax.result_info = "result.timestamps"}, tensor<2xi32> {jax.result_info = "result.counts"}, tensor<2x3xf32> {jax.result_info = "result.freq_flux_re"}, tensor<2x3xf32> {jax.result_info = "result.freq_flux_im"}, tensor<2x3xf32> {jax.result_info = "result.freq_phase_re"}, tensor<2x3xf32> {jax.result_info = "result.freq_phase_im"}, tensor<7200xf32> {jax.result_info = "result.dft_vec_re"}, tensor<7200xf32> {jax.result_info = "result.dft_vec_im"}, tensor<6xf32> {jax.result_info = "result.dft_weight_sum"}, tensor<f32> {jax.result_info = "result.t"}, tensor<i32> {jax.result_info = "result.current_step"}) {
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
    %cst_20 = stablehlo.constant dense_resource<__elided__> : tensor<2x14x25x1xf32>
    %cst_21 = stablehlo.constant dense_resource<__elided__> : tensor<2x32xf32>
    %cst_22 = stablehlo.constant dense_resource<__elided__> : tensor<2x15x24x1xf32>
    %cst_23 = stablehlo.constant dense_resource<__elided__> : tensor<2x32xf32>
    %c_24 = stablehlo.constant dense_resource<__elided__> : tensor<6x3xi32>
    %cst_25 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_26 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_27 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_28 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_29 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_30 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_31 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_32 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_33 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_34 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_35 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_36 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_37 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_38 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_39 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_40 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_41 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_42 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %c_43 = stablehlo.constant dense_resource<__elided__> : tensor<6x5xi32>
    %cst_44 = stablehlo.constant dense_resource<__elided__> : tensor<2x15x26x1xf32>
    %cst_45 = stablehlo.constant dense_resource<__elided__> : tensor<2x32xf32>
    %cst_46 = stablehlo.constant dense_resource<__elided__> : tensor<2x16x25x1xf32>
    %cst_47 = stablehlo.constant dense_resource<__elided__> : tensor<2x32xf32>
    %c_48 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_49 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_50 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_51 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_52 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_53 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_54 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_56 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_57 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_58 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_59 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_60 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %cst_61 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xf32>
    %c_62 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_63 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %c_64 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_65 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %c_66 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %cst_67 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xf32>
    %c_68 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_69 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %c_70 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_71 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %cst_72 = stablehlo.constant dense<[1.87612057E+14, 1.9341449E+14, 1.99216924E+14]> : tensor<3xf32>
    %cst_73 = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
    %c_74 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_75 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_76 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_77 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_78 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_80 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_82 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_84 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_85 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_86 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %cst_87 = stablehlo.constant dense<1.250000e-01> : tensor<200x8xf32>
    %c_88 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %cst_89 = stablehlo.constant dense<5.000000e-01> : tensor<200x2xf32>
    %c_90 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %cst_91 = stablehlo.constant dense<5.000000e-01> : tensor<200x2xf32>
    %c_92 = stablehlo.constant dense_resource<__elided__> : tensor<200x1xi32>
    %cst_93 = stablehlo.constant dense<1.000000e+00> : tensor<200x1xf32>
    %c_94 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_95 = stablehlo.constant dense<2.500000e-01> : tensor<200x4xf32>
    %c_96 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_97 = stablehlo.constant dense<2.500000e-01> : tensor<200x4xf32>
    %cst_98 = stablehlo.constant dense<[1.87612057E+14, 1.9341449E+14, 1.99216924E+14]> : tensor<3xf32>
    %cst_99 = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
    %c_100 = stablehlo.constant dense<1> : tensor<ui32>
    %c_101 = stablehlo.constant dense<1> : tensor<ui32>
    %0 = stablehlo.partition_id : tensor<ui32>
    %1 = stablehlo.divide %0, %c_100 : tensor<ui32>
    %2 = stablehlo.remainder %1, %c_101 : tensor<ui32>
    %3 = stablehlo.convert %2 : (tensor<ui32>) -> tensor<i32>
    %c_102 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.compare  EQ, %3, %c_102,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_103 = stablehlo.constant dense<0> : tensor<i32>
    %5 = call @_where(%4, %arg25, %c_103) : (tensor<i1>, tensor<7200xf32>, tensor<i32>) -> tensor<7200xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
    %c_104 = stablehlo.constant dense<1> : tensor<ui32>
    %c_105 = stablehlo.constant dense<1> : tensor<ui32>
    %7 = stablehlo.partition_id : tensor<ui32>
    %8 = stablehlo.divide %7, %c_104 : tensor<ui32>
    %9 = stablehlo.remainder %8, %c_105 : tensor<ui32>
    %10 = stablehlo.convert %9 : (tensor<ui32>) -> tensor<i32>
    %c_106 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.compare  EQ, %10, %c_106,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_107 = stablehlo.constant dense<0> : tensor<i32>
    %12 = call @_where(%11, %arg26, %c_107) : (tensor<i1>, tensor<7200xf32>, tensor<i32>) -> tensor<7200xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
    %c_108 = stablehlo.constant dense<1> : tensor<ui32>
    %c_109 = stablehlo.constant dense<1> : tensor<ui32>
    %14 = stablehlo.partition_id : tensor<ui32>
    %15 = stablehlo.divide %14, %c_108 : tensor<ui32>
    %16 = stablehlo.remainder %15, %c_109 : tensor<ui32>
    %17 = stablehlo.convert %16 : (tensor<ui32>) -> tensor<i32>
    %c_110 = stablehlo.constant dense<257> : tensor<i32>
    %18 = stablehlo.multiply %17, %c_110 : tensor<i32>
    %19 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_111 = stablehlo.constant dense<0> : tensor<i32>
    %20 = stablehlo.broadcast_in_dim %c_111, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %21 = stablehlo.compare  GE, %19, %20,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_112 = stablehlo.constant dense<256> : tensor<i32>
    %22 = stablehlo.broadcast_in_dim %c_112, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %23 = stablehlo.compare  LT, %19, %22,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %24 = stablehlo.and %21, %23 : tensor<256xi1>
    %25 = stablehlo.reshape %24 : (tensor<256xi1>) -> tensor<256x1x1xi1>
    %c_113 = stablehlo.constant dense<true> : tensor<i1>
    %26 = stablehlo.broadcast_in_dim %c_113, dims = [] : (tensor<i1>) -> tensor<256x1x1xi1>
    %27 = stablehlo.and %26, %25 : tensor<256x1x1xi1>
    %28 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_114 = stablehlo.constant dense<0> : tensor<i32>
    %29 = stablehlo.broadcast_in_dim %c_114, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %30 = stablehlo.compare  GE, %28, %29,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_115 = stablehlo.constant dense<257> : tensor<i32>
    %31 = stablehlo.broadcast_in_dim %c_115, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %32 = stablehlo.compare  LT, %28, %31,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %33 = stablehlo.and %30, %32 : tensor<257xi1>
    %34 = stablehlo.reshape %33 : (tensor<257xi1>) -> tensor<1x257x1xi1>
    %35 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2] : (tensor<256x1x1xi1>) -> tensor<256x257x1xi1>
    %36 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2] : (tensor<1x257x1xi1>) -> tensor<256x257x1xi1>
    %37 = stablehlo.and %35, %36 : tensor<256x257x1xi1>
    %38 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_116 = stablehlo.constant dense<12> : tensor<i32>
    %39 = stablehlo.broadcast_in_dim %c_116, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %40 = stablehlo.compare  LT, %38, %39,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_117 = stablehlo.constant dense<12> : tensor<i32>
    %41 = stablehlo.broadcast_in_dim %c_117, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %42 = stablehlo.subtract %38, %41 : tensor<24xi32>
    %c_118 = stablehlo.constant dense<256> : tensor<i32>
    %43 = stablehlo.broadcast_in_dim %c_118, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %44 = stablehlo.add %42, %43 : tensor<24xi32>
    %c_119 = stablehlo.constant dense<12> : tensor<i32>
    %45 = stablehlo.broadcast_in_dim %c_119, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %46 = stablehlo.subtract %44, %45 : tensor<24xi32>
    %47 = call @_where_13(%40, %38, %46) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_120 = stablehlo.constant dense<0> : tensor<i32>
    %48 = stablehlo.broadcast_in_dim %c_120, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %49 = stablehlo.compare  GE, %47, %48,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_121 = stablehlo.constant dense<256> : tensor<i32>
    %50 = stablehlo.broadcast_in_dim %c_121, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %51 = stablehlo.compare  LT, %47, %50,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %52 = stablehlo.and %49, %51 : tensor<24xi1>
    %53 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %54 = stablehlo.compare  GE, %47, %53,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_122 = stablehlo.constant dense<257> : tensor<i32>
    %55 = stablehlo.add %18, %c_122 : tensor<i32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %57 = stablehlo.compare  LT, %47, %56,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %58 = stablehlo.and %54, %57 : tensor<24xi1>
    %59 = stablehlo.and %52, %58 : tensor<24xi1>
    %60 = stablehlo.reshape %59 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %61 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2] : (tensor<256x257x1xi1>) -> tensor<256x257x24xi1>
    %62 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<256x257x24xi1>
    %63 = stablehlo.and %61, %62 : tensor<256x257x24xi1>
    %cst_123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_123, dims = [] : (tensor<f32>) -> tensor<256x257x24xf32>
    %65 = call @_where_27(%63, %arg9, %64) : (tensor<256x257x24xi1>, tensor<256x257x24xf32>, tensor<256x257x24xf32>) -> tensor<256x257x24xf32>
    %66 = stablehlo.broadcast_in_dim %65, dims = [1, 2, 3] : (tensor<256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %c_124 = stablehlo.constant dense<1> : tensor<ui32>
    %c_125 = stablehlo.constant dense<1> : tensor<ui32>
    %67 = stablehlo.partition_id : tensor<ui32>
    %68 = stablehlo.divide %67, %c_124 : tensor<ui32>
    %69 = stablehlo.remainder %68, %c_125 : tensor<ui32>
    %70 = stablehlo.convert %69 : (tensor<ui32>) -> tensor<i32>
    %c_126 = stablehlo.constant dense<257> : tensor<i32>
    %71 = stablehlo.multiply %70, %c_126 : tensor<i32>
    %72 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_127 = stablehlo.constant dense<0> : tensor<i32>
    %73 = stablehlo.broadcast_in_dim %c_127, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %74 = stablehlo.compare  GE, %72, %73,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_128 = stablehlo.constant dense<257> : tensor<i32>
    %75 = stablehlo.broadcast_in_dim %c_128, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %76 = stablehlo.compare  LT, %72, %75,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %77 = stablehlo.and %74, %76 : tensor<257xi1>
    %78 = stablehlo.reshape %77 : (tensor<257xi1>) -> tensor<257x1x1xi1>
    %c_129 = stablehlo.constant dense<true> : tensor<i1>
    %79 = stablehlo.broadcast_in_dim %c_129, dims = [] : (tensor<i1>) -> tensor<257x1x1xi1>
    %80 = stablehlo.and %79, %78 : tensor<257x1x1xi1>
    %81 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_130 = stablehlo.constant dense<0> : tensor<i32>
    %82 = stablehlo.broadcast_in_dim %c_130, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %83 = stablehlo.compare  GE, %81, %82,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_131 = stablehlo.constant dense<256> : tensor<i32>
    %84 = stablehlo.broadcast_in_dim %c_131, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %85 = stablehlo.compare  LT, %81, %84,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %86 = stablehlo.and %83, %85 : tensor<256xi1>
    %87 = stablehlo.reshape %86 : (tensor<256xi1>) -> tensor<1x256x1xi1>
    %88 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2] : (tensor<257x1x1xi1>) -> tensor<257x256x1xi1>
    %89 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 2] : (tensor<1x256x1xi1>) -> tensor<257x256x1xi1>
    %90 = stablehlo.and %88, %89 : tensor<257x256x1xi1>
    %91 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_132 = stablehlo.constant dense<12> : tensor<i32>
    %92 = stablehlo.broadcast_in_dim %c_132, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %93 = stablehlo.compare  LT, %91, %92,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_133 = stablehlo.constant dense<12> : tensor<i32>
    %94 = stablehlo.broadcast_in_dim %c_133, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %95 = stablehlo.subtract %91, %94 : tensor<24xi32>
    %c_134 = stablehlo.constant dense<256> : tensor<i32>
    %96 = stablehlo.broadcast_in_dim %c_134, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %97 = stablehlo.add %95, %96 : tensor<24xi32>
    %c_135 = stablehlo.constant dense<12> : tensor<i32>
    %98 = stablehlo.broadcast_in_dim %c_135, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %99 = stablehlo.subtract %97, %98 : tensor<24xi32>
    %100 = call @_where_13(%93, %91, %99) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_136 = stablehlo.constant dense<0> : tensor<i32>
    %101 = stablehlo.broadcast_in_dim %c_136, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %102 = stablehlo.compare  GE, %100, %101,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_137 = stablehlo.constant dense<256> : tensor<i32>
    %103 = stablehlo.broadcast_in_dim %c_137, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %104 = stablehlo.compare  LT, %100, %103,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %105 = stablehlo.and %102, %104 : tensor<24xi1>
    %106 = stablehlo.broadcast_in_dim %71, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %107 = stablehlo.compare  GE, %100, %106,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_138 = stablehlo.constant dense<257> : tensor<i32>
    %108 = stablehlo.add %71, %c_138 : tensor<i32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %110 = stablehlo.compare  LT, %100, %109,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %111 = stablehlo.and %107, %110 : tensor<24xi1>
    %112 = stablehlo.and %105, %111 : tensor<24xi1>
    %113 = stablehlo.reshape %112 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %114 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<257x256x1xi1>) -> tensor<257x256x24xi1>
    %115 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<257x256x24xi1>
    %116 = stablehlo.and %114, %115 : tensor<257x256x24xi1>
    %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %117 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f32>) -> tensor<257x256x24xf32>
    %118 = call @_where_39(%116, %arg10, %117) : (tensor<257x256x24xi1>, tensor<257x256x24xf32>, tensor<257x256x24xf32>) -> tensor<257x256x24xf32>
    %119 = stablehlo.broadcast_in_dim %118, dims = [1, 2, 3] : (tensor<257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %c_140 = stablehlo.constant dense<1> : tensor<ui32>
    %c_141 = stablehlo.constant dense<1> : tensor<ui32>
    %120 = stablehlo.partition_id : tensor<ui32>
    %121 = stablehlo.divide %120, %c_140 : tensor<ui32>
    %122 = stablehlo.remainder %121, %c_141 : tensor<ui32>
    %123 = stablehlo.convert %122 : (tensor<ui32>) -> tensor<i32>
    %c_142 = stablehlo.constant dense<257> : tensor<i32>
    %124 = stablehlo.multiply %123, %c_142 : tensor<i32>
    %125 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_143 = stablehlo.constant dense<0> : tensor<i32>
    %126 = stablehlo.broadcast_in_dim %c_143, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %127 = stablehlo.compare  GE, %125, %126,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_144 = stablehlo.constant dense<257> : tensor<i32>
    %128 = stablehlo.broadcast_in_dim %c_144, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %129 = stablehlo.compare  LT, %125, %128,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %130 = stablehlo.and %127, %129 : tensor<257xi1>
    %131 = stablehlo.reshape %130 : (tensor<257xi1>) -> tensor<257x1x1xi1>
    %c_145 = stablehlo.constant dense<true> : tensor<i1>
    %132 = stablehlo.broadcast_in_dim %c_145, dims = [] : (tensor<i1>) -> tensor<257x1x1xi1>
    %133 = stablehlo.and %132, %131 : tensor<257x1x1xi1>
    %134 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_146 = stablehlo.constant dense<0> : tensor<i32>
    %135 = stablehlo.broadcast_in_dim %c_146, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %136 = stablehlo.compare  GE, %134, %135,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_147 = stablehlo.constant dense<256> : tensor<i32>
    %137 = stablehlo.broadcast_in_dim %c_147, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %138 = stablehlo.compare  LT, %134, %137,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %139 = stablehlo.and %136, %138 : tensor<256xi1>
    %140 = stablehlo.reshape %139 : (tensor<256xi1>) -> tensor<1x256x1xi1>
    %141 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2] : (tensor<257x1x1xi1>) -> tensor<257x256x1xi1>
    %142 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2] : (tensor<1x256x1xi1>) -> tensor<257x256x1xi1>
    %143 = stablehlo.and %141, %142 : tensor<257x256x1xi1>
    %144 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_148 = stablehlo.constant dense<12> : tensor<i32>
    %145 = stablehlo.broadcast_in_dim %c_148, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %146 = stablehlo.compare  LT, %144, %145,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_149 = stablehlo.constant dense<12> : tensor<i32>
    %147 = stablehlo.broadcast_in_dim %c_149, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %148 = stablehlo.subtract %144, %147 : tensor<24xi32>
    %c_150 = stablehlo.constant dense<257> : tensor<i32>
    %149 = stablehlo.broadcast_in_dim %c_150, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %150 = stablehlo.add %148, %149 : tensor<24xi32>
    %c_151 = stablehlo.constant dense<12> : tensor<i32>
    %151 = stablehlo.broadcast_in_dim %c_151, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %152 = stablehlo.subtract %150, %151 : tensor<24xi32>
    %153 = call @_where_13(%146, %144, %152) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_152 = stablehlo.constant dense<0> : tensor<i32>
    %154 = stablehlo.broadcast_in_dim %c_152, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %155 = stablehlo.compare  GE, %153, %154,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_153 = stablehlo.constant dense<257> : tensor<i32>
    %156 = stablehlo.broadcast_in_dim %c_153, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %157 = stablehlo.compare  LT, %153, %156,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %158 = stablehlo.and %155, %157 : tensor<24xi1>
    %159 = stablehlo.broadcast_in_dim %124, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %160 = stablehlo.compare  GE, %153, %159,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_154 = stablehlo.constant dense<257> : tensor<i32>
    %161 = stablehlo.add %124, %c_154 : tensor<i32>
    %162 = stablehlo.broadcast_in_dim %161, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %163 = stablehlo.compare  LT, %153, %162,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %164 = stablehlo.and %160, %163 : tensor<24xi1>
    %165 = stablehlo.and %158, %164 : tensor<24xi1>
    %166 = stablehlo.reshape %165 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %167 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2] : (tensor<257x256x1xi1>) -> tensor<257x256x24xi1>
    %168 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<257x256x24xi1>
    %169 = stablehlo.and %167, %168 : tensor<257x256x24xi1>
    %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %170 = stablehlo.broadcast_in_dim %cst_155, dims = [] : (tensor<f32>) -> tensor<257x256x24xf32>
    %171 = call @_where_39(%169, %arg15, %170) : (tensor<257x256x24xi1>, tensor<257x256x24xf32>, tensor<257x256x24xf32>) -> tensor<257x256x24xf32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [1, 2, 3] : (tensor<257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %c_156 = stablehlo.constant dense<1> : tensor<ui32>
    %c_157 = stablehlo.constant dense<1> : tensor<ui32>
    %173 = stablehlo.partition_id : tensor<ui32>
    %174 = stablehlo.divide %173, %c_156 : tensor<ui32>
    %175 = stablehlo.remainder %174, %c_157 : tensor<ui32>
    %176 = stablehlo.convert %175 : (tensor<ui32>) -> tensor<i32>
    %c_158 = stablehlo.constant dense<257> : tensor<i32>
    %177 = stablehlo.multiply %176, %c_158 : tensor<i32>
    %178 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_159 = stablehlo.constant dense<0> : tensor<i32>
    %179 = stablehlo.broadcast_in_dim %c_159, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %180 = stablehlo.compare  GE, %178, %179,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_160 = stablehlo.constant dense<256> : tensor<i32>
    %181 = stablehlo.broadcast_in_dim %c_160, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %182 = stablehlo.compare  LT, %178, %181,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %183 = stablehlo.and %180, %182 : tensor<256xi1>
    %184 = stablehlo.reshape %183 : (tensor<256xi1>) -> tensor<256x1x1xi1>
    %c_161 = stablehlo.constant dense<true> : tensor<i1>
    %185 = stablehlo.broadcast_in_dim %c_161, dims = [] : (tensor<i1>) -> tensor<256x1x1xi1>
    %186 = stablehlo.and %185, %184 : tensor<256x1x1xi1>
    %187 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_162 = stablehlo.constant dense<0> : tensor<i32>
    %188 = stablehlo.broadcast_in_dim %c_162, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %189 = stablehlo.compare  GE, %187, %188,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_163 = stablehlo.constant dense<257> : tensor<i32>
    %190 = stablehlo.broadcast_in_dim %c_163, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %191 = stablehlo.compare  LT, %187, %190,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %192 = stablehlo.and %189, %191 : tensor<257xi1>
    %193 = stablehlo.reshape %192 : (tensor<257xi1>) -> tensor<1x257x1xi1>
    %194 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2] : (tensor<256x1x1xi1>) -> tensor<256x257x1xi1>
    %195 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2] : (tensor<1x257x1xi1>) -> tensor<256x257x1xi1>
    %196 = stablehlo.and %194, %195 : tensor<256x257x1xi1>
    %197 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_164 = stablehlo.constant dense<12> : tensor<i32>
    %198 = stablehlo.broadcast_in_dim %c_164, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %199 = stablehlo.compare  LT, %197, %198,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_165 = stablehlo.constant dense<12> : tensor<i32>
    %200 = stablehlo.broadcast_in_dim %c_165, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %201 = stablehlo.subtract %197, %200 : tensor<24xi32>
    %c_166 = stablehlo.constant dense<257> : tensor<i32>
    %202 = stablehlo.broadcast_in_dim %c_166, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %203 = stablehlo.add %201, %202 : tensor<24xi32>
    %c_167 = stablehlo.constant dense<12> : tensor<i32>
    %204 = stablehlo.broadcast_in_dim %c_167, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %205 = stablehlo.subtract %203, %204 : tensor<24xi32>
    %206 = call @_where_13(%199, %197, %205) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_168 = stablehlo.constant dense<0> : tensor<i32>
    %207 = stablehlo.broadcast_in_dim %c_168, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %208 = stablehlo.compare  GE, %206, %207,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_169 = stablehlo.constant dense<257> : tensor<i32>
    %209 = stablehlo.broadcast_in_dim %c_169, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %210 = stablehlo.compare  LT, %206, %209,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %211 = stablehlo.and %208, %210 : tensor<24xi1>
    %212 = stablehlo.broadcast_in_dim %177, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %213 = stablehlo.compare  GE, %206, %212,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_170 = stablehlo.constant dense<257> : tensor<i32>
    %214 = stablehlo.add %177, %c_170 : tensor<i32>
    %215 = stablehlo.broadcast_in_dim %214, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %216 = stablehlo.compare  LT, %206, %215,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %217 = stablehlo.and %213, %216 : tensor<24xi1>
    %218 = stablehlo.and %211, %217 : tensor<24xi1>
    %219 = stablehlo.reshape %218 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %220 = stablehlo.broadcast_in_dim %196, dims = [0, 1, 2] : (tensor<256x257x1xi1>) -> tensor<256x257x24xi1>
    %221 = stablehlo.broadcast_in_dim %219, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<256x257x24xi1>
    %222 = stablehlo.and %220, %221 : tensor<256x257x24xi1>
    %cst_171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %223 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<f32>) -> tensor<256x257x24xf32>
    %224 = call @_where_27(%222, %arg16, %223) : (tensor<256x257x24xi1>, tensor<256x257x24xf32>, tensor<256x257x24xf32>) -> tensor<256x257x24xf32>
    %225 = stablehlo.broadcast_in_dim %224, dims = [1, 2, 3] : (tensor<256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %226 = stablehlo.iota dim = 0 : tensor<32xi32>
    %c_172 = stablehlo.constant dense<0> : tensor<i32>
    %227:143 = stablehlo.while(%iterArg = %226, %iterArg_173 = %cst, %iterArg_174 = %arg28, %iterArg_175 = %c, %iterArg_176 = %arg30, %iterArg_177 = %arg32, %iterArg_178 = %arg34, %iterArg_179 = %arg31, %iterArg_180 = %arg33, %iterArg_181 = %arg35, %iterArg_182 = %cst_0, %iterArg_183 = %cst_1, %iterArg_184 = %cst_2, %iterArg_185 = %cst_3, %iterArg_186 = %cst_4, %iterArg_187 = %cst_5, %iterArg_188 = %cst_6, %iterArg_189 = %cst_7, %iterArg_190 = %cst_8, %iterArg_191 = %cst_9, %iterArg_192 = %cst_10, %iterArg_193 = %cst_11, %iterArg_194 = %cst_12, %iterArg_195 = %cst_13, %iterArg_196 = %cst_14, %iterArg_197 = %cst_15, %iterArg_198 = %cst_16, %iterArg_199 = %cst_17, %iterArg_200 = %cst_18, %iterArg_201 = %c_19, %iterArg_202 = %cst_20, %iterArg_203 = %cst_21, %iterArg_204 = %cst_22, %iterArg_205 = %cst_23, %iterArg_206 = %c_24, %iterArg_207 = %arg36, %iterArg_208 = %arg38, %iterArg_209 = %arg40, %iterArg_210 = %arg37, %iterArg_211 = %arg39, %iterArg_212 = %arg41, %iterArg_213 = %cst_25, %iterArg_214 = %cst_26, %iterArg_215 = %cst_27, %iterArg_216 = %cst_28, %iterArg_217 = %cst_29, %iterArg_218 = %cst_30, %iterArg_219 = %cst_31, %iterArg_220 = %cst_32, %iterArg_221 = %cst_33, %iterArg_222 = %cst_34, %iterArg_223 = %cst_35, %iterArg_224 = %cst_36, %iterArg_225 = %cst_37, %iterArg_226 = %cst_38, %iterArg_227 = %cst_39, %iterArg_228 = %cst_40, %iterArg_229 = %cst_41, %iterArg_230 = %cst_42, %iterArg_231 = %c_43, %iterArg_232 = %cst_44, %iterArg_233 = %cst_45, %iterArg_234 = %cst_46, %iterArg_235 = %cst_47, %iterArg_236 = %c_48, %iterArg_237 = %cst_49, %iterArg_238 = %c_50, %iterArg_239 = %cst_51, %iterArg_240 = %c_52, %iterArg_241 = %cst_53, %iterArg_242 = %c_54, %iterArg_243 = %cst_55, %iterArg_244 = %c_56, %iterArg_245 = %cst_57, %iterArg_246 = %c_58, %iterArg_247 = %cst_59, %iterArg_248 = %c_60, %iterArg_249 = %cst_61, %iterArg_250 = %c_62, %iterArg_251 = %cst_63, %iterArg_252 = %c_64, %iterArg_253 = %cst_65, %iterArg_254 = %c_66, %iterArg_255 = %cst_67, %iterArg_256 = %c_68, %iterArg_257 = %cst_69, %iterArg_258 = %c_70, %iterArg_259 = %cst_71, %iterArg_260 = %cst_72, %iterArg_261 = %cst_73, %iterArg_262 = %c_74, %iterArg_263 = %cst_75, %iterArg_264 = %c_76, %iterArg_265 = %cst_77, %iterArg_266 = %c_78, %iterArg_267 = %cst_79, %iterArg_268 = %c_80, %iterArg_269 = %cst_81, %iterArg_270 = %c_82, %iterArg_271 = %cst_83, %iterArg_272 = %c_84, %iterArg_273 = %cst_85, %iterArg_274 = %c_86, %iterArg_275 = %cst_87, %iterArg_276 = %c_88, %iterArg_277 = %cst_89, %iterArg_278 = %c_90, %iterArg_279 = %cst_91, %iterArg_280 = %c_92, %iterArg_281 = %cst_93, %iterArg_282 = %c_94, %iterArg_283 = %cst_95, %iterArg_284 = %c_96, %iterArg_285 = %cst_97, %iterArg_286 = %cst_98, %iterArg_287 = %cst_99, %iterArg_288 = %c_172, %iterArg_289 = %arg0, %iterArg_290 = %arg1, %iterArg_291 = %arg2, %iterArg_292 = %arg3, %iterArg_293 = %arg4, %iterArg_294 = %arg5, %iterArg_295 = %arg6, %iterArg_296 = %arg7, %iterArg_297 = %arg8, %iterArg_298 = %66, %iterArg_299 = %119, %iterArg_300 = %arg11, %iterArg_301 = %arg12, %iterArg_302 = %arg13, %iterArg_303 = %arg14, %iterArg_304 = %172, %iterArg_305 = %225, %iterArg_306 = %arg17, %iterArg_307 = %arg18, %iterArg_308 = %arg19, %iterArg_309 = %arg20, %iterArg_310 = %6, %iterArg_311 = %13, %iterArg_312 = %arg27, %iterArg_313 = %arg28, %iterArg_314 = %arg29) : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x14x25x1xf32>, tensor<2x32xf32>, tensor<2x15x24x1xf32>, tensor<2x32xf32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x15x26x1xf32>, tensor<2x32xf32>, tensor<2x16x25x1xf32>, tensor<2x32xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<i32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
    cond {
      %c_315 = stablehlo.constant dense<32> : tensor<i32>
      %246 = stablehlo.compare  LT, %iterArg_288, %c_315,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %246 : tensor<i1>
    } do {
      %246 = stablehlo.dynamic_slice %iterArg, %iterArg_288, sizes = [1] : (tensor<32xi32>, tensor<i32>) -> tensor<1xi32>
      %247 = stablehlo.reshape %246 : (tensor<1xi32>) -> tensor<i32>
      %248:26 = func.call @closed_call(%iterArg_173, %iterArg_174, %iterArg_175, %iterArg_176, %iterArg_177, %iterArg_178, %iterArg_179, %iterArg_180, %iterArg_181, %iterArg_182, %iterArg_183, %iterArg_184, %iterArg_185, %iterArg_186, %iterArg_187, %iterArg_188, %iterArg_189, %iterArg_190, %iterArg_191, %iterArg_192, %iterArg_193, %iterArg_194, %iterArg_195, %iterArg_196, %iterArg_197, %iterArg_198, %iterArg_199, %iterArg_200, %iterArg_201, %iterArg_202, %iterArg_203, %iterArg_204, %iterArg_205, %iterArg_206, %iterArg_207, %iterArg_208, %iterArg_209, %iterArg_210, %iterArg_211, %iterArg_212, %iterArg_213, %iterArg_214, %iterArg_215, %iterArg_216, %iterArg_217, %iterArg_218, %iterArg_219, %iterArg_220, %iterArg_221, %iterArg_222, %iterArg_223, %iterArg_224, %iterArg_225, %iterArg_226, %iterArg_227, %iterArg_228, %iterArg_229, %iterArg_230, %iterArg_231, %iterArg_232, %iterArg_233, %iterArg_234, %iterArg_235, %iterArg_236, %iterArg_237, %iterArg_238, %iterArg_239, %iterArg_240, %iterArg_241, %iterArg_242, %iterArg_243, %iterArg_244, %iterArg_245, %iterArg_246, %iterArg_247, %iterArg_248, %iterArg_249, %iterArg_250, %iterArg_251, %iterArg_252, %iterArg_253, %iterArg_254, %iterArg_255, %iterArg_256, %iterArg_257, %iterArg_258, %iterArg_259, %iterArg_260, %iterArg_261, %iterArg_262, %iterArg_263, %iterArg_264, %iterArg_265, %iterArg_266, %iterArg_267, %iterArg_268, %iterArg_269, %iterArg_270, %iterArg_271, %iterArg_272, %iterArg_273, %iterArg_274, %iterArg_275, %iterArg_276, %iterArg_277, %iterArg_278, %iterArg_279, %iterArg_280, %iterArg_281, %iterArg_282, %iterArg_283, %iterArg_284, %iterArg_285, %iterArg_286, %iterArg_287, %iterArg_289, %iterArg_290, %iterArg_291, %iterArg_292, %iterArg_293, %iterArg_294, %iterArg_295, %iterArg_296, %iterArg_297, %iterArg_298, %iterArg_299, %iterArg_300, %iterArg_301, %iterArg_302, %iterArg_303, %iterArg_304, %iterArg_305, %iterArg_306, %iterArg_307, %iterArg_308, %iterArg_309, %iterArg_310, %iterArg_311, %iterArg_312, %iterArg_313, %iterArg_314, %247) : (tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x14x25x1xf32>, tensor<2x32xf32>, tensor<2x15x24x1xf32>, tensor<2x32xf32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x15x26x1xf32>, tensor<2x32xf32>, tensor<2x16x25x1xf32>, tensor<2x32xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>, tensor<i32>) -> (tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>)
      %c_315 = stablehlo.constant dense<1> : tensor<i32>
      %249 = stablehlo.add %iterArg_288, %c_315 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_173, %iterArg_174, %iterArg_175, %iterArg_176, %iterArg_177, %iterArg_178, %iterArg_179, %iterArg_180, %iterArg_181, %iterArg_182, %iterArg_183, %iterArg_184, %iterArg_185, %iterArg_186, %iterArg_187, %iterArg_188, %iterArg_189, %iterArg_190, %iterArg_191, %iterArg_192, %iterArg_193, %iterArg_194, %iterArg_195, %iterArg_196, %iterArg_197, %iterArg_198, %iterArg_199, %iterArg_200, %iterArg_201, %iterArg_202, %iterArg_203, %iterArg_204, %iterArg_205, %iterArg_206, %iterArg_207, %iterArg_208, %iterArg_209, %iterArg_210, %iterArg_211, %iterArg_212, %iterArg_213, %iterArg_214, %iterArg_215, %iterArg_216, %iterArg_217, %iterArg_218, %iterArg_219, %iterArg_220, %iterArg_221, %iterArg_222, %iterArg_223, %iterArg_224, %iterArg_225, %iterArg_226, %iterArg_227, %iterArg_228, %iterArg_229, %iterArg_230, %iterArg_231, %iterArg_232, %iterArg_233, %iterArg_234, %iterArg_235, %iterArg_236, %iterArg_237, %iterArg_238, %iterArg_239, %iterArg_240, %iterArg_241, %iterArg_242, %iterArg_243, %iterArg_244, %iterArg_245, %iterArg_246, %iterArg_247, %iterArg_248, %iterArg_249, %iterArg_250, %iterArg_251, %iterArg_252, %iterArg_253, %iterArg_254, %iterArg_255, %iterArg_256, %iterArg_257, %iterArg_258, %iterArg_259, %iterArg_260, %iterArg_261, %iterArg_262, %iterArg_263, %iterArg_264, %iterArg_265, %iterArg_266, %iterArg_267, %iterArg_268, %iterArg_269, %iterArg_270, %iterArg_271, %iterArg_272, %iterArg_273, %iterArg_274, %iterArg_275, %iterArg_276, %iterArg_277, %iterArg_278, %iterArg_279, %iterArg_280, %iterArg_281, %iterArg_282, %iterArg_283, %iterArg_284, %iterArg_285, %iterArg_286, %iterArg_287, %249, %248#0, %248#1, %248#2, %248#3, %248#4, %248#5, %248#6, %248#7, %248#8, %248#9, %248#10, %248#11, %248#12, %248#13, %248#14, %248#15, %248#16, %248#17, %248#18, %248#19, %248#20, %248#21, %248#22, %248#23, %248#24, %248#25 : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x14x25x1xf32>, tensor<2x32xf32>, tensor<2x15x24x1xf32>, tensor<2x32xf32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x15x26x1xf32>, tensor<2x32xf32>, tensor<2x16x25x1xf32>, tensor<2x32xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<i32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
    }
    %228 = stablehlo.slice %227#126 [0:1, 0:256, 0:257, 0:24] : (tensor<1x256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %229 = stablehlo.reshape %228 : (tensor<1x256x257x24xf32>) -> tensor<256x257x24xf32>
    %230 = "stablehlo.all_reduce"(%229) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<256x257x24xf32>) -> tensor<256x257x24xf32>
    %231 = stablehlo.slice %227#127 [0:1, 0:257, 0:256, 0:24] : (tensor<1x257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %232 = stablehlo.reshape %231 : (tensor<1x257x256x24xf32>) -> tensor<257x256x24xf32>
    %233 = "stablehlo.all_reduce"(%232) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<257x256x24xf32>) -> tensor<257x256x24xf32>
    %234 = stablehlo.slice %227#132 [0:1, 0:257, 0:256, 0:24] : (tensor<1x257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %235 = stablehlo.reshape %234 : (tensor<1x257x256x24xf32>) -> tensor<257x256x24xf32>
    %236 = "stablehlo.all_reduce"(%235) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<257x256x24xf32>) -> tensor<257x256x24xf32>
    %237 = stablehlo.slice %227#133 [0:1, 0:256, 0:257, 0:24] : (tensor<1x256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %238 = stablehlo.reshape %237 : (tensor<1x256x257x24xf32>) -> tensor<256x257x24xf32>
    %239 = "stablehlo.all_reduce"(%238) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<256x257x24xf32>) -> tensor<256x257x24xf32>
    %240 = stablehlo.slice %227#138 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
    %241 = stablehlo.reshape %240 : (tensor<1x7200xf32>) -> tensor<7200xf32>
    %242 = "stablehlo.all_reduce"(%241) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<7200xf32>) -> tensor<7200xf32>
    %243 = stablehlo.slice %227#139 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
    %244 = stablehlo.reshape %243 : (tensor<1x7200xf32>) -> tensor<7200xf32>
    %245 = "stablehlo.all_reduce"(%244) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<7200xf32>) -> tensor<7200xf32>
    return %227#117, %227#118, %227#119, %227#120, %227#121, %227#122, %227#123, %227#124, %227#125, %230, %233, %227#128, %227#129, %227#130, %227#131, %236, %239, %227#134, %227#135, %227#136, %227#137, %arg21, %arg22, %arg23, %arg24, %242, %245, %227#140, %227#141, %227#142 : tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<256x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<7200xf32>, %arg2: tensor<i32>) -> tensor<7200xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<7200xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<i1>, tensor<7200xf32>
    return %2 : tensor<7200xf32>
  }
  func.func private @_where_13(%arg0: tensor<24xi1>, %arg1: tensor<24xi32>, %arg2: tensor<24xi32>) -> tensor<24xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24xi1>, tensor<24xi32>
    return %0 : tensor<24xi32>
  }
  func.func private @_where_27(%arg0: tensor<256x257x24xi1>, %arg1: tensor<256x257x24xf32>, %arg2: tensor<256x257x24xf32>) -> tensor<256x257x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<256x257x24xi1>, tensor<256x257x24xf32>
    return %0 : tensor<256x257x24xf32>
  }
  func.func private @_where_39(%arg0: tensor<257x256x24xi1>, %arg1: tensor<257x256x24xf32>, %arg2: tensor<257x256x24xf32>) -> tensor<257x256x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<257x256x24xi1>, tensor<257x256x24xf32>
    return %0 : tensor<257x256x24xf32>
  }
  func.func private @closed_call(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<6x3xi32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>, %arg9: tensor<0xf32>, %arg10: tensor<1x24x1xf32>, %arg11: tensor<1x24x1xf32>, %arg12: tensor<1x24x1xf32>, %arg13: tensor<24x1x1xf32>, %arg14: tensor<24x1x1xf32>, %arg15: tensor<24x1x1xf32>, %arg16: tensor<24x1x1xf32>, %arg17: tensor<24x1x1xf32>, %arg18: tensor<24x1x1xf32>, %arg19: tensor<1x1x24xf32>, %arg20: tensor<1x1x24xf32>, %arg21: tensor<1x1x24xf32>, %arg22: tensor<1x1x24xf32>, %arg23: tensor<1x1x24xf32>, %arg24: tensor<1x1x24xf32>, %arg25: tensor<1x24x1xf32>, %arg26: tensor<1x24x1xf32>, %arg27: tensor<1x24x1xf32>, %arg28: tensor<6x5xi32>, %arg29: tensor<2x14x25x1xf32>, %arg30: tensor<2x32xf32>, %arg31: tensor<2x15x24x1xf32>, %arg32: tensor<2x32xf32>, %arg33: tensor<6x3xi32>, %arg34: tensor<f32>, %arg35: tensor<f32>, %arg36: tensor<f32>, %arg37: tensor<257x257x257xf32>, %arg38: tensor<257x256x257xf32>, %arg39: tensor<256x257x257xf32>, %arg40: tensor<1x24x1xf32>, %arg41: tensor<1x24x1xf32>, %arg42: tensor<1x24x1xf32>, %arg43: tensor<24x1x1xf32>, %arg44: tensor<24x1x1xf32>, %arg45: tensor<24x1x1xf32>, %arg46: tensor<24x1x1xf32>, %arg47: tensor<24x1x1xf32>, %arg48: tensor<24x1x1xf32>, %arg49: tensor<1x1x24xf32>, %arg50: tensor<1x1x24xf32>, %arg51: tensor<1x1x24xf32>, %arg52: tensor<1x1x24xf32>, %arg53: tensor<1x1x24xf32>, %arg54: tensor<1x1x24xf32>, %arg55: tensor<1x24x1xf32>, %arg56: tensor<1x24x1xf32>, %arg57: tensor<1x24x1xf32>, %arg58: tensor<6x5xi32>, %arg59: tensor<2x15x26x1xf32>, %arg60: tensor<2x32xf32>, %arg61: tensor<2x16x25x1xf32>, %arg62: tensor<2x32xf32>, %arg63: tensor<200x1xi32>, %arg64: tensor<200x1xf32>, %arg65: tensor<200x1xi32>, %arg66: tensor<200x1xf32>, %arg67: tensor<200x1xi32>, %arg68: tensor<200x1xf32>, %arg69: tensor<200x1xi32>, %arg70: tensor<200x1xf32>, %arg71: tensor<200x1xi32>, %arg72: tensor<200x1xf32>, %arg73: tensor<200x1xi32>, %arg74: tensor<200x1xf32>, %arg75: tensor<200x8xi32>, %arg76: tensor<200x8xf32>, %arg77: tensor<200x4xi32>, %arg78: tensor<200x4xf32>, %arg79: tensor<200x4xi32>, %arg80: tensor<200x4xf32>, %arg81: tensor<200x2xi32>, %arg82: tensor<200x2xf32>, %arg83: tensor<200x4xi32>, %arg84: tensor<200x4xf32>, %arg85: tensor<200x4xi32>, %arg86: tensor<200x4xf32>, %arg87: tensor<3xf32>, %arg88: tensor<6xf32>, %arg89: tensor<200x1xi32>, %arg90: tensor<200x1xf32>, %arg91: tensor<200x1xi32>, %arg92: tensor<200x1xf32>, %arg93: tensor<200x1xi32>, %arg94: tensor<200x1xf32>, %arg95: tensor<200x1xi32>, %arg96: tensor<200x1xf32>, %arg97: tensor<200x1xi32>, %arg98: tensor<200x1xf32>, %arg99: tensor<200x1xi32>, %arg100: tensor<200x1xf32>, %arg101: tensor<200x8xi32>, %arg102: tensor<200x8xf32>, %arg103: tensor<200x2xi32>, %arg104: tensor<200x2xf32>, %arg105: tensor<200x2xi32>, %arg106: tensor<200x2xf32>, %arg107: tensor<200x1xi32>, %arg108: tensor<200x1xf32>, %arg109: tensor<200x4xi32>, %arg110: tensor<200x4xf32>, %arg111: tensor<200x4xi32>, %arg112: tensor<200x4xf32>, %arg113: tensor<3xf32>, %arg114: tensor<6xf32>, %arg115: tensor<257x257x257xf32>, %arg116: tensor<257x256x257xf32>, %arg117: tensor<256x257x257xf32>, %arg118: tensor<256x256x257xf32>, %arg119: tensor<256x257x257xf32>, %arg120: tensor<257x256x257xf32>, %arg121: tensor<256x24x257xf32>, %arg122: tensor<24x256x257xf32>, %arg123: tensor<24x257x257xf32>, %arg124: tensor<1x256x257x24xf32>, %arg125: tensor<1x257x256x24xf32>, %arg126: tensor<257x24x257xf32>, %arg127: tensor<257x24x257xf32>, %arg128: tensor<24x257x257xf32>, %arg129: tensor<24x256x257xf32>, %arg130: tensor<1x257x256x24xf32>, %arg131: tensor<1x256x257x24xf32>, %arg132: tensor<256x24x257xf32>, %arg133: tensor<2x1xf32>, %arg134: tensor<2x1xf32>, %arg135: tensor<2xi32>, %arg136: tensor<1x7200xf32>, %arg137: tensor<1x7200xf32>, %arg138: tensor<6xf32>, %arg139: tensor<f32>, %arg140: tensor<i32>, %arg141: tensor<i32>) -> (tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg141, %c : tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<f32>
    %3 = stablehlo.add %arg1, %2 : tensor<f32>
    %c_0 = stablehlo.constant dense<1> : tensor<ui32>
    %c_1 = stablehlo.constant dense<1> : tensor<ui32>
    %4 = stablehlo.partition_id : tensor<ui32>
    %5 = stablehlo.divide %4, %c_0 : tensor<ui32>
    %6 = stablehlo.remainder %5, %c_1 : tensor<ui32>
    %7 = stablehlo.convert %6 : (tensor<ui32>) -> tensor<i32>
    %c_2 = stablehlo.constant dense<257> : tensor<i32>
    %8 = stablehlo.multiply %7, %c_2 : tensor<i32>
    %c_3 = stablehlo.constant dense<2> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.concatenate %9, %10, %11, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [1] : (tensor<3xi32>) -> tensor<1x3xi32>
    %14 = stablehlo.concatenate %13, %arg2, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
    %15 = stablehlo.slice %arg124 [0:1, 0:256, 0:257, 0:24] : (tensor<1x256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %16 = stablehlo.reshape %15 : (tensor<1x256x257x24xf32>) -> tensor<256x257x24xf32>
    %17 = stablehlo.slice %arg125 [0:1, 0:257, 0:256, 0:24] : (tensor<1x257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %18 = stablehlo.reshape %17 : (tensor<1x257x256x24xf32>) -> tensor<257x256x24xf32>
    %19 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_5 = stablehlo.constant dense<0> : tensor<i32>
    %20 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %21 = stablehlo.compare  GE, %19, %20,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_6 = stablehlo.constant dense<256> : tensor<i32>
    %22 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %23 = stablehlo.compare  LT, %19, %22,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %24 = stablehlo.and %21, %23 : tensor<256xi1>
    %25 = stablehlo.reshape %24 : (tensor<256xi1>) -> tensor<256x1x1xi1>
    %c_7 = stablehlo.constant dense<true> : tensor<i1>
    %26 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i1>) -> tensor<256x1x1xi1>
    %27 = stablehlo.and %26, %25 : tensor<256x1x1xi1>
    %28 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_8 = stablehlo.constant dense<12> : tensor<i32>
    %29 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %30 = stablehlo.compare  LT, %28, %29,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_9 = stablehlo.constant dense<12> : tensor<i32>
    %31 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %32 = stablehlo.subtract %28, %31 : tensor<24xi32>
    %c_10 = stablehlo.constant dense<256> : tensor<i32>
    %33 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %34 = stablehlo.add %32, %33 : tensor<24xi32>
    %c_11 = stablehlo.constant dense<12> : tensor<i32>
    %35 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %36 = stablehlo.subtract %34, %35 : tensor<24xi32>
    %37 = call @_where_13(%30, %28, %36) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_12 = stablehlo.constant dense<0> : tensor<i32>
    %38 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %39 = stablehlo.compare  GE, %37, %38,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_13 = stablehlo.constant dense<256> : tensor<i32>
    %40 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %41 = stablehlo.compare  LT, %37, %40,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %42 = stablehlo.and %39, %41 : tensor<24xi1>
    %43 = stablehlo.reshape %42 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %44 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2] : (tensor<256x1x1xi1>) -> tensor<256x24x1xi1>
    %45 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<256x24x1xi1>
    %46 = stablehlo.and %44, %45 : tensor<256x24x1xi1>
    %47 = stablehlo.iota dim = 0 : tensor<257xi32>
    %48 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %49 = stablehlo.add %47, %48 : tensor<257xi32>
    %c_14 = stablehlo.constant dense<0> : tensor<i32>
    %50 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %51 = stablehlo.compare  GE, %49, %50,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_15 = stablehlo.constant dense<257> : tensor<i32>
    %52 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %53 = stablehlo.compare  LT, %49, %52,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %54 = stablehlo.and %51, %53 : tensor<257xi1>
    %55 = stablehlo.reshape %54 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %56 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<256x24x1xi1>) -> tensor<256x24x257xi1>
    %57 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<256x24x257xi1>
    %58 = stablehlo.and %56, %57 : tensor<256x24x257xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x24x257xf32>
    %60 = call @_where_73(%58, %arg121, %59) : (tensor<256x24x257xi1>, tensor<256x24x257xf32>, tensor<256x24x257xf32>) -> tensor<256x24x257xf32>
    %61 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_16 = stablehlo.constant dense<12> : tensor<i32>
    %62 = stablehlo.broadcast_in_dim %c_16, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %63 = stablehlo.compare  LT, %61, %62,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_17 = stablehlo.constant dense<12> : tensor<i32>
    %64 = stablehlo.broadcast_in_dim %c_17, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %65 = stablehlo.subtract %61, %64 : tensor<24xi32>
    %c_18 = stablehlo.constant dense<256> : tensor<i32>
    %66 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %67 = stablehlo.add %65, %66 : tensor<24xi32>
    %c_19 = stablehlo.constant dense<12> : tensor<i32>
    %68 = stablehlo.broadcast_in_dim %c_19, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %69 = stablehlo.subtract %67, %68 : tensor<24xi32>
    %70 = call @_where_13(%63, %61, %69) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_20 = stablehlo.constant dense<0> : tensor<i32>
    %71 = stablehlo.broadcast_in_dim %c_20, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %72 = stablehlo.compare  GE, %70, %71,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_21 = stablehlo.constant dense<256> : tensor<i32>
    %73 = stablehlo.broadcast_in_dim %c_21, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %74 = stablehlo.compare  LT, %70, %73,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %75 = stablehlo.and %72, %74 : tensor<24xi1>
    %76 = stablehlo.reshape %75 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_22 = stablehlo.constant dense<true> : tensor<i1>
    %77 = stablehlo.broadcast_in_dim %c_22, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %78 = stablehlo.and %77, %76 : tensor<24x1x1xi1>
    %79 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_23 = stablehlo.constant dense<0> : tensor<i32>
    %80 = stablehlo.broadcast_in_dim %c_23, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %81 = stablehlo.compare  GE, %79, %80,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_24 = stablehlo.constant dense<256> : tensor<i32>
    %82 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %83 = stablehlo.compare  LT, %79, %82,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %84 = stablehlo.and %81, %83 : tensor<256xi1>
    %85 = stablehlo.reshape %84 : (tensor<256xi1>) -> tensor<1x256x1xi1>
    %86 = stablehlo.broadcast_in_dim %78, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x256x1xi1>
    %87 = stablehlo.broadcast_in_dim %85, dims = [0, 1, 2] : (tensor<1x256x1xi1>) -> tensor<24x256x1xi1>
    %88 = stablehlo.and %86, %87 : tensor<24x256x1xi1>
    %89 = stablehlo.iota dim = 0 : tensor<257xi32>
    %90 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %91 = stablehlo.add %89, %90 : tensor<257xi32>
    %c_25 = stablehlo.constant dense<0> : tensor<i32>
    %92 = stablehlo.broadcast_in_dim %c_25, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %93 = stablehlo.compare  GE, %91, %92,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_26 = stablehlo.constant dense<257> : tensor<i32>
    %94 = stablehlo.broadcast_in_dim %c_26, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %95 = stablehlo.compare  LT, %91, %94,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %96 = stablehlo.and %93, %95 : tensor<257xi1>
    %97 = stablehlo.reshape %96 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %98 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2] : (tensor<24x256x1xi1>) -> tensor<24x256x257xi1>
    %99 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<24x256x257xi1>
    %100 = stablehlo.and %98, %99 : tensor<24x256x257xi1>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %101 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<24x256x257xf32>
    %102 = call @_where_82(%100, %arg122, %101) : (tensor<24x256x257xi1>, tensor<24x256x257xf32>, tensor<24x256x257xf32>) -> tensor<24x256x257xf32>
    %103 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_28 = stablehlo.constant dense<12> : tensor<i32>
    %104 = stablehlo.broadcast_in_dim %c_28, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %105 = stablehlo.compare  LT, %103, %104,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_29 = stablehlo.constant dense<12> : tensor<i32>
    %106 = stablehlo.broadcast_in_dim %c_29, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %107 = stablehlo.subtract %103, %106 : tensor<24xi32>
    %c_30 = stablehlo.constant dense<256> : tensor<i32>
    %108 = stablehlo.broadcast_in_dim %c_30, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %109 = stablehlo.add %107, %108 : tensor<24xi32>
    %c_31 = stablehlo.constant dense<12> : tensor<i32>
    %110 = stablehlo.broadcast_in_dim %c_31, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %111 = stablehlo.subtract %109, %110 : tensor<24xi32>
    %112 = call @_where_13(%105, %103, %111) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_32 = stablehlo.constant dense<0> : tensor<i32>
    %113 = stablehlo.broadcast_in_dim %c_32, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %114 = stablehlo.compare  GE, %112, %113,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_33 = stablehlo.constant dense<256> : tensor<i32>
    %115 = stablehlo.broadcast_in_dim %c_33, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %116 = stablehlo.compare  LT, %112, %115,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %117 = stablehlo.and %114, %116 : tensor<24xi1>
    %118 = stablehlo.reshape %117 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_34 = stablehlo.constant dense<true> : tensor<i1>
    %119 = stablehlo.broadcast_in_dim %c_34, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %120 = stablehlo.and %119, %118 : tensor<24x1x1xi1>
    %121 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_35 = stablehlo.constant dense<0> : tensor<i32>
    %122 = stablehlo.broadcast_in_dim %c_35, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %123 = stablehlo.compare  GE, %121, %122,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_36 = stablehlo.constant dense<257> : tensor<i32>
    %124 = stablehlo.broadcast_in_dim %c_36, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %125 = stablehlo.compare  LT, %121, %124,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %126 = stablehlo.and %123, %125 : tensor<257xi1>
    %127 = stablehlo.reshape %126 : (tensor<257xi1>) -> tensor<1x257x1xi1>
    %128 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x257x1xi1>
    %129 = stablehlo.broadcast_in_dim %127, dims = [0, 1, 2] : (tensor<1x257x1xi1>) -> tensor<24x257x1xi1>
    %130 = stablehlo.and %128, %129 : tensor<24x257x1xi1>
    %131 = stablehlo.iota dim = 0 : tensor<257xi32>
    %132 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %133 = stablehlo.add %131, %132 : tensor<257xi32>
    %c_37 = stablehlo.constant dense<0> : tensor<i32>
    %134 = stablehlo.broadcast_in_dim %c_37, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %135 = stablehlo.compare  GE, %133, %134,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_38 = stablehlo.constant dense<256> : tensor<i32>
    %136 = stablehlo.broadcast_in_dim %c_38, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %137 = stablehlo.compare  LT, %133, %136,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %138 = stablehlo.and %135, %137 : tensor<257xi1>
    %139 = stablehlo.reshape %138 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %140 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<24x257x1xi1>) -> tensor<24x257x257xi1>
    %141 = stablehlo.broadcast_in_dim %139, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<24x257x257xi1>
    %142 = stablehlo.and %140, %141 : tensor<24x257x257xi1>
    %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %143 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<24x257x257xf32>
    %144 = call @_where_89(%142, %arg123, %143) : (tensor<24x257x257xi1>, tensor<24x257x257xf32>, tensor<24x257x257xf32>) -> tensor<24x257x257xf32>
    %145 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_40 = stablehlo.constant dense<0> : tensor<i32>
    %146 = stablehlo.broadcast_in_dim %c_40, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %147 = stablehlo.compare  GE, %145, %146,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_41 = stablehlo.constant dense<256> : tensor<i32>
    %148 = stablehlo.broadcast_in_dim %c_41, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %149 = stablehlo.compare  LT, %145, %148,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %150 = stablehlo.and %147, %149 : tensor<256xi1>
    %151 = stablehlo.reshape %150 : (tensor<256xi1>) -> tensor<256x1x1xi1>
    %c_42 = stablehlo.constant dense<true> : tensor<i1>
    %152 = stablehlo.broadcast_in_dim %c_42, dims = [] : (tensor<i1>) -> tensor<256x1x1xi1>
    %153 = stablehlo.and %152, %151 : tensor<256x1x1xi1>
    %154 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_43 = stablehlo.constant dense<0> : tensor<i32>
    %155 = stablehlo.broadcast_in_dim %c_43, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %156 = stablehlo.compare  GE, %154, %155,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_44 = stablehlo.constant dense<257> : tensor<i32>
    %157 = stablehlo.broadcast_in_dim %c_44, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %158 = stablehlo.compare  LT, %154, %157,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %159 = stablehlo.and %156, %158 : tensor<257xi1>
    %160 = stablehlo.reshape %159 : (tensor<257xi1>) -> tensor<1x257x1xi1>
    %161 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2] : (tensor<256x1x1xi1>) -> tensor<256x257x1xi1>
    %162 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2] : (tensor<1x257x1xi1>) -> tensor<256x257x1xi1>
    %163 = stablehlo.and %161, %162 : tensor<256x257x1xi1>
    %164 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_45 = stablehlo.constant dense<12> : tensor<i32>
    %165 = stablehlo.broadcast_in_dim %c_45, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %166 = stablehlo.compare  LT, %164, %165,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_46 = stablehlo.constant dense<12> : tensor<i32>
    %167 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %168 = stablehlo.subtract %164, %167 : tensor<24xi32>
    %c_47 = stablehlo.constant dense<256> : tensor<i32>
    %169 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %170 = stablehlo.add %168, %169 : tensor<24xi32>
    %c_48 = stablehlo.constant dense<12> : tensor<i32>
    %171 = stablehlo.broadcast_in_dim %c_48, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %172 = stablehlo.subtract %170, %171 : tensor<24xi32>
    %173 = call @_where_13(%166, %164, %172) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_49 = stablehlo.constant dense<0> : tensor<i32>
    %174 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %175 = stablehlo.compare  GE, %173, %174,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_50 = stablehlo.constant dense<256> : tensor<i32>
    %176 = stablehlo.broadcast_in_dim %c_50, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %177 = stablehlo.compare  LT, %173, %176,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %178 = stablehlo.and %175, %177 : tensor<24xi1>
    %179 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %180 = stablehlo.compare  GE, %173, %179,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_51 = stablehlo.constant dense<257> : tensor<i32>
    %181 = stablehlo.add %8, %c_51 : tensor<i32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %183 = stablehlo.compare  LT, %173, %182,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %184 = stablehlo.and %180, %183 : tensor<24xi1>
    %185 = stablehlo.and %178, %184 : tensor<24xi1>
    %186 = stablehlo.reshape %185 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %187 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2] : (tensor<256x257x1xi1>) -> tensor<256x257x24xi1>
    %188 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<256x257x24xi1>
    %189 = stablehlo.and %187, %188 : tensor<256x257x24xi1>
    %cst_52 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %190 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<256x257x24xf32>
    %191 = call @_where_91(%189, %16, %190) : (tensor<256x257x24xi1>, tensor<256x257x24xf32>, tensor<256x257x24xf32>) -> tensor<256x257x24xf32>
    %192 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_53 = stablehlo.constant dense<0> : tensor<i32>
    %193 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %194 = stablehlo.compare  GE, %192, %193,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_54 = stablehlo.constant dense<257> : tensor<i32>
    %195 = stablehlo.broadcast_in_dim %c_54, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %196 = stablehlo.compare  LT, %192, %195,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %197 = stablehlo.and %194, %196 : tensor<257xi1>
    %198 = stablehlo.reshape %197 : (tensor<257xi1>) -> tensor<257x1x1xi1>
    %c_55 = stablehlo.constant dense<true> : tensor<i1>
    %199 = stablehlo.broadcast_in_dim %c_55, dims = [] : (tensor<i1>) -> tensor<257x1x1xi1>
    %200 = stablehlo.and %199, %198 : tensor<257x1x1xi1>
    %201 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_56 = stablehlo.constant dense<0> : tensor<i32>
    %202 = stablehlo.broadcast_in_dim %c_56, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %203 = stablehlo.compare  GE, %201, %202,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_57 = stablehlo.constant dense<256> : tensor<i32>
    %204 = stablehlo.broadcast_in_dim %c_57, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %205 = stablehlo.compare  LT, %201, %204,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %206 = stablehlo.and %203, %205 : tensor<256xi1>
    %207 = stablehlo.reshape %206 : (tensor<256xi1>) -> tensor<1x256x1xi1>
    %208 = stablehlo.broadcast_in_dim %200, dims = [0, 1, 2] : (tensor<257x1x1xi1>) -> tensor<257x256x1xi1>
    %209 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2] : (tensor<1x256x1xi1>) -> tensor<257x256x1xi1>
    %210 = stablehlo.and %208, %209 : tensor<257x256x1xi1>
    %211 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_58 = stablehlo.constant dense<12> : tensor<i32>
    %212 = stablehlo.broadcast_in_dim %c_58, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %213 = stablehlo.compare  LT, %211, %212,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_59 = stablehlo.constant dense<12> : tensor<i32>
    %214 = stablehlo.broadcast_in_dim %c_59, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %215 = stablehlo.subtract %211, %214 : tensor<24xi32>
    %c_60 = stablehlo.constant dense<256> : tensor<i32>
    %216 = stablehlo.broadcast_in_dim %c_60, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %217 = stablehlo.add %215, %216 : tensor<24xi32>
    %c_61 = stablehlo.constant dense<12> : tensor<i32>
    %218 = stablehlo.broadcast_in_dim %c_61, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %219 = stablehlo.subtract %217, %218 : tensor<24xi32>
    %220 = call @_where_13(%213, %211, %219) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_62 = stablehlo.constant dense<0> : tensor<i32>
    %221 = stablehlo.broadcast_in_dim %c_62, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %222 = stablehlo.compare  GE, %220, %221,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_63 = stablehlo.constant dense<256> : tensor<i32>
    %223 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %224 = stablehlo.compare  LT, %220, %223,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %225 = stablehlo.and %222, %224 : tensor<24xi1>
    %226 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %227 = stablehlo.compare  GE, %220, %226,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_64 = stablehlo.constant dense<257> : tensor<i32>
    %228 = stablehlo.add %8, %c_64 : tensor<i32>
    %229 = stablehlo.broadcast_in_dim %228, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %230 = stablehlo.compare  LT, %220, %229,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %231 = stablehlo.and %227, %230 : tensor<24xi1>
    %232 = stablehlo.and %225, %231 : tensor<24xi1>
    %233 = stablehlo.reshape %232 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %234 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<257x256x1xi1>) -> tensor<257x256x24xi1>
    %235 = stablehlo.broadcast_in_dim %233, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<257x256x24xi1>
    %236 = stablehlo.and %234, %235 : tensor<257x256x24xi1>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %237 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f32>) -> tensor<257x256x24xf32>
    %238 = call @_where_92(%236, %18, %237) : (tensor<257x256x24xi1>, tensor<257x256x24xf32>, tensor<257x256x24xf32>) -> tensor<257x256x24xf32>
    %239 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_66 = stablehlo.constant dense<0> : tensor<i32>
    %240 = stablehlo.broadcast_in_dim %c_66, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %241 = stablehlo.compare  GE, %239, %240,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_67 = stablehlo.constant dense<257> : tensor<i32>
    %242 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %243 = stablehlo.compare  LT, %239, %242,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %244 = stablehlo.and %241, %243 : tensor<257xi1>
    %245 = stablehlo.reshape %244 : (tensor<257xi1>) -> tensor<257x1x1xi1>
    %c_68 = stablehlo.constant dense<true> : tensor<i1>
    %246 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i1>) -> tensor<257x1x1xi1>
    %247 = stablehlo.and %246, %245 : tensor<257x1x1xi1>
    %248 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_69 = stablehlo.constant dense<12> : tensor<i32>
    %249 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %250 = stablehlo.compare  LT, %248, %249,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_70 = stablehlo.constant dense<12> : tensor<i32>
    %251 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %252 = stablehlo.subtract %248, %251 : tensor<24xi32>
    %c_71 = stablehlo.constant dense<256> : tensor<i32>
    %253 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %254 = stablehlo.add %252, %253 : tensor<24xi32>
    %c_72 = stablehlo.constant dense<12> : tensor<i32>
    %255 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %256 = stablehlo.subtract %254, %255 : tensor<24xi32>
    %257 = call @_where_13(%250, %248, %256) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_73 = stablehlo.constant dense<0> : tensor<i32>
    %258 = stablehlo.broadcast_in_dim %c_73, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %259 = stablehlo.compare  GE, %257, %258,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_74 = stablehlo.constant dense<256> : tensor<i32>
    %260 = stablehlo.broadcast_in_dim %c_74, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %261 = stablehlo.compare  LT, %257, %260,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %262 = stablehlo.and %259, %261 : tensor<24xi1>
    %263 = stablehlo.reshape %262 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %264 = stablehlo.broadcast_in_dim %247, dims = [0, 1, 2] : (tensor<257x1x1xi1>) -> tensor<257x24x1xi1>
    %265 = stablehlo.broadcast_in_dim %263, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<257x24x1xi1>
    %266 = stablehlo.and %264, %265 : tensor<257x24x1xi1>
    %267 = stablehlo.iota dim = 0 : tensor<257xi32>
    %268 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %269 = stablehlo.add %267, %268 : tensor<257xi32>
    %c_75 = stablehlo.constant dense<0> : tensor<i32>
    %270 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %271 = stablehlo.compare  GE, %269, %270,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_76 = stablehlo.constant dense<256> : tensor<i32>
    %272 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %273 = stablehlo.compare  LT, %269, %272,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %274 = stablehlo.and %271, %273 : tensor<257xi1>
    %275 = stablehlo.reshape %274 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %276 = stablehlo.broadcast_in_dim %266, dims = [0, 1, 2] : (tensor<257x24x1xi1>) -> tensor<257x24x257xi1>
    %277 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<257x24x257xi1>
    %278 = stablehlo.and %276, %277 : tensor<257x24x257xi1>
    %cst_77 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %279 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f32>) -> tensor<257x24x257xf32>
    %280 = call @_where_98(%278, %arg126, %279) : (tensor<257x24x257xi1>, tensor<257x24x257xf32>, tensor<257x24x257xf32>) -> tensor<257x24x257xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %281 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f32>) -> tensor<257x257x1xf32>
    %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %282 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f32>) -> tensor<257x257x1xf32>
    %283 = stablehlo.slice %arg116 [0:257, 0:256, 0:1] : (tensor<257x256x257xf32>) -> tensor<257x256x1xf32>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %284 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<257x256x1xf32>
    %285 = "stablehlo.collective_permute"(%283) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<257x256x1xf32>) -> tensor<257x256x1xf32>
    %286 = stablehlo.slice %arg117 [0:256, 0:257, 0:1] : (tensor<256x257x257xf32>) -> tensor<256x257x1xf32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %287 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<256x257x1xf32>
    %288 = "stablehlo.collective_permute"(%286) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<256x257x1xf32>) -> tensor<256x257x1xf32>
    %289:9 = stablehlo.custom_call @beamz_cuda_sharded(%arg118, %arg119, %arg120, %arg115, %arg116, %arg117, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg28, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %60, %102, %144, %191, %238, %280, %arg9, %arg9, %arg9, %14, %281, %282, %284, %285, %287, %288) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 0 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<256x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x257xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<257x257x1xf32>, tensor<257x257x1xf32>, tensor<257x256x1xf32>, tensor<257x256x1xf32>, tensor<256x257x1xf32>, tensor<256x257x1xf32>) -> (tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<256x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x257xf32>)
    %290 = stablehlo.broadcast_in_dim %289#6, dims = [1, 2, 3] : (tensor<256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %291 = stablehlo.broadcast_in_dim %289#7, dims = [1, 2, 3] : (tensor<257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %c_82 = stablehlo.constant dense<0> : tensor<i32>
    %c_83 = stablehlo.constant dense<31> : tensor<i32>
    %292 = call @clip(%arg140, %c_82, %c_83) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_84 = stablehlo.constant dense<1> : tensor<ui32>
    %c_85 = stablehlo.constant dense<1> : tensor<ui32>
    %293 = stablehlo.partition_id : tensor<ui32>
    %294 = stablehlo.divide %293, %c_84 : tensor<ui32>
    %295 = stablehlo.remainder %294, %c_85 : tensor<ui32>
    %296 = stablehlo.convert %295 : (tensor<ui32>) -> tensor<i32>
    %c_86 = stablehlo.constant dense<257> : tensor<i32>
    %297 = stablehlo.multiply %296, %c_86 : tensor<i32>
    %c_87 = stablehlo.constant dense<75> : tensor<i32>
    %298 = stablehlo.subtract %c_87, %297 : tensor<i32>
    %c_88 = stablehlo.constant dense<0> : tensor<i32>
    %c_89 = stablehlo.constant dense<256> : tensor<i32>
    %299 = call @clip_125(%298, %c_88, %c_89) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %300 = stablehlo.add %297, %299 : tensor<i32>
    %301 = stablehlo.iota dim = 0 : tensor<1xi32>
    %302 = stablehlo.broadcast_in_dim %300, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %303 = stablehlo.add %302, %301 : tensor<1xi32>
    %c_90 = stablehlo.constant dense<75> : tensor<i32>
    %304 = stablehlo.broadcast_in_dim %c_90, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %305 = stablehlo.subtract %303, %304 : tensor<1xi32>
    %c_91 = stablehlo.constant dense<0> : tensor<i32>
    %306 = stablehlo.broadcast_in_dim %c_91, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %307 = stablehlo.compare  GE, %305, %306,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_92 = stablehlo.constant dense<1> : tensor<i32>
    %308 = stablehlo.broadcast_in_dim %c_92, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %309 = stablehlo.compare  LT, %305, %308,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %310 = stablehlo.and %307, %309 : tensor<1xi1>
    %311 = stablehlo.slice %arg29 [0:1, 0:14, 0:25, 0:1] : (tensor<2x14x25x1xf32>) -> tensor<1x14x25x1xf32>
    %312 = stablehlo.reshape %311 : (tensor<1x14x25x1xf32>) -> tensor<14x25x1xf32>
    %313 = call @_take(%312, %305) : (tensor<14x25x1xf32>, tensor<1xi32>) -> tensor<14x25x1xf32>
    %314 = stablehlo.reshape %310 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_93 = stablehlo.constant dense<0> : tensor<i32>
    %315 = stablehlo.compare  LT, %292, %c_93,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_94 = stablehlo.constant dense<32> : tensor<i32>
    %316 = stablehlo.add %292, %c_94 : tensor<i32>
    %317 = stablehlo.select %315, %316, %292 : tensor<i1>, tensor<i32>
    %c_95 = stablehlo.constant dense<0> : tensor<i32>
    %318 = stablehlo.dynamic_slice %arg30, %c_95, %317, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %319 = stablehlo.reshape %318 : (tensor<1x1xf32>) -> tensor<f32>
    %320 = stablehlo.broadcast_in_dim %319, dims = [] : (tensor<f32>) -> tensor<14x25x1xf32>
    %321 = stablehlo.multiply %313, %320 : tensor<14x25x1xf32>
    %c_96 = stablehlo.constant dense<0> : tensor<i32>
    %322 = call @_where_145(%314, %321, %c_96) : (tensor<1x1x1xi1>, tensor<14x25x1xf32>, tensor<i32>) -> tensor<14x25x1xf32>
    %c_97 = stablehlo.constant dense<121> : tensor<i32>
    %c_98 = stablehlo.constant dense<0> : tensor<i32>
    %323 = stablehlo.compare  LT, %c_97, %c_98,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_99 = stablehlo.constant dense<121> : tensor<i32>
    %c_100 = stablehlo.constant dense<256> : tensor<i32>
    %324 = stablehlo.add %c_99, %c_100 : tensor<i32>
    %c_101 = stablehlo.constant dense<121> : tensor<i32>
    %325 = stablehlo.select %323, %324, %c_101 : tensor<i1>, tensor<i32>
    %c_102 = stablehlo.constant dense<116> : tensor<i32>
    %c_103 = stablehlo.constant dense<0> : tensor<i32>
    %326 = stablehlo.compare  LT, %c_102, %c_103,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_104 = stablehlo.constant dense<116> : tensor<i32>
    %c_105 = stablehlo.constant dense<257> : tensor<i32>
    %327 = stablehlo.add %c_104, %c_105 : tensor<i32>
    %c_106 = stablehlo.constant dense<116> : tensor<i32>
    %328 = stablehlo.select %326, %327, %c_106 : tensor<i1>, tensor<i32>
    %c_107 = stablehlo.constant dense<0> : tensor<i32>
    %329 = stablehlo.compare  LT, %299, %c_107,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_108 = stablehlo.constant dense<257> : tensor<i32>
    %330 = stablehlo.add %299, %c_108 : tensor<i32>
    %331 = stablehlo.select %329, %330, %299 : tensor<i1>, tensor<i32>
    %332 = stablehlo.dynamic_slice %289#1, %325, %328, %331, sizes = [14, 25, 1] : (tensor<256x257x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<14x25x1xf32>
    %333 = stablehlo.add %332, %322 : tensor<14x25x1xf32>
    %c_109 = stablehlo.constant dense<121> : tensor<i32>
    %c_110 = stablehlo.constant dense<0> : tensor<i32>
    %334 = stablehlo.compare  LT, %c_109, %c_110,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_111 = stablehlo.constant dense<121> : tensor<i32>
    %c_112 = stablehlo.constant dense<256> : tensor<i32>
    %335 = stablehlo.add %c_111, %c_112 : tensor<i32>
    %c_113 = stablehlo.constant dense<121> : tensor<i32>
    %336 = stablehlo.select %334, %335, %c_113 : tensor<i1>, tensor<i32>
    %c_114 = stablehlo.constant dense<116> : tensor<i32>
    %c_115 = stablehlo.constant dense<0> : tensor<i32>
    %337 = stablehlo.compare  LT, %c_114, %c_115,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_116 = stablehlo.constant dense<116> : tensor<i32>
    %c_117 = stablehlo.constant dense<257> : tensor<i32>
    %338 = stablehlo.add %c_116, %c_117 : tensor<i32>
    %c_118 = stablehlo.constant dense<116> : tensor<i32>
    %339 = stablehlo.select %337, %338, %c_118 : tensor<i1>, tensor<i32>
    %c_119 = stablehlo.constant dense<0> : tensor<i32>
    %340 = stablehlo.compare  LT, %299, %c_119,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_120 = stablehlo.constant dense<257> : tensor<i32>
    %341 = stablehlo.add %299, %c_120 : tensor<i32>
    %342 = stablehlo.select %340, %341, %299 : tensor<i1>, tensor<i32>
    %343 = stablehlo.dynamic_update_slice %289#1, %333, %336, %339, %342 : (tensor<256x257x257xf32>, tensor<14x25x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<256x257x257xf32>
    %c_121 = stablehlo.constant dense<75> : tensor<i32>
    %344 = stablehlo.subtract %c_121, %297 : tensor<i32>
    %c_122 = stablehlo.constant dense<0> : tensor<i32>
    %c_123 = stablehlo.constant dense<256> : tensor<i32>
    %345 = call @clip_125(%344, %c_122, %c_123) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %346 = stablehlo.add %297, %345 : tensor<i32>
    %347 = stablehlo.iota dim = 0 : tensor<1xi32>
    %348 = stablehlo.broadcast_in_dim %346, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %349 = stablehlo.add %348, %347 : tensor<1xi32>
    %c_124 = stablehlo.constant dense<75> : tensor<i32>
    %350 = stablehlo.broadcast_in_dim %c_124, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %351 = stablehlo.subtract %349, %350 : tensor<1xi32>
    %c_125 = stablehlo.constant dense<0> : tensor<i32>
    %352 = stablehlo.broadcast_in_dim %c_125, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %353 = stablehlo.compare  GE, %351, %352,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_126 = stablehlo.constant dense<1> : tensor<i32>
    %354 = stablehlo.broadcast_in_dim %c_126, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %355 = stablehlo.compare  LT, %351, %354,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %356 = stablehlo.and %353, %355 : tensor<1xi1>
    %357 = stablehlo.slice %arg29 [1:2, 0:14, 0:25, 0:1] : (tensor<2x14x25x1xf32>) -> tensor<1x14x25x1xf32>
    %358 = stablehlo.reshape %357 : (tensor<1x14x25x1xf32>) -> tensor<14x25x1xf32>
    %359 = call @_take(%358, %351) : (tensor<14x25x1xf32>, tensor<1xi32>) -> tensor<14x25x1xf32>
    %360 = stablehlo.reshape %356 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_127 = stablehlo.constant dense<0> : tensor<i32>
    %361 = stablehlo.compare  LT, %292, %c_127,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_128 = stablehlo.constant dense<32> : tensor<i32>
    %362 = stablehlo.add %292, %c_128 : tensor<i32>
    %363 = stablehlo.select %361, %362, %292 : tensor<i1>, tensor<i32>
    %c_129 = stablehlo.constant dense<1> : tensor<i32>
    %364 = stablehlo.dynamic_slice %arg30, %c_129, %363, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %365 = stablehlo.reshape %364 : (tensor<1x1xf32>) -> tensor<f32>
    %366 = stablehlo.broadcast_in_dim %365, dims = [] : (tensor<f32>) -> tensor<14x25x1xf32>
    %367 = stablehlo.multiply %359, %366 : tensor<14x25x1xf32>
    %c_130 = stablehlo.constant dense<0> : tensor<i32>
    %368 = call @_where_145(%360, %367, %c_130) : (tensor<1x1x1xi1>, tensor<14x25x1xf32>, tensor<i32>) -> tensor<14x25x1xf32>
    %c_131 = stablehlo.constant dense<121> : tensor<i32>
    %c_132 = stablehlo.constant dense<0> : tensor<i32>
    %369 = stablehlo.compare  LT, %c_131, %c_132,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_133 = stablehlo.constant dense<121> : tensor<i32>
    %c_134 = stablehlo.constant dense<256> : tensor<i32>
    %370 = stablehlo.add %c_133, %c_134 : tensor<i32>
    %c_135 = stablehlo.constant dense<121> : tensor<i32>
    %371 = stablehlo.select %369, %370, %c_135 : tensor<i1>, tensor<i32>
    %c_136 = stablehlo.constant dense<116> : tensor<i32>
    %c_137 = stablehlo.constant dense<0> : tensor<i32>
    %372 = stablehlo.compare  LT, %c_136, %c_137,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_138 = stablehlo.constant dense<116> : tensor<i32>
    %c_139 = stablehlo.constant dense<257> : tensor<i32>
    %373 = stablehlo.add %c_138, %c_139 : tensor<i32>
    %c_140 = stablehlo.constant dense<116> : tensor<i32>
    %374 = stablehlo.select %372, %373, %c_140 : tensor<i1>, tensor<i32>
    %c_141 = stablehlo.constant dense<0> : tensor<i32>
    %375 = stablehlo.compare  LT, %345, %c_141,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_142 = stablehlo.constant dense<257> : tensor<i32>
    %376 = stablehlo.add %345, %c_142 : tensor<i32>
    %377 = stablehlo.select %375, %376, %345 : tensor<i1>, tensor<i32>
    %378 = stablehlo.dynamic_slice %343, %371, %374, %377, sizes = [14, 25, 1] : (tensor<256x257x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<14x25x1xf32>
    %379 = stablehlo.add %378, %368 : tensor<14x25x1xf32>
    %c_143 = stablehlo.constant dense<121> : tensor<i32>
    %c_144 = stablehlo.constant dense<0> : tensor<i32>
    %380 = stablehlo.compare  LT, %c_143, %c_144,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_145 = stablehlo.constant dense<121> : tensor<i32>
    %c_146 = stablehlo.constant dense<256> : tensor<i32>
    %381 = stablehlo.add %c_145, %c_146 : tensor<i32>
    %c_147 = stablehlo.constant dense<121> : tensor<i32>
    %382 = stablehlo.select %380, %381, %c_147 : tensor<i1>, tensor<i32>
    %c_148 = stablehlo.constant dense<116> : tensor<i32>
    %c_149 = stablehlo.constant dense<0> : tensor<i32>
    %383 = stablehlo.compare  LT, %c_148, %c_149,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_150 = stablehlo.constant dense<116> : tensor<i32>
    %c_151 = stablehlo.constant dense<257> : tensor<i32>
    %384 = stablehlo.add %c_150, %c_151 : tensor<i32>
    %c_152 = stablehlo.constant dense<116> : tensor<i32>
    %385 = stablehlo.select %383, %384, %c_152 : tensor<i1>, tensor<i32>
    %c_153 = stablehlo.constant dense<0> : tensor<i32>
    %386 = stablehlo.compare  LT, %345, %c_153,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_154 = stablehlo.constant dense<257> : tensor<i32>
    %387 = stablehlo.add %345, %c_154 : tensor<i32>
    %388 = stablehlo.select %386, %387, %345 : tensor<i1>, tensor<i32>
    %389 = stablehlo.dynamic_update_slice %343, %379, %382, %385, %388 : (tensor<256x257x257xf32>, tensor<14x25x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<256x257x257xf32>
    %c_155 = stablehlo.constant dense<0> : tensor<i32>
    %c_156 = stablehlo.constant dense<31> : tensor<i32>
    %390 = call @clip(%arg140, %c_155, %c_156) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_157 = stablehlo.constant dense<1> : tensor<ui32>
    %c_158 = stablehlo.constant dense<1> : tensor<ui32>
    %391 = stablehlo.partition_id : tensor<ui32>
    %392 = stablehlo.divide %391, %c_157 : tensor<ui32>
    %393 = stablehlo.remainder %392, %c_158 : tensor<ui32>
    %394 = stablehlo.convert %393 : (tensor<ui32>) -> tensor<i32>
    %c_159 = stablehlo.constant dense<257> : tensor<i32>
    %395 = stablehlo.multiply %394, %c_159 : tensor<i32>
    %c_160 = stablehlo.constant dense<75> : tensor<i32>
    %396 = stablehlo.subtract %c_160, %395 : tensor<i32>
    %c_161 = stablehlo.constant dense<0> : tensor<i32>
    %c_162 = stablehlo.constant dense<256> : tensor<i32>
    %397 = call @clip_125(%396, %c_161, %c_162) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %398 = stablehlo.add %395, %397 : tensor<i32>
    %399 = stablehlo.iota dim = 0 : tensor<1xi32>
    %400 = stablehlo.broadcast_in_dim %398, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %401 = stablehlo.add %400, %399 : tensor<1xi32>
    %c_163 = stablehlo.constant dense<75> : tensor<i32>
    %402 = stablehlo.broadcast_in_dim %c_163, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %403 = stablehlo.subtract %401, %402 : tensor<1xi32>
    %c_164 = stablehlo.constant dense<0> : tensor<i32>
    %404 = stablehlo.broadcast_in_dim %c_164, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %405 = stablehlo.compare  GE, %403, %404,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_165 = stablehlo.constant dense<1> : tensor<i32>
    %406 = stablehlo.broadcast_in_dim %c_165, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %407 = stablehlo.compare  LT, %403, %406,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %408 = stablehlo.and %405, %407 : tensor<1xi1>
    %409 = stablehlo.slice %arg31 [0:1, 0:15, 0:24, 0:1] : (tensor<2x15x24x1xf32>) -> tensor<1x15x24x1xf32>
    %410 = stablehlo.reshape %409 : (tensor<1x15x24x1xf32>) -> tensor<15x24x1xf32>
    %411 = call @_take_157(%410, %403) : (tensor<15x24x1xf32>, tensor<1xi32>) -> tensor<15x24x1xf32>
    %412 = stablehlo.reshape %408 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_166 = stablehlo.constant dense<0> : tensor<i32>
    %413 = stablehlo.compare  LT, %390, %c_166,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_167 = stablehlo.constant dense<32> : tensor<i32>
    %414 = stablehlo.add %390, %c_167 : tensor<i32>
    %415 = stablehlo.select %413, %414, %390 : tensor<i1>, tensor<i32>
    %c_168 = stablehlo.constant dense<0> : tensor<i32>
    %416 = stablehlo.dynamic_slice %arg32, %c_168, %415, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %417 = stablehlo.reshape %416 : (tensor<1x1xf32>) -> tensor<f32>
    %418 = stablehlo.broadcast_in_dim %417, dims = [] : (tensor<f32>) -> tensor<15x24x1xf32>
    %419 = stablehlo.multiply %411, %418 : tensor<15x24x1xf32>
    %c_169 = stablehlo.constant dense<0> : tensor<i32>
    %420 = call @_where_161(%412, %419, %c_169) : (tensor<1x1x1xi1>, tensor<15x24x1xf32>, tensor<i32>) -> tensor<15x24x1xf32>
    %c_170 = stablehlo.constant dense<121> : tensor<i32>
    %c_171 = stablehlo.constant dense<0> : tensor<i32>
    %421 = stablehlo.compare  LT, %c_170, %c_171,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_172 = stablehlo.constant dense<121> : tensor<i32>
    %c_173 = stablehlo.constant dense<257> : tensor<i32>
    %422 = stablehlo.add %c_172, %c_173 : tensor<i32>
    %c_174 = stablehlo.constant dense<121> : tensor<i32>
    %423 = stablehlo.select %421, %422, %c_174 : tensor<i1>, tensor<i32>
    %c_175 = stablehlo.constant dense<116> : tensor<i32>
    %c_176 = stablehlo.constant dense<0> : tensor<i32>
    %424 = stablehlo.compare  LT, %c_175, %c_176,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_177 = stablehlo.constant dense<116> : tensor<i32>
    %c_178 = stablehlo.constant dense<256> : tensor<i32>
    %425 = stablehlo.add %c_177, %c_178 : tensor<i32>
    %c_179 = stablehlo.constant dense<116> : tensor<i32>
    %426 = stablehlo.select %424, %425, %c_179 : tensor<i1>, tensor<i32>
    %c_180 = stablehlo.constant dense<0> : tensor<i32>
    %427 = stablehlo.compare  LT, %397, %c_180,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_181 = stablehlo.constant dense<257> : tensor<i32>
    %428 = stablehlo.add %397, %c_181 : tensor<i32>
    %429 = stablehlo.select %427, %428, %397 : tensor<i1>, tensor<i32>
    %430 = stablehlo.dynamic_slice %289#2, %423, %426, %429, sizes = [15, 24, 1] : (tensor<257x256x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<15x24x1xf32>
    %431 = stablehlo.add %430, %420 : tensor<15x24x1xf32>
    %c_182 = stablehlo.constant dense<121> : tensor<i32>
    %c_183 = stablehlo.constant dense<0> : tensor<i32>
    %432 = stablehlo.compare  LT, %c_182, %c_183,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_184 = stablehlo.constant dense<121> : tensor<i32>
    %c_185 = stablehlo.constant dense<257> : tensor<i32>
    %433 = stablehlo.add %c_184, %c_185 : tensor<i32>
    %c_186 = stablehlo.constant dense<121> : tensor<i32>
    %434 = stablehlo.select %432, %433, %c_186 : tensor<i1>, tensor<i32>
    %c_187 = stablehlo.constant dense<116> : tensor<i32>
    %c_188 = stablehlo.constant dense<0> : tensor<i32>
    %435 = stablehlo.compare  LT, %c_187, %c_188,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_189 = stablehlo.constant dense<116> : tensor<i32>
    %c_190 = stablehlo.constant dense<256> : tensor<i32>
    %436 = stablehlo.add %c_189, %c_190 : tensor<i32>
    %c_191 = stablehlo.constant dense<116> : tensor<i32>
    %437 = stablehlo.select %435, %436, %c_191 : tensor<i1>, tensor<i32>
    %c_192 = stablehlo.constant dense<0> : tensor<i32>
    %438 = stablehlo.compare  LT, %397, %c_192,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_193 = stablehlo.constant dense<257> : tensor<i32>
    %439 = stablehlo.add %397, %c_193 : tensor<i32>
    %440 = stablehlo.select %438, %439, %397 : tensor<i1>, tensor<i32>
    %441 = stablehlo.dynamic_update_slice %289#2, %431, %434, %437, %440 : (tensor<257x256x257xf32>, tensor<15x24x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<257x256x257xf32>
    %c_194 = stablehlo.constant dense<75> : tensor<i32>
    %442 = stablehlo.subtract %c_194, %395 : tensor<i32>
    %c_195 = stablehlo.constant dense<0> : tensor<i32>
    %c_196 = stablehlo.constant dense<256> : tensor<i32>
    %443 = call @clip_125(%442, %c_195, %c_196) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %444 = stablehlo.add %395, %443 : tensor<i32>
    %445 = stablehlo.iota dim = 0 : tensor<1xi32>
    %446 = stablehlo.broadcast_in_dim %444, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %447 = stablehlo.add %446, %445 : tensor<1xi32>
    %c_197 = stablehlo.constant dense<75> : tensor<i32>
    %448 = stablehlo.broadcast_in_dim %c_197, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %449 = stablehlo.subtract %447, %448 : tensor<1xi32>
    %c_198 = stablehlo.constant dense<0> : tensor<i32>
    %450 = stablehlo.broadcast_in_dim %c_198, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %451 = stablehlo.compare  GE, %449, %450,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_199 = stablehlo.constant dense<1> : tensor<i32>
    %452 = stablehlo.broadcast_in_dim %c_199, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %453 = stablehlo.compare  LT, %449, %452,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %454 = stablehlo.and %451, %453 : tensor<1xi1>
    %455 = stablehlo.slice %arg31 [1:2, 0:15, 0:24, 0:1] : (tensor<2x15x24x1xf32>) -> tensor<1x15x24x1xf32>
    %456 = stablehlo.reshape %455 : (tensor<1x15x24x1xf32>) -> tensor<15x24x1xf32>
    %457 = call @_take_157(%456, %449) : (tensor<15x24x1xf32>, tensor<1xi32>) -> tensor<15x24x1xf32>
    %458 = stablehlo.reshape %454 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_200 = stablehlo.constant dense<0> : tensor<i32>
    %459 = stablehlo.compare  LT, %390, %c_200,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_201 = stablehlo.constant dense<32> : tensor<i32>
    %460 = stablehlo.add %390, %c_201 : tensor<i32>
    %461 = stablehlo.select %459, %460, %390 : tensor<i1>, tensor<i32>
    %c_202 = stablehlo.constant dense<1> : tensor<i32>
    %462 = stablehlo.dynamic_slice %arg32, %c_202, %461, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %463 = stablehlo.reshape %462 : (tensor<1x1xf32>) -> tensor<f32>
    %464 = stablehlo.broadcast_in_dim %463, dims = [] : (tensor<f32>) -> tensor<15x24x1xf32>
    %465 = stablehlo.multiply %457, %464 : tensor<15x24x1xf32>
    %c_203 = stablehlo.constant dense<0> : tensor<i32>
    %466 = call @_where_161(%458, %465, %c_203) : (tensor<1x1x1xi1>, tensor<15x24x1xf32>, tensor<i32>) -> tensor<15x24x1xf32>
    %c_204 = stablehlo.constant dense<121> : tensor<i32>
    %c_205 = stablehlo.constant dense<0> : tensor<i32>
    %467 = stablehlo.compare  LT, %c_204, %c_205,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_206 = stablehlo.constant dense<121> : tensor<i32>
    %c_207 = stablehlo.constant dense<257> : tensor<i32>
    %468 = stablehlo.add %c_206, %c_207 : tensor<i32>
    %c_208 = stablehlo.constant dense<121> : tensor<i32>
    %469 = stablehlo.select %467, %468, %c_208 : tensor<i1>, tensor<i32>
    %c_209 = stablehlo.constant dense<116> : tensor<i32>
    %c_210 = stablehlo.constant dense<0> : tensor<i32>
    %470 = stablehlo.compare  LT, %c_209, %c_210,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_211 = stablehlo.constant dense<116> : tensor<i32>
    %c_212 = stablehlo.constant dense<256> : tensor<i32>
    %471 = stablehlo.add %c_211, %c_212 : tensor<i32>
    %c_213 = stablehlo.constant dense<116> : tensor<i32>
    %472 = stablehlo.select %470, %471, %c_213 : tensor<i1>, tensor<i32>
    %c_214 = stablehlo.constant dense<0> : tensor<i32>
    %473 = stablehlo.compare  LT, %443, %c_214,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_215 = stablehlo.constant dense<257> : tensor<i32>
    %474 = stablehlo.add %443, %c_215 : tensor<i32>
    %475 = stablehlo.select %473, %474, %443 : tensor<i1>, tensor<i32>
    %476 = stablehlo.dynamic_slice %441, %469, %472, %475, sizes = [15, 24, 1] : (tensor<257x256x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<15x24x1xf32>
    %477 = stablehlo.add %476, %466 : tensor<15x24x1xf32>
    %c_216 = stablehlo.constant dense<121> : tensor<i32>
    %c_217 = stablehlo.constant dense<0> : tensor<i32>
    %478 = stablehlo.compare  LT, %c_216, %c_217,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_218 = stablehlo.constant dense<121> : tensor<i32>
    %c_219 = stablehlo.constant dense<257> : tensor<i32>
    %479 = stablehlo.add %c_218, %c_219 : tensor<i32>
    %c_220 = stablehlo.constant dense<121> : tensor<i32>
    %480 = stablehlo.select %478, %479, %c_220 : tensor<i1>, tensor<i32>
    %c_221 = stablehlo.constant dense<116> : tensor<i32>
    %c_222 = stablehlo.constant dense<0> : tensor<i32>
    %481 = stablehlo.compare  LT, %c_221, %c_222,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_223 = stablehlo.constant dense<116> : tensor<i32>
    %c_224 = stablehlo.constant dense<256> : tensor<i32>
    %482 = stablehlo.add %c_223, %c_224 : tensor<i32>
    %c_225 = stablehlo.constant dense<116> : tensor<i32>
    %483 = stablehlo.select %481, %482, %c_225 : tensor<i1>, tensor<i32>
    %c_226 = stablehlo.constant dense<0> : tensor<i32>
    %484 = stablehlo.compare  LT, %443, %c_226,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_227 = stablehlo.constant dense<257> : tensor<i32>
    %485 = stablehlo.add %443, %c_227 : tensor<i32>
    %486 = stablehlo.select %484, %485, %443 : tensor<i1>, tensor<i32>
    %487 = stablehlo.dynamic_update_slice %441, %477, %480, %483, %486 : (tensor<257x256x257xf32>, tensor<15x24x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<257x256x257xf32>
    %c_228 = stablehlo.constant dense<1> : tensor<ui32>
    %c_229 = stablehlo.constant dense<1> : tensor<ui32>
    %488 = stablehlo.partition_id : tensor<ui32>
    %489 = stablehlo.divide %488, %c_228 : tensor<ui32>
    %490 = stablehlo.remainder %489, %c_229 : tensor<ui32>
    %491 = stablehlo.convert %490 : (tensor<ui32>) -> tensor<i32>
    %c_230 = stablehlo.constant dense<257> : tensor<i32>
    %492 = stablehlo.multiply %491, %c_230 : tensor<i32>
    %c_231 = stablehlo.constant dense<2> : tensor<i32>
    %493 = stablehlo.broadcast_in_dim %c_231, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %494 = stablehlo.broadcast_in_dim %492, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_232 = stablehlo.constant dense<0> : tensor<i32>
    %495 = stablehlo.broadcast_in_dim %c_232, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %496 = stablehlo.concatenate %493, %494, %495, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %497 = stablehlo.broadcast_in_dim %496, dims = [1] : (tensor<3xi32>) -> tensor<1x3xi32>
    %498 = stablehlo.concatenate %497, %arg33, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
    %499 = stablehlo.slice %arg130 [0:1, 0:257, 0:256, 0:24] : (tensor<1x257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %500 = stablehlo.reshape %499 : (tensor<1x257x256x24xf32>) -> tensor<257x256x24xf32>
    %501 = stablehlo.slice %arg131 [0:1, 0:256, 0:257, 0:24] : (tensor<1x256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %502 = stablehlo.reshape %501 : (tensor<1x256x257x24xf32>) -> tensor<256x257x24xf32>
    %503 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_233 = stablehlo.constant dense<0> : tensor<i32>
    %504 = stablehlo.broadcast_in_dim %c_233, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %505 = stablehlo.compare  GE, %503, %504,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_234 = stablehlo.constant dense<257> : tensor<i32>
    %506 = stablehlo.broadcast_in_dim %c_234, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %507 = stablehlo.compare  LT, %503, %506,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %508 = stablehlo.and %505, %507 : tensor<257xi1>
    %509 = stablehlo.reshape %508 : (tensor<257xi1>) -> tensor<257x1x1xi1>
    %c_235 = stablehlo.constant dense<true> : tensor<i1>
    %510 = stablehlo.broadcast_in_dim %c_235, dims = [] : (tensor<i1>) -> tensor<257x1x1xi1>
    %511 = stablehlo.and %510, %509 : tensor<257x1x1xi1>
    %512 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_236 = stablehlo.constant dense<12> : tensor<i32>
    %513 = stablehlo.broadcast_in_dim %c_236, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %514 = stablehlo.compare  LT, %512, %513,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_237 = stablehlo.constant dense<12> : tensor<i32>
    %515 = stablehlo.broadcast_in_dim %c_237, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %516 = stablehlo.subtract %512, %515 : tensor<24xi32>
    %c_238 = stablehlo.constant dense<257> : tensor<i32>
    %517 = stablehlo.broadcast_in_dim %c_238, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %518 = stablehlo.add %516, %517 : tensor<24xi32>
    %c_239 = stablehlo.constant dense<12> : tensor<i32>
    %519 = stablehlo.broadcast_in_dim %c_239, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %520 = stablehlo.subtract %518, %519 : tensor<24xi32>
    %521 = call @_where_13(%514, %512, %520) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_240 = stablehlo.constant dense<0> : tensor<i32>
    %522 = stablehlo.broadcast_in_dim %c_240, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %523 = stablehlo.compare  GE, %521, %522,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_241 = stablehlo.constant dense<257> : tensor<i32>
    %524 = stablehlo.broadcast_in_dim %c_241, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %525 = stablehlo.compare  LT, %521, %524,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %526 = stablehlo.and %523, %525 : tensor<24xi1>
    %527 = stablehlo.reshape %526 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %528 = stablehlo.broadcast_in_dim %511, dims = [0, 1, 2] : (tensor<257x1x1xi1>) -> tensor<257x24x1xi1>
    %529 = stablehlo.broadcast_in_dim %527, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<257x24x1xi1>
    %530 = stablehlo.and %528, %529 : tensor<257x24x1xi1>
    %531 = stablehlo.iota dim = 0 : tensor<257xi32>
    %532 = stablehlo.broadcast_in_dim %492, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %533 = stablehlo.add %531, %532 : tensor<257xi32>
    %c_242 = stablehlo.constant dense<0> : tensor<i32>
    %534 = stablehlo.broadcast_in_dim %c_242, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %535 = stablehlo.compare  GE, %533, %534,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_243 = stablehlo.constant dense<256> : tensor<i32>
    %536 = stablehlo.broadcast_in_dim %c_243, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %537 = stablehlo.compare  LT, %533, %536,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %538 = stablehlo.and %535, %537 : tensor<257xi1>
    %539 = stablehlo.reshape %538 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %540 = stablehlo.broadcast_in_dim %530, dims = [0, 1, 2] : (tensor<257x24x1xi1>) -> tensor<257x24x257xi1>
    %541 = stablehlo.broadcast_in_dim %539, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<257x24x257xi1>
    %542 = stablehlo.and %540, %541 : tensor<257x24x257xi1>
    %cst_244 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %543 = stablehlo.broadcast_in_dim %cst_244, dims = [] : (tensor<f32>) -> tensor<257x24x257xf32>
    %544 = call @_where_98(%542, %arg127, %543) : (tensor<257x24x257xi1>, tensor<257x24x257xf32>, tensor<257x24x257xf32>) -> tensor<257x24x257xf32>
    %545 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_245 = stablehlo.constant dense<12> : tensor<i32>
    %546 = stablehlo.broadcast_in_dim %c_245, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %547 = stablehlo.compare  LT, %545, %546,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_246 = stablehlo.constant dense<12> : tensor<i32>
    %548 = stablehlo.broadcast_in_dim %c_246, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %549 = stablehlo.subtract %545, %548 : tensor<24xi32>
    %c_247 = stablehlo.constant dense<257> : tensor<i32>
    %550 = stablehlo.broadcast_in_dim %c_247, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %551 = stablehlo.add %549, %550 : tensor<24xi32>
    %c_248 = stablehlo.constant dense<12> : tensor<i32>
    %552 = stablehlo.broadcast_in_dim %c_248, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %553 = stablehlo.subtract %551, %552 : tensor<24xi32>
    %554 = call @_where_13(%547, %545, %553) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_249 = stablehlo.constant dense<0> : tensor<i32>
    %555 = stablehlo.broadcast_in_dim %c_249, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %556 = stablehlo.compare  GE, %554, %555,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_250 = stablehlo.constant dense<257> : tensor<i32>
    %557 = stablehlo.broadcast_in_dim %c_250, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %558 = stablehlo.compare  LT, %554, %557,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %559 = stablehlo.and %556, %558 : tensor<24xi1>
    %560 = stablehlo.reshape %559 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_251 = stablehlo.constant dense<true> : tensor<i1>
    %561 = stablehlo.broadcast_in_dim %c_251, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %562 = stablehlo.and %561, %560 : tensor<24x1x1xi1>
    %563 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_252 = stablehlo.constant dense<0> : tensor<i32>
    %564 = stablehlo.broadcast_in_dim %c_252, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %565 = stablehlo.compare  GE, %563, %564,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_253 = stablehlo.constant dense<257> : tensor<i32>
    %566 = stablehlo.broadcast_in_dim %c_253, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %567 = stablehlo.compare  LT, %563, %566,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %568 = stablehlo.and %565, %567 : tensor<257xi1>
    %569 = stablehlo.reshape %568 : (tensor<257xi1>) -> tensor<1x257x1xi1>
    %570 = stablehlo.broadcast_in_dim %562, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x257x1xi1>
    %571 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2] : (tensor<1x257x1xi1>) -> tensor<24x257x1xi1>
    %572 = stablehlo.and %570, %571 : tensor<24x257x1xi1>
    %573 = stablehlo.iota dim = 0 : tensor<257xi32>
    %574 = stablehlo.broadcast_in_dim %492, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %575 = stablehlo.add %573, %574 : tensor<257xi32>
    %c_254 = stablehlo.constant dense<0> : tensor<i32>
    %576 = stablehlo.broadcast_in_dim %c_254, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %577 = stablehlo.compare  GE, %575, %576,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_255 = stablehlo.constant dense<256> : tensor<i32>
    %578 = stablehlo.broadcast_in_dim %c_255, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %579 = stablehlo.compare  LT, %575, %578,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %580 = stablehlo.and %577, %579 : tensor<257xi1>
    %581 = stablehlo.reshape %580 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %582 = stablehlo.broadcast_in_dim %572, dims = [0, 1, 2] : (tensor<24x257x1xi1>) -> tensor<24x257x257xi1>
    %583 = stablehlo.broadcast_in_dim %581, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<24x257x257xi1>
    %584 = stablehlo.and %582, %583 : tensor<24x257x257xi1>
    %cst_256 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %585 = stablehlo.broadcast_in_dim %cst_256, dims = [] : (tensor<f32>) -> tensor<24x257x257xf32>
    %586 = call @_where_89(%584, %arg128, %585) : (tensor<24x257x257xi1>, tensor<24x257x257xf32>, tensor<24x257x257xf32>) -> tensor<24x257x257xf32>
    %587 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_257 = stablehlo.constant dense<12> : tensor<i32>
    %588 = stablehlo.broadcast_in_dim %c_257, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %589 = stablehlo.compare  LT, %587, %588,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_258 = stablehlo.constant dense<12> : tensor<i32>
    %590 = stablehlo.broadcast_in_dim %c_258, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %591 = stablehlo.subtract %587, %590 : tensor<24xi32>
    %c_259 = stablehlo.constant dense<257> : tensor<i32>
    %592 = stablehlo.broadcast_in_dim %c_259, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %593 = stablehlo.add %591, %592 : tensor<24xi32>
    %c_260 = stablehlo.constant dense<12> : tensor<i32>
    %594 = stablehlo.broadcast_in_dim %c_260, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %595 = stablehlo.subtract %593, %594 : tensor<24xi32>
    %596 = call @_where_13(%589, %587, %595) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_261 = stablehlo.constant dense<0> : tensor<i32>
    %597 = stablehlo.broadcast_in_dim %c_261, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %598 = stablehlo.compare  GE, %596, %597,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_262 = stablehlo.constant dense<257> : tensor<i32>
    %599 = stablehlo.broadcast_in_dim %c_262, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %600 = stablehlo.compare  LT, %596, %599,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %601 = stablehlo.and %598, %600 : tensor<24xi1>
    %602 = stablehlo.reshape %601 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_263 = stablehlo.constant dense<true> : tensor<i1>
    %603 = stablehlo.broadcast_in_dim %c_263, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %604 = stablehlo.and %603, %602 : tensor<24x1x1xi1>
    %605 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_264 = stablehlo.constant dense<0> : tensor<i32>
    %606 = stablehlo.broadcast_in_dim %c_264, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %607 = stablehlo.compare  GE, %605, %606,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_265 = stablehlo.constant dense<256> : tensor<i32>
    %608 = stablehlo.broadcast_in_dim %c_265, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %609 = stablehlo.compare  LT, %605, %608,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %610 = stablehlo.and %607, %609 : tensor<256xi1>
    %611 = stablehlo.reshape %610 : (tensor<256xi1>) -> tensor<1x256x1xi1>
    %612 = stablehlo.broadcast_in_dim %604, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x256x1xi1>
    %613 = stablehlo.broadcast_in_dim %611, dims = [0, 1, 2] : (tensor<1x256x1xi1>) -> tensor<24x256x1xi1>
    %614 = stablehlo.and %612, %613 : tensor<24x256x1xi1>
    %615 = stablehlo.iota dim = 0 : tensor<257xi32>
    %616 = stablehlo.broadcast_in_dim %492, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %617 = stablehlo.add %615, %616 : tensor<257xi32>
    %c_266 = stablehlo.constant dense<0> : tensor<i32>
    %618 = stablehlo.broadcast_in_dim %c_266, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %619 = stablehlo.compare  GE, %617, %618,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_267 = stablehlo.constant dense<257> : tensor<i32>
    %620 = stablehlo.broadcast_in_dim %c_267, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %621 = stablehlo.compare  LT, %617, %620,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %622 = stablehlo.and %619, %621 : tensor<257xi1>
    %623 = stablehlo.reshape %622 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %624 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2] : (tensor<24x256x1xi1>) -> tensor<24x256x257xi1>
    %625 = stablehlo.broadcast_in_dim %623, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<24x256x257xi1>
    %626 = stablehlo.and %624, %625 : tensor<24x256x257xi1>
    %cst_268 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %627 = stablehlo.broadcast_in_dim %cst_268, dims = [] : (tensor<f32>) -> tensor<24x256x257xf32>
    %628 = call @_where_82(%626, %arg129, %627) : (tensor<24x256x257xi1>, tensor<24x256x257xf32>, tensor<24x256x257xf32>) -> tensor<24x256x257xf32>
    %629 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_269 = stablehlo.constant dense<0> : tensor<i32>
    %630 = stablehlo.broadcast_in_dim %c_269, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %631 = stablehlo.compare  GE, %629, %630,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_270 = stablehlo.constant dense<257> : tensor<i32>
    %632 = stablehlo.broadcast_in_dim %c_270, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %633 = stablehlo.compare  LT, %629, %632,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %634 = stablehlo.and %631, %633 : tensor<257xi1>
    %635 = stablehlo.reshape %634 : (tensor<257xi1>) -> tensor<257x1x1xi1>
    %c_271 = stablehlo.constant dense<true> : tensor<i1>
    %636 = stablehlo.broadcast_in_dim %c_271, dims = [] : (tensor<i1>) -> tensor<257x1x1xi1>
    %637 = stablehlo.and %636, %635 : tensor<257x1x1xi1>
    %638 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_272 = stablehlo.constant dense<0> : tensor<i32>
    %639 = stablehlo.broadcast_in_dim %c_272, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %640 = stablehlo.compare  GE, %638, %639,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_273 = stablehlo.constant dense<256> : tensor<i32>
    %641 = stablehlo.broadcast_in_dim %c_273, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %642 = stablehlo.compare  LT, %638, %641,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %643 = stablehlo.and %640, %642 : tensor<256xi1>
    %644 = stablehlo.reshape %643 : (tensor<256xi1>) -> tensor<1x256x1xi1>
    %645 = stablehlo.broadcast_in_dim %637, dims = [0, 1, 2] : (tensor<257x1x1xi1>) -> tensor<257x256x1xi1>
    %646 = stablehlo.broadcast_in_dim %644, dims = [0, 1, 2] : (tensor<1x256x1xi1>) -> tensor<257x256x1xi1>
    %647 = stablehlo.and %645, %646 : tensor<257x256x1xi1>
    %648 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_274 = stablehlo.constant dense<12> : tensor<i32>
    %649 = stablehlo.broadcast_in_dim %c_274, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %650 = stablehlo.compare  LT, %648, %649,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_275 = stablehlo.constant dense<12> : tensor<i32>
    %651 = stablehlo.broadcast_in_dim %c_275, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %652 = stablehlo.subtract %648, %651 : tensor<24xi32>
    %c_276 = stablehlo.constant dense<257> : tensor<i32>
    %653 = stablehlo.broadcast_in_dim %c_276, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %654 = stablehlo.add %652, %653 : tensor<24xi32>
    %c_277 = stablehlo.constant dense<12> : tensor<i32>
    %655 = stablehlo.broadcast_in_dim %c_277, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %656 = stablehlo.subtract %654, %655 : tensor<24xi32>
    %657 = call @_where_13(%650, %648, %656) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_278 = stablehlo.constant dense<0> : tensor<i32>
    %658 = stablehlo.broadcast_in_dim %c_278, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %659 = stablehlo.compare  GE, %657, %658,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_279 = stablehlo.constant dense<257> : tensor<i32>
    %660 = stablehlo.broadcast_in_dim %c_279, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %661 = stablehlo.compare  LT, %657, %660,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %662 = stablehlo.and %659, %661 : tensor<24xi1>
    %663 = stablehlo.broadcast_in_dim %492, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %664 = stablehlo.compare  GE, %657, %663,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_280 = stablehlo.constant dense<257> : tensor<i32>
    %665 = stablehlo.add %492, %c_280 : tensor<i32>
    %666 = stablehlo.broadcast_in_dim %665, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %667 = stablehlo.compare  LT, %657, %666,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %668 = stablehlo.and %664, %667 : tensor<24xi1>
    %669 = stablehlo.and %662, %668 : tensor<24xi1>
    %670 = stablehlo.reshape %669 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %671 = stablehlo.broadcast_in_dim %647, dims = [0, 1, 2] : (tensor<257x256x1xi1>) -> tensor<257x256x24xi1>
    %672 = stablehlo.broadcast_in_dim %670, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<257x256x24xi1>
    %673 = stablehlo.and %671, %672 : tensor<257x256x24xi1>
    %cst_281 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %674 = stablehlo.broadcast_in_dim %cst_281, dims = [] : (tensor<f32>) -> tensor<257x256x24xf32>
    %675 = call @_where_92(%673, %500, %674) : (tensor<257x256x24xi1>, tensor<257x256x24xf32>, tensor<257x256x24xf32>) -> tensor<257x256x24xf32>
    %676 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_282 = stablehlo.constant dense<0> : tensor<i32>
    %677 = stablehlo.broadcast_in_dim %c_282, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %678 = stablehlo.compare  GE, %676, %677,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_283 = stablehlo.constant dense<256> : tensor<i32>
    %679 = stablehlo.broadcast_in_dim %c_283, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %680 = stablehlo.compare  LT, %676, %679,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %681 = stablehlo.and %678, %680 : tensor<256xi1>
    %682 = stablehlo.reshape %681 : (tensor<256xi1>) -> tensor<256x1x1xi1>
    %c_284 = stablehlo.constant dense<true> : tensor<i1>
    %683 = stablehlo.broadcast_in_dim %c_284, dims = [] : (tensor<i1>) -> tensor<256x1x1xi1>
    %684 = stablehlo.and %683, %682 : tensor<256x1x1xi1>
    %685 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_285 = stablehlo.constant dense<0> : tensor<i32>
    %686 = stablehlo.broadcast_in_dim %c_285, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %687 = stablehlo.compare  GE, %685, %686,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_286 = stablehlo.constant dense<257> : tensor<i32>
    %688 = stablehlo.broadcast_in_dim %c_286, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %689 = stablehlo.compare  LT, %685, %688,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %690 = stablehlo.and %687, %689 : tensor<257xi1>
    %691 = stablehlo.reshape %690 : (tensor<257xi1>) -> tensor<1x257x1xi1>
    %692 = stablehlo.broadcast_in_dim %684, dims = [0, 1, 2] : (tensor<256x1x1xi1>) -> tensor<256x257x1xi1>
    %693 = stablehlo.broadcast_in_dim %691, dims = [0, 1, 2] : (tensor<1x257x1xi1>) -> tensor<256x257x1xi1>
    %694 = stablehlo.and %692, %693 : tensor<256x257x1xi1>
    %695 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_287 = stablehlo.constant dense<12> : tensor<i32>
    %696 = stablehlo.broadcast_in_dim %c_287, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %697 = stablehlo.compare  LT, %695, %696,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_288 = stablehlo.constant dense<12> : tensor<i32>
    %698 = stablehlo.broadcast_in_dim %c_288, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %699 = stablehlo.subtract %695, %698 : tensor<24xi32>
    %c_289 = stablehlo.constant dense<257> : tensor<i32>
    %700 = stablehlo.broadcast_in_dim %c_289, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %701 = stablehlo.add %699, %700 : tensor<24xi32>
    %c_290 = stablehlo.constant dense<12> : tensor<i32>
    %702 = stablehlo.broadcast_in_dim %c_290, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %703 = stablehlo.subtract %701, %702 : tensor<24xi32>
    %704 = call @_where_13(%697, %695, %703) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_291 = stablehlo.constant dense<0> : tensor<i32>
    %705 = stablehlo.broadcast_in_dim %c_291, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %706 = stablehlo.compare  GE, %704, %705,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_292 = stablehlo.constant dense<257> : tensor<i32>
    %707 = stablehlo.broadcast_in_dim %c_292, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %708 = stablehlo.compare  LT, %704, %707,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %709 = stablehlo.and %706, %708 : tensor<24xi1>
    %710 = stablehlo.broadcast_in_dim %492, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %711 = stablehlo.compare  GE, %704, %710,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_293 = stablehlo.constant dense<257> : tensor<i32>
    %712 = stablehlo.add %492, %c_293 : tensor<i32>
    %713 = stablehlo.broadcast_in_dim %712, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %714 = stablehlo.compare  LT, %704, %713,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %715 = stablehlo.and %711, %714 : tensor<24xi1>
    %716 = stablehlo.and %709, %715 : tensor<24xi1>
    %717 = stablehlo.reshape %716 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %718 = stablehlo.broadcast_in_dim %694, dims = [0, 1, 2] : (tensor<256x257x1xi1>) -> tensor<256x257x24xi1>
    %719 = stablehlo.broadcast_in_dim %717, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<256x257x24xi1>
    %720 = stablehlo.and %718, %719 : tensor<256x257x24xi1>
    %cst_294 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %721 = stablehlo.broadcast_in_dim %cst_294, dims = [] : (tensor<f32>) -> tensor<256x257x24xf32>
    %722 = call @_where_91(%720, %502, %721) : (tensor<256x257x24xi1>, tensor<256x257x24xf32>, tensor<256x257x24xf32>) -> tensor<256x257x24xf32>
    %723 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_295 = stablehlo.constant dense<0> : tensor<i32>
    %724 = stablehlo.broadcast_in_dim %c_295, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %725 = stablehlo.compare  GE, %723, %724,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_296 = stablehlo.constant dense<256> : tensor<i32>
    %726 = stablehlo.broadcast_in_dim %c_296, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %727 = stablehlo.compare  LT, %723, %726,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %728 = stablehlo.and %725, %727 : tensor<256xi1>
    %729 = stablehlo.reshape %728 : (tensor<256xi1>) -> tensor<256x1x1xi1>
    %c_297 = stablehlo.constant dense<true> : tensor<i1>
    %730 = stablehlo.broadcast_in_dim %c_297, dims = [] : (tensor<i1>) -> tensor<256x1x1xi1>
    %731 = stablehlo.and %730, %729 : tensor<256x1x1xi1>
    %732 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_298 = stablehlo.constant dense<12> : tensor<i32>
    %733 = stablehlo.broadcast_in_dim %c_298, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %734 = stablehlo.compare  LT, %732, %733,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_299 = stablehlo.constant dense<12> : tensor<i32>
    %735 = stablehlo.broadcast_in_dim %c_299, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %736 = stablehlo.subtract %732, %735 : tensor<24xi32>
    %c_300 = stablehlo.constant dense<257> : tensor<i32>
    %737 = stablehlo.broadcast_in_dim %c_300, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %738 = stablehlo.add %736, %737 : tensor<24xi32>
    %c_301 = stablehlo.constant dense<12> : tensor<i32>
    %739 = stablehlo.broadcast_in_dim %c_301, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %740 = stablehlo.subtract %738, %739 : tensor<24xi32>
    %741 = call @_where_13(%734, %732, %740) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_302 = stablehlo.constant dense<0> : tensor<i32>
    %742 = stablehlo.broadcast_in_dim %c_302, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %743 = stablehlo.compare  GE, %741, %742,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_303 = stablehlo.constant dense<257> : tensor<i32>
    %744 = stablehlo.broadcast_in_dim %c_303, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %745 = stablehlo.compare  LT, %741, %744,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %746 = stablehlo.and %743, %745 : tensor<24xi1>
    %747 = stablehlo.reshape %746 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %748 = stablehlo.broadcast_in_dim %731, dims = [0, 1, 2] : (tensor<256x1x1xi1>) -> tensor<256x24x1xi1>
    %749 = stablehlo.broadcast_in_dim %747, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<256x24x1xi1>
    %750 = stablehlo.and %748, %749 : tensor<256x24x1xi1>
    %751 = stablehlo.iota dim = 0 : tensor<257xi32>
    %752 = stablehlo.broadcast_in_dim %492, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %753 = stablehlo.add %751, %752 : tensor<257xi32>
    %c_304 = stablehlo.constant dense<0> : tensor<i32>
    %754 = stablehlo.broadcast_in_dim %c_304, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %755 = stablehlo.compare  GE, %753, %754,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_305 = stablehlo.constant dense<257> : tensor<i32>
    %756 = stablehlo.broadcast_in_dim %c_305, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %757 = stablehlo.compare  LT, %753, %756,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %758 = stablehlo.and %755, %757 : tensor<257xi1>
    %759 = stablehlo.reshape %758 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %760 = stablehlo.broadcast_in_dim %750, dims = [0, 1, 2] : (tensor<256x24x1xi1>) -> tensor<256x24x257xi1>
    %761 = stablehlo.broadcast_in_dim %759, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<256x24x257xi1>
    %762 = stablehlo.and %760, %761 : tensor<256x24x257xi1>
    %cst_306 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %763 = stablehlo.broadcast_in_dim %cst_306, dims = [] : (tensor<f32>) -> tensor<256x24x257xf32>
    %764 = call @_where_73(%762, %arg132, %763) : (tensor<256x24x257xi1>, tensor<256x24x257xf32>, tensor<256x24x257xf32>) -> tensor<256x24x257xf32>
    %cst_307 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %765 = stablehlo.broadcast_in_dim %cst_307, dims = [] : (tensor<f32>) -> tensor<256x256x1xf32>
    %cst_308 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %766 = stablehlo.broadcast_in_dim %cst_308, dims = [] : (tensor<f32>) -> tensor<256x256x1xf32>
    %767 = stablehlo.slice %389 [0:256, 0:257, 256:257] : (tensor<256x257x257xf32>) -> tensor<256x257x1xf32>
    %768 = "stablehlo.collective_permute"(%767) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<256x257x1xf32>) -> tensor<256x257x1xf32>
    %cst_309 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %769 = stablehlo.broadcast_in_dim %cst_309, dims = [] : (tensor<f32>) -> tensor<256x257x1xf32>
    %770 = stablehlo.slice %487 [0:257, 0:256, 256:257] : (tensor<257x256x257xf32>) -> tensor<257x256x1xf32>
    %771 = "stablehlo.collective_permute"(%770) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<257x256x1xf32>) -> tensor<257x256x1xf32>
    %cst_310 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %772 = stablehlo.broadcast_in_dim %cst_310, dims = [] : (tensor<f32>) -> tensor<257x256x1xf32>
    %773:9 = stablehlo.custom_call @beamz_cuda_sharded(%arg115, %arg116, %arg117, %289#0, %389, %487, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg58, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %544, %586, %628, %675, %722, %764, %arg9, %arg9, %arg9, %498, %765, %766, %768, %769, %771, %772) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 1 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<256x257x24xf32>, tensor<256x24x257xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<256x256x1xf32>, tensor<256x256x1xf32>, tensor<256x257x1xf32>, tensor<256x257x1xf32>, tensor<257x256x1xf32>, tensor<257x256x1xf32>) -> (tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<256x257x24xf32>, tensor<256x24x257xf32>)
    %774 = stablehlo.broadcast_in_dim %773#6, dims = [1, 2, 3] : (tensor<257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %775 = stablehlo.broadcast_in_dim %773#7, dims = [1, 2, 3] : (tensor<256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %c_311 = stablehlo.constant dense<0> : tensor<i32>
    %c_312 = stablehlo.constant dense<31> : tensor<i32>
    %776 = call @clip(%arg140, %c_311, %c_312) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_313 = stablehlo.constant dense<1> : tensor<ui32>
    %c_314 = stablehlo.constant dense<1> : tensor<ui32>
    %777 = stablehlo.partition_id : tensor<ui32>
    %778 = stablehlo.divide %777, %c_313 : tensor<ui32>
    %779 = stablehlo.remainder %778, %c_314 : tensor<ui32>
    %780 = stablehlo.convert %779 : (tensor<ui32>) -> tensor<i32>
    %c_315 = stablehlo.constant dense<257> : tensor<i32>
    %781 = stablehlo.multiply %780, %c_315 : tensor<i32>
    %c_316 = stablehlo.constant dense<75> : tensor<i32>
    %782 = stablehlo.subtract %c_316, %781 : tensor<i32>
    %c_317 = stablehlo.constant dense<0> : tensor<i32>
    %c_318 = stablehlo.constant dense<256> : tensor<i32>
    %783 = call @clip_125(%782, %c_317, %c_318) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %784 = stablehlo.add %781, %783 : tensor<i32>
    %785 = stablehlo.iota dim = 0 : tensor<1xi32>
    %786 = stablehlo.broadcast_in_dim %784, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %787 = stablehlo.add %786, %785 : tensor<1xi32>
    %c_319 = stablehlo.constant dense<75> : tensor<i32>
    %788 = stablehlo.broadcast_in_dim %c_319, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %789 = stablehlo.subtract %787, %788 : tensor<1xi32>
    %c_320 = stablehlo.constant dense<0> : tensor<i32>
    %790 = stablehlo.broadcast_in_dim %c_320, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %791 = stablehlo.compare  GE, %789, %790,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_321 = stablehlo.constant dense<1> : tensor<i32>
    %792 = stablehlo.broadcast_in_dim %c_321, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %793 = stablehlo.compare  LT, %789, %792,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %794 = stablehlo.and %791, %793 : tensor<1xi1>
    %795 = stablehlo.slice %arg59 [0:1, 0:15, 0:26, 0:1] : (tensor<2x15x26x1xf32>) -> tensor<1x15x26x1xf32>
    %796 = stablehlo.reshape %795 : (tensor<1x15x26x1xf32>) -> tensor<15x26x1xf32>
    %797 = call @_take_179(%796, %789) : (tensor<15x26x1xf32>, tensor<1xi32>) -> tensor<15x26x1xf32>
    %798 = stablehlo.reshape %794 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_322 = stablehlo.constant dense<0> : tensor<i32>
    %799 = stablehlo.compare  LT, %776, %c_322,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_323 = stablehlo.constant dense<32> : tensor<i32>
    %800 = stablehlo.add %776, %c_323 : tensor<i32>
    %801 = stablehlo.select %799, %800, %776 : tensor<i1>, tensor<i32>
    %c_324 = stablehlo.constant dense<0> : tensor<i32>
    %802 = stablehlo.dynamic_slice %arg60, %c_324, %801, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %803 = stablehlo.reshape %802 : (tensor<1x1xf32>) -> tensor<f32>
    %804 = stablehlo.broadcast_in_dim %803, dims = [] : (tensor<f32>) -> tensor<15x26x1xf32>
    %805 = stablehlo.multiply %797, %804 : tensor<15x26x1xf32>
    %c_325 = stablehlo.constant dense<0> : tensor<i32>
    %806 = call @_where_183(%798, %805, %c_325) : (tensor<1x1x1xi1>, tensor<15x26x1xf32>, tensor<i32>) -> tensor<15x26x1xf32>
    %c_326 = stablehlo.constant dense<121> : tensor<i32>
    %c_327 = stablehlo.constant dense<0> : tensor<i32>
    %807 = stablehlo.compare  LT, %c_326, %c_327,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_328 = stablehlo.constant dense<121> : tensor<i32>
    %c_329 = stablehlo.constant dense<257> : tensor<i32>
    %808 = stablehlo.add %c_328, %c_329 : tensor<i32>
    %c_330 = stablehlo.constant dense<121> : tensor<i32>
    %809 = stablehlo.select %807, %808, %c_330 : tensor<i1>, tensor<i32>
    %c_331 = stablehlo.constant dense<115> : tensor<i32>
    %c_332 = stablehlo.constant dense<0> : tensor<i32>
    %810 = stablehlo.compare  LT, %c_331, %c_332,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_333 = stablehlo.constant dense<115> : tensor<i32>
    %c_334 = stablehlo.constant dense<256> : tensor<i32>
    %811 = stablehlo.add %c_333, %c_334 : tensor<i32>
    %c_335 = stablehlo.constant dense<115> : tensor<i32>
    %812 = stablehlo.select %810, %811, %c_335 : tensor<i1>, tensor<i32>
    %c_336 = stablehlo.constant dense<0> : tensor<i32>
    %813 = stablehlo.compare  LT, %783, %c_336,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_337 = stablehlo.constant dense<257> : tensor<i32>
    %814 = stablehlo.add %783, %c_337 : tensor<i32>
    %815 = stablehlo.select %813, %814, %783 : tensor<i1>, tensor<i32>
    %816 = stablehlo.dynamic_slice %773#1, %809, %812, %815, sizes = [15, 26, 1] : (tensor<257x256x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<15x26x1xf32>
    %817 = stablehlo.add %816, %806 : tensor<15x26x1xf32>
    %c_338 = stablehlo.constant dense<121> : tensor<i32>
    %c_339 = stablehlo.constant dense<0> : tensor<i32>
    %818 = stablehlo.compare  LT, %c_338, %c_339,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_340 = stablehlo.constant dense<121> : tensor<i32>
    %c_341 = stablehlo.constant dense<257> : tensor<i32>
    %819 = stablehlo.add %c_340, %c_341 : tensor<i32>
    %c_342 = stablehlo.constant dense<121> : tensor<i32>
    %820 = stablehlo.select %818, %819, %c_342 : tensor<i1>, tensor<i32>
    %c_343 = stablehlo.constant dense<115> : tensor<i32>
    %c_344 = stablehlo.constant dense<0> : tensor<i32>
    %821 = stablehlo.compare  LT, %c_343, %c_344,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_345 = stablehlo.constant dense<115> : tensor<i32>
    %c_346 = stablehlo.constant dense<256> : tensor<i32>
    %822 = stablehlo.add %c_345, %c_346 : tensor<i32>
    %c_347 = stablehlo.constant dense<115> : tensor<i32>
    %823 = stablehlo.select %821, %822, %c_347 : tensor<i1>, tensor<i32>
    %c_348 = stablehlo.constant dense<0> : tensor<i32>
    %824 = stablehlo.compare  LT, %783, %c_348,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_349 = stablehlo.constant dense<257> : tensor<i32>
    %825 = stablehlo.add %783, %c_349 : tensor<i32>
    %826 = stablehlo.select %824, %825, %783 : tensor<i1>, tensor<i32>
    %827 = stablehlo.dynamic_update_slice %773#1, %817, %820, %823, %826 : (tensor<257x256x257xf32>, tensor<15x26x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<257x256x257xf32>
    %c_350 = stablehlo.constant dense<75> : tensor<i32>
    %828 = stablehlo.subtract %c_350, %781 : tensor<i32>
    %c_351 = stablehlo.constant dense<0> : tensor<i32>
    %c_352 = stablehlo.constant dense<256> : tensor<i32>
    %829 = call @clip_125(%828, %c_351, %c_352) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %830 = stablehlo.add %781, %829 : tensor<i32>
    %831 = stablehlo.iota dim = 0 : tensor<1xi32>
    %832 = stablehlo.broadcast_in_dim %830, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %833 = stablehlo.add %832, %831 : tensor<1xi32>
    %c_353 = stablehlo.constant dense<75> : tensor<i32>
    %834 = stablehlo.broadcast_in_dim %c_353, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %835 = stablehlo.subtract %833, %834 : tensor<1xi32>
    %c_354 = stablehlo.constant dense<0> : tensor<i32>
    %836 = stablehlo.broadcast_in_dim %c_354, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %837 = stablehlo.compare  GE, %835, %836,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_355 = stablehlo.constant dense<1> : tensor<i32>
    %838 = stablehlo.broadcast_in_dim %c_355, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %839 = stablehlo.compare  LT, %835, %838,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %840 = stablehlo.and %837, %839 : tensor<1xi1>
    %841 = stablehlo.slice %arg59 [1:2, 0:15, 0:26, 0:1] : (tensor<2x15x26x1xf32>) -> tensor<1x15x26x1xf32>
    %842 = stablehlo.reshape %841 : (tensor<1x15x26x1xf32>) -> tensor<15x26x1xf32>
    %843 = call @_take_179(%842, %835) : (tensor<15x26x1xf32>, tensor<1xi32>) -> tensor<15x26x1xf32>
    %844 = stablehlo.reshape %840 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_356 = stablehlo.constant dense<0> : tensor<i32>
    %845 = stablehlo.compare  LT, %776, %c_356,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_357 = stablehlo.constant dense<32> : tensor<i32>
    %846 = stablehlo.add %776, %c_357 : tensor<i32>
    %847 = stablehlo.select %845, %846, %776 : tensor<i1>, tensor<i32>
    %c_358 = stablehlo.constant dense<1> : tensor<i32>
    %848 = stablehlo.dynamic_slice %arg60, %c_358, %847, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %849 = stablehlo.reshape %848 : (tensor<1x1xf32>) -> tensor<f32>
    %850 = stablehlo.broadcast_in_dim %849, dims = [] : (tensor<f32>) -> tensor<15x26x1xf32>
    %851 = stablehlo.multiply %843, %850 : tensor<15x26x1xf32>
    %c_359 = stablehlo.constant dense<0> : tensor<i32>
    %852 = call @_where_183(%844, %851, %c_359) : (tensor<1x1x1xi1>, tensor<15x26x1xf32>, tensor<i32>) -> tensor<15x26x1xf32>
    %c_360 = stablehlo.constant dense<121> : tensor<i32>
    %c_361 = stablehlo.constant dense<0> : tensor<i32>
    %853 = stablehlo.compare  LT, %c_360, %c_361,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_362 = stablehlo.constant dense<121> : tensor<i32>
    %c_363 = stablehlo.constant dense<257> : tensor<i32>
    %854 = stablehlo.add %c_362, %c_363 : tensor<i32>
    %c_364 = stablehlo.constant dense<121> : tensor<i32>
    %855 = stablehlo.select %853, %854, %c_364 : tensor<i1>, tensor<i32>
    %c_365 = stablehlo.constant dense<115> : tensor<i32>
    %c_366 = stablehlo.constant dense<0> : tensor<i32>
    %856 = stablehlo.compare  LT, %c_365, %c_366,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_367 = stablehlo.constant dense<115> : tensor<i32>
    %c_368 = stablehlo.constant dense<256> : tensor<i32>
    %857 = stablehlo.add %c_367, %c_368 : tensor<i32>
    %c_369 = stablehlo.constant dense<115> : tensor<i32>
    %858 = stablehlo.select %856, %857, %c_369 : tensor<i1>, tensor<i32>
    %c_370 = stablehlo.constant dense<0> : tensor<i32>
    %859 = stablehlo.compare  LT, %829, %c_370,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_371 = stablehlo.constant dense<257> : tensor<i32>
    %860 = stablehlo.add %829, %c_371 : tensor<i32>
    %861 = stablehlo.select %859, %860, %829 : tensor<i1>, tensor<i32>
    %862 = stablehlo.dynamic_slice %827, %855, %858, %861, sizes = [15, 26, 1] : (tensor<257x256x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<15x26x1xf32>
    %863 = stablehlo.add %862, %852 : tensor<15x26x1xf32>
    %c_372 = stablehlo.constant dense<121> : tensor<i32>
    %c_373 = stablehlo.constant dense<0> : tensor<i32>
    %864 = stablehlo.compare  LT, %c_372, %c_373,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_374 = stablehlo.constant dense<121> : tensor<i32>
    %c_375 = stablehlo.constant dense<257> : tensor<i32>
    %865 = stablehlo.add %c_374, %c_375 : tensor<i32>
    %c_376 = stablehlo.constant dense<121> : tensor<i32>
    %866 = stablehlo.select %864, %865, %c_376 : tensor<i1>, tensor<i32>
    %c_377 = stablehlo.constant dense<115> : tensor<i32>
    %c_378 = stablehlo.constant dense<0> : tensor<i32>
    %867 = stablehlo.compare  LT, %c_377, %c_378,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_379 = stablehlo.constant dense<115> : tensor<i32>
    %c_380 = stablehlo.constant dense<256> : tensor<i32>
    %868 = stablehlo.add %c_379, %c_380 : tensor<i32>
    %c_381 = stablehlo.constant dense<115> : tensor<i32>
    %869 = stablehlo.select %867, %868, %c_381 : tensor<i1>, tensor<i32>
    %c_382 = stablehlo.constant dense<0> : tensor<i32>
    %870 = stablehlo.compare  LT, %829, %c_382,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_383 = stablehlo.constant dense<257> : tensor<i32>
    %871 = stablehlo.add %829, %c_383 : tensor<i32>
    %872 = stablehlo.select %870, %871, %829 : tensor<i1>, tensor<i32>
    %873 = stablehlo.dynamic_update_slice %827, %863, %866, %869, %872 : (tensor<257x256x257xf32>, tensor<15x26x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<257x256x257xf32>
    %c_384 = stablehlo.constant dense<0> : tensor<i32>
    %c_385 = stablehlo.constant dense<31> : tensor<i32>
    %874 = call @clip(%arg140, %c_384, %c_385) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_386 = stablehlo.constant dense<1> : tensor<ui32>
    %c_387 = stablehlo.constant dense<1> : tensor<ui32>
    %875 = stablehlo.partition_id : tensor<ui32>
    %876 = stablehlo.divide %875, %c_386 : tensor<ui32>
    %877 = stablehlo.remainder %876, %c_387 : tensor<ui32>
    %878 = stablehlo.convert %877 : (tensor<ui32>) -> tensor<i32>
    %c_388 = stablehlo.constant dense<257> : tensor<i32>
    %879 = stablehlo.multiply %878, %c_388 : tensor<i32>
    %c_389 = stablehlo.constant dense<75> : tensor<i32>
    %880 = stablehlo.subtract %c_389, %879 : tensor<i32>
    %c_390 = stablehlo.constant dense<0> : tensor<i32>
    %c_391 = stablehlo.constant dense<256> : tensor<i32>
    %881 = call @clip_125(%880, %c_390, %c_391) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %882 = stablehlo.add %879, %881 : tensor<i32>
    %883 = stablehlo.iota dim = 0 : tensor<1xi32>
    %884 = stablehlo.broadcast_in_dim %882, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %885 = stablehlo.add %884, %883 : tensor<1xi32>
    %c_392 = stablehlo.constant dense<75> : tensor<i32>
    %886 = stablehlo.broadcast_in_dim %c_392, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %887 = stablehlo.subtract %885, %886 : tensor<1xi32>
    %c_393 = stablehlo.constant dense<0> : tensor<i32>
    %888 = stablehlo.broadcast_in_dim %c_393, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %889 = stablehlo.compare  GE, %887, %888,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_394 = stablehlo.constant dense<1> : tensor<i32>
    %890 = stablehlo.broadcast_in_dim %c_394, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %891 = stablehlo.compare  LT, %887, %890,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %892 = stablehlo.and %889, %891 : tensor<1xi1>
    %893 = stablehlo.slice %arg61 [0:1, 0:16, 0:25, 0:1] : (tensor<2x16x25x1xf32>) -> tensor<1x16x25x1xf32>
    %894 = stablehlo.reshape %893 : (tensor<1x16x25x1xf32>) -> tensor<16x25x1xf32>
    %895 = call @_take_194(%894, %887) : (tensor<16x25x1xf32>, tensor<1xi32>) -> tensor<16x25x1xf32>
    %896 = stablehlo.reshape %892 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_395 = stablehlo.constant dense<0> : tensor<i32>
    %897 = stablehlo.compare  LT, %874, %c_395,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_396 = stablehlo.constant dense<32> : tensor<i32>
    %898 = stablehlo.add %874, %c_396 : tensor<i32>
    %899 = stablehlo.select %897, %898, %874 : tensor<i1>, tensor<i32>
    %c_397 = stablehlo.constant dense<0> : tensor<i32>
    %900 = stablehlo.dynamic_slice %arg62, %c_397, %899, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %901 = stablehlo.reshape %900 : (tensor<1x1xf32>) -> tensor<f32>
    %902 = stablehlo.broadcast_in_dim %901, dims = [] : (tensor<f32>) -> tensor<16x25x1xf32>
    %903 = stablehlo.multiply %895, %902 : tensor<16x25x1xf32>
    %c_398 = stablehlo.constant dense<0> : tensor<i32>
    %904 = call @_where_198(%896, %903, %c_398) : (tensor<1x1x1xi1>, tensor<16x25x1xf32>, tensor<i32>) -> tensor<16x25x1xf32>
    %c_399 = stablehlo.constant dense<120> : tensor<i32>
    %c_400 = stablehlo.constant dense<0> : tensor<i32>
    %905 = stablehlo.compare  LT, %c_399, %c_400,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_401 = stablehlo.constant dense<120> : tensor<i32>
    %c_402 = stablehlo.constant dense<256> : tensor<i32>
    %906 = stablehlo.add %c_401, %c_402 : tensor<i32>
    %c_403 = stablehlo.constant dense<120> : tensor<i32>
    %907 = stablehlo.select %905, %906, %c_403 : tensor<i1>, tensor<i32>
    %c_404 = stablehlo.constant dense<116> : tensor<i32>
    %c_405 = stablehlo.constant dense<0> : tensor<i32>
    %908 = stablehlo.compare  LT, %c_404, %c_405,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_406 = stablehlo.constant dense<116> : tensor<i32>
    %c_407 = stablehlo.constant dense<257> : tensor<i32>
    %909 = stablehlo.add %c_406, %c_407 : tensor<i32>
    %c_408 = stablehlo.constant dense<116> : tensor<i32>
    %910 = stablehlo.select %908, %909, %c_408 : tensor<i1>, tensor<i32>
    %c_409 = stablehlo.constant dense<0> : tensor<i32>
    %911 = stablehlo.compare  LT, %881, %c_409,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_410 = stablehlo.constant dense<257> : tensor<i32>
    %912 = stablehlo.add %881, %c_410 : tensor<i32>
    %913 = stablehlo.select %911, %912, %881 : tensor<i1>, tensor<i32>
    %914 = stablehlo.dynamic_slice %773#2, %907, %910, %913, sizes = [16, 25, 1] : (tensor<256x257x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x25x1xf32>
    %915 = stablehlo.add %914, %904 : tensor<16x25x1xf32>
    %c_411 = stablehlo.constant dense<120> : tensor<i32>
    %c_412 = stablehlo.constant dense<0> : tensor<i32>
    %916 = stablehlo.compare  LT, %c_411, %c_412,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_413 = stablehlo.constant dense<120> : tensor<i32>
    %c_414 = stablehlo.constant dense<256> : tensor<i32>
    %917 = stablehlo.add %c_413, %c_414 : tensor<i32>
    %c_415 = stablehlo.constant dense<120> : tensor<i32>
    %918 = stablehlo.select %916, %917, %c_415 : tensor<i1>, tensor<i32>
    %c_416 = stablehlo.constant dense<116> : tensor<i32>
    %c_417 = stablehlo.constant dense<0> : tensor<i32>
    %919 = stablehlo.compare  LT, %c_416, %c_417,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_418 = stablehlo.constant dense<116> : tensor<i32>
    %c_419 = stablehlo.constant dense<257> : tensor<i32>
    %920 = stablehlo.add %c_418, %c_419 : tensor<i32>
    %c_420 = stablehlo.constant dense<116> : tensor<i32>
    %921 = stablehlo.select %919, %920, %c_420 : tensor<i1>, tensor<i32>
    %c_421 = stablehlo.constant dense<0> : tensor<i32>
    %922 = stablehlo.compare  LT, %881, %c_421,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_422 = stablehlo.constant dense<257> : tensor<i32>
    %923 = stablehlo.add %881, %c_422 : tensor<i32>
    %924 = stablehlo.select %922, %923, %881 : tensor<i1>, tensor<i32>
    %925 = stablehlo.dynamic_update_slice %773#2, %915, %918, %921, %924 : (tensor<256x257x257xf32>, tensor<16x25x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<256x257x257xf32>
    %c_423 = stablehlo.constant dense<75> : tensor<i32>
    %926 = stablehlo.subtract %c_423, %879 : tensor<i32>
    %c_424 = stablehlo.constant dense<0> : tensor<i32>
    %c_425 = stablehlo.constant dense<256> : tensor<i32>
    %927 = call @clip_125(%926, %c_424, %c_425) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %928 = stablehlo.add %879, %927 : tensor<i32>
    %929 = stablehlo.iota dim = 0 : tensor<1xi32>
    %930 = stablehlo.broadcast_in_dim %928, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %931 = stablehlo.add %930, %929 : tensor<1xi32>
    %c_426 = stablehlo.constant dense<75> : tensor<i32>
    %932 = stablehlo.broadcast_in_dim %c_426, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %933 = stablehlo.subtract %931, %932 : tensor<1xi32>
    %c_427 = stablehlo.constant dense<0> : tensor<i32>
    %934 = stablehlo.broadcast_in_dim %c_427, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %935 = stablehlo.compare  GE, %933, %934,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_428 = stablehlo.constant dense<1> : tensor<i32>
    %936 = stablehlo.broadcast_in_dim %c_428, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %937 = stablehlo.compare  LT, %933, %936,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %938 = stablehlo.and %935, %937 : tensor<1xi1>
    %939 = stablehlo.slice %arg61 [1:2, 0:16, 0:25, 0:1] : (tensor<2x16x25x1xf32>) -> tensor<1x16x25x1xf32>
    %940 = stablehlo.reshape %939 : (tensor<1x16x25x1xf32>) -> tensor<16x25x1xf32>
    %941 = call @_take_194(%940, %933) : (tensor<16x25x1xf32>, tensor<1xi32>) -> tensor<16x25x1xf32>
    %942 = stablehlo.reshape %938 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_429 = stablehlo.constant dense<0> : tensor<i32>
    %943 = stablehlo.compare  LT, %874, %c_429,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_430 = stablehlo.constant dense<32> : tensor<i32>
    %944 = stablehlo.add %874, %c_430 : tensor<i32>
    %945 = stablehlo.select %943, %944, %874 : tensor<i1>, tensor<i32>
    %c_431 = stablehlo.constant dense<1> : tensor<i32>
    %946 = stablehlo.dynamic_slice %arg62, %c_431, %945, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %947 = stablehlo.reshape %946 : (tensor<1x1xf32>) -> tensor<f32>
    %948 = stablehlo.broadcast_in_dim %947, dims = [] : (tensor<f32>) -> tensor<16x25x1xf32>
    %949 = stablehlo.multiply %941, %948 : tensor<16x25x1xf32>
    %c_432 = stablehlo.constant dense<0> : tensor<i32>
    %950 = call @_where_198(%942, %949, %c_432) : (tensor<1x1x1xi1>, tensor<16x25x1xf32>, tensor<i32>) -> tensor<16x25x1xf32>
    %c_433 = stablehlo.constant dense<120> : tensor<i32>
    %c_434 = stablehlo.constant dense<0> : tensor<i32>
    %951 = stablehlo.compare  LT, %c_433, %c_434,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_435 = stablehlo.constant dense<120> : tensor<i32>
    %c_436 = stablehlo.constant dense<256> : tensor<i32>
    %952 = stablehlo.add %c_435, %c_436 : tensor<i32>
    %c_437 = stablehlo.constant dense<120> : tensor<i32>
    %953 = stablehlo.select %951, %952, %c_437 : tensor<i1>, tensor<i32>
    %c_438 = stablehlo.constant dense<116> : tensor<i32>
    %c_439 = stablehlo.constant dense<0> : tensor<i32>
    %954 = stablehlo.compare  LT, %c_438, %c_439,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_440 = stablehlo.constant dense<116> : tensor<i32>
    %c_441 = stablehlo.constant dense<257> : tensor<i32>
    %955 = stablehlo.add %c_440, %c_441 : tensor<i32>
    %c_442 = stablehlo.constant dense<116> : tensor<i32>
    %956 = stablehlo.select %954, %955, %c_442 : tensor<i1>, tensor<i32>
    %c_443 = stablehlo.constant dense<0> : tensor<i32>
    %957 = stablehlo.compare  LT, %927, %c_443,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_444 = stablehlo.constant dense<257> : tensor<i32>
    %958 = stablehlo.add %927, %c_444 : tensor<i32>
    %959 = stablehlo.select %957, %958, %927 : tensor<i1>, tensor<i32>
    %960 = stablehlo.dynamic_slice %925, %953, %956, %959, sizes = [16, 25, 1] : (tensor<256x257x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x25x1xf32>
    %961 = stablehlo.add %960, %950 : tensor<16x25x1xf32>
    %c_445 = stablehlo.constant dense<120> : tensor<i32>
    %c_446 = stablehlo.constant dense<0> : tensor<i32>
    %962 = stablehlo.compare  LT, %c_445, %c_446,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_447 = stablehlo.constant dense<120> : tensor<i32>
    %c_448 = stablehlo.constant dense<256> : tensor<i32>
    %963 = stablehlo.add %c_447, %c_448 : tensor<i32>
    %c_449 = stablehlo.constant dense<120> : tensor<i32>
    %964 = stablehlo.select %962, %963, %c_449 : tensor<i1>, tensor<i32>
    %c_450 = stablehlo.constant dense<116> : tensor<i32>
    %c_451 = stablehlo.constant dense<0> : tensor<i32>
    %965 = stablehlo.compare  LT, %c_450, %c_451,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_452 = stablehlo.constant dense<116> : tensor<i32>
    %c_453 = stablehlo.constant dense<257> : tensor<i32>
    %966 = stablehlo.add %c_452, %c_453 : tensor<i32>
    %c_454 = stablehlo.constant dense<116> : tensor<i32>
    %967 = stablehlo.select %965, %966, %c_454 : tensor<i1>, tensor<i32>
    %c_455 = stablehlo.constant dense<0> : tensor<i32>
    %968 = stablehlo.compare  LT, %927, %c_455,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_456 = stablehlo.constant dense<257> : tensor<i32>
    %969 = stablehlo.add %927, %c_456 : tensor<i32>
    %970 = stablehlo.select %968, %969, %927 : tensor<i1>, tensor<i32>
    %971 = stablehlo.dynamic_update_slice %925, %961, %964, %967, %970 : (tensor<256x257x257xf32>, tensor<16x25x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<256x257x257xf32>
    %c_457 = stablehlo.constant dense<1> : tensor<i32>
    %972 = stablehlo.add %arg140, %c_457 : tensor<i32>
    %c_458 = stablehlo.constant dense<1> : tensor<i32>
    %c_459 = stablehlo.constant dense<1> : tensor<i32>
    %973 = stablehlo.maximum %c_458, %c_459 : tensor<i32>
    %974 = call @remainder(%972, %973) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_460 = stablehlo.constant dense<0> : tensor<i32>
    %975 = stablehlo.compare  EQ, %974, %c_460,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %976 = stablehlo.slice %arg135 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %977 = stablehlo.reshape %976 : (tensor<1xi32>) -> tensor<i32>
    %c_461 = stablehlo.constant dense<1> : tensor<i32>
    %978 = stablehlo.compare  LT, %977, %c_461,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %979 = stablehlo.and %975, %978 : tensor<i1>
    %c_462 = stablehlo.constant dense<false> : tensor<i1>
    %980 = stablehlo.and %979, %c_462 : tensor<i1>
    %c_463 = stablehlo.constant dense<false> : tensor<i1>
    %981 = stablehlo.or %980, %c_463 : tensor<i1>
    %982 = stablehlo.convert %981 : (tensor<i1>) -> tensor<i32>
    %983 = "stablehlo.case"(%982) ({
      %cst_518 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      stablehlo.return %cst_518 : tensor<f32>
    }, {
      %c_518 = stablehlo.constant dense<257> : tensor<i32>
      %1091:2 = func.call @divmod(%arg63, %c_518) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_519 = stablehlo.constant dense<257> : tensor<i32>
      %1092:2 = func.call @divmod(%1091#0, %c_519) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_520 = stablehlo.constant dense<257> : tensor<i32>
      %1093:2 = func.call @divmod(%1092#0, %c_520) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_521 = stablehlo.constant dense<0> : tensor<i32>
      %1094 = stablehlo.broadcast_in_dim %c_521, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1095 = stablehlo.compare  GT, %1093#0, %1094,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_522 = stablehlo.constant dense<-1> : tensor<i32>
      %1096 = stablehlo.broadcast_in_dim %c_522, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1097 = stablehlo.compare  LT, %1093#0, %1096,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_523 = stablehlo.constant dense<0> : tensor<i32>
      %1098 = func.call @_where_233(%1097, %c_523, %1093#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_524 = stablehlo.constant dense<256> : tensor<i32>
      %1099 = func.call @_where_233(%1095, %c_524, %1098) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_525 = stablehlo.constant dense<0> : tensor<i32>
      %1100 = func.call @_where_233(%1097, %c_525, %1092#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_526 = stablehlo.constant dense<256> : tensor<i32>
      %1101 = func.call @_where_233(%1095, %c_526, %1100) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_527 = stablehlo.constant dense<0> : tensor<i32>
      %1102 = func.call @_where_233(%1097, %c_527, %1091#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_528 = stablehlo.constant dense<256> : tensor<i32>
      %1103 = func.call @_where_233(%1095, %c_528, %1102) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_529 = stablehlo.constant dense<0> : tensor<i32>
      %1104 = stablehlo.broadcast_in_dim %c_529, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1105 = stablehlo.compare  LT, %1099, %1104,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_530 = stablehlo.constant dense<257> : tensor<i32>
      %1106 = stablehlo.broadcast_in_dim %c_530, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1107 = stablehlo.add %1099, %1106 : tensor<200x1xi32>
      %1108 = stablehlo.select %1105, %1107, %1099 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_531 = stablehlo.constant dense<0> : tensor<i32>
      %1109 = stablehlo.broadcast_in_dim %c_531, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1110 = stablehlo.compare  LT, %1101, %1109,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_532 = stablehlo.constant dense<257> : tensor<i32>
      %1111 = stablehlo.broadcast_in_dim %c_532, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1112 = stablehlo.add %1101, %1111 : tensor<200x1xi32>
      %1113 = stablehlo.select %1110, %1112, %1101 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_533 = stablehlo.constant dense<0> : tensor<i32>
      %1114 = stablehlo.broadcast_in_dim %c_533, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1115 = stablehlo.compare  LT, %1103, %1114,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_534 = stablehlo.constant dense<257> : tensor<i32>
      %1116 = stablehlo.broadcast_in_dim %c_534, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1117 = stablehlo.add %1103, %1116 : tensor<200x1xi32>
      %1118 = stablehlo.select %1115, %1117, %1103 : tensor<200x1xi1>, tensor<200x1xi32>
      %1119 = stablehlo.broadcast_in_dim %1108, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1120 = stablehlo.broadcast_in_dim %1113, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1121 = stablehlo.broadcast_in_dim %1118, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1122 = stablehlo.concatenate %1119, %1120, %1121, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1123 = "stablehlo.gather"(%773#0, %1122) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1124 = stablehlo.multiply %1123, %arg64 : tensor<200x1xf32>
      %cst_535 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1125 = stablehlo.reduce(%1124 init: %cst_535) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_536 = stablehlo.constant dense<257> : tensor<i32>
      %1126:2 = func.call @divmod(%arg65, %c_536) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_537 = stablehlo.constant dense<256> : tensor<i32>
      %1127:2 = func.call @divmod(%1126#0, %c_537) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_538 = stablehlo.constant dense<257> : tensor<i32>
      %1128:2 = func.call @divmod(%1127#0, %c_538) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_539 = stablehlo.constant dense<0> : tensor<i32>
      %1129 = stablehlo.broadcast_in_dim %c_539, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1130 = stablehlo.compare  GT, %1128#0, %1129,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_540 = stablehlo.constant dense<-1> : tensor<i32>
      %1131 = stablehlo.broadcast_in_dim %c_540, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1132 = stablehlo.compare  LT, %1128#0, %1131,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_541 = stablehlo.constant dense<0> : tensor<i32>
      %1133 = func.call @_where_233(%1132, %c_541, %1128#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_542 = stablehlo.constant dense<256> : tensor<i32>
      %1134 = func.call @_where_233(%1130, %c_542, %1133) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_543 = stablehlo.constant dense<0> : tensor<i32>
      %1135 = func.call @_where_233(%1132, %c_543, %1127#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_544 = stablehlo.constant dense<255> : tensor<i32>
      %1136 = func.call @_where_233(%1130, %c_544, %1135) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_545 = stablehlo.constant dense<0> : tensor<i32>
      %1137 = func.call @_where_233(%1132, %c_545, %1126#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_546 = stablehlo.constant dense<256> : tensor<i32>
      %1138 = func.call @_where_233(%1130, %c_546, %1137) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_547 = stablehlo.constant dense<0> : tensor<i32>
      %1139 = stablehlo.broadcast_in_dim %c_547, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1140 = stablehlo.compare  LT, %1134, %1139,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_548 = stablehlo.constant dense<257> : tensor<i32>
      %1141 = stablehlo.broadcast_in_dim %c_548, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1142 = stablehlo.add %1134, %1141 : tensor<200x1xi32>
      %1143 = stablehlo.select %1140, %1142, %1134 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_549 = stablehlo.constant dense<0> : tensor<i32>
      %1144 = stablehlo.broadcast_in_dim %c_549, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1145 = stablehlo.compare  LT, %1136, %1144,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_550 = stablehlo.constant dense<256> : tensor<i32>
      %1146 = stablehlo.broadcast_in_dim %c_550, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1147 = stablehlo.add %1136, %1146 : tensor<200x1xi32>
      %1148 = stablehlo.select %1145, %1147, %1136 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_551 = stablehlo.constant dense<0> : tensor<i32>
      %1149 = stablehlo.broadcast_in_dim %c_551, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1150 = stablehlo.compare  LT, %1138, %1149,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_552 = stablehlo.constant dense<257> : tensor<i32>
      %1151 = stablehlo.broadcast_in_dim %c_552, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1152 = stablehlo.add %1138, %1151 : tensor<200x1xi32>
      %1153 = stablehlo.select %1150, %1152, %1138 : tensor<200x1xi1>, tensor<200x1xi32>
      %1154 = stablehlo.broadcast_in_dim %1143, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1155 = stablehlo.broadcast_in_dim %1148, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1156 = stablehlo.broadcast_in_dim %1153, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1157 = stablehlo.concatenate %1154, %1155, %1156, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1158 = "stablehlo.gather"(%873, %1157) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1159 = stablehlo.multiply %1158, %arg66 : tensor<200x1xf32>
      %cst_553 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1160 = stablehlo.reduce(%1159 init: %cst_553) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_554 = stablehlo.constant dense<257> : tensor<i32>
      %1161:2 = func.call @divmod(%arg67, %c_554) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_555 = stablehlo.constant dense<257> : tensor<i32>
      %1162:2 = func.call @divmod(%1161#0, %c_555) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_556 = stablehlo.constant dense<256> : tensor<i32>
      %1163:2 = func.call @divmod(%1162#0, %c_556) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_557 = stablehlo.constant dense<0> : tensor<i32>
      %1164 = stablehlo.broadcast_in_dim %c_557, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1165 = stablehlo.compare  GT, %1163#0, %1164,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_558 = stablehlo.constant dense<-1> : tensor<i32>
      %1166 = stablehlo.broadcast_in_dim %c_558, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1167 = stablehlo.compare  LT, %1163#0, %1166,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_559 = stablehlo.constant dense<0> : tensor<i32>
      %1168 = func.call @_where_233(%1167, %c_559, %1163#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_560 = stablehlo.constant dense<255> : tensor<i32>
      %1169 = func.call @_where_233(%1165, %c_560, %1168) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_561 = stablehlo.constant dense<0> : tensor<i32>
      %1170 = func.call @_where_233(%1167, %c_561, %1162#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_562 = stablehlo.constant dense<256> : tensor<i32>
      %1171 = func.call @_where_233(%1165, %c_562, %1170) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_563 = stablehlo.constant dense<0> : tensor<i32>
      %1172 = func.call @_where_233(%1167, %c_563, %1161#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_564 = stablehlo.constant dense<256> : tensor<i32>
      %1173 = func.call @_where_233(%1165, %c_564, %1172) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_565 = stablehlo.constant dense<0> : tensor<i32>
      %1174 = stablehlo.broadcast_in_dim %c_565, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1175 = stablehlo.compare  LT, %1169, %1174,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_566 = stablehlo.constant dense<256> : tensor<i32>
      %1176 = stablehlo.broadcast_in_dim %c_566, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1177 = stablehlo.add %1169, %1176 : tensor<200x1xi32>
      %1178 = stablehlo.select %1175, %1177, %1169 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_567 = stablehlo.constant dense<0> : tensor<i32>
      %1179 = stablehlo.broadcast_in_dim %c_567, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1180 = stablehlo.compare  LT, %1171, %1179,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_568 = stablehlo.constant dense<257> : tensor<i32>
      %1181 = stablehlo.broadcast_in_dim %c_568, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1182 = stablehlo.add %1171, %1181 : tensor<200x1xi32>
      %1183 = stablehlo.select %1180, %1182, %1171 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_569 = stablehlo.constant dense<0> : tensor<i32>
      %1184 = stablehlo.broadcast_in_dim %c_569, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1185 = stablehlo.compare  LT, %1173, %1184,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_570 = stablehlo.constant dense<257> : tensor<i32>
      %1186 = stablehlo.broadcast_in_dim %c_570, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1187 = stablehlo.add %1173, %1186 : tensor<200x1xi32>
      %1188 = stablehlo.select %1185, %1187, %1173 : tensor<200x1xi1>, tensor<200x1xi32>
      %1189 = stablehlo.broadcast_in_dim %1178, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1190 = stablehlo.broadcast_in_dim %1183, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1191 = stablehlo.broadcast_in_dim %1188, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1192 = stablehlo.concatenate %1189, %1190, %1191, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1193 = "stablehlo.gather"(%971, %1192) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1194 = stablehlo.multiply %1193, %arg68 : tensor<200x1xf32>
      %cst_571 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1195 = stablehlo.reduce(%1194 init: %cst_571) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_572 = stablehlo.constant dense<257> : tensor<i32>
      %1196:2 = func.call @divmod(%arg69, %c_572) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_573 = stablehlo.constant dense<256> : tensor<i32>
      %1197:2 = func.call @divmod(%1196#0, %c_573) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_574 = stablehlo.constant dense<256> : tensor<i32>
      %1198:2 = func.call @divmod(%1197#0, %c_574) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_575 = stablehlo.constant dense<0> : tensor<i32>
      %1199 = stablehlo.broadcast_in_dim %c_575, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1200 = stablehlo.compare  GT, %1198#0, %1199,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_576 = stablehlo.constant dense<-1> : tensor<i32>
      %1201 = stablehlo.broadcast_in_dim %c_576, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1202 = stablehlo.compare  LT, %1198#0, %1201,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_577 = stablehlo.constant dense<0> : tensor<i32>
      %1203 = func.call @_where_233(%1202, %c_577, %1198#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_578 = stablehlo.constant dense<255> : tensor<i32>
      %1204 = func.call @_where_233(%1200, %c_578, %1203) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_579 = stablehlo.constant dense<0> : tensor<i32>
      %1205 = func.call @_where_233(%1202, %c_579, %1197#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_580 = stablehlo.constant dense<255> : tensor<i32>
      %1206 = func.call @_where_233(%1200, %c_580, %1205) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_581 = stablehlo.constant dense<0> : tensor<i32>
      %1207 = func.call @_where_233(%1202, %c_581, %1196#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_582 = stablehlo.constant dense<256> : tensor<i32>
      %1208 = func.call @_where_233(%1200, %c_582, %1207) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_583 = stablehlo.constant dense<0> : tensor<i32>
      %1209 = stablehlo.broadcast_in_dim %c_583, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1210 = stablehlo.compare  LT, %1204, %1209,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_584 = stablehlo.constant dense<256> : tensor<i32>
      %1211 = stablehlo.broadcast_in_dim %c_584, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1212 = stablehlo.add %1204, %1211 : tensor<200x1xi32>
      %1213 = stablehlo.select %1210, %1212, %1204 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_585 = stablehlo.constant dense<0> : tensor<i32>
      %1214 = stablehlo.broadcast_in_dim %c_585, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1215 = stablehlo.compare  LT, %1206, %1214,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_586 = stablehlo.constant dense<256> : tensor<i32>
      %1216 = stablehlo.broadcast_in_dim %c_586, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1217 = stablehlo.add %1206, %1216 : tensor<200x1xi32>
      %1218 = stablehlo.select %1215, %1217, %1206 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_587 = stablehlo.constant dense<0> : tensor<i32>
      %1219 = stablehlo.broadcast_in_dim %c_587, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1220 = stablehlo.compare  LT, %1208, %1219,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_588 = stablehlo.constant dense<257> : tensor<i32>
      %1221 = stablehlo.broadcast_in_dim %c_588, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1222 = stablehlo.add %1208, %1221 : tensor<200x1xi32>
      %1223 = stablehlo.select %1220, %1222, %1208 : tensor<200x1xi1>, tensor<200x1xi32>
      %1224 = stablehlo.broadcast_in_dim %1213, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1225 = stablehlo.broadcast_in_dim %1218, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1226 = stablehlo.broadcast_in_dim %1223, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1227 = stablehlo.concatenate %1224, %1225, %1226, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1228 = "stablehlo.gather"(%289#0, %1227) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1229 = stablehlo.multiply %1228, %arg70 : tensor<200x1xf32>
      %cst_589 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1230 = stablehlo.reduce(%1229 init: %cst_589) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_590 = stablehlo.constant dense<257> : tensor<i32>
      %1231:2 = func.call @divmod(%arg71, %c_590) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_591 = stablehlo.constant dense<257> : tensor<i32>
      %1232:2 = func.call @divmod(%1231#0, %c_591) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_592 = stablehlo.constant dense<256> : tensor<i32>
      %1233:2 = func.call @divmod(%1232#0, %c_592) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_593 = stablehlo.constant dense<0> : tensor<i32>
      %1234 = stablehlo.broadcast_in_dim %c_593, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1235 = stablehlo.compare  GT, %1233#0, %1234,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_594 = stablehlo.constant dense<-1> : tensor<i32>
      %1236 = stablehlo.broadcast_in_dim %c_594, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1237 = stablehlo.compare  LT, %1233#0, %1236,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_595 = stablehlo.constant dense<0> : tensor<i32>
      %1238 = func.call @_where_233(%1237, %c_595, %1233#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_596 = stablehlo.constant dense<255> : tensor<i32>
      %1239 = func.call @_where_233(%1235, %c_596, %1238) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_597 = stablehlo.constant dense<0> : tensor<i32>
      %1240 = func.call @_where_233(%1237, %c_597, %1232#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_598 = stablehlo.constant dense<256> : tensor<i32>
      %1241 = func.call @_where_233(%1235, %c_598, %1240) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_599 = stablehlo.constant dense<0> : tensor<i32>
      %1242 = func.call @_where_233(%1237, %c_599, %1231#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_600 = stablehlo.constant dense<256> : tensor<i32>
      %1243 = func.call @_where_233(%1235, %c_600, %1242) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_601 = stablehlo.constant dense<0> : tensor<i32>
      %1244 = stablehlo.broadcast_in_dim %c_601, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1245 = stablehlo.compare  LT, %1239, %1244,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_602 = stablehlo.constant dense<256> : tensor<i32>
      %1246 = stablehlo.broadcast_in_dim %c_602, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1247 = stablehlo.add %1239, %1246 : tensor<200x1xi32>
      %1248 = stablehlo.select %1245, %1247, %1239 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_603 = stablehlo.constant dense<0> : tensor<i32>
      %1249 = stablehlo.broadcast_in_dim %c_603, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1250 = stablehlo.compare  LT, %1241, %1249,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_604 = stablehlo.constant dense<257> : tensor<i32>
      %1251 = stablehlo.broadcast_in_dim %c_604, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1252 = stablehlo.add %1241, %1251 : tensor<200x1xi32>
      %1253 = stablehlo.select %1250, %1252, %1241 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_605 = stablehlo.constant dense<0> : tensor<i32>
      %1254 = stablehlo.broadcast_in_dim %c_605, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1255 = stablehlo.compare  LT, %1243, %1254,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_606 = stablehlo.constant dense<257> : tensor<i32>
      %1256 = stablehlo.broadcast_in_dim %c_606, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1257 = stablehlo.add %1243, %1256 : tensor<200x1xi32>
      %1258 = stablehlo.select %1255, %1257, %1243 : tensor<200x1xi1>, tensor<200x1xi32>
      %1259 = stablehlo.broadcast_in_dim %1248, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1260 = stablehlo.broadcast_in_dim %1253, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1261 = stablehlo.broadcast_in_dim %1258, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1262 = stablehlo.concatenate %1259, %1260, %1261, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1263 = "stablehlo.gather"(%389, %1262) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1264 = stablehlo.multiply %1263, %arg72 : tensor<200x1xf32>
      %cst_607 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1265 = stablehlo.reduce(%1264 init: %cst_607) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_608 = stablehlo.constant dense<257> : tensor<i32>
      %1266:2 = func.call @divmod(%arg73, %c_608) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_609 = stablehlo.constant dense<256> : tensor<i32>
      %1267:2 = func.call @divmod(%1266#0, %c_609) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_610 = stablehlo.constant dense<257> : tensor<i32>
      %1268:2 = func.call @divmod(%1267#0, %c_610) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_611 = stablehlo.constant dense<0> : tensor<i32>
      %1269 = stablehlo.broadcast_in_dim %c_611, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1270 = stablehlo.compare  GT, %1268#0, %1269,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_612 = stablehlo.constant dense<-1> : tensor<i32>
      %1271 = stablehlo.broadcast_in_dim %c_612, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1272 = stablehlo.compare  LT, %1268#0, %1271,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_613 = stablehlo.constant dense<0> : tensor<i32>
      %1273 = func.call @_where_233(%1272, %c_613, %1268#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_614 = stablehlo.constant dense<256> : tensor<i32>
      %1274 = func.call @_where_233(%1270, %c_614, %1273) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_615 = stablehlo.constant dense<0> : tensor<i32>
      %1275 = func.call @_where_233(%1272, %c_615, %1267#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_616 = stablehlo.constant dense<255> : tensor<i32>
      %1276 = func.call @_where_233(%1270, %c_616, %1275) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_617 = stablehlo.constant dense<0> : tensor<i32>
      %1277 = func.call @_where_233(%1272, %c_617, %1266#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_618 = stablehlo.constant dense<256> : tensor<i32>
      %1278 = func.call @_where_233(%1270, %c_618, %1277) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_619 = stablehlo.constant dense<0> : tensor<i32>
      %1279 = stablehlo.broadcast_in_dim %c_619, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1280 = stablehlo.compare  LT, %1274, %1279,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_620 = stablehlo.constant dense<257> : tensor<i32>
      %1281 = stablehlo.broadcast_in_dim %c_620, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1282 = stablehlo.add %1274, %1281 : tensor<200x1xi32>
      %1283 = stablehlo.select %1280, %1282, %1274 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_621 = stablehlo.constant dense<0> : tensor<i32>
      %1284 = stablehlo.broadcast_in_dim %c_621, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1285 = stablehlo.compare  LT, %1276, %1284,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_622 = stablehlo.constant dense<256> : tensor<i32>
      %1286 = stablehlo.broadcast_in_dim %c_622, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1287 = stablehlo.add %1276, %1286 : tensor<200x1xi32>
      %1288 = stablehlo.select %1285, %1287, %1276 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_623 = stablehlo.constant dense<0> : tensor<i32>
      %1289 = stablehlo.broadcast_in_dim %c_623, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1290 = stablehlo.compare  LT, %1278, %1289,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_624 = stablehlo.constant dense<257> : tensor<i32>
      %1291 = stablehlo.broadcast_in_dim %c_624, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1292 = stablehlo.add %1278, %1291 : tensor<200x1xi32>
      %1293 = stablehlo.select %1290, %1292, %1278 : tensor<200x1xi1>, tensor<200x1xi32>
      %1294 = stablehlo.broadcast_in_dim %1283, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1295 = stablehlo.broadcast_in_dim %1288, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1296 = stablehlo.broadcast_in_dim %1293, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1297 = stablehlo.concatenate %1294, %1295, %1296, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1298 = "stablehlo.gather"(%487, %1297) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1299 = stablehlo.multiply %1298, %arg74 : tensor<200x1xf32>
      %cst_625 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1300 = stablehlo.reduce(%1299 init: %cst_625) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %1301 = stablehlo.broadcast_in_dim %1125, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1302 = stablehlo.broadcast_in_dim %1160, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1303 = stablehlo.broadcast_in_dim %1195, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1304 = stablehlo.broadcast_in_dim %1230, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1305 = stablehlo.broadcast_in_dim %1265, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1306 = stablehlo.broadcast_in_dim %1300, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1307 = stablehlo.concatenate %1301, %1302, %1303, %1304, %1305, %1306, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %1308 = stablehlo.slice %1307 [1:2, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1309 = stablehlo.reshape %1308 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1310 = stablehlo.slice %1307 [2:3, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1311 = stablehlo.reshape %1310 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1312 = stablehlo.slice %1307 [4:5, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1313 = stablehlo.reshape %1312 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1314 = stablehlo.slice %1307 [5:6, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1315 = stablehlo.reshape %1314 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1316 = stablehlo.multiply %1309, %1315 : tensor<200xf32>
      %1317 = stablehlo.multiply %1311, %1313 : tensor<200xf32>
      %1318 = stablehlo.subtract %1316, %1317 : tensor<200xf32>
      %cst_626 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1319 = stablehlo.broadcast_in_dim %cst_626, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %1320 = stablehlo.multiply %1318, %1319 : tensor<200xf32>
      %cst_627 = stablehlo.constant dense<7.485380e-15> : tensor<f32>
      %1321 = stablehlo.broadcast_in_dim %cst_627, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %1322 = stablehlo.multiply %1320, %1321 : tensor<200xf32>
      %cst_628 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1323 = stablehlo.reduce(%1322 init: %cst_628) applies stablehlo.add across dimensions = [0] : (tensor<200xf32>, tensor<f32>) -> tensor<f32>
      stablehlo.return %1323 : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %cst_464 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %984 = call @_where_253(%980, %983, %cst_464) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %985 = stablehlo.slice %arg135 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %986 = stablehlo.reshape %985 : (tensor<1xi32>) -> tensor<i32>
    %c_465 = stablehlo.constant dense<0> : tensor<i32>
    %987 = stablehlo.minimum %986, %c_465 : tensor<i32>
    %c_466 = stablehlo.constant dense<0> : tensor<i32>
    %988 = stablehlo.compare  LT, %987, %c_466,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_467 = stablehlo.constant dense<1> : tensor<i32>
    %989 = stablehlo.add %987, %c_467 : tensor<i32>
    %990 = stablehlo.select %988, %989, %987 : tensor<i1>, tensor<i32>
    %c_468 = stablehlo.constant dense<0> : tensor<i32>
    %991 = stablehlo.dynamic_slice %arg133, %c_468, %990, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %992 = stablehlo.reshape %991 : (tensor<1x1xf32>) -> tensor<f32>
    %c_469 = stablehlo.constant dense<0> : tensor<i32>
    %993 = stablehlo.compare  LT, %987, %c_469,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_470 = stablehlo.constant dense<1> : tensor<i32>
    %994 = stablehlo.add %987, %c_470 : tensor<i32>
    %995 = stablehlo.select %993, %994, %987 : tensor<i1>, tensor<i32>
    %c_471 = stablehlo.constant dense<0> : tensor<i32>
    %996 = stablehlo.dynamic_slice %arg134, %c_471, %995, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %997 = stablehlo.reshape %996 : (tensor<1x1xf32>) -> tensor<f32>
    %998 = call @_where_258(%980, %984, %992) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_472 = stablehlo.constant dense<0> : tensor<i32>
    %999 = stablehlo.compare  LT, %987, %c_472,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_473 = stablehlo.constant dense<1> : tensor<i32>
    %1000 = stablehlo.add %987, %c_473 : tensor<i32>
    %1001 = stablehlo.select %999, %1000, %987 : tensor<i1>, tensor<i32>
    %c_474 = stablehlo.constant dense<0> : tensor<i32>
    %1002 = stablehlo.broadcast_in_dim %c_474, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1003 = stablehlo.broadcast_in_dim %1001, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1004 = stablehlo.concatenate %1002, %1003, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %1005 = "stablehlo.scatter"(%arg133, %1004, %998) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg142: tensor<f32>, %arg143: tensor<f32>):
      stablehlo.return %arg143 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %1006 = call @_where_258(%980, %3, %997) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_475 = stablehlo.constant dense<0> : tensor<i32>
    %1007 = stablehlo.compare  LT, %987, %c_475,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_476 = stablehlo.constant dense<1> : tensor<i32>
    %1008 = stablehlo.add %987, %c_476 : tensor<i32>
    %1009 = stablehlo.select %1007, %1008, %987 : tensor<i1>, tensor<i32>
    %c_477 = stablehlo.constant dense<0> : tensor<i32>
    %1010 = stablehlo.broadcast_in_dim %c_477, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1011 = stablehlo.broadcast_in_dim %1009, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1012 = stablehlo.concatenate %1010, %1011, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %1013 = "stablehlo.scatter"(%arg134, %1012, %1006) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg142: tensor<f32>, %arg143: tensor<f32>):
      stablehlo.return %arg143 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %1014 = stablehlo.slice %arg135 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %1015 = stablehlo.reshape %1014 : (tensor<1xi32>) -> tensor<i32>
    %c_478 = stablehlo.constant dense<1> : tensor<i32>
    %c_479 = stablehlo.constant dense<0> : tensor<i32>
    %1016 = call @_where_263(%980, %c_478, %c_479) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %1017 = stablehlo.convert %1016 : tensor<i32>
    %1018 = stablehlo.add %1015, %1017 : tensor<i32>
    %c_480 = stablehlo.constant dense<0> : tensor<i32>
    %1019 = stablehlo.broadcast_in_dim %c_480, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1020 = "stablehlo.scatter"(%arg135, %1019, %1018) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg142: tensor<i32>, %arg143: tensor<i32>):
      stablehlo.return %arg143 : tensor<i32>
    }) : (tensor<2xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
    %cst_481 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1021 = stablehlo.compare  GE, %3, %cst_481,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %c_482 = stablehlo.constant dense<true> : tensor<i1>
    %1022 = stablehlo.and %c_482, %1021 : tensor<i1>
    %cst_483 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %1023 = stablehlo.compare  LE, %3, %cst_483,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %1024 = stablehlo.and %1022, %1023 : tensor<i1>
    %c_484 = stablehlo.constant dense<1> : tensor<i32>
    %c_485 = stablehlo.constant dense<1> : tensor<i32>
    %1025 = stablehlo.maximum %c_484, %c_485 : tensor<i32>
    %1026 = call @remainder(%arg140, %1025) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_486 = stablehlo.constant dense<0> : tensor<i32>
    %1027 = stablehlo.compare  EQ, %1026, %c_486,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1028 = stablehlo.and %1024, %1027 : tensor<i1>
    %1029 = stablehlo.convert %1028 : (tensor<i1>) -> tensor<i32>
    %1030:3 = "stablehlo.case"(%1029) ({
      stablehlo.return %arg136, %arg137, %arg138 : tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>
    }, {
      %c_518 = stablehlo.constant dense<257> : tensor<i32>
      %1091:2 = func.call @divmod_272(%arg75, %c_518) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_519 = stablehlo.constant dense<257> : tensor<i32>
      %1092:2 = func.call @divmod_289(%1091#0, %c_519) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_520 = stablehlo.constant dense<257> : tensor<i32>
      %1093:2 = func.call @divmod_289(%1092#0, %c_520) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_521 = stablehlo.constant dense<0> : tensor<i32>
      %1094 = stablehlo.broadcast_in_dim %c_521, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1095 = stablehlo.compare  GT, %1093#0, %1094,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_522 = stablehlo.constant dense<-1> : tensor<i32>
      %1096 = stablehlo.broadcast_in_dim %c_522, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1097 = stablehlo.compare  LT, %1093#0, %1096,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_523 = stablehlo.constant dense<0> : tensor<i32>
      %1098 = func.call @_where_296(%1097, %c_523, %1093#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_524 = stablehlo.constant dense<256> : tensor<i32>
      %1099 = func.call @_where_296(%1095, %c_524, %1098) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_525 = stablehlo.constant dense<0> : tensor<i32>
      %1100 = func.call @_where_296(%1097, %c_525, %1092#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_526 = stablehlo.constant dense<256> : tensor<i32>
      %1101 = func.call @_where_296(%1095, %c_526, %1100) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_527 = stablehlo.constant dense<0> : tensor<i32>
      %1102 = func.call @_where_296(%1097, %c_527, %1091#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_528 = stablehlo.constant dense<256> : tensor<i32>
      %1103 = func.call @_where_296(%1095, %c_528, %1102) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_529 = stablehlo.constant dense<1> : tensor<ui32>
      %c_530 = stablehlo.constant dense<1> : tensor<ui32>
      %1104 = stablehlo.partition_id : tensor<ui32>
      %1105 = stablehlo.divide %1104, %c_529 : tensor<ui32>
      %1106 = stablehlo.remainder %1105, %c_530 : tensor<ui32>
      %1107 = stablehlo.convert %1106 : (tensor<ui32>) -> tensor<i32>
      %c_531 = stablehlo.constant dense<257> : tensor<i32>
      %1108 = stablehlo.multiply %1107, %c_531 : tensor<i32>
      %1109 = stablehlo.broadcast_in_dim %1108, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1110 = stablehlo.subtract %1103, %1109 : tensor<200x8xi32>
      %c_532 = stablehlo.constant dense<0> : tensor<i32>
      %1111 = stablehlo.broadcast_in_dim %c_532, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1112 = stablehlo.compare  GE, %1110, %1111,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_533 = stablehlo.constant dense<257> : tensor<i32>
      %1113 = stablehlo.broadcast_in_dim %c_533, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1114 = stablehlo.compare  LT, %1110, %1113,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %1115 = stablehlo.and %1112, %1114 : tensor<200x8xi1>
      %c_534 = stablehlo.constant dense<0> : tensor<i32>
      %c_535 = stablehlo.constant dense<256> : tensor<i32>
      %1116 = func.call @clip_303(%1110, %c_534, %c_535) : (tensor<200x8xi32>, tensor<i32>, tensor<i32>) -> tensor<200x8xi32>
      %c_536 = stablehlo.constant dense<0> : tensor<i32>
      %1117 = stablehlo.broadcast_in_dim %c_536, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1118 = stablehlo.compare  LT, %1099, %1117,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_537 = stablehlo.constant dense<257> : tensor<i32>
      %1119 = stablehlo.broadcast_in_dim %c_537, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1120 = stablehlo.add %1099, %1119 : tensor<200x8xi32>
      %1121 = stablehlo.select %1118, %1120, %1099 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_538 = stablehlo.constant dense<0> : tensor<i32>
      %1122 = stablehlo.broadcast_in_dim %c_538, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1123 = stablehlo.compare  LT, %1101, %1122,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_539 = stablehlo.constant dense<257> : tensor<i32>
      %1124 = stablehlo.broadcast_in_dim %c_539, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1125 = stablehlo.add %1101, %1124 : tensor<200x8xi32>
      %1126 = stablehlo.select %1123, %1125, %1101 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_540 = stablehlo.constant dense<0> : tensor<i32>
      %1127 = stablehlo.broadcast_in_dim %c_540, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1128 = stablehlo.compare  LT, %1116, %1127,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_541 = stablehlo.constant dense<257> : tensor<i32>
      %1129 = stablehlo.broadcast_in_dim %c_541, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1130 = stablehlo.add %1116, %1129 : tensor<200x8xi32>
      %1131 = stablehlo.select %1128, %1130, %1116 : tensor<200x8xi1>, tensor<200x8xi32>
      %1132 = stablehlo.broadcast_in_dim %1121, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1133 = stablehlo.broadcast_in_dim %1126, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1134 = stablehlo.broadcast_in_dim %1131, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1135 = stablehlo.concatenate %1132, %1133, %1134, dim = 2 : (tensor<200x8x1xi32>, tensor<200x8x1xi32>, tensor<200x8x1xi32>) -> tensor<200x8x3xi32>
      %1136 = "stablehlo.gather"(%773#0, %1135) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x257xf32>, tensor<200x8x3xi32>) -> tensor<200x8xf32>
      %c_542 = stablehlo.constant dense<0> : tensor<i32>
      %1137 = func.call @_where_315(%1115, %1136, %c_542) : (tensor<200x8xi1>, tensor<200x8xf32>, tensor<i32>) -> tensor<200x8xf32>
      %1138 = stablehlo.multiply %1137, %arg76 : tensor<200x8xf32>
      %cst_543 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1139 = stablehlo.reduce(%1138 init: %cst_543) applies stablehlo.add across dimensions = [1] : (tensor<200x8xf32>, tensor<f32>) -> tensor<200xf32>
      %c_544 = stablehlo.constant dense<257> : tensor<i32>
      %1140:2 = func.call @divmod_322(%arg77, %c_544) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_545 = stablehlo.constant dense<256> : tensor<i32>
      %1141:2 = func.call @divmod_336(%1140#0, %c_545) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_546 = stablehlo.constant dense<257> : tensor<i32>
      %1142:2 = func.call @divmod_336(%1141#0, %c_546) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_547 = stablehlo.constant dense<0> : tensor<i32>
      %1143 = stablehlo.broadcast_in_dim %c_547, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1144 = stablehlo.compare  GT, %1142#0, %1143,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_548 = stablehlo.constant dense<-1> : tensor<i32>
      %1145 = stablehlo.broadcast_in_dim %c_548, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1146 = stablehlo.compare  LT, %1142#0, %1145,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_549 = stablehlo.constant dense<0> : tensor<i32>
      %1147 = func.call @_where_343(%1146, %c_549, %1142#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_550 = stablehlo.constant dense<256> : tensor<i32>
      %1148 = func.call @_where_343(%1144, %c_550, %1147) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_551 = stablehlo.constant dense<0> : tensor<i32>
      %1149 = func.call @_where_343(%1146, %c_551, %1141#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_552 = stablehlo.constant dense<255> : tensor<i32>
      %1150 = func.call @_where_343(%1144, %c_552, %1149) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_553 = stablehlo.constant dense<0> : tensor<i32>
      %1151 = func.call @_where_343(%1146, %c_553, %1140#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_554 = stablehlo.constant dense<256> : tensor<i32>
      %1152 = func.call @_where_343(%1144, %c_554, %1151) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_555 = stablehlo.constant dense<1> : tensor<ui32>
      %c_556 = stablehlo.constant dense<1> : tensor<ui32>
      %1153 = stablehlo.partition_id : tensor<ui32>
      %1154 = stablehlo.divide %1153, %c_555 : tensor<ui32>
      %1155 = stablehlo.remainder %1154, %c_556 : tensor<ui32>
      %1156 = stablehlo.convert %1155 : (tensor<ui32>) -> tensor<i32>
      %c_557 = stablehlo.constant dense<257> : tensor<i32>
      %1157 = stablehlo.multiply %1156, %c_557 : tensor<i32>
      %1158 = stablehlo.broadcast_in_dim %1157, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1159 = stablehlo.subtract %1152, %1158 : tensor<200x4xi32>
      %c_558 = stablehlo.constant dense<0> : tensor<i32>
      %1160 = stablehlo.broadcast_in_dim %c_558, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1161 = stablehlo.compare  GE, %1159, %1160,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_559 = stablehlo.constant dense<257> : tensor<i32>
      %1162 = stablehlo.broadcast_in_dim %c_559, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1163 = stablehlo.compare  LT, %1159, %1162,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1164 = stablehlo.and %1161, %1163 : tensor<200x4xi1>
      %c_560 = stablehlo.constant dense<0> : tensor<i32>
      %c_561 = stablehlo.constant dense<256> : tensor<i32>
      %1165 = func.call @clip_350(%1159, %c_560, %c_561) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_562 = stablehlo.constant dense<0> : tensor<i32>
      %1166 = stablehlo.broadcast_in_dim %c_562, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1167 = stablehlo.compare  LT, %1148, %1166,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_563 = stablehlo.constant dense<257> : tensor<i32>
      %1168 = stablehlo.broadcast_in_dim %c_563, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1169 = stablehlo.add %1148, %1168 : tensor<200x4xi32>
      %1170 = stablehlo.select %1167, %1169, %1148 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_564 = stablehlo.constant dense<0> : tensor<i32>
      %1171 = stablehlo.broadcast_in_dim %c_564, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1172 = stablehlo.compare  LT, %1150, %1171,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_565 = stablehlo.constant dense<256> : tensor<i32>
      %1173 = stablehlo.broadcast_in_dim %c_565, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1174 = stablehlo.add %1150, %1173 : tensor<200x4xi32>
      %1175 = stablehlo.select %1172, %1174, %1150 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_566 = stablehlo.constant dense<0> : tensor<i32>
      %1176 = stablehlo.broadcast_in_dim %c_566, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1177 = stablehlo.compare  LT, %1165, %1176,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_567 = stablehlo.constant dense<257> : tensor<i32>
      %1178 = stablehlo.broadcast_in_dim %c_567, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1179 = stablehlo.add %1165, %1178 : tensor<200x4xi32>
      %1180 = stablehlo.select %1177, %1179, %1165 : tensor<200x4xi1>, tensor<200x4xi32>
      %1181 = stablehlo.broadcast_in_dim %1170, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1182 = stablehlo.broadcast_in_dim %1175, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1183 = stablehlo.broadcast_in_dim %1180, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1184 = stablehlo.concatenate %1181, %1182, %1183, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1185 = "stablehlo.gather"(%873, %1184) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_568 = stablehlo.constant dense<0> : tensor<i32>
      %1186 = func.call @_where_361(%1164, %1185, %c_568) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1187 = stablehlo.multiply %1186, %arg78 : tensor<200x4xf32>
      %cst_569 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1188 = stablehlo.reduce(%1187 init: %cst_569) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_570 = stablehlo.constant dense<257> : tensor<i32>
      %1189:2 = func.call @divmod_322(%arg79, %c_570) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_571 = stablehlo.constant dense<257> : tensor<i32>
      %1190:2 = func.call @divmod_336(%1189#0, %c_571) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_572 = stablehlo.constant dense<256> : tensor<i32>
      %1191:2 = func.call @divmod_336(%1190#0, %c_572) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_573 = stablehlo.constant dense<0> : tensor<i32>
      %1192 = stablehlo.broadcast_in_dim %c_573, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1193 = stablehlo.compare  GT, %1191#0, %1192,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_574 = stablehlo.constant dense<-1> : tensor<i32>
      %1194 = stablehlo.broadcast_in_dim %c_574, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1195 = stablehlo.compare  LT, %1191#0, %1194,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_575 = stablehlo.constant dense<0> : tensor<i32>
      %1196 = func.call @_where_343(%1195, %c_575, %1191#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_576 = stablehlo.constant dense<255> : tensor<i32>
      %1197 = func.call @_where_343(%1193, %c_576, %1196) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_577 = stablehlo.constant dense<0> : tensor<i32>
      %1198 = func.call @_where_343(%1195, %c_577, %1190#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_578 = stablehlo.constant dense<256> : tensor<i32>
      %1199 = func.call @_where_343(%1193, %c_578, %1198) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_579 = stablehlo.constant dense<0> : tensor<i32>
      %1200 = func.call @_where_343(%1195, %c_579, %1189#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_580 = stablehlo.constant dense<256> : tensor<i32>
      %1201 = func.call @_where_343(%1193, %c_580, %1200) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_581 = stablehlo.constant dense<1> : tensor<ui32>
      %c_582 = stablehlo.constant dense<1> : tensor<ui32>
      %1202 = stablehlo.partition_id : tensor<ui32>
      %1203 = stablehlo.divide %1202, %c_581 : tensor<ui32>
      %1204 = stablehlo.remainder %1203, %c_582 : tensor<ui32>
      %1205 = stablehlo.convert %1204 : (tensor<ui32>) -> tensor<i32>
      %c_583 = stablehlo.constant dense<257> : tensor<i32>
      %1206 = stablehlo.multiply %1205, %c_583 : tensor<i32>
      %1207 = stablehlo.broadcast_in_dim %1206, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1208 = stablehlo.subtract %1201, %1207 : tensor<200x4xi32>
      %c_584 = stablehlo.constant dense<0> : tensor<i32>
      %1209 = stablehlo.broadcast_in_dim %c_584, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1210 = stablehlo.compare  GE, %1208, %1209,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_585 = stablehlo.constant dense<257> : tensor<i32>
      %1211 = stablehlo.broadcast_in_dim %c_585, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1212 = stablehlo.compare  LT, %1208, %1211,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1213 = stablehlo.and %1210, %1212 : tensor<200x4xi1>
      %c_586 = stablehlo.constant dense<0> : tensor<i32>
      %c_587 = stablehlo.constant dense<256> : tensor<i32>
      %1214 = func.call @clip_350(%1208, %c_586, %c_587) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_588 = stablehlo.constant dense<0> : tensor<i32>
      %1215 = stablehlo.broadcast_in_dim %c_588, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1216 = stablehlo.compare  LT, %1197, %1215,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_589 = stablehlo.constant dense<256> : tensor<i32>
      %1217 = stablehlo.broadcast_in_dim %c_589, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1218 = stablehlo.add %1197, %1217 : tensor<200x4xi32>
      %1219 = stablehlo.select %1216, %1218, %1197 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_590 = stablehlo.constant dense<0> : tensor<i32>
      %1220 = stablehlo.broadcast_in_dim %c_590, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1221 = stablehlo.compare  LT, %1199, %1220,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_591 = stablehlo.constant dense<257> : tensor<i32>
      %1222 = stablehlo.broadcast_in_dim %c_591, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1223 = stablehlo.add %1199, %1222 : tensor<200x4xi32>
      %1224 = stablehlo.select %1221, %1223, %1199 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_592 = stablehlo.constant dense<0> : tensor<i32>
      %1225 = stablehlo.broadcast_in_dim %c_592, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1226 = stablehlo.compare  LT, %1214, %1225,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_593 = stablehlo.constant dense<257> : tensor<i32>
      %1227 = stablehlo.broadcast_in_dim %c_593, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1228 = stablehlo.add %1214, %1227 : tensor<200x4xi32>
      %1229 = stablehlo.select %1226, %1228, %1214 : tensor<200x4xi1>, tensor<200x4xi32>
      %1230 = stablehlo.broadcast_in_dim %1219, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1231 = stablehlo.broadcast_in_dim %1224, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1232 = stablehlo.broadcast_in_dim %1229, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1233 = stablehlo.concatenate %1230, %1231, %1232, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1234 = "stablehlo.gather"(%971, %1233) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_594 = stablehlo.constant dense<0> : tensor<i32>
      %1235 = func.call @_where_361(%1213, %1234, %c_594) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1236 = stablehlo.multiply %1235, %arg80 : tensor<200x4xf32>
      %cst_595 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1237 = stablehlo.reduce(%1236 init: %cst_595) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_596 = stablehlo.constant dense<257> : tensor<i32>
      %1238:2 = func.call @divmod_369(%arg81, %c_596) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_597 = stablehlo.constant dense<256> : tensor<i32>
      %1239:2 = func.call @divmod_383(%1238#0, %c_597) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_598 = stablehlo.constant dense<256> : tensor<i32>
      %1240:2 = func.call @divmod_383(%1239#0, %c_598) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_599 = stablehlo.constant dense<0> : tensor<i32>
      %1241 = stablehlo.broadcast_in_dim %c_599, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1242 = stablehlo.compare  GT, %1240#0, %1241,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_600 = stablehlo.constant dense<-1> : tensor<i32>
      %1243 = stablehlo.broadcast_in_dim %c_600, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1244 = stablehlo.compare  LT, %1240#0, %1243,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_601 = stablehlo.constant dense<0> : tensor<i32>
      %1245 = func.call @_where_390(%1244, %c_601, %1240#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_602 = stablehlo.constant dense<255> : tensor<i32>
      %1246 = func.call @_where_390(%1242, %c_602, %1245) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_603 = stablehlo.constant dense<0> : tensor<i32>
      %1247 = func.call @_where_390(%1244, %c_603, %1239#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_604 = stablehlo.constant dense<255> : tensor<i32>
      %1248 = func.call @_where_390(%1242, %c_604, %1247) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_605 = stablehlo.constant dense<0> : tensor<i32>
      %1249 = func.call @_where_390(%1244, %c_605, %1238#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_606 = stablehlo.constant dense<256> : tensor<i32>
      %1250 = func.call @_where_390(%1242, %c_606, %1249) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_607 = stablehlo.constant dense<1> : tensor<ui32>
      %c_608 = stablehlo.constant dense<1> : tensor<ui32>
      %1251 = stablehlo.partition_id : tensor<ui32>
      %1252 = stablehlo.divide %1251, %c_607 : tensor<ui32>
      %1253 = stablehlo.remainder %1252, %c_608 : tensor<ui32>
      %1254 = stablehlo.convert %1253 : (tensor<ui32>) -> tensor<i32>
      %c_609 = stablehlo.constant dense<257> : tensor<i32>
      %1255 = stablehlo.multiply %1254, %c_609 : tensor<i32>
      %1256 = stablehlo.broadcast_in_dim %1255, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1257 = stablehlo.subtract %1250, %1256 : tensor<200x2xi32>
      %c_610 = stablehlo.constant dense<0> : tensor<i32>
      %1258 = stablehlo.broadcast_in_dim %c_610, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1259 = stablehlo.compare  GE, %1257, %1258,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_611 = stablehlo.constant dense<257> : tensor<i32>
      %1260 = stablehlo.broadcast_in_dim %c_611, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1261 = stablehlo.compare  LT, %1257, %1260,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %1262 = stablehlo.and %1259, %1261 : tensor<200x2xi1>
      %c_612 = stablehlo.constant dense<0> : tensor<i32>
      %c_613 = stablehlo.constant dense<256> : tensor<i32>
      %1263 = func.call @clip_397(%1257, %c_612, %c_613) : (tensor<200x2xi32>, tensor<i32>, tensor<i32>) -> tensor<200x2xi32>
      %c_614 = stablehlo.constant dense<0> : tensor<i32>
      %1264 = stablehlo.broadcast_in_dim %c_614, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1265 = stablehlo.compare  LT, %1246, %1264,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_615 = stablehlo.constant dense<256> : tensor<i32>
      %1266 = stablehlo.broadcast_in_dim %c_615, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1267 = stablehlo.add %1246, %1266 : tensor<200x2xi32>
      %1268 = stablehlo.select %1265, %1267, %1246 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_616 = stablehlo.constant dense<0> : tensor<i32>
      %1269 = stablehlo.broadcast_in_dim %c_616, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1270 = stablehlo.compare  LT, %1248, %1269,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_617 = stablehlo.constant dense<256> : tensor<i32>
      %1271 = stablehlo.broadcast_in_dim %c_617, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1272 = stablehlo.add %1248, %1271 : tensor<200x2xi32>
      %1273 = stablehlo.select %1270, %1272, %1248 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_618 = stablehlo.constant dense<0> : tensor<i32>
      %1274 = stablehlo.broadcast_in_dim %c_618, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1275 = stablehlo.compare  LT, %1263, %1274,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_619 = stablehlo.constant dense<257> : tensor<i32>
      %1276 = stablehlo.broadcast_in_dim %c_619, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1277 = stablehlo.add %1263, %1276 : tensor<200x2xi32>
      %1278 = stablehlo.select %1275, %1277, %1263 : tensor<200x2xi1>, tensor<200x2xi32>
      %1279 = stablehlo.broadcast_in_dim %1268, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1280 = stablehlo.broadcast_in_dim %1273, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1281 = stablehlo.broadcast_in_dim %1278, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1282 = stablehlo.concatenate %1279, %1280, %1281, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
      %1283 = "stablehlo.gather"(%289#0, %1282) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
      %c_620 = stablehlo.constant dense<0> : tensor<i32>
      %1284 = func.call @_where_408(%1262, %1283, %c_620) : (tensor<200x2xi1>, tensor<200x2xf32>, tensor<i32>) -> tensor<200x2xf32>
      %1285 = stablehlo.multiply %1284, %arg82 : tensor<200x2xf32>
      %cst_621 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1286 = stablehlo.reduce(%1285 init: %cst_621) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
      %c_622 = stablehlo.constant dense<257> : tensor<i32>
      %1287:2 = func.call @divmod_322(%arg83, %c_622) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_623 = stablehlo.constant dense<257> : tensor<i32>
      %1288:2 = func.call @divmod_336(%1287#0, %c_623) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_624 = stablehlo.constant dense<256> : tensor<i32>
      %1289:2 = func.call @divmod_336(%1288#0, %c_624) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_625 = stablehlo.constant dense<0> : tensor<i32>
      %1290 = stablehlo.broadcast_in_dim %c_625, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1291 = stablehlo.compare  GT, %1289#0, %1290,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_626 = stablehlo.constant dense<-1> : tensor<i32>
      %1292 = stablehlo.broadcast_in_dim %c_626, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1293 = stablehlo.compare  LT, %1289#0, %1292,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_627 = stablehlo.constant dense<0> : tensor<i32>
      %1294 = func.call @_where_343(%1293, %c_627, %1289#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_628 = stablehlo.constant dense<255> : tensor<i32>
      %1295 = func.call @_where_343(%1291, %c_628, %1294) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_629 = stablehlo.constant dense<0> : tensor<i32>
      %1296 = func.call @_where_343(%1293, %c_629, %1288#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_630 = stablehlo.constant dense<256> : tensor<i32>
      %1297 = func.call @_where_343(%1291, %c_630, %1296) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_631 = stablehlo.constant dense<0> : tensor<i32>
      %1298 = func.call @_where_343(%1293, %c_631, %1287#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_632 = stablehlo.constant dense<256> : tensor<i32>
      %1299 = func.call @_where_343(%1291, %c_632, %1298) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_633 = stablehlo.constant dense<1> : tensor<ui32>
      %c_634 = stablehlo.constant dense<1> : tensor<ui32>
      %1300 = stablehlo.partition_id : tensor<ui32>
      %1301 = stablehlo.divide %1300, %c_633 : tensor<ui32>
      %1302 = stablehlo.remainder %1301, %c_634 : tensor<ui32>
      %1303 = stablehlo.convert %1302 : (tensor<ui32>) -> tensor<i32>
      %c_635 = stablehlo.constant dense<257> : tensor<i32>
      %1304 = stablehlo.multiply %1303, %c_635 : tensor<i32>
      %1305 = stablehlo.broadcast_in_dim %1304, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1306 = stablehlo.subtract %1299, %1305 : tensor<200x4xi32>
      %c_636 = stablehlo.constant dense<0> : tensor<i32>
      %1307 = stablehlo.broadcast_in_dim %c_636, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1308 = stablehlo.compare  GE, %1306, %1307,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_637 = stablehlo.constant dense<257> : tensor<i32>
      %1309 = stablehlo.broadcast_in_dim %c_637, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1310 = stablehlo.compare  LT, %1306, %1309,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1311 = stablehlo.and %1308, %1310 : tensor<200x4xi1>
      %c_638 = stablehlo.constant dense<0> : tensor<i32>
      %c_639 = stablehlo.constant dense<256> : tensor<i32>
      %1312 = func.call @clip_350(%1306, %c_638, %c_639) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_640 = stablehlo.constant dense<0> : tensor<i32>
      %1313 = stablehlo.broadcast_in_dim %c_640, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1314 = stablehlo.compare  LT, %1295, %1313,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_641 = stablehlo.constant dense<256> : tensor<i32>
      %1315 = stablehlo.broadcast_in_dim %c_641, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1316 = stablehlo.add %1295, %1315 : tensor<200x4xi32>
      %1317 = stablehlo.select %1314, %1316, %1295 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_642 = stablehlo.constant dense<0> : tensor<i32>
      %1318 = stablehlo.broadcast_in_dim %c_642, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1319 = stablehlo.compare  LT, %1297, %1318,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_643 = stablehlo.constant dense<257> : tensor<i32>
      %1320 = stablehlo.broadcast_in_dim %c_643, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1321 = stablehlo.add %1297, %1320 : tensor<200x4xi32>
      %1322 = stablehlo.select %1319, %1321, %1297 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_644 = stablehlo.constant dense<0> : tensor<i32>
      %1323 = stablehlo.broadcast_in_dim %c_644, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1324 = stablehlo.compare  LT, %1312, %1323,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_645 = stablehlo.constant dense<257> : tensor<i32>
      %1325 = stablehlo.broadcast_in_dim %c_645, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1326 = stablehlo.add %1312, %1325 : tensor<200x4xi32>
      %1327 = stablehlo.select %1324, %1326, %1312 : tensor<200x4xi1>, tensor<200x4xi32>
      %1328 = stablehlo.broadcast_in_dim %1317, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1329 = stablehlo.broadcast_in_dim %1322, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1330 = stablehlo.broadcast_in_dim %1327, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1331 = stablehlo.concatenate %1328, %1329, %1330, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1332 = "stablehlo.gather"(%389, %1331) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_646 = stablehlo.constant dense<0> : tensor<i32>
      %1333 = func.call @_where_361(%1311, %1332, %c_646) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1334 = stablehlo.multiply %1333, %arg84 : tensor<200x4xf32>
      %cst_647 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1335 = stablehlo.reduce(%1334 init: %cst_647) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_648 = stablehlo.constant dense<257> : tensor<i32>
      %1336:2 = func.call @divmod_322(%arg85, %c_648) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_649 = stablehlo.constant dense<256> : tensor<i32>
      %1337:2 = func.call @divmod_336(%1336#0, %c_649) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_650 = stablehlo.constant dense<257> : tensor<i32>
      %1338:2 = func.call @divmod_336(%1337#0, %c_650) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_651 = stablehlo.constant dense<0> : tensor<i32>
      %1339 = stablehlo.broadcast_in_dim %c_651, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1340 = stablehlo.compare  GT, %1338#0, %1339,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_652 = stablehlo.constant dense<-1> : tensor<i32>
      %1341 = stablehlo.broadcast_in_dim %c_652, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1342 = stablehlo.compare  LT, %1338#0, %1341,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_653 = stablehlo.constant dense<0> : tensor<i32>
      %1343 = func.call @_where_343(%1342, %c_653, %1338#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_654 = stablehlo.constant dense<256> : tensor<i32>
      %1344 = func.call @_where_343(%1340, %c_654, %1343) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_655 = stablehlo.constant dense<0> : tensor<i32>
      %1345 = func.call @_where_343(%1342, %c_655, %1337#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_656 = stablehlo.constant dense<255> : tensor<i32>
      %1346 = func.call @_where_343(%1340, %c_656, %1345) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_657 = stablehlo.constant dense<0> : tensor<i32>
      %1347 = func.call @_where_343(%1342, %c_657, %1336#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_658 = stablehlo.constant dense<256> : tensor<i32>
      %1348 = func.call @_where_343(%1340, %c_658, %1347) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_659 = stablehlo.constant dense<1> : tensor<ui32>
      %c_660 = stablehlo.constant dense<1> : tensor<ui32>
      %1349 = stablehlo.partition_id : tensor<ui32>
      %1350 = stablehlo.divide %1349, %c_659 : tensor<ui32>
      %1351 = stablehlo.remainder %1350, %c_660 : tensor<ui32>
      %1352 = stablehlo.convert %1351 : (tensor<ui32>) -> tensor<i32>
      %c_661 = stablehlo.constant dense<257> : tensor<i32>
      %1353 = stablehlo.multiply %1352, %c_661 : tensor<i32>
      %1354 = stablehlo.broadcast_in_dim %1353, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1355 = stablehlo.subtract %1348, %1354 : tensor<200x4xi32>
      %c_662 = stablehlo.constant dense<0> : tensor<i32>
      %1356 = stablehlo.broadcast_in_dim %c_662, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1357 = stablehlo.compare  GE, %1355, %1356,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_663 = stablehlo.constant dense<257> : tensor<i32>
      %1358 = stablehlo.broadcast_in_dim %c_663, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1359 = stablehlo.compare  LT, %1355, %1358,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1360 = stablehlo.and %1357, %1359 : tensor<200x4xi1>
      %c_664 = stablehlo.constant dense<0> : tensor<i32>
      %c_665 = stablehlo.constant dense<256> : tensor<i32>
      %1361 = func.call @clip_350(%1355, %c_664, %c_665) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_666 = stablehlo.constant dense<0> : tensor<i32>
      %1362 = stablehlo.broadcast_in_dim %c_666, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1363 = stablehlo.compare  LT, %1344, %1362,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_667 = stablehlo.constant dense<257> : tensor<i32>
      %1364 = stablehlo.broadcast_in_dim %c_667, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1365 = stablehlo.add %1344, %1364 : tensor<200x4xi32>
      %1366 = stablehlo.select %1363, %1365, %1344 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_668 = stablehlo.constant dense<0> : tensor<i32>
      %1367 = stablehlo.broadcast_in_dim %c_668, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1368 = stablehlo.compare  LT, %1346, %1367,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_669 = stablehlo.constant dense<256> : tensor<i32>
      %1369 = stablehlo.broadcast_in_dim %c_669, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1370 = stablehlo.add %1346, %1369 : tensor<200x4xi32>
      %1371 = stablehlo.select %1368, %1370, %1346 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_670 = stablehlo.constant dense<0> : tensor<i32>
      %1372 = stablehlo.broadcast_in_dim %c_670, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1373 = stablehlo.compare  LT, %1361, %1372,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_671 = stablehlo.constant dense<257> : tensor<i32>
      %1374 = stablehlo.broadcast_in_dim %c_671, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1375 = stablehlo.add %1361, %1374 : tensor<200x4xi32>
      %1376 = stablehlo.select %1373, %1375, %1361 : tensor<200x4xi1>, tensor<200x4xi32>
      %1377 = stablehlo.broadcast_in_dim %1366, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1378 = stablehlo.broadcast_in_dim %1371, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1379 = stablehlo.broadcast_in_dim %1376, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1380 = stablehlo.concatenate %1377, %1378, %1379, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1381 = "stablehlo.gather"(%487, %1380) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_672 = stablehlo.constant dense<0> : tensor<i32>
      %1382 = func.call @_where_361(%1360, %1381, %c_672) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1383 = stablehlo.multiply %1382, %arg86 : tensor<200x4xf32>
      %cst_673 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1384 = stablehlo.reduce(%1383 init: %cst_673) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %1385 = stablehlo.slice %arg136 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
      %1386 = stablehlo.reshape %1385 : (tensor<1x7200xf32>) -> tensor<7200xf32>
      %1387 = stablehlo.slice %arg137 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
      %1388 = stablehlo.reshape %1387 : (tensor<1x7200xf32>) -> tensor<7200xf32>
      %1389 = stablehlo.broadcast_in_dim %1139, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1390 = stablehlo.broadcast_in_dim %1188, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1391 = stablehlo.broadcast_in_dim %1237, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1392 = stablehlo.broadcast_in_dim %1286, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1393 = stablehlo.broadcast_in_dim %1335, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1394 = stablehlo.broadcast_in_dim %1384, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1395 = stablehlo.concatenate %1389, %1390, %1391, %1392, %1393, %1394, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %cst_674 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1396 = stablehlo.broadcast_in_dim %cst_674, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %1397 = stablehlo.multiply %1396, %arg87 : tensor<3xf32>
      %1398 = stablehlo.optimization_barrier %1397 : tensor<3xf32>
      %1399 = stablehlo.optimization_barrier %3 : tensor<f32>
      %1400 = stablehlo.broadcast_in_dim %1399, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %1401 = stablehlo.multiply %1398, %1400 : tensor<3xf32>
      %1402 = stablehlo.cosine %1401 : tensor<3xf32>
      %1403 = stablehlo.sine %1401 : tensor<3xf32>
      %cst_675 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_676 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1404 = stablehlo.subtract %cst_675, %cst_676 : tensor<f32>
      %cst_677 = stablehlo.constant dense<1.000000e-30> : tensor<f32>
      %1405 = stablehlo.maximum %1404, %cst_677 : tensor<f32>
      %cst_678 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1406 = stablehlo.subtract %3, %cst_678 : tensor<f32>
      %1407 = stablehlo.divide %1406, %1405 : tensor<f32>
      %cst_679 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %cst_680 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1408 = func.call @clip_425(%1407, %cst_679, %cst_680) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_681 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1409 = stablehlo.multiply %cst_681, %1408 : tensor<f32>
      %1410 = stablehlo.cosine %1409 : tensor<f32>
      %cst_682 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1411 = stablehlo.subtract %cst_682, %1410 : tensor<f32>
      %cst_683 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %1412 = stablehlo.multiply %cst_683, %1411 : tensor<f32>
      %cst_684 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %1413 = stablehlo.is_finite %cst_684 : (tensor<f32>) -> tensor<i1>
      %c_685 = stablehlo.constant dense<false> : tensor<i1>
      %1414 = stablehlo.and %c_685, %1413 : tensor<i1>
      %cst_686 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_687 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1415 = stablehlo.compare  GT, %cst_686, %cst_687,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %1416 = stablehlo.and %1414, %1415 : tensor<i1>
      %cst_688 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1417 = func.call @_where_432(%1416, %1412, %cst_688) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_689 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1418 = stablehlo.maximum %1417, %cst_689 : tensor<f32>
      %cst_690 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1419 = stablehlo.multiply %arg0, %cst_690 : tensor<f32>
      %cst_691 = stablehlo.constant dense<0x4D8EF3C2> : tensor<f32>
      %1420 = stablehlo.multiply %1419, %cst_691 : tensor<f32>
      %cst_692 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %1421 = stablehlo.divide %1420, %cst_692 : tensor<f32>
      %cst_693 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1422 = stablehlo.sqrt %cst_693 : tensor<f32>
      %1423 = stablehlo.divide %1421, %1422 : tensor<f32>
      %1424 = stablehlo.multiply %1418, %1423 : tensor<f32>
      %c_694 = stablehlo.constant dense<0> : tensor<i32>
      %c_695 = stablehlo.constant dense<1> : tensor<i32>
      %1425 = stablehlo.compare  EQ, %c_694, %c_695,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1426 = func.call @_where_435(%1425, %1424, %1418) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %1427 = stablehlo.broadcast_in_dim %arg88, dims = [0] : (tensor<6xf32>) -> tensor<6x1x1xf32>
      %1428 = stablehlo.broadcast_in_dim %1426, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %1429 = stablehlo.multiply %1428, %1427 : tensor<6x1x1xf32>
      %1430 = stablehlo.dot_general %1395, %1402, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %1431 = stablehlo.transpose %1430, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %1432 = stablehlo.broadcast_in_dim %1429, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %1433 = stablehlo.multiply %1432, %1431 : tensor<6x3x200xf32>
      %1434 = stablehlo.broadcast_in_dim %1426, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %1435 = stablehlo.multiply %1434, %1427 : tensor<6x1x1xf32>
      %1436 = stablehlo.dot_general %1395, %1403, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %1437 = stablehlo.transpose %1436, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %1438 = stablehlo.broadcast_in_dim %1435, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %1439 = stablehlo.multiply %1438, %1437 : tensor<6x3x200xf32>
      %1440 = stablehlo.reshape %1433 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_696 = stablehlo.constant dense<0> : tensor<i32>
      %1441 = stablehlo.broadcast_in_dim %c_696, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1442 = "stablehlo.scatter"(%1386, %1441, %1440) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg142: tensor<f32>, %arg143: tensor<f32>):
        %1451 = stablehlo.add %arg142, %arg143 : tensor<f32>
        stablehlo.return %1451 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %1443 = stablehlo.reshape %1439 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_697 = stablehlo.constant dense<0> : tensor<i32>
      %1444 = stablehlo.broadcast_in_dim %c_697, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1445 = "stablehlo.scatter"(%1388, %1444, %1443) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg142: tensor<f32>, %arg143: tensor<f32>):
        %1451 = stablehlo.add %arg142, %arg143 : tensor<f32>
        stablehlo.return %1451 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %1446 = stablehlo.broadcast_in_dim %1417, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %c_698 = stablehlo.constant dense<0> : tensor<i32>
      %1447 = stablehlo.broadcast_in_dim %c_698, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1448 = "stablehlo.scatter"(%arg138, %1447, %1446) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg142: tensor<f32>, %arg143: tensor<f32>):
        %1451 = stablehlo.add %arg142, %arg143 : tensor<f32>
        stablehlo.return %1451 : tensor<f32>
      }) : (tensor<6xf32>, tensor<1xi32>, tensor<3xf32>) -> tensor<6xf32>
      %1449 = stablehlo.broadcast_in_dim %1442, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
      %1450 = stablehlo.broadcast_in_dim %1445, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
      stablehlo.return %1449, %1450, %1448 : tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>
    }) : (tensor<i32>) -> (tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>)
    %c_487 = stablehlo.constant dense<1> : tensor<i32>
    %1031 = stablehlo.add %arg140, %c_487 : tensor<i32>
    %c_488 = stablehlo.constant dense<1> : tensor<i32>
    %c_489 = stablehlo.constant dense<1> : tensor<i32>
    %1032 = stablehlo.maximum %c_488, %c_489 : tensor<i32>
    %1033 = call @remainder(%1031, %1032) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_490 = stablehlo.constant dense<0> : tensor<i32>
    %1034 = stablehlo.compare  EQ, %1033, %c_490,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1035 = stablehlo.slice %1020 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %1036 = stablehlo.reshape %1035 : (tensor<1xi32>) -> tensor<i32>
    %c_491 = stablehlo.constant dense<1> : tensor<i32>
    %1037 = stablehlo.compare  LT, %1036, %c_491,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1038 = stablehlo.and %1034, %1037 : tensor<i1>
    %c_492 = stablehlo.constant dense<false> : tensor<i1>
    %1039 = stablehlo.and %1038, %c_492 : tensor<i1>
    %c_493 = stablehlo.constant dense<false> : tensor<i1>
    %1040 = stablehlo.or %1039, %c_493 : tensor<i1>
    %1041 = stablehlo.convert %1040 : (tensor<i1>) -> tensor<i32>
    %1042 = "stablehlo.case"(%1041) ({
      %cst_518 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      stablehlo.return %cst_518 : tensor<f32>
    }, {
      %c_518 = stablehlo.constant dense<257> : tensor<i32>
      %1091:2 = func.call @divmod(%arg89, %c_518) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_519 = stablehlo.constant dense<257> : tensor<i32>
      %1092:2 = func.call @divmod(%1091#0, %c_519) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_520 = stablehlo.constant dense<257> : tensor<i32>
      %1093:2 = func.call @divmod(%1092#0, %c_520) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_521 = stablehlo.constant dense<0> : tensor<i32>
      %1094 = stablehlo.broadcast_in_dim %c_521, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1095 = stablehlo.compare  GT, %1093#0, %1094,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_522 = stablehlo.constant dense<-1> : tensor<i32>
      %1096 = stablehlo.broadcast_in_dim %c_522, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1097 = stablehlo.compare  LT, %1093#0, %1096,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_523 = stablehlo.constant dense<0> : tensor<i32>
      %1098 = func.call @_where_233(%1097, %c_523, %1093#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_524 = stablehlo.constant dense<256> : tensor<i32>
      %1099 = func.call @_where_233(%1095, %c_524, %1098) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_525 = stablehlo.constant dense<0> : tensor<i32>
      %1100 = func.call @_where_233(%1097, %c_525, %1092#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_526 = stablehlo.constant dense<256> : tensor<i32>
      %1101 = func.call @_where_233(%1095, %c_526, %1100) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_527 = stablehlo.constant dense<0> : tensor<i32>
      %1102 = func.call @_where_233(%1097, %c_527, %1091#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_528 = stablehlo.constant dense<256> : tensor<i32>
      %1103 = func.call @_where_233(%1095, %c_528, %1102) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_529 = stablehlo.constant dense<0> : tensor<i32>
      %1104 = stablehlo.broadcast_in_dim %c_529, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1105 = stablehlo.compare  LT, %1099, %1104,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_530 = stablehlo.constant dense<257> : tensor<i32>
      %1106 = stablehlo.broadcast_in_dim %c_530, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1107 = stablehlo.add %1099, %1106 : tensor<200x1xi32>
      %1108 = stablehlo.select %1105, %1107, %1099 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_531 = stablehlo.constant dense<0> : tensor<i32>
      %1109 = stablehlo.broadcast_in_dim %c_531, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1110 = stablehlo.compare  LT, %1101, %1109,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_532 = stablehlo.constant dense<257> : tensor<i32>
      %1111 = stablehlo.broadcast_in_dim %c_532, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1112 = stablehlo.add %1101, %1111 : tensor<200x1xi32>
      %1113 = stablehlo.select %1110, %1112, %1101 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_533 = stablehlo.constant dense<0> : tensor<i32>
      %1114 = stablehlo.broadcast_in_dim %c_533, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1115 = stablehlo.compare  LT, %1103, %1114,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_534 = stablehlo.constant dense<257> : tensor<i32>
      %1116 = stablehlo.broadcast_in_dim %c_534, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1117 = stablehlo.add %1103, %1116 : tensor<200x1xi32>
      %1118 = stablehlo.select %1115, %1117, %1103 : tensor<200x1xi1>, tensor<200x1xi32>
      %1119 = stablehlo.broadcast_in_dim %1108, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1120 = stablehlo.broadcast_in_dim %1113, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1121 = stablehlo.broadcast_in_dim %1118, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1122 = stablehlo.concatenate %1119, %1120, %1121, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1123 = "stablehlo.gather"(%773#0, %1122) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1124 = stablehlo.multiply %1123, %arg90 : tensor<200x1xf32>
      %cst_535 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1125 = stablehlo.reduce(%1124 init: %cst_535) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_536 = stablehlo.constant dense<257> : tensor<i32>
      %1126:2 = func.call @divmod(%arg91, %c_536) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_537 = stablehlo.constant dense<256> : tensor<i32>
      %1127:2 = func.call @divmod(%1126#0, %c_537) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_538 = stablehlo.constant dense<257> : tensor<i32>
      %1128:2 = func.call @divmod(%1127#0, %c_538) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_539 = stablehlo.constant dense<0> : tensor<i32>
      %1129 = stablehlo.broadcast_in_dim %c_539, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1130 = stablehlo.compare  GT, %1128#0, %1129,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_540 = stablehlo.constant dense<-1> : tensor<i32>
      %1131 = stablehlo.broadcast_in_dim %c_540, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1132 = stablehlo.compare  LT, %1128#0, %1131,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_541 = stablehlo.constant dense<0> : tensor<i32>
      %1133 = func.call @_where_233(%1132, %c_541, %1128#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_542 = stablehlo.constant dense<256> : tensor<i32>
      %1134 = func.call @_where_233(%1130, %c_542, %1133) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_543 = stablehlo.constant dense<0> : tensor<i32>
      %1135 = func.call @_where_233(%1132, %c_543, %1127#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_544 = stablehlo.constant dense<255> : tensor<i32>
      %1136 = func.call @_where_233(%1130, %c_544, %1135) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_545 = stablehlo.constant dense<0> : tensor<i32>
      %1137 = func.call @_where_233(%1132, %c_545, %1126#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_546 = stablehlo.constant dense<256> : tensor<i32>
      %1138 = func.call @_where_233(%1130, %c_546, %1137) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_547 = stablehlo.constant dense<0> : tensor<i32>
      %1139 = stablehlo.broadcast_in_dim %c_547, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1140 = stablehlo.compare  LT, %1134, %1139,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_548 = stablehlo.constant dense<257> : tensor<i32>
      %1141 = stablehlo.broadcast_in_dim %c_548, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1142 = stablehlo.add %1134, %1141 : tensor<200x1xi32>
      %1143 = stablehlo.select %1140, %1142, %1134 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_549 = stablehlo.constant dense<0> : tensor<i32>
      %1144 = stablehlo.broadcast_in_dim %c_549, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1145 = stablehlo.compare  LT, %1136, %1144,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_550 = stablehlo.constant dense<256> : tensor<i32>
      %1146 = stablehlo.broadcast_in_dim %c_550, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1147 = stablehlo.add %1136, %1146 : tensor<200x1xi32>
      %1148 = stablehlo.select %1145, %1147, %1136 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_551 = stablehlo.constant dense<0> : tensor<i32>
      %1149 = stablehlo.broadcast_in_dim %c_551, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1150 = stablehlo.compare  LT, %1138, %1149,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_552 = stablehlo.constant dense<257> : tensor<i32>
      %1151 = stablehlo.broadcast_in_dim %c_552, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1152 = stablehlo.add %1138, %1151 : tensor<200x1xi32>
      %1153 = stablehlo.select %1150, %1152, %1138 : tensor<200x1xi1>, tensor<200x1xi32>
      %1154 = stablehlo.broadcast_in_dim %1143, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1155 = stablehlo.broadcast_in_dim %1148, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1156 = stablehlo.broadcast_in_dim %1153, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1157 = stablehlo.concatenate %1154, %1155, %1156, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1158 = "stablehlo.gather"(%873, %1157) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1159 = stablehlo.multiply %1158, %arg92 : tensor<200x1xf32>
      %cst_553 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1160 = stablehlo.reduce(%1159 init: %cst_553) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_554 = stablehlo.constant dense<257> : tensor<i32>
      %1161:2 = func.call @divmod(%arg93, %c_554) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_555 = stablehlo.constant dense<257> : tensor<i32>
      %1162:2 = func.call @divmod(%1161#0, %c_555) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_556 = stablehlo.constant dense<256> : tensor<i32>
      %1163:2 = func.call @divmod(%1162#0, %c_556) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_557 = stablehlo.constant dense<0> : tensor<i32>
      %1164 = stablehlo.broadcast_in_dim %c_557, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1165 = stablehlo.compare  GT, %1163#0, %1164,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_558 = stablehlo.constant dense<-1> : tensor<i32>
      %1166 = stablehlo.broadcast_in_dim %c_558, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1167 = stablehlo.compare  LT, %1163#0, %1166,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_559 = stablehlo.constant dense<0> : tensor<i32>
      %1168 = func.call @_where_233(%1167, %c_559, %1163#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_560 = stablehlo.constant dense<255> : tensor<i32>
      %1169 = func.call @_where_233(%1165, %c_560, %1168) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_561 = stablehlo.constant dense<0> : tensor<i32>
      %1170 = func.call @_where_233(%1167, %c_561, %1162#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_562 = stablehlo.constant dense<256> : tensor<i32>
      %1171 = func.call @_where_233(%1165, %c_562, %1170) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_563 = stablehlo.constant dense<0> : tensor<i32>
      %1172 = func.call @_where_233(%1167, %c_563, %1161#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_564 = stablehlo.constant dense<256> : tensor<i32>
      %1173 = func.call @_where_233(%1165, %c_564, %1172) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_565 = stablehlo.constant dense<0> : tensor<i32>
      %1174 = stablehlo.broadcast_in_dim %c_565, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1175 = stablehlo.compare  LT, %1169, %1174,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_566 = stablehlo.constant dense<256> : tensor<i32>
      %1176 = stablehlo.broadcast_in_dim %c_566, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1177 = stablehlo.add %1169, %1176 : tensor<200x1xi32>
      %1178 = stablehlo.select %1175, %1177, %1169 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_567 = stablehlo.constant dense<0> : tensor<i32>
      %1179 = stablehlo.broadcast_in_dim %c_567, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1180 = stablehlo.compare  LT, %1171, %1179,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_568 = stablehlo.constant dense<257> : tensor<i32>
      %1181 = stablehlo.broadcast_in_dim %c_568, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1182 = stablehlo.add %1171, %1181 : tensor<200x1xi32>
      %1183 = stablehlo.select %1180, %1182, %1171 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_569 = stablehlo.constant dense<0> : tensor<i32>
      %1184 = stablehlo.broadcast_in_dim %c_569, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1185 = stablehlo.compare  LT, %1173, %1184,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_570 = stablehlo.constant dense<257> : tensor<i32>
      %1186 = stablehlo.broadcast_in_dim %c_570, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1187 = stablehlo.add %1173, %1186 : tensor<200x1xi32>
      %1188 = stablehlo.select %1185, %1187, %1173 : tensor<200x1xi1>, tensor<200x1xi32>
      %1189 = stablehlo.broadcast_in_dim %1178, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1190 = stablehlo.broadcast_in_dim %1183, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1191 = stablehlo.broadcast_in_dim %1188, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1192 = stablehlo.concatenate %1189, %1190, %1191, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1193 = "stablehlo.gather"(%971, %1192) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1194 = stablehlo.multiply %1193, %arg94 : tensor<200x1xf32>
      %cst_571 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1195 = stablehlo.reduce(%1194 init: %cst_571) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_572 = stablehlo.constant dense<257> : tensor<i32>
      %1196:2 = func.call @divmod(%arg95, %c_572) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_573 = stablehlo.constant dense<256> : tensor<i32>
      %1197:2 = func.call @divmod(%1196#0, %c_573) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_574 = stablehlo.constant dense<256> : tensor<i32>
      %1198:2 = func.call @divmod(%1197#0, %c_574) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_575 = stablehlo.constant dense<0> : tensor<i32>
      %1199 = stablehlo.broadcast_in_dim %c_575, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1200 = stablehlo.compare  GT, %1198#0, %1199,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_576 = stablehlo.constant dense<-1> : tensor<i32>
      %1201 = stablehlo.broadcast_in_dim %c_576, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1202 = stablehlo.compare  LT, %1198#0, %1201,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_577 = stablehlo.constant dense<0> : tensor<i32>
      %1203 = func.call @_where_233(%1202, %c_577, %1198#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_578 = stablehlo.constant dense<255> : tensor<i32>
      %1204 = func.call @_where_233(%1200, %c_578, %1203) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_579 = stablehlo.constant dense<0> : tensor<i32>
      %1205 = func.call @_where_233(%1202, %c_579, %1197#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_580 = stablehlo.constant dense<255> : tensor<i32>
      %1206 = func.call @_where_233(%1200, %c_580, %1205) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_581 = stablehlo.constant dense<0> : tensor<i32>
      %1207 = func.call @_where_233(%1202, %c_581, %1196#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_582 = stablehlo.constant dense<256> : tensor<i32>
      %1208 = func.call @_where_233(%1200, %c_582, %1207) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_583 = stablehlo.constant dense<0> : tensor<i32>
      %1209 = stablehlo.broadcast_in_dim %c_583, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1210 = stablehlo.compare  LT, %1204, %1209,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_584 = stablehlo.constant dense<256> : tensor<i32>
      %1211 = stablehlo.broadcast_in_dim %c_584, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1212 = stablehlo.add %1204, %1211 : tensor<200x1xi32>
      %1213 = stablehlo.select %1210, %1212, %1204 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_585 = stablehlo.constant dense<0> : tensor<i32>
      %1214 = stablehlo.broadcast_in_dim %c_585, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1215 = stablehlo.compare  LT, %1206, %1214,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_586 = stablehlo.constant dense<256> : tensor<i32>
      %1216 = stablehlo.broadcast_in_dim %c_586, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1217 = stablehlo.add %1206, %1216 : tensor<200x1xi32>
      %1218 = stablehlo.select %1215, %1217, %1206 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_587 = stablehlo.constant dense<0> : tensor<i32>
      %1219 = stablehlo.broadcast_in_dim %c_587, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1220 = stablehlo.compare  LT, %1208, %1219,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_588 = stablehlo.constant dense<257> : tensor<i32>
      %1221 = stablehlo.broadcast_in_dim %c_588, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1222 = stablehlo.add %1208, %1221 : tensor<200x1xi32>
      %1223 = stablehlo.select %1220, %1222, %1208 : tensor<200x1xi1>, tensor<200x1xi32>
      %1224 = stablehlo.broadcast_in_dim %1213, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1225 = stablehlo.broadcast_in_dim %1218, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1226 = stablehlo.broadcast_in_dim %1223, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1227 = stablehlo.concatenate %1224, %1225, %1226, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1228 = "stablehlo.gather"(%289#0, %1227) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1229 = stablehlo.multiply %1228, %arg96 : tensor<200x1xf32>
      %cst_589 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1230 = stablehlo.reduce(%1229 init: %cst_589) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_590 = stablehlo.constant dense<257> : tensor<i32>
      %1231:2 = func.call @divmod(%arg97, %c_590) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_591 = stablehlo.constant dense<257> : tensor<i32>
      %1232:2 = func.call @divmod(%1231#0, %c_591) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_592 = stablehlo.constant dense<256> : tensor<i32>
      %1233:2 = func.call @divmod(%1232#0, %c_592) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_593 = stablehlo.constant dense<0> : tensor<i32>
      %1234 = stablehlo.broadcast_in_dim %c_593, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1235 = stablehlo.compare  GT, %1233#0, %1234,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_594 = stablehlo.constant dense<-1> : tensor<i32>
      %1236 = stablehlo.broadcast_in_dim %c_594, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1237 = stablehlo.compare  LT, %1233#0, %1236,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_595 = stablehlo.constant dense<0> : tensor<i32>
      %1238 = func.call @_where_233(%1237, %c_595, %1233#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_596 = stablehlo.constant dense<255> : tensor<i32>
      %1239 = func.call @_where_233(%1235, %c_596, %1238) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_597 = stablehlo.constant dense<0> : tensor<i32>
      %1240 = func.call @_where_233(%1237, %c_597, %1232#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_598 = stablehlo.constant dense<256> : tensor<i32>
      %1241 = func.call @_where_233(%1235, %c_598, %1240) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_599 = stablehlo.constant dense<0> : tensor<i32>
      %1242 = func.call @_where_233(%1237, %c_599, %1231#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_600 = stablehlo.constant dense<256> : tensor<i32>
      %1243 = func.call @_where_233(%1235, %c_600, %1242) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_601 = stablehlo.constant dense<0> : tensor<i32>
      %1244 = stablehlo.broadcast_in_dim %c_601, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1245 = stablehlo.compare  LT, %1239, %1244,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_602 = stablehlo.constant dense<256> : tensor<i32>
      %1246 = stablehlo.broadcast_in_dim %c_602, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1247 = stablehlo.add %1239, %1246 : tensor<200x1xi32>
      %1248 = stablehlo.select %1245, %1247, %1239 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_603 = stablehlo.constant dense<0> : tensor<i32>
      %1249 = stablehlo.broadcast_in_dim %c_603, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1250 = stablehlo.compare  LT, %1241, %1249,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_604 = stablehlo.constant dense<257> : tensor<i32>
      %1251 = stablehlo.broadcast_in_dim %c_604, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1252 = stablehlo.add %1241, %1251 : tensor<200x1xi32>
      %1253 = stablehlo.select %1250, %1252, %1241 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_605 = stablehlo.constant dense<0> : tensor<i32>
      %1254 = stablehlo.broadcast_in_dim %c_605, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1255 = stablehlo.compare  LT, %1243, %1254,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_606 = stablehlo.constant dense<257> : tensor<i32>
      %1256 = stablehlo.broadcast_in_dim %c_606, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1257 = stablehlo.add %1243, %1256 : tensor<200x1xi32>
      %1258 = stablehlo.select %1255, %1257, %1243 : tensor<200x1xi1>, tensor<200x1xi32>
      %1259 = stablehlo.broadcast_in_dim %1248, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1260 = stablehlo.broadcast_in_dim %1253, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1261 = stablehlo.broadcast_in_dim %1258, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1262 = stablehlo.concatenate %1259, %1260, %1261, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1263 = "stablehlo.gather"(%389, %1262) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1264 = stablehlo.multiply %1263, %arg98 : tensor<200x1xf32>
      %cst_607 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1265 = stablehlo.reduce(%1264 init: %cst_607) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_608 = stablehlo.constant dense<257> : tensor<i32>
      %1266:2 = func.call @divmod(%arg99, %c_608) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_609 = stablehlo.constant dense<256> : tensor<i32>
      %1267:2 = func.call @divmod(%1266#0, %c_609) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_610 = stablehlo.constant dense<257> : tensor<i32>
      %1268:2 = func.call @divmod(%1267#0, %c_610) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_611 = stablehlo.constant dense<0> : tensor<i32>
      %1269 = stablehlo.broadcast_in_dim %c_611, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1270 = stablehlo.compare  GT, %1268#0, %1269,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_612 = stablehlo.constant dense<-1> : tensor<i32>
      %1271 = stablehlo.broadcast_in_dim %c_612, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1272 = stablehlo.compare  LT, %1268#0, %1271,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_613 = stablehlo.constant dense<0> : tensor<i32>
      %1273 = func.call @_where_233(%1272, %c_613, %1268#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_614 = stablehlo.constant dense<256> : tensor<i32>
      %1274 = func.call @_where_233(%1270, %c_614, %1273) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_615 = stablehlo.constant dense<0> : tensor<i32>
      %1275 = func.call @_where_233(%1272, %c_615, %1267#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_616 = stablehlo.constant dense<255> : tensor<i32>
      %1276 = func.call @_where_233(%1270, %c_616, %1275) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_617 = stablehlo.constant dense<0> : tensor<i32>
      %1277 = func.call @_where_233(%1272, %c_617, %1266#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_618 = stablehlo.constant dense<256> : tensor<i32>
      %1278 = func.call @_where_233(%1270, %c_618, %1277) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_619 = stablehlo.constant dense<0> : tensor<i32>
      %1279 = stablehlo.broadcast_in_dim %c_619, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1280 = stablehlo.compare  LT, %1274, %1279,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_620 = stablehlo.constant dense<257> : tensor<i32>
      %1281 = stablehlo.broadcast_in_dim %c_620, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1282 = stablehlo.add %1274, %1281 : tensor<200x1xi32>
      %1283 = stablehlo.select %1280, %1282, %1274 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_621 = stablehlo.constant dense<0> : tensor<i32>
      %1284 = stablehlo.broadcast_in_dim %c_621, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1285 = stablehlo.compare  LT, %1276, %1284,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_622 = stablehlo.constant dense<256> : tensor<i32>
      %1286 = stablehlo.broadcast_in_dim %c_622, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1287 = stablehlo.add %1276, %1286 : tensor<200x1xi32>
      %1288 = stablehlo.select %1285, %1287, %1276 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_623 = stablehlo.constant dense<0> : tensor<i32>
      %1289 = stablehlo.broadcast_in_dim %c_623, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1290 = stablehlo.compare  LT, %1278, %1289,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_624 = stablehlo.constant dense<257> : tensor<i32>
      %1291 = stablehlo.broadcast_in_dim %c_624, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1292 = stablehlo.add %1278, %1291 : tensor<200x1xi32>
      %1293 = stablehlo.select %1290, %1292, %1278 : tensor<200x1xi1>, tensor<200x1xi32>
      %1294 = stablehlo.broadcast_in_dim %1283, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1295 = stablehlo.broadcast_in_dim %1288, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1296 = stablehlo.broadcast_in_dim %1293, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1297 = stablehlo.concatenate %1294, %1295, %1296, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1298 = "stablehlo.gather"(%487, %1297) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1299 = stablehlo.multiply %1298, %arg100 : tensor<200x1xf32>
      %cst_625 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1300 = stablehlo.reduce(%1299 init: %cst_625) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %1301 = stablehlo.broadcast_in_dim %1125, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1302 = stablehlo.broadcast_in_dim %1160, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1303 = stablehlo.broadcast_in_dim %1195, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1304 = stablehlo.broadcast_in_dim %1230, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1305 = stablehlo.broadcast_in_dim %1265, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1306 = stablehlo.broadcast_in_dim %1300, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1307 = stablehlo.concatenate %1301, %1302, %1303, %1304, %1305, %1306, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %1308 = stablehlo.slice %1307 [1:2, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1309 = stablehlo.reshape %1308 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1310 = stablehlo.slice %1307 [2:3, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1311 = stablehlo.reshape %1310 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1312 = stablehlo.slice %1307 [4:5, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1313 = stablehlo.reshape %1312 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1314 = stablehlo.slice %1307 [5:6, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1315 = stablehlo.reshape %1314 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1316 = stablehlo.multiply %1309, %1315 : tensor<200xf32>
      %1317 = stablehlo.multiply %1311, %1313 : tensor<200xf32>
      %1318 = stablehlo.subtract %1316, %1317 : tensor<200xf32>
      %cst_626 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1319 = stablehlo.broadcast_in_dim %cst_626, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %1320 = stablehlo.multiply %1318, %1319 : tensor<200xf32>
      %cst_627 = stablehlo.constant dense<7.485380e-15> : tensor<f32>
      %1321 = stablehlo.broadcast_in_dim %cst_627, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %1322 = stablehlo.multiply %1320, %1321 : tensor<200xf32>
      %cst_628 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1323 = stablehlo.reduce(%1322 init: %cst_628) applies stablehlo.add across dimensions = [0] : (tensor<200xf32>, tensor<f32>) -> tensor<f32>
      stablehlo.return %1323 : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %cst_494 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1043 = call @_where_253(%1039, %1042, %cst_494) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %1044 = stablehlo.slice %1020 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %1045 = stablehlo.reshape %1044 : (tensor<1xi32>) -> tensor<i32>
    %c_495 = stablehlo.constant dense<0> : tensor<i32>
    %1046 = stablehlo.minimum %1045, %c_495 : tensor<i32>
    %c_496 = stablehlo.constant dense<0> : tensor<i32>
    %1047 = stablehlo.compare  LT, %1046, %c_496,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_497 = stablehlo.constant dense<1> : tensor<i32>
    %1048 = stablehlo.add %1046, %c_497 : tensor<i32>
    %1049 = stablehlo.select %1047, %1048, %1046 : tensor<i1>, tensor<i32>
    %c_498 = stablehlo.constant dense<1> : tensor<i32>
    %1050 = stablehlo.dynamic_slice %1005, %c_498, %1049, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %1051 = stablehlo.reshape %1050 : (tensor<1x1xf32>) -> tensor<f32>
    %c_499 = stablehlo.constant dense<0> : tensor<i32>
    %1052 = stablehlo.compare  LT, %1046, %c_499,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_500 = stablehlo.constant dense<1> : tensor<i32>
    %1053 = stablehlo.add %1046, %c_500 : tensor<i32>
    %1054 = stablehlo.select %1052, %1053, %1046 : tensor<i1>, tensor<i32>
    %c_501 = stablehlo.constant dense<1> : tensor<i32>
    %1055 = stablehlo.dynamic_slice %1013, %c_501, %1054, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %1056 = stablehlo.reshape %1055 : (tensor<1x1xf32>) -> tensor<f32>
    %1057 = call @_where_258(%1039, %1043, %1051) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_502 = stablehlo.constant dense<0> : tensor<i32>
    %1058 = stablehlo.compare  LT, %1046, %c_502,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_503 = stablehlo.constant dense<1> : tensor<i32>
    %1059 = stablehlo.add %1046, %c_503 : tensor<i32>
    %1060 = stablehlo.select %1058, %1059, %1046 : tensor<i1>, tensor<i32>
    %c_504 = stablehlo.constant dense<1> : tensor<i32>
    %1061 = stablehlo.broadcast_in_dim %c_504, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1062 = stablehlo.broadcast_in_dim %1060, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1063 = stablehlo.concatenate %1061, %1062, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %1064 = "stablehlo.scatter"(%1005, %1063, %1057) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg142: tensor<f32>, %arg143: tensor<f32>):
      stablehlo.return %arg143 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %1065 = call @_where_258(%1039, %3, %1056) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_505 = stablehlo.constant dense<0> : tensor<i32>
    %1066 = stablehlo.compare  LT, %1046, %c_505,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_506 = stablehlo.constant dense<1> : tensor<i32>
    %1067 = stablehlo.add %1046, %c_506 : tensor<i32>
    %1068 = stablehlo.select %1066, %1067, %1046 : tensor<i1>, tensor<i32>
    %c_507 = stablehlo.constant dense<1> : tensor<i32>
    %1069 = stablehlo.broadcast_in_dim %c_507, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1070 = stablehlo.broadcast_in_dim %1068, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1071 = stablehlo.concatenate %1069, %1070, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %1072 = "stablehlo.scatter"(%1013, %1071, %1065) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg142: tensor<f32>, %arg143: tensor<f32>):
      stablehlo.return %arg143 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %1073 = stablehlo.slice %1020 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %1074 = stablehlo.reshape %1073 : (tensor<1xi32>) -> tensor<i32>
    %c_508 = stablehlo.constant dense<1> : tensor<i32>
    %c_509 = stablehlo.constant dense<0> : tensor<i32>
    %1075 = call @_where_263(%1039, %c_508, %c_509) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %1076 = stablehlo.convert %1075 : tensor<i32>
    %1077 = stablehlo.add %1074, %1076 : tensor<i32>
    %c_510 = stablehlo.constant dense<1> : tensor<i32>
    %1078 = stablehlo.broadcast_in_dim %c_510, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1079 = "stablehlo.scatter"(%1020, %1078, %1077) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg142: tensor<i32>, %arg143: tensor<i32>):
      stablehlo.return %arg143 : tensor<i32>
    }) : (tensor<2xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
    %cst_511 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1080 = stablehlo.compare  GE, %3, %cst_511,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %c_512 = stablehlo.constant dense<true> : tensor<i1>
    %1081 = stablehlo.and %c_512, %1080 : tensor<i1>
    %cst_513 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %1082 = stablehlo.compare  LE, %3, %cst_513,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %1083 = stablehlo.and %1081, %1082 : tensor<i1>
    %c_514 = stablehlo.constant dense<1> : tensor<i32>
    %c_515 = stablehlo.constant dense<1> : tensor<i32>
    %1084 = stablehlo.maximum %c_514, %c_515 : tensor<i32>
    %1085 = call @remainder(%arg140, %1084) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_516 = stablehlo.constant dense<0> : tensor<i32>
    %1086 = stablehlo.compare  EQ, %1085, %c_516,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1087 = stablehlo.and %1083, %1086 : tensor<i1>
    %1088 = stablehlo.convert %1087 : (tensor<i1>) -> tensor<i32>
    %1089:3 = "stablehlo.case"(%1088) ({
      stablehlo.return %1030#0, %1030#1, %1030#2 : tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>
    }, {
      %c_518 = stablehlo.constant dense<257> : tensor<i32>
      %1091:2 = func.call @divmod_272(%arg101, %c_518) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_519 = stablehlo.constant dense<257> : tensor<i32>
      %1092:2 = func.call @divmod_289(%1091#0, %c_519) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_520 = stablehlo.constant dense<257> : tensor<i32>
      %1093:2 = func.call @divmod_289(%1092#0, %c_520) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_521 = stablehlo.constant dense<0> : tensor<i32>
      %1094 = stablehlo.broadcast_in_dim %c_521, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1095 = stablehlo.compare  GT, %1093#0, %1094,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_522 = stablehlo.constant dense<-1> : tensor<i32>
      %1096 = stablehlo.broadcast_in_dim %c_522, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1097 = stablehlo.compare  LT, %1093#0, %1096,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_523 = stablehlo.constant dense<0> : tensor<i32>
      %1098 = func.call @_where_296(%1097, %c_523, %1093#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_524 = stablehlo.constant dense<256> : tensor<i32>
      %1099 = func.call @_where_296(%1095, %c_524, %1098) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_525 = stablehlo.constant dense<0> : tensor<i32>
      %1100 = func.call @_where_296(%1097, %c_525, %1092#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_526 = stablehlo.constant dense<256> : tensor<i32>
      %1101 = func.call @_where_296(%1095, %c_526, %1100) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_527 = stablehlo.constant dense<0> : tensor<i32>
      %1102 = func.call @_where_296(%1097, %c_527, %1091#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_528 = stablehlo.constant dense<256> : tensor<i32>
      %1103 = func.call @_where_296(%1095, %c_528, %1102) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_529 = stablehlo.constant dense<1> : tensor<ui32>
      %c_530 = stablehlo.constant dense<1> : tensor<ui32>
      %1104 = stablehlo.partition_id : tensor<ui32>
      %1105 = stablehlo.divide %1104, %c_529 : tensor<ui32>
      %1106 = stablehlo.remainder %1105, %c_530 : tensor<ui32>
      %1107 = stablehlo.convert %1106 : (tensor<ui32>) -> tensor<i32>
      %c_531 = stablehlo.constant dense<257> : tensor<i32>
      %1108 = stablehlo.multiply %1107, %c_531 : tensor<i32>
      %1109 = stablehlo.broadcast_in_dim %1108, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1110 = stablehlo.subtract %1103, %1109 : tensor<200x8xi32>
      %c_532 = stablehlo.constant dense<0> : tensor<i32>
      %1111 = stablehlo.broadcast_in_dim %c_532, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1112 = stablehlo.compare  GE, %1110, %1111,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_533 = stablehlo.constant dense<257> : tensor<i32>
      %1113 = stablehlo.broadcast_in_dim %c_533, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1114 = stablehlo.compare  LT, %1110, %1113,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %1115 = stablehlo.and %1112, %1114 : tensor<200x8xi1>
      %c_534 = stablehlo.constant dense<0> : tensor<i32>
      %c_535 = stablehlo.constant dense<256> : tensor<i32>
      %1116 = func.call @clip_303(%1110, %c_534, %c_535) : (tensor<200x8xi32>, tensor<i32>, tensor<i32>) -> tensor<200x8xi32>
      %c_536 = stablehlo.constant dense<0> : tensor<i32>
      %1117 = stablehlo.broadcast_in_dim %c_536, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1118 = stablehlo.compare  LT, %1099, %1117,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_537 = stablehlo.constant dense<257> : tensor<i32>
      %1119 = stablehlo.broadcast_in_dim %c_537, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1120 = stablehlo.add %1099, %1119 : tensor<200x8xi32>
      %1121 = stablehlo.select %1118, %1120, %1099 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_538 = stablehlo.constant dense<0> : tensor<i32>
      %1122 = stablehlo.broadcast_in_dim %c_538, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1123 = stablehlo.compare  LT, %1101, %1122,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_539 = stablehlo.constant dense<257> : tensor<i32>
      %1124 = stablehlo.broadcast_in_dim %c_539, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1125 = stablehlo.add %1101, %1124 : tensor<200x8xi32>
      %1126 = stablehlo.select %1123, %1125, %1101 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_540 = stablehlo.constant dense<0> : tensor<i32>
      %1127 = stablehlo.broadcast_in_dim %c_540, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1128 = stablehlo.compare  LT, %1116, %1127,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_541 = stablehlo.constant dense<257> : tensor<i32>
      %1129 = stablehlo.broadcast_in_dim %c_541, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1130 = stablehlo.add %1116, %1129 : tensor<200x8xi32>
      %1131 = stablehlo.select %1128, %1130, %1116 : tensor<200x8xi1>, tensor<200x8xi32>
      %1132 = stablehlo.broadcast_in_dim %1121, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1133 = stablehlo.broadcast_in_dim %1126, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1134 = stablehlo.broadcast_in_dim %1131, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1135 = stablehlo.concatenate %1132, %1133, %1134, dim = 2 : (tensor<200x8x1xi32>, tensor<200x8x1xi32>, tensor<200x8x1xi32>) -> tensor<200x8x3xi32>
      %1136 = "stablehlo.gather"(%773#0, %1135) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x257xf32>, tensor<200x8x3xi32>) -> tensor<200x8xf32>
      %c_542 = stablehlo.constant dense<0> : tensor<i32>
      %1137 = func.call @_where_315(%1115, %1136, %c_542) : (tensor<200x8xi1>, tensor<200x8xf32>, tensor<i32>) -> tensor<200x8xf32>
      %1138 = stablehlo.multiply %1137, %arg102 : tensor<200x8xf32>
      %cst_543 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1139 = stablehlo.reduce(%1138 init: %cst_543) applies stablehlo.add across dimensions = [1] : (tensor<200x8xf32>, tensor<f32>) -> tensor<200xf32>
      %c_544 = stablehlo.constant dense<257> : tensor<i32>
      %1140:2 = func.call @divmod_369(%arg103, %c_544) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_545 = stablehlo.constant dense<256> : tensor<i32>
      %1141:2 = func.call @divmod_383(%1140#0, %c_545) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_546 = stablehlo.constant dense<257> : tensor<i32>
      %1142:2 = func.call @divmod_383(%1141#0, %c_546) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_547 = stablehlo.constant dense<0> : tensor<i32>
      %1143 = stablehlo.broadcast_in_dim %c_547, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1144 = stablehlo.compare  GT, %1142#0, %1143,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_548 = stablehlo.constant dense<-1> : tensor<i32>
      %1145 = stablehlo.broadcast_in_dim %c_548, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1146 = stablehlo.compare  LT, %1142#0, %1145,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_549 = stablehlo.constant dense<0> : tensor<i32>
      %1147 = func.call @_where_390(%1146, %c_549, %1142#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_550 = stablehlo.constant dense<256> : tensor<i32>
      %1148 = func.call @_where_390(%1144, %c_550, %1147) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_551 = stablehlo.constant dense<0> : tensor<i32>
      %1149 = func.call @_where_390(%1146, %c_551, %1141#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_552 = stablehlo.constant dense<255> : tensor<i32>
      %1150 = func.call @_where_390(%1144, %c_552, %1149) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_553 = stablehlo.constant dense<0> : tensor<i32>
      %1151 = func.call @_where_390(%1146, %c_553, %1140#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_554 = stablehlo.constant dense<256> : tensor<i32>
      %1152 = func.call @_where_390(%1144, %c_554, %1151) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_555 = stablehlo.constant dense<1> : tensor<ui32>
      %c_556 = stablehlo.constant dense<1> : tensor<ui32>
      %1153 = stablehlo.partition_id : tensor<ui32>
      %1154 = stablehlo.divide %1153, %c_555 : tensor<ui32>
      %1155 = stablehlo.remainder %1154, %c_556 : tensor<ui32>
      %1156 = stablehlo.convert %1155 : (tensor<ui32>) -> tensor<i32>
      %c_557 = stablehlo.constant dense<257> : tensor<i32>
      %1157 = stablehlo.multiply %1156, %c_557 : tensor<i32>
      %1158 = stablehlo.broadcast_in_dim %1157, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1159 = stablehlo.subtract %1152, %1158 : tensor<200x2xi32>
      %c_558 = stablehlo.constant dense<0> : tensor<i32>
      %1160 = stablehlo.broadcast_in_dim %c_558, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1161 = stablehlo.compare  GE, %1159, %1160,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_559 = stablehlo.constant dense<257> : tensor<i32>
      %1162 = stablehlo.broadcast_in_dim %c_559, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1163 = stablehlo.compare  LT, %1159, %1162,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %1164 = stablehlo.and %1161, %1163 : tensor<200x2xi1>
      %c_560 = stablehlo.constant dense<0> : tensor<i32>
      %c_561 = stablehlo.constant dense<256> : tensor<i32>
      %1165 = func.call @clip_397(%1159, %c_560, %c_561) : (tensor<200x2xi32>, tensor<i32>, tensor<i32>) -> tensor<200x2xi32>
      %c_562 = stablehlo.constant dense<0> : tensor<i32>
      %1166 = stablehlo.broadcast_in_dim %c_562, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1167 = stablehlo.compare  LT, %1148, %1166,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_563 = stablehlo.constant dense<257> : tensor<i32>
      %1168 = stablehlo.broadcast_in_dim %c_563, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1169 = stablehlo.add %1148, %1168 : tensor<200x2xi32>
      %1170 = stablehlo.select %1167, %1169, %1148 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_564 = stablehlo.constant dense<0> : tensor<i32>
      %1171 = stablehlo.broadcast_in_dim %c_564, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1172 = stablehlo.compare  LT, %1150, %1171,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_565 = stablehlo.constant dense<256> : tensor<i32>
      %1173 = stablehlo.broadcast_in_dim %c_565, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1174 = stablehlo.add %1150, %1173 : tensor<200x2xi32>
      %1175 = stablehlo.select %1172, %1174, %1150 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_566 = stablehlo.constant dense<0> : tensor<i32>
      %1176 = stablehlo.broadcast_in_dim %c_566, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1177 = stablehlo.compare  LT, %1165, %1176,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_567 = stablehlo.constant dense<257> : tensor<i32>
      %1178 = stablehlo.broadcast_in_dim %c_567, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1179 = stablehlo.add %1165, %1178 : tensor<200x2xi32>
      %1180 = stablehlo.select %1177, %1179, %1165 : tensor<200x2xi1>, tensor<200x2xi32>
      %1181 = stablehlo.broadcast_in_dim %1170, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1182 = stablehlo.broadcast_in_dim %1175, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1183 = stablehlo.broadcast_in_dim %1180, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1184 = stablehlo.concatenate %1181, %1182, %1183, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
      %1185 = "stablehlo.gather"(%873, %1184) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
      %c_568 = stablehlo.constant dense<0> : tensor<i32>
      %1186 = func.call @_where_408(%1164, %1185, %c_568) : (tensor<200x2xi1>, tensor<200x2xf32>, tensor<i32>) -> tensor<200x2xf32>
      %1187 = stablehlo.multiply %1186, %arg104 : tensor<200x2xf32>
      %cst_569 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1188 = stablehlo.reduce(%1187 init: %cst_569) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
      %c_570 = stablehlo.constant dense<257> : tensor<i32>
      %1189:2 = func.call @divmod_369(%arg105, %c_570) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_571 = stablehlo.constant dense<257> : tensor<i32>
      %1190:2 = func.call @divmod_383(%1189#0, %c_571) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_572 = stablehlo.constant dense<256> : tensor<i32>
      %1191:2 = func.call @divmod_383(%1190#0, %c_572) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_573 = stablehlo.constant dense<0> : tensor<i32>
      %1192 = stablehlo.broadcast_in_dim %c_573, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1193 = stablehlo.compare  GT, %1191#0, %1192,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_574 = stablehlo.constant dense<-1> : tensor<i32>
      %1194 = stablehlo.broadcast_in_dim %c_574, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1195 = stablehlo.compare  LT, %1191#0, %1194,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_575 = stablehlo.constant dense<0> : tensor<i32>
      %1196 = func.call @_where_390(%1195, %c_575, %1191#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_576 = stablehlo.constant dense<255> : tensor<i32>
      %1197 = func.call @_where_390(%1193, %c_576, %1196) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_577 = stablehlo.constant dense<0> : tensor<i32>
      %1198 = func.call @_where_390(%1195, %c_577, %1190#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_578 = stablehlo.constant dense<256> : tensor<i32>
      %1199 = func.call @_where_390(%1193, %c_578, %1198) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_579 = stablehlo.constant dense<0> : tensor<i32>
      %1200 = func.call @_where_390(%1195, %c_579, %1189#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_580 = stablehlo.constant dense<256> : tensor<i32>
      %1201 = func.call @_where_390(%1193, %c_580, %1200) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_581 = stablehlo.constant dense<1> : tensor<ui32>
      %c_582 = stablehlo.constant dense<1> : tensor<ui32>
      %1202 = stablehlo.partition_id : tensor<ui32>
      %1203 = stablehlo.divide %1202, %c_581 : tensor<ui32>
      %1204 = stablehlo.remainder %1203, %c_582 : tensor<ui32>
      %1205 = stablehlo.convert %1204 : (tensor<ui32>) -> tensor<i32>
      %c_583 = stablehlo.constant dense<257> : tensor<i32>
      %1206 = stablehlo.multiply %1205, %c_583 : tensor<i32>
      %1207 = stablehlo.broadcast_in_dim %1206, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1208 = stablehlo.subtract %1201, %1207 : tensor<200x2xi32>
      %c_584 = stablehlo.constant dense<0> : tensor<i32>
      %1209 = stablehlo.broadcast_in_dim %c_584, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1210 = stablehlo.compare  GE, %1208, %1209,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_585 = stablehlo.constant dense<257> : tensor<i32>
      %1211 = stablehlo.broadcast_in_dim %c_585, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1212 = stablehlo.compare  LT, %1208, %1211,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %1213 = stablehlo.and %1210, %1212 : tensor<200x2xi1>
      %c_586 = stablehlo.constant dense<0> : tensor<i32>
      %c_587 = stablehlo.constant dense<256> : tensor<i32>
      %1214 = func.call @clip_397(%1208, %c_586, %c_587) : (tensor<200x2xi32>, tensor<i32>, tensor<i32>) -> tensor<200x2xi32>
      %c_588 = stablehlo.constant dense<0> : tensor<i32>
      %1215 = stablehlo.broadcast_in_dim %c_588, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1216 = stablehlo.compare  LT, %1197, %1215,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_589 = stablehlo.constant dense<256> : tensor<i32>
      %1217 = stablehlo.broadcast_in_dim %c_589, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1218 = stablehlo.add %1197, %1217 : tensor<200x2xi32>
      %1219 = stablehlo.select %1216, %1218, %1197 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_590 = stablehlo.constant dense<0> : tensor<i32>
      %1220 = stablehlo.broadcast_in_dim %c_590, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1221 = stablehlo.compare  LT, %1199, %1220,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_591 = stablehlo.constant dense<257> : tensor<i32>
      %1222 = stablehlo.broadcast_in_dim %c_591, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1223 = stablehlo.add %1199, %1222 : tensor<200x2xi32>
      %1224 = stablehlo.select %1221, %1223, %1199 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_592 = stablehlo.constant dense<0> : tensor<i32>
      %1225 = stablehlo.broadcast_in_dim %c_592, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1226 = stablehlo.compare  LT, %1214, %1225,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_593 = stablehlo.constant dense<257> : tensor<i32>
      %1227 = stablehlo.broadcast_in_dim %c_593, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1228 = stablehlo.add %1214, %1227 : tensor<200x2xi32>
      %1229 = stablehlo.select %1226, %1228, %1214 : tensor<200x2xi1>, tensor<200x2xi32>
      %1230 = stablehlo.broadcast_in_dim %1219, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1231 = stablehlo.broadcast_in_dim %1224, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1232 = stablehlo.broadcast_in_dim %1229, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1233 = stablehlo.concatenate %1230, %1231, %1232, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
      %1234 = "stablehlo.gather"(%971, %1233) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
      %c_594 = stablehlo.constant dense<0> : tensor<i32>
      %1235 = func.call @_where_408(%1213, %1234, %c_594) : (tensor<200x2xi1>, tensor<200x2xf32>, tensor<i32>) -> tensor<200x2xf32>
      %1236 = stablehlo.multiply %1235, %arg106 : tensor<200x2xf32>
      %cst_595 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1237 = stablehlo.reduce(%1236 init: %cst_595) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
      %c_596 = stablehlo.constant dense<257> : tensor<i32>
      %1238:2 = func.call @divmod_452(%arg107, %c_596) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_597 = stablehlo.constant dense<256> : tensor<i32>
      %1239:2 = func.call @divmod_466(%1238#0, %c_597) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_598 = stablehlo.constant dense<256> : tensor<i32>
      %1240:2 = func.call @divmod_466(%1239#0, %c_598) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_599 = stablehlo.constant dense<0> : tensor<i32>
      %1241 = stablehlo.broadcast_in_dim %c_599, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1242 = stablehlo.compare  GT, %1240#0, %1241,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_600 = stablehlo.constant dense<-1> : tensor<i32>
      %1243 = stablehlo.broadcast_in_dim %c_600, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1244 = stablehlo.compare  LT, %1240#0, %1243,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_601 = stablehlo.constant dense<0> : tensor<i32>
      %1245 = func.call @_where_473(%1244, %c_601, %1240#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_602 = stablehlo.constant dense<255> : tensor<i32>
      %1246 = func.call @_where_473(%1242, %c_602, %1245) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_603 = stablehlo.constant dense<0> : tensor<i32>
      %1247 = func.call @_where_473(%1244, %c_603, %1239#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_604 = stablehlo.constant dense<255> : tensor<i32>
      %1248 = func.call @_where_473(%1242, %c_604, %1247) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_605 = stablehlo.constant dense<0> : tensor<i32>
      %1249 = func.call @_where_473(%1244, %c_605, %1238#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_606 = stablehlo.constant dense<256> : tensor<i32>
      %1250 = func.call @_where_473(%1242, %c_606, %1249) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_607 = stablehlo.constant dense<1> : tensor<ui32>
      %c_608 = stablehlo.constant dense<1> : tensor<ui32>
      %1251 = stablehlo.partition_id : tensor<ui32>
      %1252 = stablehlo.divide %1251, %c_607 : tensor<ui32>
      %1253 = stablehlo.remainder %1252, %c_608 : tensor<ui32>
      %1254 = stablehlo.convert %1253 : (tensor<ui32>) -> tensor<i32>
      %c_609 = stablehlo.constant dense<257> : tensor<i32>
      %1255 = stablehlo.multiply %1254, %c_609 : tensor<i32>
      %1256 = stablehlo.broadcast_in_dim %1255, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1257 = stablehlo.subtract %1250, %1256 : tensor<200x1xi32>
      %c_610 = stablehlo.constant dense<0> : tensor<i32>
      %1258 = stablehlo.broadcast_in_dim %c_610, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1259 = stablehlo.compare  GE, %1257, %1258,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_611 = stablehlo.constant dense<257> : tensor<i32>
      %1260 = stablehlo.broadcast_in_dim %c_611, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1261 = stablehlo.compare  LT, %1257, %1260,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %1262 = stablehlo.and %1259, %1261 : tensor<200x1xi1>
      %c_612 = stablehlo.constant dense<0> : tensor<i32>
      %c_613 = stablehlo.constant dense<256> : tensor<i32>
      %1263 = func.call @clip_480(%1257, %c_612, %c_613) : (tensor<200x1xi32>, tensor<i32>, tensor<i32>) -> tensor<200x1xi32>
      %c_614 = stablehlo.constant dense<0> : tensor<i32>
      %1264 = stablehlo.broadcast_in_dim %c_614, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1265 = stablehlo.compare  LT, %1246, %1264,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_615 = stablehlo.constant dense<256> : tensor<i32>
      %1266 = stablehlo.broadcast_in_dim %c_615, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1267 = stablehlo.add %1246, %1266 : tensor<200x1xi32>
      %1268 = stablehlo.select %1265, %1267, %1246 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_616 = stablehlo.constant dense<0> : tensor<i32>
      %1269 = stablehlo.broadcast_in_dim %c_616, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1270 = stablehlo.compare  LT, %1248, %1269,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_617 = stablehlo.constant dense<256> : tensor<i32>
      %1271 = stablehlo.broadcast_in_dim %c_617, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1272 = stablehlo.add %1248, %1271 : tensor<200x1xi32>
      %1273 = stablehlo.select %1270, %1272, %1248 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_618 = stablehlo.constant dense<0> : tensor<i32>
      %1274 = stablehlo.broadcast_in_dim %c_618, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1275 = stablehlo.compare  LT, %1263, %1274,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_619 = stablehlo.constant dense<257> : tensor<i32>
      %1276 = stablehlo.broadcast_in_dim %c_619, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1277 = stablehlo.add %1263, %1276 : tensor<200x1xi32>
      %1278 = stablehlo.select %1275, %1277, %1263 : tensor<200x1xi1>, tensor<200x1xi32>
      %1279 = stablehlo.broadcast_in_dim %1268, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1280 = stablehlo.broadcast_in_dim %1273, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1281 = stablehlo.broadcast_in_dim %1278, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1282 = stablehlo.concatenate %1279, %1280, %1281, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1283 = "stablehlo.gather"(%289#0, %1282) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %c_620 = stablehlo.constant dense<0> : tensor<i32>
      %1284 = func.call @_where_491(%1262, %1283, %c_620) : (tensor<200x1xi1>, tensor<200x1xf32>, tensor<i32>) -> tensor<200x1xf32>
      %1285 = stablehlo.multiply %1284, %arg108 : tensor<200x1xf32>
      %cst_621 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1286 = stablehlo.reduce(%1285 init: %cst_621) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_622 = stablehlo.constant dense<257> : tensor<i32>
      %1287:2 = func.call @divmod_322(%arg109, %c_622) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_623 = stablehlo.constant dense<257> : tensor<i32>
      %1288:2 = func.call @divmod_336(%1287#0, %c_623) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_624 = stablehlo.constant dense<256> : tensor<i32>
      %1289:2 = func.call @divmod_336(%1288#0, %c_624) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_625 = stablehlo.constant dense<0> : tensor<i32>
      %1290 = stablehlo.broadcast_in_dim %c_625, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1291 = stablehlo.compare  GT, %1289#0, %1290,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_626 = stablehlo.constant dense<-1> : tensor<i32>
      %1292 = stablehlo.broadcast_in_dim %c_626, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1293 = stablehlo.compare  LT, %1289#0, %1292,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_627 = stablehlo.constant dense<0> : tensor<i32>
      %1294 = func.call @_where_343(%1293, %c_627, %1289#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_628 = stablehlo.constant dense<255> : tensor<i32>
      %1295 = func.call @_where_343(%1291, %c_628, %1294) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_629 = stablehlo.constant dense<0> : tensor<i32>
      %1296 = func.call @_where_343(%1293, %c_629, %1288#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_630 = stablehlo.constant dense<256> : tensor<i32>
      %1297 = func.call @_where_343(%1291, %c_630, %1296) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_631 = stablehlo.constant dense<0> : tensor<i32>
      %1298 = func.call @_where_343(%1293, %c_631, %1287#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_632 = stablehlo.constant dense<256> : tensor<i32>
      %1299 = func.call @_where_343(%1291, %c_632, %1298) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_633 = stablehlo.constant dense<1> : tensor<ui32>
      %c_634 = stablehlo.constant dense<1> : tensor<ui32>
      %1300 = stablehlo.partition_id : tensor<ui32>
      %1301 = stablehlo.divide %1300, %c_633 : tensor<ui32>
      %1302 = stablehlo.remainder %1301, %c_634 : tensor<ui32>
      %1303 = stablehlo.convert %1302 : (tensor<ui32>) -> tensor<i32>
      %c_635 = stablehlo.constant dense<257> : tensor<i32>
      %1304 = stablehlo.multiply %1303, %c_635 : tensor<i32>
      %1305 = stablehlo.broadcast_in_dim %1304, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1306 = stablehlo.subtract %1299, %1305 : tensor<200x4xi32>
      %c_636 = stablehlo.constant dense<0> : tensor<i32>
      %1307 = stablehlo.broadcast_in_dim %c_636, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1308 = stablehlo.compare  GE, %1306, %1307,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_637 = stablehlo.constant dense<257> : tensor<i32>
      %1309 = stablehlo.broadcast_in_dim %c_637, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1310 = stablehlo.compare  LT, %1306, %1309,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1311 = stablehlo.and %1308, %1310 : tensor<200x4xi1>
      %c_638 = stablehlo.constant dense<0> : tensor<i32>
      %c_639 = stablehlo.constant dense<256> : tensor<i32>
      %1312 = func.call @clip_350(%1306, %c_638, %c_639) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_640 = stablehlo.constant dense<0> : tensor<i32>
      %1313 = stablehlo.broadcast_in_dim %c_640, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1314 = stablehlo.compare  LT, %1295, %1313,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_641 = stablehlo.constant dense<256> : tensor<i32>
      %1315 = stablehlo.broadcast_in_dim %c_641, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1316 = stablehlo.add %1295, %1315 : tensor<200x4xi32>
      %1317 = stablehlo.select %1314, %1316, %1295 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_642 = stablehlo.constant dense<0> : tensor<i32>
      %1318 = stablehlo.broadcast_in_dim %c_642, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1319 = stablehlo.compare  LT, %1297, %1318,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_643 = stablehlo.constant dense<257> : tensor<i32>
      %1320 = stablehlo.broadcast_in_dim %c_643, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1321 = stablehlo.add %1297, %1320 : tensor<200x4xi32>
      %1322 = stablehlo.select %1319, %1321, %1297 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_644 = stablehlo.constant dense<0> : tensor<i32>
      %1323 = stablehlo.broadcast_in_dim %c_644, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1324 = stablehlo.compare  LT, %1312, %1323,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_645 = stablehlo.constant dense<257> : tensor<i32>
      %1325 = stablehlo.broadcast_in_dim %c_645, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1326 = stablehlo.add %1312, %1325 : tensor<200x4xi32>
      %1327 = stablehlo.select %1324, %1326, %1312 : tensor<200x4xi1>, tensor<200x4xi32>
      %1328 = stablehlo.broadcast_in_dim %1317, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1329 = stablehlo.broadcast_in_dim %1322, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1330 = stablehlo.broadcast_in_dim %1327, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1331 = stablehlo.concatenate %1328, %1329, %1330, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1332 = "stablehlo.gather"(%389, %1331) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_646 = stablehlo.constant dense<0> : tensor<i32>
      %1333 = func.call @_where_361(%1311, %1332, %c_646) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1334 = stablehlo.multiply %1333, %arg110 : tensor<200x4xf32>
      %cst_647 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1335 = stablehlo.reduce(%1334 init: %cst_647) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_648 = stablehlo.constant dense<257> : tensor<i32>
      %1336:2 = func.call @divmod_322(%arg111, %c_648) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_649 = stablehlo.constant dense<256> : tensor<i32>
      %1337:2 = func.call @divmod_336(%1336#0, %c_649) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_650 = stablehlo.constant dense<257> : tensor<i32>
      %1338:2 = func.call @divmod_336(%1337#0, %c_650) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_651 = stablehlo.constant dense<0> : tensor<i32>
      %1339 = stablehlo.broadcast_in_dim %c_651, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1340 = stablehlo.compare  GT, %1338#0, %1339,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_652 = stablehlo.constant dense<-1> : tensor<i32>
      %1341 = stablehlo.broadcast_in_dim %c_652, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1342 = stablehlo.compare  LT, %1338#0, %1341,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_653 = stablehlo.constant dense<0> : tensor<i32>
      %1343 = func.call @_where_343(%1342, %c_653, %1338#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_654 = stablehlo.constant dense<256> : tensor<i32>
      %1344 = func.call @_where_343(%1340, %c_654, %1343) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_655 = stablehlo.constant dense<0> : tensor<i32>
      %1345 = func.call @_where_343(%1342, %c_655, %1337#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_656 = stablehlo.constant dense<255> : tensor<i32>
      %1346 = func.call @_where_343(%1340, %c_656, %1345) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_657 = stablehlo.constant dense<0> : tensor<i32>
      %1347 = func.call @_where_343(%1342, %c_657, %1336#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_658 = stablehlo.constant dense<256> : tensor<i32>
      %1348 = func.call @_where_343(%1340, %c_658, %1347) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_659 = stablehlo.constant dense<1> : tensor<ui32>
      %c_660 = stablehlo.constant dense<1> : tensor<ui32>
      %1349 = stablehlo.partition_id : tensor<ui32>
      %1350 = stablehlo.divide %1349, %c_659 : tensor<ui32>
      %1351 = stablehlo.remainder %1350, %c_660 : tensor<ui32>
      %1352 = stablehlo.convert %1351 : (tensor<ui32>) -> tensor<i32>
      %c_661 = stablehlo.constant dense<257> : tensor<i32>
      %1353 = stablehlo.multiply %1352, %c_661 : tensor<i32>
      %1354 = stablehlo.broadcast_in_dim %1353, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1355 = stablehlo.subtract %1348, %1354 : tensor<200x4xi32>
      %c_662 = stablehlo.constant dense<0> : tensor<i32>
      %1356 = stablehlo.broadcast_in_dim %c_662, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1357 = stablehlo.compare  GE, %1355, %1356,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_663 = stablehlo.constant dense<257> : tensor<i32>
      %1358 = stablehlo.broadcast_in_dim %c_663, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1359 = stablehlo.compare  LT, %1355, %1358,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1360 = stablehlo.and %1357, %1359 : tensor<200x4xi1>
      %c_664 = stablehlo.constant dense<0> : tensor<i32>
      %c_665 = stablehlo.constant dense<256> : tensor<i32>
      %1361 = func.call @clip_350(%1355, %c_664, %c_665) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_666 = stablehlo.constant dense<0> : tensor<i32>
      %1362 = stablehlo.broadcast_in_dim %c_666, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1363 = stablehlo.compare  LT, %1344, %1362,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_667 = stablehlo.constant dense<257> : tensor<i32>
      %1364 = stablehlo.broadcast_in_dim %c_667, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1365 = stablehlo.add %1344, %1364 : tensor<200x4xi32>
      %1366 = stablehlo.select %1363, %1365, %1344 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_668 = stablehlo.constant dense<0> : tensor<i32>
      %1367 = stablehlo.broadcast_in_dim %c_668, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1368 = stablehlo.compare  LT, %1346, %1367,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_669 = stablehlo.constant dense<256> : tensor<i32>
      %1369 = stablehlo.broadcast_in_dim %c_669, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1370 = stablehlo.add %1346, %1369 : tensor<200x4xi32>
      %1371 = stablehlo.select %1368, %1370, %1346 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_670 = stablehlo.constant dense<0> : tensor<i32>
      %1372 = stablehlo.broadcast_in_dim %c_670, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1373 = stablehlo.compare  LT, %1361, %1372,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_671 = stablehlo.constant dense<257> : tensor<i32>
      %1374 = stablehlo.broadcast_in_dim %c_671, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1375 = stablehlo.add %1361, %1374 : tensor<200x4xi32>
      %1376 = stablehlo.select %1373, %1375, %1361 : tensor<200x4xi1>, tensor<200x4xi32>
      %1377 = stablehlo.broadcast_in_dim %1366, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1378 = stablehlo.broadcast_in_dim %1371, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1379 = stablehlo.broadcast_in_dim %1376, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1380 = stablehlo.concatenate %1377, %1378, %1379, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1381 = "stablehlo.gather"(%487, %1380) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_672 = stablehlo.constant dense<0> : tensor<i32>
      %1382 = func.call @_where_361(%1360, %1381, %c_672) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1383 = stablehlo.multiply %1382, %arg112 : tensor<200x4xf32>
      %cst_673 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1384 = stablehlo.reduce(%1383 init: %cst_673) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %1385 = stablehlo.slice %1030#0 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
      %1386 = stablehlo.reshape %1385 : (tensor<1x7200xf32>) -> tensor<7200xf32>
      %1387 = stablehlo.slice %1030#1 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
      %1388 = stablehlo.reshape %1387 : (tensor<1x7200xf32>) -> tensor<7200xf32>
      %1389 = stablehlo.broadcast_in_dim %1139, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1390 = stablehlo.broadcast_in_dim %1188, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1391 = stablehlo.broadcast_in_dim %1237, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1392 = stablehlo.broadcast_in_dim %1286, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1393 = stablehlo.broadcast_in_dim %1335, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1394 = stablehlo.broadcast_in_dim %1384, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1395 = stablehlo.concatenate %1389, %1390, %1391, %1392, %1393, %1394, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %cst_674 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1396 = stablehlo.broadcast_in_dim %cst_674, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %1397 = stablehlo.multiply %1396, %arg113 : tensor<3xf32>
      %1398 = stablehlo.optimization_barrier %1397 : tensor<3xf32>
      %1399 = stablehlo.optimization_barrier %3 : tensor<f32>
      %1400 = stablehlo.broadcast_in_dim %1399, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %1401 = stablehlo.multiply %1398, %1400 : tensor<3xf32>
      %1402 = stablehlo.cosine %1401 : tensor<3xf32>
      %1403 = stablehlo.sine %1401 : tensor<3xf32>
      %cst_675 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_676 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1404 = stablehlo.subtract %cst_675, %cst_676 : tensor<f32>
      %cst_677 = stablehlo.constant dense<1.000000e-30> : tensor<f32>
      %1405 = stablehlo.maximum %1404, %cst_677 : tensor<f32>
      %cst_678 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1406 = stablehlo.subtract %3, %cst_678 : tensor<f32>
      %1407 = stablehlo.divide %1406, %1405 : tensor<f32>
      %cst_679 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %cst_680 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1408 = func.call @clip_425(%1407, %cst_679, %cst_680) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_681 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1409 = stablehlo.multiply %cst_681, %1408 : tensor<f32>
      %1410 = stablehlo.cosine %1409 : tensor<f32>
      %cst_682 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1411 = stablehlo.subtract %cst_682, %1410 : tensor<f32>
      %cst_683 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %1412 = stablehlo.multiply %cst_683, %1411 : tensor<f32>
      %cst_684 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %1413 = stablehlo.is_finite %cst_684 : (tensor<f32>) -> tensor<i1>
      %c_685 = stablehlo.constant dense<false> : tensor<i1>
      %1414 = stablehlo.and %c_685, %1413 : tensor<i1>
      %cst_686 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_687 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1415 = stablehlo.compare  GT, %cst_686, %cst_687,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %1416 = stablehlo.and %1414, %1415 : tensor<i1>
      %cst_688 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1417 = func.call @_where_432(%1416, %1412, %cst_688) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_689 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1418 = stablehlo.maximum %1417, %cst_689 : tensor<f32>
      %cst_690 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1419 = stablehlo.multiply %arg0, %cst_690 : tensor<f32>
      %cst_691 = stablehlo.constant dense<0x4D8EF3C2> : tensor<f32>
      %1420 = stablehlo.multiply %1419, %cst_691 : tensor<f32>
      %cst_692 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %1421 = stablehlo.divide %1420, %cst_692 : tensor<f32>
      %cst_693 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1422 = stablehlo.sqrt %cst_693 : tensor<f32>
      %1423 = stablehlo.divide %1421, %1422 : tensor<f32>
      %1424 = stablehlo.multiply %1418, %1423 : tensor<f32>
      %c_694 = stablehlo.constant dense<0> : tensor<i32>
      %c_695 = stablehlo.constant dense<1> : tensor<i32>
      %1425 = stablehlo.compare  EQ, %c_694, %c_695,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1426 = func.call @_where_435(%1425, %1424, %1418) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %1427 = stablehlo.broadcast_in_dim %arg114, dims = [0] : (tensor<6xf32>) -> tensor<6x1x1xf32>
      %1428 = stablehlo.broadcast_in_dim %1426, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %1429 = stablehlo.multiply %1428, %1427 : tensor<6x1x1xf32>
      %1430 = stablehlo.dot_general %1395, %1402, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %1431 = stablehlo.transpose %1430, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %1432 = stablehlo.broadcast_in_dim %1429, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %1433 = stablehlo.multiply %1432, %1431 : tensor<6x3x200xf32>
      %1434 = stablehlo.broadcast_in_dim %1426, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %1435 = stablehlo.multiply %1434, %1427 : tensor<6x1x1xf32>
      %1436 = stablehlo.dot_general %1395, %1403, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %1437 = stablehlo.transpose %1436, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %1438 = stablehlo.broadcast_in_dim %1435, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %1439 = stablehlo.multiply %1438, %1437 : tensor<6x3x200xf32>
      %1440 = stablehlo.reshape %1433 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_696 = stablehlo.constant dense<3600> : tensor<i32>
      %1441 = stablehlo.broadcast_in_dim %c_696, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1442 = "stablehlo.scatter"(%1386, %1441, %1440) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg142: tensor<f32>, %arg143: tensor<f32>):
        %1451 = stablehlo.add %arg142, %arg143 : tensor<f32>
        stablehlo.return %1451 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %1443 = stablehlo.reshape %1439 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_697 = stablehlo.constant dense<3600> : tensor<i32>
      %1444 = stablehlo.broadcast_in_dim %c_697, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1445 = "stablehlo.scatter"(%1388, %1444, %1443) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg142: tensor<f32>, %arg143: tensor<f32>):
        %1451 = stablehlo.add %arg142, %arg143 : tensor<f32>
        stablehlo.return %1451 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %1446 = stablehlo.broadcast_in_dim %1417, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %c_698 = stablehlo.constant dense<3> : tensor<i32>
      %1447 = stablehlo.broadcast_in_dim %c_698, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1448 = "stablehlo.scatter"(%1030#2, %1447, %1446) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg142: tensor<f32>, %arg143: tensor<f32>):
        %1451 = stablehlo.add %arg142, %arg143 : tensor<f32>
        stablehlo.return %1451 : tensor<f32>
      }) : (tensor<6xf32>, tensor<1xi32>, tensor<3xf32>) -> tensor<6xf32>
      %1449 = stablehlo.broadcast_in_dim %1442, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
      %1450 = stablehlo.broadcast_in_dim %1445, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
      stablehlo.return %1449, %1450, %1448 : tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>
    }) : (tensor<i32>) -> (tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>)
    %c_517 = stablehlo.constant dense<1> : tensor<i32>
    %1090 = stablehlo.add %arg140, %c_517 : tensor<i32>
    return %773#0, %873, %971, %289#0, %389, %487, %289#3, %289#4, %289#5, %290, %291, %289#8, %773#3, %773#4, %773#5, %774, %775, %773#8, %1064, %1072, %1079, %1089#0, %1089#1, %1089#2, %3, %1090 : tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @_where_73(%arg0: tensor<256x24x257xi1>, %arg1: tensor<256x24x257xf32>, %arg2: tensor<256x24x257xf32>) -> tensor<256x24x257xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<256x24x257xi1>, tensor<256x24x257xf32>
    return %0 : tensor<256x24x257xf32>
  }
  func.func private @_where_82(%arg0: tensor<24x256x257xi1>, %arg1: tensor<24x256x257xf32>, %arg2: tensor<24x256x257xf32>) -> tensor<24x256x257xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x256x257xi1>, tensor<24x256x257xf32>
    return %0 : tensor<24x256x257xf32>
  }
  func.func private @_where_89(%arg0: tensor<24x257x257xi1>, %arg1: tensor<24x257x257xf32>, %arg2: tensor<24x257x257xf32>) -> tensor<24x257x257xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x257x257xi1>, tensor<24x257x257xf32>
    return %0 : tensor<24x257x257xf32>
  }
  func.func private @_where_91(%arg0: tensor<256x257x24xi1>, %arg1: tensor<256x257x24xf32>, %arg2: tensor<256x257x24xf32>) -> tensor<256x257x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<256x257x24xi1>, tensor<256x257x24xf32>
    return %0 : tensor<256x257x24xf32>
  }
  func.func private @_where_92(%arg0: tensor<257x256x24xi1>, %arg1: tensor<257x256x24xf32>, %arg2: tensor<257x256x24xf32>) -> tensor<257x256x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<257x256x24xi1>, tensor<257x256x24xf32>
    return %0 : tensor<257x256x24xf32>
  }
  func.func private @_where_98(%arg0: tensor<257x24x257xi1>, %arg1: tensor<257x24x257xf32>, %arg2: tensor<257x24x257xf32>) -> tensor<257x24x257xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<257x24x257xi1>, tensor<257x24x257xf32>
    return %0 : tensor<257x24x257xf32>
  }
  func.func private @clip(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %arg2 : tensor<i32>
    %3 = stablehlo.minimum %2, %1 : tensor<i32>
    return %3 : tensor<i32>
  }
  func.func private @clip_125(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %arg2 : tensor<i32>
    %3 = stablehlo.minimum %2, %1 : tensor<i32>
    return %3 : tensor<i32>
  }
  func.func private @_take(%arg0: tensor<14x25x1xf32>, %arg1: tensor<1xi32>) -> tensor<14x25x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 14, 25, 1>}> : (tensor<14x25x1xf32>, tensor<1x1xi32>) -> tensor<14x25x1xf32>
    return %1 : tensor<14x25x1xf32>
  }
  func.func private @_where_145(%arg0: tensor<1x1x1xi1>, %arg1: tensor<14x25x1xf32>, %arg2: tensor<i32>) -> tensor<14x25x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x1x1xi1>) -> tensor<14x25x1xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<14x25x1xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<14x25x1xi1>, tensor<14x25x1xf32>
    return %3 : tensor<14x25x1xf32>
  }
  func.func private @_take_157(%arg0: tensor<15x24x1xf32>, %arg1: tensor<1xi32>) -> tensor<15x24x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 15, 24, 1>}> : (tensor<15x24x1xf32>, tensor<1x1xi32>) -> tensor<15x24x1xf32>
    return %1 : tensor<15x24x1xf32>
  }
  func.func private @_where_161(%arg0: tensor<1x1x1xi1>, %arg1: tensor<15x24x1xf32>, %arg2: tensor<i32>) -> tensor<15x24x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x1x1xi1>) -> tensor<15x24x1xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<15x24x1xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<15x24x1xi1>, tensor<15x24x1xf32>
    return %3 : tensor<15x24x1xf32>
  }
  func.func private @_take_179(%arg0: tensor<15x26x1xf32>, %arg1: tensor<1xi32>) -> tensor<15x26x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 15, 26, 1>}> : (tensor<15x26x1xf32>, tensor<1x1xi32>) -> tensor<15x26x1xf32>
    return %1 : tensor<15x26x1xf32>
  }
  func.func private @_where_183(%arg0: tensor<1x1x1xi1>, %arg1: tensor<15x26x1xf32>, %arg2: tensor<i32>) -> tensor<15x26x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x1x1xi1>) -> tensor<15x26x1xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<15x26x1xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<15x26x1xi1>, tensor<15x26x1xf32>
    return %3 : tensor<15x26x1xf32>
  }
  func.func private @_take_194(%arg0: tensor<16x25x1xf32>, %arg1: tensor<1xi32>) -> tensor<16x25x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 25, 1>}> : (tensor<16x25x1xf32>, tensor<1x1xi32>) -> tensor<16x25x1xf32>
    return %1 : tensor<16x25x1xf32>
  }
  func.func private @_where_198(%arg0: tensor<1x1x1xi1>, %arg1: tensor<16x25x1xf32>, %arg2: tensor<i32>) -> tensor<16x25x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x1x1xi1>) -> tensor<16x25x1xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<16x25x1xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<16x25x1xi1>, tensor<16x25x1xf32>
    return %3 : tensor<16x25x1xf32>
  }
  func.func private @remainder(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_209(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.remainder %arg0, %1 : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.compare  NE, %2, %c_1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.compare  LT, %2, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.compare  LT, %1, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = stablehlo.compare  NE, %4, %5,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = stablehlo.and %6, %3 : tensor<i1>
    %8 = stablehlo.add %2, %1 : tensor<i32>
    %9 = stablehlo.select %7, %8, %2 : tensor<i1>, tensor<i32>
    return %9 : tensor<i32>
  }
  func.func private @_where_209(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @divmod(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    %2 = call @remainder_229(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    return %1, %2 : tensor<200x1xi32>, tensor<200x1xi32>
  }
  func.func private @floor_divide(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %1 = stablehlo.divide %arg0, %0 : tensor<200x1xi32>
    %2 = stablehlo.sign %arg0 : tensor<200x1xi32>
    %3 = stablehlo.sign %arg1 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %5 = stablehlo.compare  NE, %2, %4,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %7 = stablehlo.remainder %arg0, %6 : tensor<200x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %9 = stablehlo.compare  NE, %7, %8,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %10 = stablehlo.and %5, %9 : tensor<200x1xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %12 = stablehlo.subtract %1, %11 : tensor<200x1xi32>
    %13 = call @_where_227(%10, %12, %1) : (tensor<200x1xi1>, tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi32>
    return %13 : tensor<200x1xi32>
  }
  func.func private @_where_227(%arg0: tensor<200x1xi1>, %arg1: tensor<200x1xi32>, %arg2: tensor<200x1xi32>) -> tensor<200x1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x1xi1>, tensor<200x1xi32>
    return %0 : tensor<200x1xi32>
  }
  func.func private @remainder_229(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_209(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %3 = stablehlo.remainder %arg0, %2 : tensor<200x1xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %5 = stablehlo.compare  NE, %3, %4,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %7 = stablehlo.compare  LT, %3, %6,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %1, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<200x1xi1>
    %10 = stablehlo.compare  NE, %7, %9,  UNSIGNED : (tensor<200x1xi1>, tensor<200x1xi1>) -> tensor<200x1xi1>
    %11 = stablehlo.and %10, %5 : tensor<200x1xi1>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %13 = stablehlo.add %3, %12 : tensor<200x1xi32>
    %14 = stablehlo.select %11, %13, %3 : tensor<200x1xi1>, tensor<200x1xi32>
    return %14 : tensor<200x1xi32>
  }
  func.func private @_where_233(%arg0: tensor<200x1xi1>, %arg1: tensor<i32>, %arg2: tensor<200x1xi32>) -> tensor<200x1xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x1xi1>, tensor<200x1xi32>
    return %2 : tensor<200x1xi32>
  }
  func.func private @_where_253(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_258(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_263(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @divmod_272(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_273(%arg0, %0) : (tensor<200x8xi32>, tensor<i32>) -> tensor<200x8xi32>
    %2 = call @remainder_283(%arg0, %0) : (tensor<200x8xi32>, tensor<i32>) -> tensor<200x8xi32>
    return %1, %2 : tensor<200x8xi32>, tensor<200x8xi32>
  }
  func.func private @floor_divide_273(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> tensor<200x8xi32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %1 = stablehlo.divide %arg0, %0 : tensor<200x8xi32>
    %2 = stablehlo.sign %arg0 : tensor<200x8xi32>
    %3 = stablehlo.sign %arg1 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %5 = stablehlo.compare  NE, %2, %4,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %7 = stablehlo.remainder %arg0, %6 : tensor<200x8xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %9 = stablehlo.compare  NE, %7, %8,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
    %10 = stablehlo.and %5, %9 : tensor<200x8xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %12 = stablehlo.subtract %1, %11 : tensor<200x8xi32>
    %13 = call @_where_281(%10, %12, %1) : (tensor<200x8xi1>, tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi32>
    return %13 : tensor<200x8xi32>
  }
  func.func private @_where_281(%arg0: tensor<200x8xi1>, %arg1: tensor<200x8xi32>, %arg2: tensor<200x8xi32>) -> tensor<200x8xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x8xi1>, tensor<200x8xi32>
    return %0 : tensor<200x8xi32>
  }
  func.func private @remainder_283(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> tensor<200x8xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_285(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %3 = stablehlo.remainder %arg0, %2 : tensor<200x8xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %5 = stablehlo.compare  NE, %3, %4,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %7 = stablehlo.compare  LT, %3, %6,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %1, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<200x8xi1>
    %10 = stablehlo.compare  NE, %7, %9,  UNSIGNED : (tensor<200x8xi1>, tensor<200x8xi1>) -> tensor<200x8xi1>
    %11 = stablehlo.and %10, %5 : tensor<200x8xi1>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %13 = stablehlo.add %3, %12 : tensor<200x8xi32>
    %14 = stablehlo.select %11, %13, %3 : tensor<200x8xi1>, tensor<200x8xi32>
    return %14 : tensor<200x8xi32>
  }
  func.func private @_where_285(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @divmod_289(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_290(%arg0, %0) : (tensor<200x8xi32>, tensor<i32>) -> tensor<200x8xi32>
    %2 = call @remainder_294(%arg0, %0) : (tensor<200x8xi32>, tensor<i32>) -> tensor<200x8xi32>
    return %1, %2 : tensor<200x8xi32>, tensor<200x8xi32>
  }
  func.func private @floor_divide_290(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> tensor<200x8xi32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %1 = stablehlo.divide %arg0, %0 : tensor<200x8xi32>
    %2 = stablehlo.sign %arg0 : tensor<200x8xi32>
    %3 = stablehlo.sign %arg1 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %5 = stablehlo.compare  NE, %2, %4,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %7 = stablehlo.remainder %arg0, %6 : tensor<200x8xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %9 = stablehlo.compare  NE, %7, %8,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
    %10 = stablehlo.and %5, %9 : tensor<200x8xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %12 = stablehlo.subtract %1, %11 : tensor<200x8xi32>
    %13 = call @_where_281(%10, %12, %1) : (tensor<200x8xi1>, tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi32>
    return %13 : tensor<200x8xi32>
  }
  func.func private @remainder_294(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> tensor<200x8xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_285(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %3 = stablehlo.remainder %arg0, %2 : tensor<200x8xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %5 = stablehlo.compare  NE, %3, %4,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %7 = stablehlo.compare  LT, %3, %6,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %1, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<200x8xi1>
    %10 = stablehlo.compare  NE, %7, %9,  UNSIGNED : (tensor<200x8xi1>, tensor<200x8xi1>) -> tensor<200x8xi1>
    %11 = stablehlo.and %10, %5 : tensor<200x8xi1>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %13 = stablehlo.add %3, %12 : tensor<200x8xi32>
    %14 = stablehlo.select %11, %13, %3 : tensor<200x8xi1>, tensor<200x8xi32>
    return %14 : tensor<200x8xi32>
  }
  func.func private @_where_296(%arg0: tensor<200x8xi1>, %arg1: tensor<i32>, %arg2: tensor<200x8xi32>) -> tensor<200x8xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x8xi1>, tensor<200x8xi32>
    return %2 : tensor<200x8xi32>
  }
  func.func private @clip_303(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x8xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x8xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x8xi32>
    return %5 : tensor<200x8xi32>
  }
  func.func private @_where_315(%arg0: tensor<200x8xi1>, %arg1: tensor<200x8xf32>, %arg2: tensor<i32>) -> tensor<200x8xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x8xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x8xi1>, tensor<200x8xf32>
    return %2 : tensor<200x8xf32>
  }
  func.func private @divmod_322(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_323(%arg0, %0) : (tensor<200x4xi32>, tensor<i32>) -> tensor<200x4xi32>
    %2 = call @remainder_332(%arg0, %0) : (tensor<200x4xi32>, tensor<i32>) -> tensor<200x4xi32>
    return %1, %2 : tensor<200x4xi32>, tensor<200x4xi32>
  }
  func.func private @floor_divide_323(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> tensor<200x4xi32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %1 = stablehlo.divide %arg0, %0 : tensor<200x4xi32>
    %2 = stablehlo.sign %arg0 : tensor<200x4xi32>
    %3 = stablehlo.sign %arg1 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %5 = stablehlo.compare  NE, %2, %4,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %7 = stablehlo.remainder %arg0, %6 : tensor<200x4xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %9 = stablehlo.compare  NE, %7, %8,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
    %10 = stablehlo.and %5, %9 : tensor<200x4xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %12 = stablehlo.subtract %1, %11 : tensor<200x4xi32>
    %13 = call @_where_330(%10, %12, %1) : (tensor<200x4xi1>, tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi32>
    return %13 : tensor<200x4xi32>
  }
  func.func private @_where_330(%arg0: tensor<200x4xi1>, %arg1: tensor<200x4xi32>, %arg2: tensor<200x4xi32>) -> tensor<200x4xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x4xi1>, tensor<200x4xi32>
    return %0 : tensor<200x4xi32>
  }
  func.func private @remainder_332(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> tensor<200x4xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_285(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %3 = stablehlo.remainder %arg0, %2 : tensor<200x4xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %5 = stablehlo.compare  NE, %3, %4,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %7 = stablehlo.compare  LT, %3, %6,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %1, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<200x4xi1>
    %10 = stablehlo.compare  NE, %7, %9,  UNSIGNED : (tensor<200x4xi1>, tensor<200x4xi1>) -> tensor<200x4xi1>
    %11 = stablehlo.and %10, %5 : tensor<200x4xi1>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %13 = stablehlo.add %3, %12 : tensor<200x4xi32>
    %14 = stablehlo.select %11, %13, %3 : tensor<200x4xi1>, tensor<200x4xi32>
    return %14 : tensor<200x4xi32>
  }
  func.func private @divmod_336(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_337(%arg0, %0) : (tensor<200x4xi32>, tensor<i32>) -> tensor<200x4xi32>
    %2 = call @remainder_341(%arg0, %0) : (tensor<200x4xi32>, tensor<i32>) -> tensor<200x4xi32>
    return %1, %2 : tensor<200x4xi32>, tensor<200x4xi32>
  }
  func.func private @floor_divide_337(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> tensor<200x4xi32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %1 = stablehlo.divide %arg0, %0 : tensor<200x4xi32>
    %2 = stablehlo.sign %arg0 : tensor<200x4xi32>
    %3 = stablehlo.sign %arg1 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %5 = stablehlo.compare  NE, %2, %4,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %7 = stablehlo.remainder %arg0, %6 : tensor<200x4xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %9 = stablehlo.compare  NE, %7, %8,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
    %10 = stablehlo.and %5, %9 : tensor<200x4xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %12 = stablehlo.subtract %1, %11 : tensor<200x4xi32>
    %13 = call @_where_330(%10, %12, %1) : (tensor<200x4xi1>, tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi32>
    return %13 : tensor<200x4xi32>
  }
  func.func private @remainder_341(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> tensor<200x4xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_285(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %3 = stablehlo.remainder %arg0, %2 : tensor<200x4xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %5 = stablehlo.compare  NE, %3, %4,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %7 = stablehlo.compare  LT, %3, %6,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %1, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<200x4xi1>
    %10 = stablehlo.compare  NE, %7, %9,  UNSIGNED : (tensor<200x4xi1>, tensor<200x4xi1>) -> tensor<200x4xi1>
    %11 = stablehlo.and %10, %5 : tensor<200x4xi1>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %13 = stablehlo.add %3, %12 : tensor<200x4xi32>
    %14 = stablehlo.select %11, %13, %3 : tensor<200x4xi1>, tensor<200x4xi32>
    return %14 : tensor<200x4xi32>
  }
  func.func private @_where_343(%arg0: tensor<200x4xi1>, %arg1: tensor<i32>, %arg2: tensor<200x4xi32>) -> tensor<200x4xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x4xi1>, tensor<200x4xi32>
    return %2 : tensor<200x4xi32>
  }
  func.func private @clip_350(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x4xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x4xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x4xi32>
    return %5 : tensor<200x4xi32>
  }
  func.func private @_where_361(%arg0: tensor<200x4xi1>, %arg1: tensor<200x4xf32>, %arg2: tensor<i32>) -> tensor<200x4xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x4xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x4xi1>, tensor<200x4xf32>
    return %2 : tensor<200x4xf32>
  }
  func.func private @divmod_369(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_370(%arg0, %0) : (tensor<200x2xi32>, tensor<i32>) -> tensor<200x2xi32>
    %2 = call @remainder_379(%arg0, %0) : (tensor<200x2xi32>, tensor<i32>) -> tensor<200x2xi32>
    return %1, %2 : tensor<200x2xi32>, tensor<200x2xi32>
  }
  func.func private @floor_divide_370(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> tensor<200x2xi32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %1 = stablehlo.divide %arg0, %0 : tensor<200x2xi32>
    %2 = stablehlo.sign %arg0 : tensor<200x2xi32>
    %3 = stablehlo.sign %arg1 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %5 = stablehlo.compare  NE, %2, %4,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %7 = stablehlo.remainder %arg0, %6 : tensor<200x2xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %9 = stablehlo.compare  NE, %7, %8,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
    %10 = stablehlo.and %5, %9 : tensor<200x2xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %12 = stablehlo.subtract %1, %11 : tensor<200x2xi32>
    %13 = call @_where_377(%10, %12, %1) : (tensor<200x2xi1>, tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi32>
    return %13 : tensor<200x2xi32>
  }
  func.func private @_where_377(%arg0: tensor<200x2xi1>, %arg1: tensor<200x2xi32>, %arg2: tensor<200x2xi32>) -> tensor<200x2xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x2xi1>, tensor<200x2xi32>
    return %0 : tensor<200x2xi32>
  }
  func.func private @remainder_379(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> tensor<200x2xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_285(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %3 = stablehlo.remainder %arg0, %2 : tensor<200x2xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %5 = stablehlo.compare  NE, %3, %4,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %7 = stablehlo.compare  LT, %3, %6,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %1, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<200x2xi1>
    %10 = stablehlo.compare  NE, %7, %9,  UNSIGNED : (tensor<200x2xi1>, tensor<200x2xi1>) -> tensor<200x2xi1>
    %11 = stablehlo.and %10, %5 : tensor<200x2xi1>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %13 = stablehlo.add %3, %12 : tensor<200x2xi32>
    %14 = stablehlo.select %11, %13, %3 : tensor<200x2xi1>, tensor<200x2xi32>
    return %14 : tensor<200x2xi32>
  }
  func.func private @divmod_383(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_384(%arg0, %0) : (tensor<200x2xi32>, tensor<i32>) -> tensor<200x2xi32>
    %2 = call @remainder_388(%arg0, %0) : (tensor<200x2xi32>, tensor<i32>) -> tensor<200x2xi32>
    return %1, %2 : tensor<200x2xi32>, tensor<200x2xi32>
  }
  func.func private @floor_divide_384(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> tensor<200x2xi32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %1 = stablehlo.divide %arg0, %0 : tensor<200x2xi32>
    %2 = stablehlo.sign %arg0 : tensor<200x2xi32>
    %3 = stablehlo.sign %arg1 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %5 = stablehlo.compare  NE, %2, %4,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %7 = stablehlo.remainder %arg0, %6 : tensor<200x2xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %9 = stablehlo.compare  NE, %7, %8,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
    %10 = stablehlo.and %5, %9 : tensor<200x2xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %12 = stablehlo.subtract %1, %11 : tensor<200x2xi32>
    %13 = call @_where_377(%10, %12, %1) : (tensor<200x2xi1>, tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi32>
    return %13 : tensor<200x2xi32>
  }
  func.func private @remainder_388(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> tensor<200x2xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_285(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %3 = stablehlo.remainder %arg0, %2 : tensor<200x2xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %5 = stablehlo.compare  NE, %3, %4,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %7 = stablehlo.compare  LT, %3, %6,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %1, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<200x2xi1>
    %10 = stablehlo.compare  NE, %7, %9,  UNSIGNED : (tensor<200x2xi1>, tensor<200x2xi1>) -> tensor<200x2xi1>
    %11 = stablehlo.and %10, %5 : tensor<200x2xi1>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %13 = stablehlo.add %3, %12 : tensor<200x2xi32>
    %14 = stablehlo.select %11, %13, %3 : tensor<200x2xi1>, tensor<200x2xi32>
    return %14 : tensor<200x2xi32>
  }
  func.func private @_where_390(%arg0: tensor<200x2xi1>, %arg1: tensor<i32>, %arg2: tensor<200x2xi32>) -> tensor<200x2xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x2xi1>, tensor<200x2xi32>
    return %2 : tensor<200x2xi32>
  }
  func.func private @clip_397(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x2xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x2xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x2xi32>
    return %5 : tensor<200x2xi32>
  }
  func.func private @_where_408(%arg0: tensor<200x2xi1>, %arg1: tensor<200x2xf32>, %arg2: tensor<i32>) -> tensor<200x2xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x2xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x2xi1>, tensor<200x2xf32>
    return %2 : tensor<200x2xf32>
  }
  func.func private @clip_425(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg1 : tensor<f32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<f32>
    %2 = stablehlo.convert %arg2 : tensor<f32>
    %3 = stablehlo.minimum %2, %1 : tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @_where_432(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @_where_435(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @divmod_452(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_453(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    %2 = call @remainder_462(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    return %1, %2 : tensor<200x1xi32>, tensor<200x1xi32>
  }
  func.func private @floor_divide_453(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %1 = stablehlo.divide %arg0, %0 : tensor<200x1xi32>
    %2 = stablehlo.sign %arg0 : tensor<200x1xi32>
    %3 = stablehlo.sign %arg1 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %5 = stablehlo.compare  NE, %2, %4,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %7 = stablehlo.remainder %arg0, %6 : tensor<200x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %9 = stablehlo.compare  NE, %7, %8,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %10 = stablehlo.and %5, %9 : tensor<200x1xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %12 = stablehlo.subtract %1, %11 : tensor<200x1xi32>
    %13 = call @_where_460(%10, %12, %1) : (tensor<200x1xi1>, tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi32>
    return %13 : tensor<200x1xi32>
  }
  func.func private @_where_460(%arg0: tensor<200x1xi1>, %arg1: tensor<200x1xi32>, %arg2: tensor<200x1xi32>) -> tensor<200x1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x1xi1>, tensor<200x1xi32>
    return %0 : tensor<200x1xi32>
  }
  func.func private @remainder_462(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_285(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %3 = stablehlo.remainder %arg0, %2 : tensor<200x1xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %5 = stablehlo.compare  NE, %3, %4,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %7 = stablehlo.compare  LT, %3, %6,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %1, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<200x1xi1>
    %10 = stablehlo.compare  NE, %7, %9,  UNSIGNED : (tensor<200x1xi1>, tensor<200x1xi1>) -> tensor<200x1xi1>
    %11 = stablehlo.and %10, %5 : tensor<200x1xi1>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %13 = stablehlo.add %3, %12 : tensor<200x1xi32>
    %14 = stablehlo.select %11, %13, %3 : tensor<200x1xi1>, tensor<200x1xi32>
    return %14 : tensor<200x1xi32>
  }
  func.func private @divmod_466(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_467(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    %2 = call @remainder_471(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    return %1, %2 : tensor<200x1xi32>, tensor<200x1xi32>
  }
  func.func private @floor_divide_467(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %1 = stablehlo.divide %arg0, %0 : tensor<200x1xi32>
    %2 = stablehlo.sign %arg0 : tensor<200x1xi32>
    %3 = stablehlo.sign %arg1 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %5 = stablehlo.compare  NE, %2, %4,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %7 = stablehlo.remainder %arg0, %6 : tensor<200x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %9 = stablehlo.compare  NE, %7, %8,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %10 = stablehlo.and %5, %9 : tensor<200x1xi1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %12 = stablehlo.subtract %1, %11 : tensor<200x1xi32>
    %13 = call @_where_460(%10, %12, %1) : (tensor<200x1xi1>, tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi32>
    return %13 : tensor<200x1xi32>
  }
  func.func private @remainder_471(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_285(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %3 = stablehlo.remainder %arg0, %2 : tensor<200x1xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %5 = stablehlo.compare  NE, %3, %4,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %7 = stablehlo.compare  LT, %3, %6,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %1, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<200x1xi1>
    %10 = stablehlo.compare  NE, %7, %9,  UNSIGNED : (tensor<200x1xi1>, tensor<200x1xi1>) -> tensor<200x1xi1>
    %11 = stablehlo.and %10, %5 : tensor<200x1xi1>
    %12 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %13 = stablehlo.add %3, %12 : tensor<200x1xi32>
    %14 = stablehlo.select %11, %13, %3 : tensor<200x1xi1>, tensor<200x1xi32>
    return %14 : tensor<200x1xi32>
  }
  func.func private @_where_473(%arg0: tensor<200x1xi1>, %arg1: tensor<i32>, %arg2: tensor<200x1xi32>) -> tensor<200x1xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x1xi1>, tensor<200x1xi32>
    return %2 : tensor<200x1xi32>
  }
  func.func private @clip_480(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x1xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x1xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x1xi32>
    return %5 : tensor<200x1xi32>
  }
  func.func private @_where_491(%arg0: tensor<200x1xi1>, %arg1: tensor<200x1xf32>, %arg2: tensor<i32>) -> tensor<200x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x1xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x1xi1>, tensor<200x1xf32>
    return %2 : tensor<200x1xf32>
  }
}
