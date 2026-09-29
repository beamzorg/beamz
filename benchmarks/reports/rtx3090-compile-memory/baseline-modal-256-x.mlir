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
    %c_19 = stablehlo.constant dense<[1, 2, 0]> : tensor<3xi32>
    %c_20 = stablehlo.constant dense<[2, 3, 1, 5, 6, 4]> : tensor<6xi32>
    %c_21 = stablehlo.constant dense<[2, 0, 1]> : tensor<3xi32>
    %c_22 = stablehlo.constant dense_resource<__elided__> : tensor<6x5xi32>
    %cst_23 = stablehlo.constant dense_resource<__elided__> : tensor<2x14x25x1xf32>
    %cst_24 = stablehlo.constant dense_resource<__elided__> : tensor<2x32xf32>
    %cst_25 = stablehlo.constant dense_resource<__elided__> : tensor<2x15x24x1xf32>
    %cst_26 = stablehlo.constant dense_resource<__elided__> : tensor<2x32xf32>
    %c_27 = stablehlo.constant dense_resource<__elided__> : tensor<6x3xi32>
    %cst_28 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_29 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_30 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_31 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_32 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_33 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_34 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_35 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_36 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_37 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_38 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_39 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_40 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_41 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_42 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_43 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_44 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_45 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %c_46 = stablehlo.constant dense<[1, 2, 0]> : tensor<3xi32>
    %c_47 = stablehlo.constant dense<[2, 3, 1, 5, 6, 4]> : tensor<6xi32>
    %c_48 = stablehlo.constant dense<[2, 0, 1]> : tensor<3xi32>
    %c_49 = stablehlo.constant dense_resource<__elided__> : tensor<6x5xi32>
    %cst_50 = stablehlo.constant dense_resource<__elided__> : tensor<2x15x26x1xf32>
    %cst_51 = stablehlo.constant dense_resource<__elided__> : tensor<2x32xf32>
    %cst_52 = stablehlo.constant dense_resource<__elided__> : tensor<2x16x25x1xf32>
    %cst_53 = stablehlo.constant dense_resource<__elided__> : tensor<2x32xf32>
    %c_54 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_56 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_57 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_58 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_59 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_60 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_62 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_63 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_64 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_66 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %cst_67 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xf32>
    %c_68 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_69 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %c_70 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_71 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %c_72 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %cst_73 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xf32>
    %c_74 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_75 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %c_76 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_77 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %cst_78 = stablehlo.constant dense<[1.87612057E+14, 1.9341449E+14, 1.99216924E+14]> : tensor<3xf32>
    %cst_79 = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
    %c_80 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_82 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_84 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_85 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_86 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_88 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_90 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_91 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_92 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %cst_93 = stablehlo.constant dense<1.250000e-01> : tensor<200x8xf32>
    %c_94 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %cst_95 = stablehlo.constant dense<5.000000e-01> : tensor<200x2xf32>
    %c_96 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %cst_97 = stablehlo.constant dense<5.000000e-01> : tensor<200x2xf32>
    %c_98 = stablehlo.constant dense_resource<__elided__> : tensor<200x1xi32>
    %cst_99 = stablehlo.constant dense<1.000000e+00> : tensor<200x1xf32>
    %c_100 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_101 = stablehlo.constant dense<2.500000e-01> : tensor<200x4xf32>
    %c_102 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_103 = stablehlo.constant dense<2.500000e-01> : tensor<200x4xf32>
    %cst_104 = stablehlo.constant dense<[1.87612057E+14, 1.9341449E+14, 1.99216924E+14]> : tensor<3xf32>
    %cst_105 = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
    %c_106 = stablehlo.constant dense<1> : tensor<ui32>
    %c_107 = stablehlo.constant dense<1> : tensor<ui32>
    %0 = stablehlo.partition_id : tensor<ui32>
    %1 = stablehlo.divide %0, %c_106 : tensor<ui32>
    %2 = stablehlo.remainder %1, %c_107 : tensor<ui32>
    %3 = stablehlo.convert %2 : (tensor<ui32>) -> tensor<i32>
    %c_108 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.compare  EQ, %3, %c_108,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_109 = stablehlo.constant dense<0> : tensor<i32>
    %5 = call @_where(%4, %arg25, %c_109) : (tensor<i1>, tensor<7200xf32>, tensor<i32>) -> tensor<7200xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
    %c_110 = stablehlo.constant dense<1> : tensor<ui32>
    %c_111 = stablehlo.constant dense<1> : tensor<ui32>
    %7 = stablehlo.partition_id : tensor<ui32>
    %8 = stablehlo.divide %7, %c_110 : tensor<ui32>
    %9 = stablehlo.remainder %8, %c_111 : tensor<ui32>
    %10 = stablehlo.convert %9 : (tensor<ui32>) -> tensor<i32>
    %c_112 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.compare  EQ, %10, %c_112,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_113 = stablehlo.constant dense<0> : tensor<i32>
    %12 = call @_where(%11, %arg26, %c_113) : (tensor<i1>, tensor<7200xf32>, tensor<i32>) -> tensor<7200xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
    %c_114 = stablehlo.constant dense<1> : tensor<ui32>
    %c_115 = stablehlo.constant dense<1> : tensor<ui32>
    %14 = stablehlo.partition_id : tensor<ui32>
    %15 = stablehlo.divide %14, %c_114 : tensor<ui32>
    %16 = stablehlo.remainder %15, %c_115 : tensor<ui32>
    %17 = stablehlo.convert %16 : (tensor<ui32>) -> tensor<i32>
    %c_116 = stablehlo.constant dense<257> : tensor<i32>
    %18 = stablehlo.multiply %17, %c_116 : tensor<i32>
    %19 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_117 = stablehlo.constant dense<0> : tensor<i32>
    %20 = stablehlo.broadcast_in_dim %c_117, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %21 = stablehlo.compare  GE, %19, %20,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_118 = stablehlo.constant dense<256> : tensor<i32>
    %22 = stablehlo.broadcast_in_dim %c_118, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %23 = stablehlo.compare  LT, %19, %22,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %24 = stablehlo.and %21, %23 : tensor<256xi1>
    %25 = stablehlo.reshape %24 : (tensor<256xi1>) -> tensor<256x1x1xi1>
    %c_119 = stablehlo.constant dense<true> : tensor<i1>
    %26 = stablehlo.broadcast_in_dim %c_119, dims = [] : (tensor<i1>) -> tensor<256x1x1xi1>
    %27 = stablehlo.and %26, %25 : tensor<256x1x1xi1>
    %28 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_120 = stablehlo.constant dense<0> : tensor<i32>
    %29 = stablehlo.broadcast_in_dim %c_120, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %30 = stablehlo.compare  GE, %28, %29,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_121 = stablehlo.constant dense<257> : tensor<i32>
    %31 = stablehlo.broadcast_in_dim %c_121, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %32 = stablehlo.compare  LT, %28, %31,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %33 = stablehlo.and %30, %32 : tensor<257xi1>
    %34 = stablehlo.reshape %33 : (tensor<257xi1>) -> tensor<1x257x1xi1>
    %35 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2] : (tensor<256x1x1xi1>) -> tensor<256x257x1xi1>
    %36 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2] : (tensor<1x257x1xi1>) -> tensor<256x257x1xi1>
    %37 = stablehlo.and %35, %36 : tensor<256x257x1xi1>
    %38 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_122 = stablehlo.constant dense<12> : tensor<i32>
    %39 = stablehlo.broadcast_in_dim %c_122, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %40 = stablehlo.compare  LT, %38, %39,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_123 = stablehlo.constant dense<12> : tensor<i32>
    %41 = stablehlo.broadcast_in_dim %c_123, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %42 = stablehlo.subtract %38, %41 : tensor<24xi32>
    %c_124 = stablehlo.constant dense<256> : tensor<i32>
    %43 = stablehlo.broadcast_in_dim %c_124, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %44 = stablehlo.add %42, %43 : tensor<24xi32>
    %c_125 = stablehlo.constant dense<12> : tensor<i32>
    %45 = stablehlo.broadcast_in_dim %c_125, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %46 = stablehlo.subtract %44, %45 : tensor<24xi32>
    %47 = call @_where_13(%40, %38, %46) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_126 = stablehlo.constant dense<0> : tensor<i32>
    %48 = stablehlo.broadcast_in_dim %c_126, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %49 = stablehlo.compare  GE, %47, %48,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_127 = stablehlo.constant dense<256> : tensor<i32>
    %50 = stablehlo.broadcast_in_dim %c_127, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %51 = stablehlo.compare  LT, %47, %50,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %52 = stablehlo.and %49, %51 : tensor<24xi1>
    %53 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %54 = stablehlo.compare  GE, %47, %53,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_128 = stablehlo.constant dense<257> : tensor<i32>
    %55 = stablehlo.add %18, %c_128 : tensor<i32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %57 = stablehlo.compare  LT, %47, %56,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %58 = stablehlo.and %54, %57 : tensor<24xi1>
    %59 = stablehlo.and %52, %58 : tensor<24xi1>
    %60 = stablehlo.reshape %59 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %61 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2] : (tensor<256x257x1xi1>) -> tensor<256x257x24xi1>
    %62 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<256x257x24xi1>
    %63 = stablehlo.and %61, %62 : tensor<256x257x24xi1>
    %cst_129 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<f32>) -> tensor<256x257x24xf32>
    %65 = call @_where_27(%63, %arg9, %64) : (tensor<256x257x24xi1>, tensor<256x257x24xf32>, tensor<256x257x24xf32>) -> tensor<256x257x24xf32>
    %66 = stablehlo.broadcast_in_dim %65, dims = [1, 2, 3] : (tensor<256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %c_130 = stablehlo.constant dense<1> : tensor<ui32>
    %c_131 = stablehlo.constant dense<1> : tensor<ui32>
    %67 = stablehlo.partition_id : tensor<ui32>
    %68 = stablehlo.divide %67, %c_130 : tensor<ui32>
    %69 = stablehlo.remainder %68, %c_131 : tensor<ui32>
    %70 = stablehlo.convert %69 : (tensor<ui32>) -> tensor<i32>
    %c_132 = stablehlo.constant dense<257> : tensor<i32>
    %71 = stablehlo.multiply %70, %c_132 : tensor<i32>
    %72 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_133 = stablehlo.constant dense<0> : tensor<i32>
    %73 = stablehlo.broadcast_in_dim %c_133, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %74 = stablehlo.compare  GE, %72, %73,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_134 = stablehlo.constant dense<257> : tensor<i32>
    %75 = stablehlo.broadcast_in_dim %c_134, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %76 = stablehlo.compare  LT, %72, %75,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %77 = stablehlo.and %74, %76 : tensor<257xi1>
    %78 = stablehlo.reshape %77 : (tensor<257xi1>) -> tensor<257x1x1xi1>
    %c_135 = stablehlo.constant dense<true> : tensor<i1>
    %79 = stablehlo.broadcast_in_dim %c_135, dims = [] : (tensor<i1>) -> tensor<257x1x1xi1>
    %80 = stablehlo.and %79, %78 : tensor<257x1x1xi1>
    %81 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_136 = stablehlo.constant dense<0> : tensor<i32>
    %82 = stablehlo.broadcast_in_dim %c_136, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %83 = stablehlo.compare  GE, %81, %82,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_137 = stablehlo.constant dense<256> : tensor<i32>
    %84 = stablehlo.broadcast_in_dim %c_137, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %85 = stablehlo.compare  LT, %81, %84,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %86 = stablehlo.and %83, %85 : tensor<256xi1>
    %87 = stablehlo.reshape %86 : (tensor<256xi1>) -> tensor<1x256x1xi1>
    %88 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2] : (tensor<257x1x1xi1>) -> tensor<257x256x1xi1>
    %89 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 2] : (tensor<1x256x1xi1>) -> tensor<257x256x1xi1>
    %90 = stablehlo.and %88, %89 : tensor<257x256x1xi1>
    %91 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_138 = stablehlo.constant dense<12> : tensor<i32>
    %92 = stablehlo.broadcast_in_dim %c_138, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %93 = stablehlo.compare  LT, %91, %92,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_139 = stablehlo.constant dense<12> : tensor<i32>
    %94 = stablehlo.broadcast_in_dim %c_139, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %95 = stablehlo.subtract %91, %94 : tensor<24xi32>
    %c_140 = stablehlo.constant dense<256> : tensor<i32>
    %96 = stablehlo.broadcast_in_dim %c_140, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %97 = stablehlo.add %95, %96 : tensor<24xi32>
    %c_141 = stablehlo.constant dense<12> : tensor<i32>
    %98 = stablehlo.broadcast_in_dim %c_141, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %99 = stablehlo.subtract %97, %98 : tensor<24xi32>
    %100 = call @_where_13(%93, %91, %99) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_142 = stablehlo.constant dense<0> : tensor<i32>
    %101 = stablehlo.broadcast_in_dim %c_142, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %102 = stablehlo.compare  GE, %100, %101,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_143 = stablehlo.constant dense<256> : tensor<i32>
    %103 = stablehlo.broadcast_in_dim %c_143, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %104 = stablehlo.compare  LT, %100, %103,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %105 = stablehlo.and %102, %104 : tensor<24xi1>
    %106 = stablehlo.broadcast_in_dim %71, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %107 = stablehlo.compare  GE, %100, %106,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_144 = stablehlo.constant dense<257> : tensor<i32>
    %108 = stablehlo.add %71, %c_144 : tensor<i32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %110 = stablehlo.compare  LT, %100, %109,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %111 = stablehlo.and %107, %110 : tensor<24xi1>
    %112 = stablehlo.and %105, %111 : tensor<24xi1>
    %113 = stablehlo.reshape %112 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %114 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<257x256x1xi1>) -> tensor<257x256x24xi1>
    %115 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<257x256x24xi1>
    %116 = stablehlo.and %114, %115 : tensor<257x256x24xi1>
    %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %117 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f32>) -> tensor<257x256x24xf32>
    %118 = call @_where_39(%116, %arg10, %117) : (tensor<257x256x24xi1>, tensor<257x256x24xf32>, tensor<257x256x24xf32>) -> tensor<257x256x24xf32>
    %119 = stablehlo.broadcast_in_dim %118, dims = [1, 2, 3] : (tensor<257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %c_146 = stablehlo.constant dense<1> : tensor<ui32>
    %c_147 = stablehlo.constant dense<1> : tensor<ui32>
    %120 = stablehlo.partition_id : tensor<ui32>
    %121 = stablehlo.divide %120, %c_146 : tensor<ui32>
    %122 = stablehlo.remainder %121, %c_147 : tensor<ui32>
    %123 = stablehlo.convert %122 : (tensor<ui32>) -> tensor<i32>
    %c_148 = stablehlo.constant dense<257> : tensor<i32>
    %124 = stablehlo.multiply %123, %c_148 : tensor<i32>
    %125 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_149 = stablehlo.constant dense<0> : tensor<i32>
    %126 = stablehlo.broadcast_in_dim %c_149, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %127 = stablehlo.compare  GE, %125, %126,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_150 = stablehlo.constant dense<257> : tensor<i32>
    %128 = stablehlo.broadcast_in_dim %c_150, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %129 = stablehlo.compare  LT, %125, %128,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %130 = stablehlo.and %127, %129 : tensor<257xi1>
    %131 = stablehlo.reshape %130 : (tensor<257xi1>) -> tensor<257x1x1xi1>
    %c_151 = stablehlo.constant dense<true> : tensor<i1>
    %132 = stablehlo.broadcast_in_dim %c_151, dims = [] : (tensor<i1>) -> tensor<257x1x1xi1>
    %133 = stablehlo.and %132, %131 : tensor<257x1x1xi1>
    %134 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_152 = stablehlo.constant dense<0> : tensor<i32>
    %135 = stablehlo.broadcast_in_dim %c_152, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %136 = stablehlo.compare  GE, %134, %135,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_153 = stablehlo.constant dense<256> : tensor<i32>
    %137 = stablehlo.broadcast_in_dim %c_153, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %138 = stablehlo.compare  LT, %134, %137,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %139 = stablehlo.and %136, %138 : tensor<256xi1>
    %140 = stablehlo.reshape %139 : (tensor<256xi1>) -> tensor<1x256x1xi1>
    %141 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2] : (tensor<257x1x1xi1>) -> tensor<257x256x1xi1>
    %142 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2] : (tensor<1x256x1xi1>) -> tensor<257x256x1xi1>
    %143 = stablehlo.and %141, %142 : tensor<257x256x1xi1>
    %144 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_154 = stablehlo.constant dense<12> : tensor<i32>
    %145 = stablehlo.broadcast_in_dim %c_154, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %146 = stablehlo.compare  LT, %144, %145,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_155 = stablehlo.constant dense<12> : tensor<i32>
    %147 = stablehlo.broadcast_in_dim %c_155, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %148 = stablehlo.subtract %144, %147 : tensor<24xi32>
    %c_156 = stablehlo.constant dense<257> : tensor<i32>
    %149 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %150 = stablehlo.add %148, %149 : tensor<24xi32>
    %c_157 = stablehlo.constant dense<12> : tensor<i32>
    %151 = stablehlo.broadcast_in_dim %c_157, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %152 = stablehlo.subtract %150, %151 : tensor<24xi32>
    %153 = call @_where_13(%146, %144, %152) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_158 = stablehlo.constant dense<0> : tensor<i32>
    %154 = stablehlo.broadcast_in_dim %c_158, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %155 = stablehlo.compare  GE, %153, %154,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_159 = stablehlo.constant dense<257> : tensor<i32>
    %156 = stablehlo.broadcast_in_dim %c_159, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %157 = stablehlo.compare  LT, %153, %156,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %158 = stablehlo.and %155, %157 : tensor<24xi1>
    %159 = stablehlo.broadcast_in_dim %124, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %160 = stablehlo.compare  GE, %153, %159,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_160 = stablehlo.constant dense<257> : tensor<i32>
    %161 = stablehlo.add %124, %c_160 : tensor<i32>
    %162 = stablehlo.broadcast_in_dim %161, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %163 = stablehlo.compare  LT, %153, %162,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %164 = stablehlo.and %160, %163 : tensor<24xi1>
    %165 = stablehlo.and %158, %164 : tensor<24xi1>
    %166 = stablehlo.reshape %165 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %167 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2] : (tensor<257x256x1xi1>) -> tensor<257x256x24xi1>
    %168 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<257x256x24xi1>
    %169 = stablehlo.and %167, %168 : tensor<257x256x24xi1>
    %cst_161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %170 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f32>) -> tensor<257x256x24xf32>
    %171 = call @_where_39(%169, %arg15, %170) : (tensor<257x256x24xi1>, tensor<257x256x24xf32>, tensor<257x256x24xf32>) -> tensor<257x256x24xf32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [1, 2, 3] : (tensor<257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %c_162 = stablehlo.constant dense<1> : tensor<ui32>
    %c_163 = stablehlo.constant dense<1> : tensor<ui32>
    %173 = stablehlo.partition_id : tensor<ui32>
    %174 = stablehlo.divide %173, %c_162 : tensor<ui32>
    %175 = stablehlo.remainder %174, %c_163 : tensor<ui32>
    %176 = stablehlo.convert %175 : (tensor<ui32>) -> tensor<i32>
    %c_164 = stablehlo.constant dense<257> : tensor<i32>
    %177 = stablehlo.multiply %176, %c_164 : tensor<i32>
    %178 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_165 = stablehlo.constant dense<0> : tensor<i32>
    %179 = stablehlo.broadcast_in_dim %c_165, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %180 = stablehlo.compare  GE, %178, %179,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_166 = stablehlo.constant dense<256> : tensor<i32>
    %181 = stablehlo.broadcast_in_dim %c_166, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %182 = stablehlo.compare  LT, %178, %181,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %183 = stablehlo.and %180, %182 : tensor<256xi1>
    %184 = stablehlo.reshape %183 : (tensor<256xi1>) -> tensor<256x1x1xi1>
    %c_167 = stablehlo.constant dense<true> : tensor<i1>
    %185 = stablehlo.broadcast_in_dim %c_167, dims = [] : (tensor<i1>) -> tensor<256x1x1xi1>
    %186 = stablehlo.and %185, %184 : tensor<256x1x1xi1>
    %187 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_168 = stablehlo.constant dense<0> : tensor<i32>
    %188 = stablehlo.broadcast_in_dim %c_168, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %189 = stablehlo.compare  GE, %187, %188,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_169 = stablehlo.constant dense<257> : tensor<i32>
    %190 = stablehlo.broadcast_in_dim %c_169, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %191 = stablehlo.compare  LT, %187, %190,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %192 = stablehlo.and %189, %191 : tensor<257xi1>
    %193 = stablehlo.reshape %192 : (tensor<257xi1>) -> tensor<1x257x1xi1>
    %194 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2] : (tensor<256x1x1xi1>) -> tensor<256x257x1xi1>
    %195 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2] : (tensor<1x257x1xi1>) -> tensor<256x257x1xi1>
    %196 = stablehlo.and %194, %195 : tensor<256x257x1xi1>
    %197 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_170 = stablehlo.constant dense<12> : tensor<i32>
    %198 = stablehlo.broadcast_in_dim %c_170, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %199 = stablehlo.compare  LT, %197, %198,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_171 = stablehlo.constant dense<12> : tensor<i32>
    %200 = stablehlo.broadcast_in_dim %c_171, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %201 = stablehlo.subtract %197, %200 : tensor<24xi32>
    %c_172 = stablehlo.constant dense<257> : tensor<i32>
    %202 = stablehlo.broadcast_in_dim %c_172, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %203 = stablehlo.add %201, %202 : tensor<24xi32>
    %c_173 = stablehlo.constant dense<12> : tensor<i32>
    %204 = stablehlo.broadcast_in_dim %c_173, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %205 = stablehlo.subtract %203, %204 : tensor<24xi32>
    %206 = call @_where_13(%199, %197, %205) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_174 = stablehlo.constant dense<0> : tensor<i32>
    %207 = stablehlo.broadcast_in_dim %c_174, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %208 = stablehlo.compare  GE, %206, %207,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_175 = stablehlo.constant dense<257> : tensor<i32>
    %209 = stablehlo.broadcast_in_dim %c_175, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %210 = stablehlo.compare  LT, %206, %209,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %211 = stablehlo.and %208, %210 : tensor<24xi1>
    %212 = stablehlo.broadcast_in_dim %177, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %213 = stablehlo.compare  GE, %206, %212,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_176 = stablehlo.constant dense<257> : tensor<i32>
    %214 = stablehlo.add %177, %c_176 : tensor<i32>
    %215 = stablehlo.broadcast_in_dim %214, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %216 = stablehlo.compare  LT, %206, %215,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %217 = stablehlo.and %213, %216 : tensor<24xi1>
    %218 = stablehlo.and %211, %217 : tensor<24xi1>
    %219 = stablehlo.reshape %218 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %220 = stablehlo.broadcast_in_dim %196, dims = [0, 1, 2] : (tensor<256x257x1xi1>) -> tensor<256x257x24xi1>
    %221 = stablehlo.broadcast_in_dim %219, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<256x257x24xi1>
    %222 = stablehlo.and %220, %221 : tensor<256x257x24xi1>
    %cst_177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %223 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<f32>) -> tensor<256x257x24xf32>
    %224 = call @_where_27(%222, %arg16, %223) : (tensor<256x257x24xi1>, tensor<256x257x24xf32>, tensor<256x257x24xf32>) -> tensor<256x257x24xf32>
    %225 = stablehlo.broadcast_in_dim %224, dims = [1, 2, 3] : (tensor<256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %226 = stablehlo.iota dim = 0 : tensor<32xi32>
    %c_178 = stablehlo.constant dense<0> : tensor<i32>
    %227:149 = stablehlo.while(%iterArg = %226, %iterArg_179 = %cst, %iterArg_180 = %arg28, %iterArg_181 = %c, %iterArg_182 = %arg30, %iterArg_183 = %arg32, %iterArg_184 = %arg34, %iterArg_185 = %arg31, %iterArg_186 = %arg33, %iterArg_187 = %arg35, %iterArg_188 = %cst_0, %iterArg_189 = %cst_1, %iterArg_190 = %cst_2, %iterArg_191 = %cst_3, %iterArg_192 = %cst_4, %iterArg_193 = %cst_5, %iterArg_194 = %cst_6, %iterArg_195 = %cst_7, %iterArg_196 = %cst_8, %iterArg_197 = %cst_9, %iterArg_198 = %cst_10, %iterArg_199 = %cst_11, %iterArg_200 = %cst_12, %iterArg_201 = %cst_13, %iterArg_202 = %cst_14, %iterArg_203 = %cst_15, %iterArg_204 = %cst_16, %iterArg_205 = %cst_17, %iterArg_206 = %cst_18, %iterArg_207 = %c_19, %iterArg_208 = %c_20, %iterArg_209 = %c_21, %iterArg_210 = %c_22, %iterArg_211 = %cst_23, %iterArg_212 = %cst_24, %iterArg_213 = %cst_25, %iterArg_214 = %cst_26, %iterArg_215 = %c_27, %iterArg_216 = %arg36, %iterArg_217 = %arg38, %iterArg_218 = %arg40, %iterArg_219 = %arg37, %iterArg_220 = %arg39, %iterArg_221 = %arg41, %iterArg_222 = %cst_28, %iterArg_223 = %cst_29, %iterArg_224 = %cst_30, %iterArg_225 = %cst_31, %iterArg_226 = %cst_32, %iterArg_227 = %cst_33, %iterArg_228 = %cst_34, %iterArg_229 = %cst_35, %iterArg_230 = %cst_36, %iterArg_231 = %cst_37, %iterArg_232 = %cst_38, %iterArg_233 = %cst_39, %iterArg_234 = %cst_40, %iterArg_235 = %cst_41, %iterArg_236 = %cst_42, %iterArg_237 = %cst_43, %iterArg_238 = %cst_44, %iterArg_239 = %cst_45, %iterArg_240 = %c_46, %iterArg_241 = %c_47, %iterArg_242 = %c_48, %iterArg_243 = %c_49, %iterArg_244 = %cst_50, %iterArg_245 = %cst_51, %iterArg_246 = %cst_52, %iterArg_247 = %cst_53, %iterArg_248 = %c_54, %iterArg_249 = %cst_55, %iterArg_250 = %c_56, %iterArg_251 = %cst_57, %iterArg_252 = %c_58, %iterArg_253 = %cst_59, %iterArg_254 = %c_60, %iterArg_255 = %cst_61, %iterArg_256 = %c_62, %iterArg_257 = %cst_63, %iterArg_258 = %c_64, %iterArg_259 = %cst_65, %iterArg_260 = %c_66, %iterArg_261 = %cst_67, %iterArg_262 = %c_68, %iterArg_263 = %cst_69, %iterArg_264 = %c_70, %iterArg_265 = %cst_71, %iterArg_266 = %c_72, %iterArg_267 = %cst_73, %iterArg_268 = %c_74, %iterArg_269 = %cst_75, %iterArg_270 = %c_76, %iterArg_271 = %cst_77, %iterArg_272 = %cst_78, %iterArg_273 = %cst_79, %iterArg_274 = %c_80, %iterArg_275 = %cst_81, %iterArg_276 = %c_82, %iterArg_277 = %cst_83, %iterArg_278 = %c_84, %iterArg_279 = %cst_85, %iterArg_280 = %c_86, %iterArg_281 = %cst_87, %iterArg_282 = %c_88, %iterArg_283 = %cst_89, %iterArg_284 = %c_90, %iterArg_285 = %cst_91, %iterArg_286 = %c_92, %iterArg_287 = %cst_93, %iterArg_288 = %c_94, %iterArg_289 = %cst_95, %iterArg_290 = %c_96, %iterArg_291 = %cst_97, %iterArg_292 = %c_98, %iterArg_293 = %cst_99, %iterArg_294 = %c_100, %iterArg_295 = %cst_101, %iterArg_296 = %c_102, %iterArg_297 = %cst_103, %iterArg_298 = %cst_104, %iterArg_299 = %cst_105, %iterArg_300 = %c_178, %iterArg_301 = %arg0, %iterArg_302 = %arg1, %iterArg_303 = %arg2, %iterArg_304 = %arg3, %iterArg_305 = %arg4, %iterArg_306 = %arg5, %iterArg_307 = %arg6, %iterArg_308 = %arg7, %iterArg_309 = %arg8, %iterArg_310 = %66, %iterArg_311 = %119, %iterArg_312 = %arg11, %iterArg_313 = %arg12, %iterArg_314 = %arg13, %iterArg_315 = %arg14, %iterArg_316 = %172, %iterArg_317 = %225, %iterArg_318 = %arg17, %iterArg_319 = %arg18, %iterArg_320 = %arg19, %iterArg_321 = %arg20, %iterArg_322 = %6, %iterArg_323 = %13, %iterArg_324 = %arg27, %iterArg_325 = %arg28, %iterArg_326 = %arg29) : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<2x14x25x1xf32>, tensor<2x32xf32>, tensor<2x15x24x1xf32>, tensor<2x32xf32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<2x15x26x1xf32>, tensor<2x32xf32>, tensor<2x16x25x1xf32>, tensor<2x32xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<i32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
    cond {
      %c_327 = stablehlo.constant dense<32> : tensor<i32>
      %246 = stablehlo.compare  LT, %iterArg_300, %c_327,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %246 : tensor<i1>
    } do {
      %246 = stablehlo.dynamic_slice %iterArg, %iterArg_300, sizes = [1] : (tensor<32xi32>, tensor<i32>) -> tensor<1xi32>
      %247 = stablehlo.reshape %246 : (tensor<1xi32>) -> tensor<i32>
      %248:26 = func.call @closed_call(%iterArg_179, %iterArg_180, %iterArg_181, %iterArg_182, %iterArg_183, %iterArg_184, %iterArg_185, %iterArg_186, %iterArg_187, %iterArg_188, %iterArg_189, %iterArg_190, %iterArg_191, %iterArg_192, %iterArg_193, %iterArg_194, %iterArg_195, %iterArg_196, %iterArg_197, %iterArg_198, %iterArg_199, %iterArg_200, %iterArg_201, %iterArg_202, %iterArg_203, %iterArg_204, %iterArg_205, %iterArg_206, %iterArg_207, %iterArg_208, %iterArg_209, %iterArg_210, %iterArg_211, %iterArg_212, %iterArg_213, %iterArg_214, %iterArg_215, %iterArg_216, %iterArg_217, %iterArg_218, %iterArg_219, %iterArg_220, %iterArg_221, %iterArg_222, %iterArg_223, %iterArg_224, %iterArg_225, %iterArg_226, %iterArg_227, %iterArg_228, %iterArg_229, %iterArg_230, %iterArg_231, %iterArg_232, %iterArg_233, %iterArg_234, %iterArg_235, %iterArg_236, %iterArg_237, %iterArg_238, %iterArg_239, %iterArg_240, %iterArg_241, %iterArg_242, %iterArg_243, %iterArg_244, %iterArg_245, %iterArg_246, %iterArg_247, %iterArg_248, %iterArg_249, %iterArg_250, %iterArg_251, %iterArg_252, %iterArg_253, %iterArg_254, %iterArg_255, %iterArg_256, %iterArg_257, %iterArg_258, %iterArg_259, %iterArg_260, %iterArg_261, %iterArg_262, %iterArg_263, %iterArg_264, %iterArg_265, %iterArg_266, %iterArg_267, %iterArg_268, %iterArg_269, %iterArg_270, %iterArg_271, %iterArg_272, %iterArg_273, %iterArg_274, %iterArg_275, %iterArg_276, %iterArg_277, %iterArg_278, %iterArg_279, %iterArg_280, %iterArg_281, %iterArg_282, %iterArg_283, %iterArg_284, %iterArg_285, %iterArg_286, %iterArg_287, %iterArg_288, %iterArg_289, %iterArg_290, %iterArg_291, %iterArg_292, %iterArg_293, %iterArg_294, %iterArg_295, %iterArg_296, %iterArg_297, %iterArg_298, %iterArg_299, %iterArg_301, %iterArg_302, %iterArg_303, %iterArg_304, %iterArg_305, %iterArg_306, %iterArg_307, %iterArg_308, %iterArg_309, %iterArg_310, %iterArg_311, %iterArg_312, %iterArg_313, %iterArg_314, %iterArg_315, %iterArg_316, %iterArg_317, %iterArg_318, %iterArg_319, %iterArg_320, %iterArg_321, %iterArg_322, %iterArg_323, %iterArg_324, %iterArg_325, %iterArg_326, %247) : (tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<2x14x25x1xf32>, tensor<2x32xf32>, tensor<2x15x24x1xf32>, tensor<2x32xf32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<2x15x26x1xf32>, tensor<2x32xf32>, tensor<2x16x25x1xf32>, tensor<2x32xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>, tensor<i32>) -> (tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>)
      %c_327 = stablehlo.constant dense<1> : tensor<i32>
      %249 = stablehlo.add %iterArg_300, %c_327 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_179, %iterArg_180, %iterArg_181, %iterArg_182, %iterArg_183, %iterArg_184, %iterArg_185, %iterArg_186, %iterArg_187, %iterArg_188, %iterArg_189, %iterArg_190, %iterArg_191, %iterArg_192, %iterArg_193, %iterArg_194, %iterArg_195, %iterArg_196, %iterArg_197, %iterArg_198, %iterArg_199, %iterArg_200, %iterArg_201, %iterArg_202, %iterArg_203, %iterArg_204, %iterArg_205, %iterArg_206, %iterArg_207, %iterArg_208, %iterArg_209, %iterArg_210, %iterArg_211, %iterArg_212, %iterArg_213, %iterArg_214, %iterArg_215, %iterArg_216, %iterArg_217, %iterArg_218, %iterArg_219, %iterArg_220, %iterArg_221, %iterArg_222, %iterArg_223, %iterArg_224, %iterArg_225, %iterArg_226, %iterArg_227, %iterArg_228, %iterArg_229, %iterArg_230, %iterArg_231, %iterArg_232, %iterArg_233, %iterArg_234, %iterArg_235, %iterArg_236, %iterArg_237, %iterArg_238, %iterArg_239, %iterArg_240, %iterArg_241, %iterArg_242, %iterArg_243, %iterArg_244, %iterArg_245, %iterArg_246, %iterArg_247, %iterArg_248, %iterArg_249, %iterArg_250, %iterArg_251, %iterArg_252, %iterArg_253, %iterArg_254, %iterArg_255, %iterArg_256, %iterArg_257, %iterArg_258, %iterArg_259, %iterArg_260, %iterArg_261, %iterArg_262, %iterArg_263, %iterArg_264, %iterArg_265, %iterArg_266, %iterArg_267, %iterArg_268, %iterArg_269, %iterArg_270, %iterArg_271, %iterArg_272, %iterArg_273, %iterArg_274, %iterArg_275, %iterArg_276, %iterArg_277, %iterArg_278, %iterArg_279, %iterArg_280, %iterArg_281, %iterArg_282, %iterArg_283, %iterArg_284, %iterArg_285, %iterArg_286, %iterArg_287, %iterArg_288, %iterArg_289, %iterArg_290, %iterArg_291, %iterArg_292, %iterArg_293, %iterArg_294, %iterArg_295, %iterArg_296, %iterArg_297, %iterArg_298, %iterArg_299, %249, %248#0, %248#1, %248#2, %248#3, %248#4, %248#5, %248#6, %248#7, %248#8, %248#9, %248#10, %248#11, %248#12, %248#13, %248#14, %248#15, %248#16, %248#17, %248#18, %248#19, %248#20, %248#21, %248#22, %248#23, %248#24, %248#25 : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<2x14x25x1xf32>, tensor<2x32xf32>, tensor<2x15x24x1xf32>, tensor<2x32xf32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<3xi32>, tensor<6xi32>, tensor<3xi32>, tensor<6x5xi32>, tensor<2x15x26x1xf32>, tensor<2x32xf32>, tensor<2x16x25x1xf32>, tensor<2x32xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<3xf32>, tensor<6xf32>, tensor<i32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
    }
    %228 = stablehlo.slice %227#132 [0:1, 0:256, 0:257, 0:24] : (tensor<1x256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %229 = stablehlo.reshape %228 : (tensor<1x256x257x24xf32>) -> tensor<256x257x24xf32>
    %230 = "stablehlo.all_reduce"(%229) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<256x257x24xf32>) -> tensor<256x257x24xf32>
    %231 = stablehlo.slice %227#133 [0:1, 0:257, 0:256, 0:24] : (tensor<1x257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %232 = stablehlo.reshape %231 : (tensor<1x257x256x24xf32>) -> tensor<257x256x24xf32>
    %233 = "stablehlo.all_reduce"(%232) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<257x256x24xf32>) -> tensor<257x256x24xf32>
    %234 = stablehlo.slice %227#138 [0:1, 0:257, 0:256, 0:24] : (tensor<1x257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %235 = stablehlo.reshape %234 : (tensor<1x257x256x24xf32>) -> tensor<257x256x24xf32>
    %236 = "stablehlo.all_reduce"(%235) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<257x256x24xf32>) -> tensor<257x256x24xf32>
    %237 = stablehlo.slice %227#139 [0:1, 0:256, 0:257, 0:24] : (tensor<1x256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %238 = stablehlo.reshape %237 : (tensor<1x256x257x24xf32>) -> tensor<256x257x24xf32>
    %239 = "stablehlo.all_reduce"(%238) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<256x257x24xf32>) -> tensor<256x257x24xf32>
    %240 = stablehlo.slice %227#144 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
    %241 = stablehlo.reshape %240 : (tensor<1x7200xf32>) -> tensor<7200xf32>
    %242 = "stablehlo.all_reduce"(%241) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<7200xf32>) -> tensor<7200xf32>
    %243 = stablehlo.slice %227#145 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
    %244 = stablehlo.reshape %243 : (tensor<1x7200xf32>) -> tensor<7200xf32>
    %245 = "stablehlo.all_reduce"(%244) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<0> : tensor<1x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg42: tensor<f32>, %arg43: tensor<f32>):
      %246 = stablehlo.add %arg42, %arg43 : tensor<f32>
      stablehlo.return %246 : tensor<f32>
    }) : (tensor<7200xf32>) -> tensor<7200xf32>
    return %227#123, %227#124, %227#125, %227#126, %227#127, %227#128, %227#129, %227#130, %227#131, %230, %233, %227#134, %227#135, %227#136, %227#137, %236, %239, %227#140, %227#141, %227#142, %227#143, %arg21, %arg22, %arg23, %arg24, %242, %245, %227#146, %227#147, %227#148 : tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<256x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
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
  func.func private @closed_call(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<6x3xi32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>, %arg9: tensor<0xf32>, %arg10: tensor<1x24x1xf32>, %arg11: tensor<1x24x1xf32>, %arg12: tensor<1x24x1xf32>, %arg13: tensor<24x1x1xf32>, %arg14: tensor<24x1x1xf32>, %arg15: tensor<24x1x1xf32>, %arg16: tensor<24x1x1xf32>, %arg17: tensor<24x1x1xf32>, %arg18: tensor<24x1x1xf32>, %arg19: tensor<1x1x24xf32>, %arg20: tensor<1x1x24xf32>, %arg21: tensor<1x1x24xf32>, %arg22: tensor<1x1x24xf32>, %arg23: tensor<1x1x24xf32>, %arg24: tensor<1x1x24xf32>, %arg25: tensor<1x24x1xf32>, %arg26: tensor<1x24x1xf32>, %arg27: tensor<1x24x1xf32>, %arg28: tensor<3xi32>, %arg29: tensor<6xi32>, %arg30: tensor<3xi32>, %arg31: tensor<6x5xi32>, %arg32: tensor<2x14x25x1xf32>, %arg33: tensor<2x32xf32>, %arg34: tensor<2x15x24x1xf32>, %arg35: tensor<2x32xf32>, %arg36: tensor<6x3xi32>, %arg37: tensor<f32>, %arg38: tensor<f32>, %arg39: tensor<f32>, %arg40: tensor<257x257x257xf32>, %arg41: tensor<257x256x257xf32>, %arg42: tensor<256x257x257xf32>, %arg43: tensor<1x24x1xf32>, %arg44: tensor<1x24x1xf32>, %arg45: tensor<1x24x1xf32>, %arg46: tensor<24x1x1xf32>, %arg47: tensor<24x1x1xf32>, %arg48: tensor<24x1x1xf32>, %arg49: tensor<24x1x1xf32>, %arg50: tensor<24x1x1xf32>, %arg51: tensor<24x1x1xf32>, %arg52: tensor<1x1x24xf32>, %arg53: tensor<1x1x24xf32>, %arg54: tensor<1x1x24xf32>, %arg55: tensor<1x1x24xf32>, %arg56: tensor<1x1x24xf32>, %arg57: tensor<1x1x24xf32>, %arg58: tensor<1x24x1xf32>, %arg59: tensor<1x24x1xf32>, %arg60: tensor<1x24x1xf32>, %arg61: tensor<3xi32>, %arg62: tensor<6xi32>, %arg63: tensor<3xi32>, %arg64: tensor<6x5xi32>, %arg65: tensor<2x15x26x1xf32>, %arg66: tensor<2x32xf32>, %arg67: tensor<2x16x25x1xf32>, %arg68: tensor<2x32xf32>, %arg69: tensor<200x1xi32>, %arg70: tensor<200x1xf32>, %arg71: tensor<200x1xi32>, %arg72: tensor<200x1xf32>, %arg73: tensor<200x1xi32>, %arg74: tensor<200x1xf32>, %arg75: tensor<200x1xi32>, %arg76: tensor<200x1xf32>, %arg77: tensor<200x1xi32>, %arg78: tensor<200x1xf32>, %arg79: tensor<200x1xi32>, %arg80: tensor<200x1xf32>, %arg81: tensor<200x8xi32>, %arg82: tensor<200x8xf32>, %arg83: tensor<200x4xi32>, %arg84: tensor<200x4xf32>, %arg85: tensor<200x4xi32>, %arg86: tensor<200x4xf32>, %arg87: tensor<200x2xi32>, %arg88: tensor<200x2xf32>, %arg89: tensor<200x4xi32>, %arg90: tensor<200x4xf32>, %arg91: tensor<200x4xi32>, %arg92: tensor<200x4xf32>, %arg93: tensor<3xf32>, %arg94: tensor<6xf32>, %arg95: tensor<200x1xi32>, %arg96: tensor<200x1xf32>, %arg97: tensor<200x1xi32>, %arg98: tensor<200x1xf32>, %arg99: tensor<200x1xi32>, %arg100: tensor<200x1xf32>, %arg101: tensor<200x1xi32>, %arg102: tensor<200x1xf32>, %arg103: tensor<200x1xi32>, %arg104: tensor<200x1xf32>, %arg105: tensor<200x1xi32>, %arg106: tensor<200x1xf32>, %arg107: tensor<200x8xi32>, %arg108: tensor<200x8xf32>, %arg109: tensor<200x2xi32>, %arg110: tensor<200x2xf32>, %arg111: tensor<200x2xi32>, %arg112: tensor<200x2xf32>, %arg113: tensor<200x1xi32>, %arg114: tensor<200x1xf32>, %arg115: tensor<200x4xi32>, %arg116: tensor<200x4xf32>, %arg117: tensor<200x4xi32>, %arg118: tensor<200x4xf32>, %arg119: tensor<3xf32>, %arg120: tensor<6xf32>, %arg121: tensor<257x257x257xf32>, %arg122: tensor<257x256x257xf32>, %arg123: tensor<256x257x257xf32>, %arg124: tensor<256x256x257xf32>, %arg125: tensor<256x257x257xf32>, %arg126: tensor<257x256x257xf32>, %arg127: tensor<256x24x257xf32>, %arg128: tensor<24x256x257xf32>, %arg129: tensor<24x257x257xf32>, %arg130: tensor<1x256x257x24xf32>, %arg131: tensor<1x257x256x24xf32>, %arg132: tensor<257x24x257xf32>, %arg133: tensor<257x24x257xf32>, %arg134: tensor<24x257x257xf32>, %arg135: tensor<24x256x257xf32>, %arg136: tensor<1x257x256x24xf32>, %arg137: tensor<1x256x257x24xf32>, %arg138: tensor<256x24x257xf32>, %arg139: tensor<2x1xf32>, %arg140: tensor<2x1xf32>, %arg141: tensor<2xi32>, %arg142: tensor<1x7200xf32>, %arg143: tensor<1x7200xf32>, %arg144: tensor<6xf32>, %arg145: tensor<f32>, %arg146: tensor<i32>, %arg147: tensor<i32>) -> (tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg147, %c : tensor<i32>
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
    %15 = stablehlo.slice %arg130 [0:1, 0:256, 0:257, 0:24] : (tensor<1x256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %16 = stablehlo.reshape %15 : (tensor<1x256x257x24xf32>) -> tensor<256x257x24xf32>
    %17 = stablehlo.slice %arg131 [0:1, 0:257, 0:256, 0:24] : (tensor<1x257x256x24xf32>) -> tensor<1x257x256x24xf32>
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
    %60 = call @_where_73(%58, %arg127, %59) : (tensor<256x24x257xi1>, tensor<256x24x257xf32>, tensor<256x24x257xf32>) -> tensor<256x24x257xf32>
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
    %102 = call @_where_82(%100, %arg128, %101) : (tensor<24x256x257xi1>, tensor<24x256x257xf32>, tensor<24x256x257xf32>) -> tensor<24x256x257xf32>
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
    %144 = call @_where_89(%142, %arg129, %143) : (tensor<24x257x257xi1>, tensor<24x257x257xf32>, tensor<24x257x257xf32>) -> tensor<24x257x257xf32>
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
    %280 = call @_where_98(%278, %arg132, %279) : (tensor<257x24x257xi1>, tensor<257x24x257xf32>, tensor<257x24x257xf32>) -> tensor<257x24x257xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %281 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f32>) -> tensor<257x257x1xf32>
    %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %282 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f32>) -> tensor<257x257x1xf32>
    %283 = stablehlo.slice %arg122 [0:257, 0:256, 0:1] : (tensor<257x256x257xf32>) -> tensor<257x256x1xf32>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %284 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<257x256x1xf32>
    %285 = "stablehlo.collective_permute"(%283) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<257x256x1xf32>) -> tensor<257x256x1xf32>
    %286 = stablehlo.slice %arg123 [0:256, 0:257, 0:1] : (tensor<256x257x257xf32>) -> tensor<256x257x1xf32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %287 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<256x257x1xf32>
    %288 = "stablehlo.collective_permute"(%286) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<256x257x1xf32>) -> tensor<256x257x1xf32>
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
    ^bb0(%arg148: tensor<i32>, %arg149: tensor<i32>):
      stablehlo.return %arg149 : tensor<i32>
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
    %334 = stablehlo.transpose %284, dims = [2, 0, 1] : (tensor<257x256x1xf32>) -> tensor<1x257x256xf32>
    %335 = stablehlo.transpose %285, dims = [2, 0, 1] : (tensor<257x256x1xf32>) -> tensor<1x257x256xf32>
    %336 = stablehlo.transpose %287, dims = [2, 0, 1] : (tensor<256x257x1xf32>) -> tensor<1x256x257xf32>
    %337 = stablehlo.transpose %288, dims = [2, 0, 1] : (tensor<256x257x1xf32>) -> tensor<1x256x257xf32>
    %338 = stablehlo.transpose %281, dims = [2, 0, 1] : (tensor<257x257x1xf32>) -> tensor<1x257x257xf32>
    %339 = stablehlo.transpose %282, dims = [2, 0, 1] : (tensor<257x257x1xf32>) -> tensor<1x257x257xf32>
    %340 = stablehlo.transpose %arg125, dims = [2, 0, 1] : (tensor<256x257x257xf32>) -> tensor<257x256x257xf32>
    %341 = stablehlo.transpose %arg126, dims = [2, 0, 1] : (tensor<257x256x257xf32>) -> tensor<257x257x256xf32>
    %342 = stablehlo.transpose %arg124, dims = [2, 0, 1] : (tensor<256x256x257xf32>) -> tensor<257x256x256xf32>
    %343 = stablehlo.transpose %arg122, dims = [2, 0, 1] : (tensor<257x256x257xf32>) -> tensor<257x257x256xf32>
    %344 = stablehlo.transpose %arg123, dims = [2, 0, 1] : (tensor<256x257x257xf32>) -> tensor<257x256x257xf32>
    %345 = stablehlo.transpose %arg121, dims = [2, 0, 1] : (tensor<257x257x257xf32>) -> tensor<257x257x257xf32>
    %346 = stablehlo.transpose %144, dims = [2, 0, 1] : (tensor<24x257x257xf32>) -> tensor<257x24x257xf32>
    %347 = stablehlo.transpose %191, dims = [2, 0, 1] : (tensor<256x257x24xf32>) -> tensor<24x256x257xf32>
    %348 = stablehlo.transpose %238, dims = [2, 0, 1] : (tensor<257x256x24xf32>) -> tensor<24x257x256xf32>
    %349 = stablehlo.transpose %280, dims = [2, 0, 1] : (tensor<257x24x257xf32>) -> tensor<257x257x24xf32>
    %350 = stablehlo.transpose %60, dims = [2, 0, 1] : (tensor<256x24x257xf32>) -> tensor<257x256x24xf32>
    %351 = stablehlo.transpose %102, dims = [2, 0, 1] : (tensor<24x256x257xf32>) -> tensor<257x24x256xf32>
    %352:9 = stablehlo.custom_call @beamz_cuda_sharded(%340, %341, %342, %343, %344, %345, %arg4, %arg5, %arg3, %arg7, %arg8, %arg6, %arg31, %289, %290, %291, %292, %293, %294, %295, %296, %297, %298, %299, %300, %301, %302, %303, %304, %305, %306, %346, %347, %348, %349, %350, %351, %arg9, %arg9, %arg9, %333, %334, %335, %336, %337, %338, %339) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 0 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<257x256x257xf32>, tensor<257x257x256xf32>, tensor<257x256x256xf32>, tensor<257x257x256xf32>, tensor<257x256x257xf32>, tensor<257x257x257xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<257x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x256xf32>, tensor<257x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x256xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<1x257x256xf32>, tensor<1x257x256xf32>, tensor<1x256x257xf32>, tensor<1x256x257xf32>, tensor<1x257x257xf32>, tensor<1x257x257xf32>) -> (tensor<257x256x257xf32>, tensor<257x257x256xf32>, tensor<257x256x256xf32>, tensor<257x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x256xf32>, tensor<257x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x256xf32>)
    %353 = stablehlo.transpose %352#2, dims = [1, 2, 0] : (tensor<257x256x256xf32>) -> tensor<256x256x257xf32>
    %354 = stablehlo.transpose %352#0, dims = [1, 2, 0] : (tensor<257x256x257xf32>) -> tensor<256x257x257xf32>
    %355 = stablehlo.transpose %352#1, dims = [1, 2, 0] : (tensor<257x257x256xf32>) -> tensor<257x256x257xf32>
    %356 = stablehlo.transpose %352#7, dims = [1, 2, 0] : (tensor<257x256x24xf32>) -> tensor<256x24x257xf32>
    %357 = stablehlo.transpose %352#8, dims = [1, 2, 0] : (tensor<257x24x256xf32>) -> tensor<24x256x257xf32>
    %358 = stablehlo.transpose %352#3, dims = [1, 2, 0] : (tensor<257x24x257xf32>) -> tensor<24x257x257xf32>
    %359 = stablehlo.transpose %352#4, dims = [1, 2, 0] : (tensor<24x256x257xf32>) -> tensor<256x257x24xf32>
    %360 = stablehlo.transpose %352#5, dims = [1, 2, 0] : (tensor<24x257x256xf32>) -> tensor<257x256x24xf32>
    %361 = stablehlo.transpose %352#6, dims = [1, 2, 0] : (tensor<257x257x24xf32>) -> tensor<257x24x257xf32>
    %362 = stablehlo.broadcast_in_dim %359, dims = [1, 2, 3] : (tensor<256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %363 = stablehlo.broadcast_in_dim %360, dims = [1, 2, 3] : (tensor<257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %c_90 = stablehlo.constant dense<0> : tensor<i32>
    %c_91 = stablehlo.constant dense<31> : tensor<i32>
    %364 = call @clip(%arg146, %c_90, %c_91) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_92 = stablehlo.constant dense<1> : tensor<ui32>
    %c_93 = stablehlo.constant dense<1> : tensor<ui32>
    %365 = stablehlo.partition_id : tensor<ui32>
    %366 = stablehlo.divide %365, %c_92 : tensor<ui32>
    %367 = stablehlo.remainder %366, %c_93 : tensor<ui32>
    %368 = stablehlo.convert %367 : (tensor<ui32>) -> tensor<i32>
    %c_94 = stablehlo.constant dense<257> : tensor<i32>
    %369 = stablehlo.multiply %368, %c_94 : tensor<i32>
    %c_95 = stablehlo.constant dense<75> : tensor<i32>
    %370 = stablehlo.subtract %c_95, %369 : tensor<i32>
    %c_96 = stablehlo.constant dense<0> : tensor<i32>
    %c_97 = stablehlo.constant dense<256> : tensor<i32>
    %371 = call @clip_170(%370, %c_96, %c_97) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %372 = stablehlo.add %369, %371 : tensor<i32>
    %373 = stablehlo.iota dim = 0 : tensor<1xi32>
    %374 = stablehlo.broadcast_in_dim %372, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %375 = stablehlo.add %374, %373 : tensor<1xi32>
    %c_98 = stablehlo.constant dense<75> : tensor<i32>
    %376 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %377 = stablehlo.subtract %375, %376 : tensor<1xi32>
    %c_99 = stablehlo.constant dense<0> : tensor<i32>
    %378 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %379 = stablehlo.compare  GE, %377, %378,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_100 = stablehlo.constant dense<1> : tensor<i32>
    %380 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %381 = stablehlo.compare  LT, %377, %380,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %382 = stablehlo.and %379, %381 : tensor<1xi1>
    %383 = stablehlo.slice %arg32 [0:1, 0:14, 0:25, 0:1] : (tensor<2x14x25x1xf32>) -> tensor<1x14x25x1xf32>
    %384 = stablehlo.reshape %383 : (tensor<1x14x25x1xf32>) -> tensor<14x25x1xf32>
    %385 = call @_take(%384, %377) : (tensor<14x25x1xf32>, tensor<1xi32>) -> tensor<14x25x1xf32>
    %386 = stablehlo.reshape %382 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_101 = stablehlo.constant dense<0> : tensor<i32>
    %387 = stablehlo.compare  LT, %364, %c_101,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_102 = stablehlo.constant dense<32> : tensor<i32>
    %388 = stablehlo.add %364, %c_102 : tensor<i32>
    %389 = stablehlo.select %387, %388, %364 : tensor<i1>, tensor<i32>
    %c_103 = stablehlo.constant dense<0> : tensor<i32>
    %390 = stablehlo.dynamic_slice %arg33, %c_103, %389, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %391 = stablehlo.reshape %390 : (tensor<1x1xf32>) -> tensor<f32>
    %392 = stablehlo.broadcast_in_dim %391, dims = [] : (tensor<f32>) -> tensor<14x25x1xf32>
    %393 = stablehlo.multiply %385, %392 : tensor<14x25x1xf32>
    %c_104 = stablehlo.constant dense<0> : tensor<i32>
    %394 = call @_where_191(%386, %393, %c_104) : (tensor<1x1x1xi1>, tensor<14x25x1xf32>, tensor<i32>) -> tensor<14x25x1xf32>
    %c_105 = stablehlo.constant dense<121> : tensor<i32>
    %c_106 = stablehlo.constant dense<0> : tensor<i32>
    %395 = stablehlo.compare  LT, %c_105, %c_106,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_107 = stablehlo.constant dense<121> : tensor<i32>
    %c_108 = stablehlo.constant dense<256> : tensor<i32>
    %396 = stablehlo.add %c_107, %c_108 : tensor<i32>
    %c_109 = stablehlo.constant dense<121> : tensor<i32>
    %397 = stablehlo.select %395, %396, %c_109 : tensor<i1>, tensor<i32>
    %c_110 = stablehlo.constant dense<116> : tensor<i32>
    %c_111 = stablehlo.constant dense<0> : tensor<i32>
    %398 = stablehlo.compare  LT, %c_110, %c_111,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_112 = stablehlo.constant dense<116> : tensor<i32>
    %c_113 = stablehlo.constant dense<257> : tensor<i32>
    %399 = stablehlo.add %c_112, %c_113 : tensor<i32>
    %c_114 = stablehlo.constant dense<116> : tensor<i32>
    %400 = stablehlo.select %398, %399, %c_114 : tensor<i1>, tensor<i32>
    %c_115 = stablehlo.constant dense<0> : tensor<i32>
    %401 = stablehlo.compare  LT, %371, %c_115,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_116 = stablehlo.constant dense<257> : tensor<i32>
    %402 = stablehlo.add %371, %c_116 : tensor<i32>
    %403 = stablehlo.select %401, %402, %371 : tensor<i1>, tensor<i32>
    %404 = stablehlo.dynamic_slice %354, %397, %400, %403, sizes = [14, 25, 1] : (tensor<256x257x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<14x25x1xf32>
    %405 = stablehlo.add %404, %394 : tensor<14x25x1xf32>
    %c_117 = stablehlo.constant dense<121> : tensor<i32>
    %c_118 = stablehlo.constant dense<0> : tensor<i32>
    %406 = stablehlo.compare  LT, %c_117, %c_118,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_119 = stablehlo.constant dense<121> : tensor<i32>
    %c_120 = stablehlo.constant dense<256> : tensor<i32>
    %407 = stablehlo.add %c_119, %c_120 : tensor<i32>
    %c_121 = stablehlo.constant dense<121> : tensor<i32>
    %408 = stablehlo.select %406, %407, %c_121 : tensor<i1>, tensor<i32>
    %c_122 = stablehlo.constant dense<116> : tensor<i32>
    %c_123 = stablehlo.constant dense<0> : tensor<i32>
    %409 = stablehlo.compare  LT, %c_122, %c_123,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_124 = stablehlo.constant dense<116> : tensor<i32>
    %c_125 = stablehlo.constant dense<257> : tensor<i32>
    %410 = stablehlo.add %c_124, %c_125 : tensor<i32>
    %c_126 = stablehlo.constant dense<116> : tensor<i32>
    %411 = stablehlo.select %409, %410, %c_126 : tensor<i1>, tensor<i32>
    %c_127 = stablehlo.constant dense<0> : tensor<i32>
    %412 = stablehlo.compare  LT, %371, %c_127,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_128 = stablehlo.constant dense<257> : tensor<i32>
    %413 = stablehlo.add %371, %c_128 : tensor<i32>
    %414 = stablehlo.select %412, %413, %371 : tensor<i1>, tensor<i32>
    %415 = stablehlo.dynamic_update_slice %354, %405, %408, %411, %414 : (tensor<256x257x257xf32>, tensor<14x25x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<256x257x257xf32>
    %c_129 = stablehlo.constant dense<75> : tensor<i32>
    %416 = stablehlo.subtract %c_129, %369 : tensor<i32>
    %c_130 = stablehlo.constant dense<0> : tensor<i32>
    %c_131 = stablehlo.constant dense<256> : tensor<i32>
    %417 = call @clip_170(%416, %c_130, %c_131) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %418 = stablehlo.add %369, %417 : tensor<i32>
    %419 = stablehlo.iota dim = 0 : tensor<1xi32>
    %420 = stablehlo.broadcast_in_dim %418, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %421 = stablehlo.add %420, %419 : tensor<1xi32>
    %c_132 = stablehlo.constant dense<75> : tensor<i32>
    %422 = stablehlo.broadcast_in_dim %c_132, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %423 = stablehlo.subtract %421, %422 : tensor<1xi32>
    %c_133 = stablehlo.constant dense<0> : tensor<i32>
    %424 = stablehlo.broadcast_in_dim %c_133, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %425 = stablehlo.compare  GE, %423, %424,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_134 = stablehlo.constant dense<1> : tensor<i32>
    %426 = stablehlo.broadcast_in_dim %c_134, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %427 = stablehlo.compare  LT, %423, %426,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %428 = stablehlo.and %425, %427 : tensor<1xi1>
    %429 = stablehlo.slice %arg32 [1:2, 0:14, 0:25, 0:1] : (tensor<2x14x25x1xf32>) -> tensor<1x14x25x1xf32>
    %430 = stablehlo.reshape %429 : (tensor<1x14x25x1xf32>) -> tensor<14x25x1xf32>
    %431 = call @_take(%430, %423) : (tensor<14x25x1xf32>, tensor<1xi32>) -> tensor<14x25x1xf32>
    %432 = stablehlo.reshape %428 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_135 = stablehlo.constant dense<0> : tensor<i32>
    %433 = stablehlo.compare  LT, %364, %c_135,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_136 = stablehlo.constant dense<32> : tensor<i32>
    %434 = stablehlo.add %364, %c_136 : tensor<i32>
    %435 = stablehlo.select %433, %434, %364 : tensor<i1>, tensor<i32>
    %c_137 = stablehlo.constant dense<1> : tensor<i32>
    %436 = stablehlo.dynamic_slice %arg33, %c_137, %435, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %437 = stablehlo.reshape %436 : (tensor<1x1xf32>) -> tensor<f32>
    %438 = stablehlo.broadcast_in_dim %437, dims = [] : (tensor<f32>) -> tensor<14x25x1xf32>
    %439 = stablehlo.multiply %431, %438 : tensor<14x25x1xf32>
    %c_138 = stablehlo.constant dense<0> : tensor<i32>
    %440 = call @_where_191(%432, %439, %c_138) : (tensor<1x1x1xi1>, tensor<14x25x1xf32>, tensor<i32>) -> tensor<14x25x1xf32>
    %c_139 = stablehlo.constant dense<121> : tensor<i32>
    %c_140 = stablehlo.constant dense<0> : tensor<i32>
    %441 = stablehlo.compare  LT, %c_139, %c_140,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_141 = stablehlo.constant dense<121> : tensor<i32>
    %c_142 = stablehlo.constant dense<256> : tensor<i32>
    %442 = stablehlo.add %c_141, %c_142 : tensor<i32>
    %c_143 = stablehlo.constant dense<121> : tensor<i32>
    %443 = stablehlo.select %441, %442, %c_143 : tensor<i1>, tensor<i32>
    %c_144 = stablehlo.constant dense<116> : tensor<i32>
    %c_145 = stablehlo.constant dense<0> : tensor<i32>
    %444 = stablehlo.compare  LT, %c_144, %c_145,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_146 = stablehlo.constant dense<116> : tensor<i32>
    %c_147 = stablehlo.constant dense<257> : tensor<i32>
    %445 = stablehlo.add %c_146, %c_147 : tensor<i32>
    %c_148 = stablehlo.constant dense<116> : tensor<i32>
    %446 = stablehlo.select %444, %445, %c_148 : tensor<i1>, tensor<i32>
    %c_149 = stablehlo.constant dense<0> : tensor<i32>
    %447 = stablehlo.compare  LT, %417, %c_149,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_150 = stablehlo.constant dense<257> : tensor<i32>
    %448 = stablehlo.add %417, %c_150 : tensor<i32>
    %449 = stablehlo.select %447, %448, %417 : tensor<i1>, tensor<i32>
    %450 = stablehlo.dynamic_slice %415, %443, %446, %449, sizes = [14, 25, 1] : (tensor<256x257x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<14x25x1xf32>
    %451 = stablehlo.add %450, %440 : tensor<14x25x1xf32>
    %c_151 = stablehlo.constant dense<121> : tensor<i32>
    %c_152 = stablehlo.constant dense<0> : tensor<i32>
    %452 = stablehlo.compare  LT, %c_151, %c_152,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_153 = stablehlo.constant dense<121> : tensor<i32>
    %c_154 = stablehlo.constant dense<256> : tensor<i32>
    %453 = stablehlo.add %c_153, %c_154 : tensor<i32>
    %c_155 = stablehlo.constant dense<121> : tensor<i32>
    %454 = stablehlo.select %452, %453, %c_155 : tensor<i1>, tensor<i32>
    %c_156 = stablehlo.constant dense<116> : tensor<i32>
    %c_157 = stablehlo.constant dense<0> : tensor<i32>
    %455 = stablehlo.compare  LT, %c_156, %c_157,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_158 = stablehlo.constant dense<116> : tensor<i32>
    %c_159 = stablehlo.constant dense<257> : tensor<i32>
    %456 = stablehlo.add %c_158, %c_159 : tensor<i32>
    %c_160 = stablehlo.constant dense<116> : tensor<i32>
    %457 = stablehlo.select %455, %456, %c_160 : tensor<i1>, tensor<i32>
    %c_161 = stablehlo.constant dense<0> : tensor<i32>
    %458 = stablehlo.compare  LT, %417, %c_161,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_162 = stablehlo.constant dense<257> : tensor<i32>
    %459 = stablehlo.add %417, %c_162 : tensor<i32>
    %460 = stablehlo.select %458, %459, %417 : tensor<i1>, tensor<i32>
    %461 = stablehlo.dynamic_update_slice %415, %451, %454, %457, %460 : (tensor<256x257x257xf32>, tensor<14x25x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<256x257x257xf32>
    %c_163 = stablehlo.constant dense<0> : tensor<i32>
    %c_164 = stablehlo.constant dense<31> : tensor<i32>
    %462 = call @clip(%arg146, %c_163, %c_164) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_165 = stablehlo.constant dense<1> : tensor<ui32>
    %c_166 = stablehlo.constant dense<1> : tensor<ui32>
    %463 = stablehlo.partition_id : tensor<ui32>
    %464 = stablehlo.divide %463, %c_165 : tensor<ui32>
    %465 = stablehlo.remainder %464, %c_166 : tensor<ui32>
    %466 = stablehlo.convert %465 : (tensor<ui32>) -> tensor<i32>
    %c_167 = stablehlo.constant dense<257> : tensor<i32>
    %467 = stablehlo.multiply %466, %c_167 : tensor<i32>
    %c_168 = stablehlo.constant dense<75> : tensor<i32>
    %468 = stablehlo.subtract %c_168, %467 : tensor<i32>
    %c_169 = stablehlo.constant dense<0> : tensor<i32>
    %c_170 = stablehlo.constant dense<256> : tensor<i32>
    %469 = call @clip_170(%468, %c_169, %c_170) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %470 = stablehlo.add %467, %469 : tensor<i32>
    %471 = stablehlo.iota dim = 0 : tensor<1xi32>
    %472 = stablehlo.broadcast_in_dim %470, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %473 = stablehlo.add %472, %471 : tensor<1xi32>
    %c_171 = stablehlo.constant dense<75> : tensor<i32>
    %474 = stablehlo.broadcast_in_dim %c_171, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %475 = stablehlo.subtract %473, %474 : tensor<1xi32>
    %c_172 = stablehlo.constant dense<0> : tensor<i32>
    %476 = stablehlo.broadcast_in_dim %c_172, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %477 = stablehlo.compare  GE, %475, %476,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_173 = stablehlo.constant dense<1> : tensor<i32>
    %478 = stablehlo.broadcast_in_dim %c_173, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %479 = stablehlo.compare  LT, %475, %478,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %480 = stablehlo.and %477, %479 : tensor<1xi1>
    %481 = stablehlo.slice %arg34 [0:1, 0:15, 0:24, 0:1] : (tensor<2x15x24x1xf32>) -> tensor<1x15x24x1xf32>
    %482 = stablehlo.reshape %481 : (tensor<1x15x24x1xf32>) -> tensor<15x24x1xf32>
    %483 = call @_take_201(%482, %475) : (tensor<15x24x1xf32>, tensor<1xi32>) -> tensor<15x24x1xf32>
    %484 = stablehlo.reshape %480 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_174 = stablehlo.constant dense<0> : tensor<i32>
    %485 = stablehlo.compare  LT, %462, %c_174,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_175 = stablehlo.constant dense<32> : tensor<i32>
    %486 = stablehlo.add %462, %c_175 : tensor<i32>
    %487 = stablehlo.select %485, %486, %462 : tensor<i1>, tensor<i32>
    %c_176 = stablehlo.constant dense<0> : tensor<i32>
    %488 = stablehlo.dynamic_slice %arg35, %c_176, %487, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %489 = stablehlo.reshape %488 : (tensor<1x1xf32>) -> tensor<f32>
    %490 = stablehlo.broadcast_in_dim %489, dims = [] : (tensor<f32>) -> tensor<15x24x1xf32>
    %491 = stablehlo.multiply %483, %490 : tensor<15x24x1xf32>
    %c_177 = stablehlo.constant dense<0> : tensor<i32>
    %492 = call @_where_205(%484, %491, %c_177) : (tensor<1x1x1xi1>, tensor<15x24x1xf32>, tensor<i32>) -> tensor<15x24x1xf32>
    %c_178 = stablehlo.constant dense<121> : tensor<i32>
    %c_179 = stablehlo.constant dense<0> : tensor<i32>
    %493 = stablehlo.compare  LT, %c_178, %c_179,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_180 = stablehlo.constant dense<121> : tensor<i32>
    %c_181 = stablehlo.constant dense<257> : tensor<i32>
    %494 = stablehlo.add %c_180, %c_181 : tensor<i32>
    %c_182 = stablehlo.constant dense<121> : tensor<i32>
    %495 = stablehlo.select %493, %494, %c_182 : tensor<i1>, tensor<i32>
    %c_183 = stablehlo.constant dense<116> : tensor<i32>
    %c_184 = stablehlo.constant dense<0> : tensor<i32>
    %496 = stablehlo.compare  LT, %c_183, %c_184,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_185 = stablehlo.constant dense<116> : tensor<i32>
    %c_186 = stablehlo.constant dense<256> : tensor<i32>
    %497 = stablehlo.add %c_185, %c_186 : tensor<i32>
    %c_187 = stablehlo.constant dense<116> : tensor<i32>
    %498 = stablehlo.select %496, %497, %c_187 : tensor<i1>, tensor<i32>
    %c_188 = stablehlo.constant dense<0> : tensor<i32>
    %499 = stablehlo.compare  LT, %469, %c_188,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_189 = stablehlo.constant dense<257> : tensor<i32>
    %500 = stablehlo.add %469, %c_189 : tensor<i32>
    %501 = stablehlo.select %499, %500, %469 : tensor<i1>, tensor<i32>
    %502 = stablehlo.dynamic_slice %355, %495, %498, %501, sizes = [15, 24, 1] : (tensor<257x256x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<15x24x1xf32>
    %503 = stablehlo.add %502, %492 : tensor<15x24x1xf32>
    %c_190 = stablehlo.constant dense<121> : tensor<i32>
    %c_191 = stablehlo.constant dense<0> : tensor<i32>
    %504 = stablehlo.compare  LT, %c_190, %c_191,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_192 = stablehlo.constant dense<121> : tensor<i32>
    %c_193 = stablehlo.constant dense<257> : tensor<i32>
    %505 = stablehlo.add %c_192, %c_193 : tensor<i32>
    %c_194 = stablehlo.constant dense<121> : tensor<i32>
    %506 = stablehlo.select %504, %505, %c_194 : tensor<i1>, tensor<i32>
    %c_195 = stablehlo.constant dense<116> : tensor<i32>
    %c_196 = stablehlo.constant dense<0> : tensor<i32>
    %507 = stablehlo.compare  LT, %c_195, %c_196,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_197 = stablehlo.constant dense<116> : tensor<i32>
    %c_198 = stablehlo.constant dense<256> : tensor<i32>
    %508 = stablehlo.add %c_197, %c_198 : tensor<i32>
    %c_199 = stablehlo.constant dense<116> : tensor<i32>
    %509 = stablehlo.select %507, %508, %c_199 : tensor<i1>, tensor<i32>
    %c_200 = stablehlo.constant dense<0> : tensor<i32>
    %510 = stablehlo.compare  LT, %469, %c_200,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_201 = stablehlo.constant dense<257> : tensor<i32>
    %511 = stablehlo.add %469, %c_201 : tensor<i32>
    %512 = stablehlo.select %510, %511, %469 : tensor<i1>, tensor<i32>
    %513 = stablehlo.dynamic_update_slice %355, %503, %506, %509, %512 : (tensor<257x256x257xf32>, tensor<15x24x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<257x256x257xf32>
    %c_202 = stablehlo.constant dense<75> : tensor<i32>
    %514 = stablehlo.subtract %c_202, %467 : tensor<i32>
    %c_203 = stablehlo.constant dense<0> : tensor<i32>
    %c_204 = stablehlo.constant dense<256> : tensor<i32>
    %515 = call @clip_170(%514, %c_203, %c_204) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %516 = stablehlo.add %467, %515 : tensor<i32>
    %517 = stablehlo.iota dim = 0 : tensor<1xi32>
    %518 = stablehlo.broadcast_in_dim %516, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %519 = stablehlo.add %518, %517 : tensor<1xi32>
    %c_205 = stablehlo.constant dense<75> : tensor<i32>
    %520 = stablehlo.broadcast_in_dim %c_205, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %521 = stablehlo.subtract %519, %520 : tensor<1xi32>
    %c_206 = stablehlo.constant dense<0> : tensor<i32>
    %522 = stablehlo.broadcast_in_dim %c_206, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %523 = stablehlo.compare  GE, %521, %522,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_207 = stablehlo.constant dense<1> : tensor<i32>
    %524 = stablehlo.broadcast_in_dim %c_207, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %525 = stablehlo.compare  LT, %521, %524,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %526 = stablehlo.and %523, %525 : tensor<1xi1>
    %527 = stablehlo.slice %arg34 [1:2, 0:15, 0:24, 0:1] : (tensor<2x15x24x1xf32>) -> tensor<1x15x24x1xf32>
    %528 = stablehlo.reshape %527 : (tensor<1x15x24x1xf32>) -> tensor<15x24x1xf32>
    %529 = call @_take_201(%528, %521) : (tensor<15x24x1xf32>, tensor<1xi32>) -> tensor<15x24x1xf32>
    %530 = stablehlo.reshape %526 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_208 = stablehlo.constant dense<0> : tensor<i32>
    %531 = stablehlo.compare  LT, %462, %c_208,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_209 = stablehlo.constant dense<32> : tensor<i32>
    %532 = stablehlo.add %462, %c_209 : tensor<i32>
    %533 = stablehlo.select %531, %532, %462 : tensor<i1>, tensor<i32>
    %c_210 = stablehlo.constant dense<1> : tensor<i32>
    %534 = stablehlo.dynamic_slice %arg35, %c_210, %533, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %535 = stablehlo.reshape %534 : (tensor<1x1xf32>) -> tensor<f32>
    %536 = stablehlo.broadcast_in_dim %535, dims = [] : (tensor<f32>) -> tensor<15x24x1xf32>
    %537 = stablehlo.multiply %529, %536 : tensor<15x24x1xf32>
    %c_211 = stablehlo.constant dense<0> : tensor<i32>
    %538 = call @_where_205(%530, %537, %c_211) : (tensor<1x1x1xi1>, tensor<15x24x1xf32>, tensor<i32>) -> tensor<15x24x1xf32>
    %c_212 = stablehlo.constant dense<121> : tensor<i32>
    %c_213 = stablehlo.constant dense<0> : tensor<i32>
    %539 = stablehlo.compare  LT, %c_212, %c_213,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_214 = stablehlo.constant dense<121> : tensor<i32>
    %c_215 = stablehlo.constant dense<257> : tensor<i32>
    %540 = stablehlo.add %c_214, %c_215 : tensor<i32>
    %c_216 = stablehlo.constant dense<121> : tensor<i32>
    %541 = stablehlo.select %539, %540, %c_216 : tensor<i1>, tensor<i32>
    %c_217 = stablehlo.constant dense<116> : tensor<i32>
    %c_218 = stablehlo.constant dense<0> : tensor<i32>
    %542 = stablehlo.compare  LT, %c_217, %c_218,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_219 = stablehlo.constant dense<116> : tensor<i32>
    %c_220 = stablehlo.constant dense<256> : tensor<i32>
    %543 = stablehlo.add %c_219, %c_220 : tensor<i32>
    %c_221 = stablehlo.constant dense<116> : tensor<i32>
    %544 = stablehlo.select %542, %543, %c_221 : tensor<i1>, tensor<i32>
    %c_222 = stablehlo.constant dense<0> : tensor<i32>
    %545 = stablehlo.compare  LT, %515, %c_222,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_223 = stablehlo.constant dense<257> : tensor<i32>
    %546 = stablehlo.add %515, %c_223 : tensor<i32>
    %547 = stablehlo.select %545, %546, %515 : tensor<i1>, tensor<i32>
    %548 = stablehlo.dynamic_slice %513, %541, %544, %547, sizes = [15, 24, 1] : (tensor<257x256x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<15x24x1xf32>
    %549 = stablehlo.add %548, %538 : tensor<15x24x1xf32>
    %c_224 = stablehlo.constant dense<121> : tensor<i32>
    %c_225 = stablehlo.constant dense<0> : tensor<i32>
    %550 = stablehlo.compare  LT, %c_224, %c_225,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_226 = stablehlo.constant dense<121> : tensor<i32>
    %c_227 = stablehlo.constant dense<257> : tensor<i32>
    %551 = stablehlo.add %c_226, %c_227 : tensor<i32>
    %c_228 = stablehlo.constant dense<121> : tensor<i32>
    %552 = stablehlo.select %550, %551, %c_228 : tensor<i1>, tensor<i32>
    %c_229 = stablehlo.constant dense<116> : tensor<i32>
    %c_230 = stablehlo.constant dense<0> : tensor<i32>
    %553 = stablehlo.compare  LT, %c_229, %c_230,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_231 = stablehlo.constant dense<116> : tensor<i32>
    %c_232 = stablehlo.constant dense<256> : tensor<i32>
    %554 = stablehlo.add %c_231, %c_232 : tensor<i32>
    %c_233 = stablehlo.constant dense<116> : tensor<i32>
    %555 = stablehlo.select %553, %554, %c_233 : tensor<i1>, tensor<i32>
    %c_234 = stablehlo.constant dense<0> : tensor<i32>
    %556 = stablehlo.compare  LT, %515, %c_234,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_235 = stablehlo.constant dense<257> : tensor<i32>
    %557 = stablehlo.add %515, %c_235 : tensor<i32>
    %558 = stablehlo.select %556, %557, %515 : tensor<i1>, tensor<i32>
    %559 = stablehlo.dynamic_update_slice %513, %549, %552, %555, %558 : (tensor<257x256x257xf32>, tensor<15x24x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<257x256x257xf32>
    %c_236 = stablehlo.constant dense<1> : tensor<ui32>
    %c_237 = stablehlo.constant dense<1> : tensor<ui32>
    %560 = stablehlo.partition_id : tensor<ui32>
    %561 = stablehlo.divide %560, %c_236 : tensor<ui32>
    %562 = stablehlo.remainder %561, %c_237 : tensor<ui32>
    %563 = stablehlo.convert %562 : (tensor<ui32>) -> tensor<i32>
    %c_238 = stablehlo.constant dense<257> : tensor<i32>
    %564 = stablehlo.multiply %563, %c_238 : tensor<i32>
    %c_239 = stablehlo.constant dense<2> : tensor<i32>
    %565 = stablehlo.broadcast_in_dim %c_239, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %566 = stablehlo.broadcast_in_dim %564, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_240 = stablehlo.constant dense<0> : tensor<i32>
    %567 = stablehlo.broadcast_in_dim %c_240, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %568 = stablehlo.concatenate %565, %566, %567, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %569 = stablehlo.broadcast_in_dim %568, dims = [1] : (tensor<3xi32>) -> tensor<1x3xi32>
    %570 = stablehlo.concatenate %569, %arg36, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
    %571 = stablehlo.slice %arg136 [0:1, 0:257, 0:256, 0:24] : (tensor<1x257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %572 = stablehlo.reshape %571 : (tensor<1x257x256x24xf32>) -> tensor<257x256x24xf32>
    %573 = stablehlo.slice %arg137 [0:1, 0:256, 0:257, 0:24] : (tensor<1x256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %574 = stablehlo.reshape %573 : (tensor<1x256x257x24xf32>) -> tensor<256x257x24xf32>
    %575 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_241 = stablehlo.constant dense<0> : tensor<i32>
    %576 = stablehlo.broadcast_in_dim %c_241, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %577 = stablehlo.compare  GE, %575, %576,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_242 = stablehlo.constant dense<257> : tensor<i32>
    %578 = stablehlo.broadcast_in_dim %c_242, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %579 = stablehlo.compare  LT, %575, %578,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %580 = stablehlo.and %577, %579 : tensor<257xi1>
    %581 = stablehlo.reshape %580 : (tensor<257xi1>) -> tensor<257x1x1xi1>
    %c_243 = stablehlo.constant dense<true> : tensor<i1>
    %582 = stablehlo.broadcast_in_dim %c_243, dims = [] : (tensor<i1>) -> tensor<257x1x1xi1>
    %583 = stablehlo.and %582, %581 : tensor<257x1x1xi1>
    %584 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_244 = stablehlo.constant dense<12> : tensor<i32>
    %585 = stablehlo.broadcast_in_dim %c_244, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %586 = stablehlo.compare  LT, %584, %585,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_245 = stablehlo.constant dense<12> : tensor<i32>
    %587 = stablehlo.broadcast_in_dim %c_245, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %588 = stablehlo.subtract %584, %587 : tensor<24xi32>
    %c_246 = stablehlo.constant dense<257> : tensor<i32>
    %589 = stablehlo.broadcast_in_dim %c_246, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %590 = stablehlo.add %588, %589 : tensor<24xi32>
    %c_247 = stablehlo.constant dense<12> : tensor<i32>
    %591 = stablehlo.broadcast_in_dim %c_247, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %592 = stablehlo.subtract %590, %591 : tensor<24xi32>
    %593 = call @_where_13(%586, %584, %592) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_248 = stablehlo.constant dense<0> : tensor<i32>
    %594 = stablehlo.broadcast_in_dim %c_248, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %595 = stablehlo.compare  GE, %593, %594,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_249 = stablehlo.constant dense<257> : tensor<i32>
    %596 = stablehlo.broadcast_in_dim %c_249, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %597 = stablehlo.compare  LT, %593, %596,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %598 = stablehlo.and %595, %597 : tensor<24xi1>
    %599 = stablehlo.reshape %598 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %600 = stablehlo.broadcast_in_dim %583, dims = [0, 1, 2] : (tensor<257x1x1xi1>) -> tensor<257x24x1xi1>
    %601 = stablehlo.broadcast_in_dim %599, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<257x24x1xi1>
    %602 = stablehlo.and %600, %601 : tensor<257x24x1xi1>
    %603 = stablehlo.iota dim = 0 : tensor<257xi32>
    %604 = stablehlo.broadcast_in_dim %564, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %605 = stablehlo.add %603, %604 : tensor<257xi32>
    %c_250 = stablehlo.constant dense<0> : tensor<i32>
    %606 = stablehlo.broadcast_in_dim %c_250, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %607 = stablehlo.compare  GE, %605, %606,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_251 = stablehlo.constant dense<256> : tensor<i32>
    %608 = stablehlo.broadcast_in_dim %c_251, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %609 = stablehlo.compare  LT, %605, %608,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %610 = stablehlo.and %607, %609 : tensor<257xi1>
    %611 = stablehlo.reshape %610 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %612 = stablehlo.broadcast_in_dim %602, dims = [0, 1, 2] : (tensor<257x24x1xi1>) -> tensor<257x24x257xi1>
    %613 = stablehlo.broadcast_in_dim %611, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<257x24x257xi1>
    %614 = stablehlo.and %612, %613 : tensor<257x24x257xi1>
    %cst_252 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %615 = stablehlo.broadcast_in_dim %cst_252, dims = [] : (tensor<f32>) -> tensor<257x24x257xf32>
    %616 = call @_where_98(%614, %arg133, %615) : (tensor<257x24x257xi1>, tensor<257x24x257xf32>, tensor<257x24x257xf32>) -> tensor<257x24x257xf32>
    %617 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_253 = stablehlo.constant dense<12> : tensor<i32>
    %618 = stablehlo.broadcast_in_dim %c_253, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %619 = stablehlo.compare  LT, %617, %618,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_254 = stablehlo.constant dense<12> : tensor<i32>
    %620 = stablehlo.broadcast_in_dim %c_254, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %621 = stablehlo.subtract %617, %620 : tensor<24xi32>
    %c_255 = stablehlo.constant dense<257> : tensor<i32>
    %622 = stablehlo.broadcast_in_dim %c_255, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %623 = stablehlo.add %621, %622 : tensor<24xi32>
    %c_256 = stablehlo.constant dense<12> : tensor<i32>
    %624 = stablehlo.broadcast_in_dim %c_256, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %625 = stablehlo.subtract %623, %624 : tensor<24xi32>
    %626 = call @_where_13(%619, %617, %625) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_257 = stablehlo.constant dense<0> : tensor<i32>
    %627 = stablehlo.broadcast_in_dim %c_257, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %628 = stablehlo.compare  GE, %626, %627,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_258 = stablehlo.constant dense<257> : tensor<i32>
    %629 = stablehlo.broadcast_in_dim %c_258, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %630 = stablehlo.compare  LT, %626, %629,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %631 = stablehlo.and %628, %630 : tensor<24xi1>
    %632 = stablehlo.reshape %631 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_259 = stablehlo.constant dense<true> : tensor<i1>
    %633 = stablehlo.broadcast_in_dim %c_259, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %634 = stablehlo.and %633, %632 : tensor<24x1x1xi1>
    %635 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_260 = stablehlo.constant dense<0> : tensor<i32>
    %636 = stablehlo.broadcast_in_dim %c_260, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %637 = stablehlo.compare  GE, %635, %636,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_261 = stablehlo.constant dense<257> : tensor<i32>
    %638 = stablehlo.broadcast_in_dim %c_261, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %639 = stablehlo.compare  LT, %635, %638,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %640 = stablehlo.and %637, %639 : tensor<257xi1>
    %641 = stablehlo.reshape %640 : (tensor<257xi1>) -> tensor<1x257x1xi1>
    %642 = stablehlo.broadcast_in_dim %634, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x257x1xi1>
    %643 = stablehlo.broadcast_in_dim %641, dims = [0, 1, 2] : (tensor<1x257x1xi1>) -> tensor<24x257x1xi1>
    %644 = stablehlo.and %642, %643 : tensor<24x257x1xi1>
    %645 = stablehlo.iota dim = 0 : tensor<257xi32>
    %646 = stablehlo.broadcast_in_dim %564, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %647 = stablehlo.add %645, %646 : tensor<257xi32>
    %c_262 = stablehlo.constant dense<0> : tensor<i32>
    %648 = stablehlo.broadcast_in_dim %c_262, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %649 = stablehlo.compare  GE, %647, %648,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_263 = stablehlo.constant dense<256> : tensor<i32>
    %650 = stablehlo.broadcast_in_dim %c_263, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %651 = stablehlo.compare  LT, %647, %650,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %652 = stablehlo.and %649, %651 : tensor<257xi1>
    %653 = stablehlo.reshape %652 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %654 = stablehlo.broadcast_in_dim %644, dims = [0, 1, 2] : (tensor<24x257x1xi1>) -> tensor<24x257x257xi1>
    %655 = stablehlo.broadcast_in_dim %653, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<24x257x257xi1>
    %656 = stablehlo.and %654, %655 : tensor<24x257x257xi1>
    %cst_264 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %657 = stablehlo.broadcast_in_dim %cst_264, dims = [] : (tensor<f32>) -> tensor<24x257x257xf32>
    %658 = call @_where_89(%656, %arg134, %657) : (tensor<24x257x257xi1>, tensor<24x257x257xf32>, tensor<24x257x257xf32>) -> tensor<24x257x257xf32>
    %659 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_265 = stablehlo.constant dense<12> : tensor<i32>
    %660 = stablehlo.broadcast_in_dim %c_265, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %661 = stablehlo.compare  LT, %659, %660,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_266 = stablehlo.constant dense<12> : tensor<i32>
    %662 = stablehlo.broadcast_in_dim %c_266, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %663 = stablehlo.subtract %659, %662 : tensor<24xi32>
    %c_267 = stablehlo.constant dense<257> : tensor<i32>
    %664 = stablehlo.broadcast_in_dim %c_267, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %665 = stablehlo.add %663, %664 : tensor<24xi32>
    %c_268 = stablehlo.constant dense<12> : tensor<i32>
    %666 = stablehlo.broadcast_in_dim %c_268, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %667 = stablehlo.subtract %665, %666 : tensor<24xi32>
    %668 = call @_where_13(%661, %659, %667) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_269 = stablehlo.constant dense<0> : tensor<i32>
    %669 = stablehlo.broadcast_in_dim %c_269, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %670 = stablehlo.compare  GE, %668, %669,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_270 = stablehlo.constant dense<257> : tensor<i32>
    %671 = stablehlo.broadcast_in_dim %c_270, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %672 = stablehlo.compare  LT, %668, %671,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %673 = stablehlo.and %670, %672 : tensor<24xi1>
    %674 = stablehlo.reshape %673 : (tensor<24xi1>) -> tensor<24x1x1xi1>
    %c_271 = stablehlo.constant dense<true> : tensor<i1>
    %675 = stablehlo.broadcast_in_dim %c_271, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
    %676 = stablehlo.and %675, %674 : tensor<24x1x1xi1>
    %677 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_272 = stablehlo.constant dense<0> : tensor<i32>
    %678 = stablehlo.broadcast_in_dim %c_272, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %679 = stablehlo.compare  GE, %677, %678,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_273 = stablehlo.constant dense<256> : tensor<i32>
    %680 = stablehlo.broadcast_in_dim %c_273, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %681 = stablehlo.compare  LT, %677, %680,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %682 = stablehlo.and %679, %681 : tensor<256xi1>
    %683 = stablehlo.reshape %682 : (tensor<256xi1>) -> tensor<1x256x1xi1>
    %684 = stablehlo.broadcast_in_dim %676, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x256x1xi1>
    %685 = stablehlo.broadcast_in_dim %683, dims = [0, 1, 2] : (tensor<1x256x1xi1>) -> tensor<24x256x1xi1>
    %686 = stablehlo.and %684, %685 : tensor<24x256x1xi1>
    %687 = stablehlo.iota dim = 0 : tensor<257xi32>
    %688 = stablehlo.broadcast_in_dim %564, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %689 = stablehlo.add %687, %688 : tensor<257xi32>
    %c_274 = stablehlo.constant dense<0> : tensor<i32>
    %690 = stablehlo.broadcast_in_dim %c_274, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %691 = stablehlo.compare  GE, %689, %690,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_275 = stablehlo.constant dense<257> : tensor<i32>
    %692 = stablehlo.broadcast_in_dim %c_275, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %693 = stablehlo.compare  LT, %689, %692,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %694 = stablehlo.and %691, %693 : tensor<257xi1>
    %695 = stablehlo.reshape %694 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %696 = stablehlo.broadcast_in_dim %686, dims = [0, 1, 2] : (tensor<24x256x1xi1>) -> tensor<24x256x257xi1>
    %697 = stablehlo.broadcast_in_dim %695, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<24x256x257xi1>
    %698 = stablehlo.and %696, %697 : tensor<24x256x257xi1>
    %cst_276 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %699 = stablehlo.broadcast_in_dim %cst_276, dims = [] : (tensor<f32>) -> tensor<24x256x257xf32>
    %700 = call @_where_82(%698, %arg135, %699) : (tensor<24x256x257xi1>, tensor<24x256x257xf32>, tensor<24x256x257xf32>) -> tensor<24x256x257xf32>
    %701 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_277 = stablehlo.constant dense<0> : tensor<i32>
    %702 = stablehlo.broadcast_in_dim %c_277, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %703 = stablehlo.compare  GE, %701, %702,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_278 = stablehlo.constant dense<257> : tensor<i32>
    %704 = stablehlo.broadcast_in_dim %c_278, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %705 = stablehlo.compare  LT, %701, %704,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %706 = stablehlo.and %703, %705 : tensor<257xi1>
    %707 = stablehlo.reshape %706 : (tensor<257xi1>) -> tensor<257x1x1xi1>
    %c_279 = stablehlo.constant dense<true> : tensor<i1>
    %708 = stablehlo.broadcast_in_dim %c_279, dims = [] : (tensor<i1>) -> tensor<257x1x1xi1>
    %709 = stablehlo.and %708, %707 : tensor<257x1x1xi1>
    %710 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_280 = stablehlo.constant dense<0> : tensor<i32>
    %711 = stablehlo.broadcast_in_dim %c_280, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %712 = stablehlo.compare  GE, %710, %711,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_281 = stablehlo.constant dense<256> : tensor<i32>
    %713 = stablehlo.broadcast_in_dim %c_281, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %714 = stablehlo.compare  LT, %710, %713,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %715 = stablehlo.and %712, %714 : tensor<256xi1>
    %716 = stablehlo.reshape %715 : (tensor<256xi1>) -> tensor<1x256x1xi1>
    %717 = stablehlo.broadcast_in_dim %709, dims = [0, 1, 2] : (tensor<257x1x1xi1>) -> tensor<257x256x1xi1>
    %718 = stablehlo.broadcast_in_dim %716, dims = [0, 1, 2] : (tensor<1x256x1xi1>) -> tensor<257x256x1xi1>
    %719 = stablehlo.and %717, %718 : tensor<257x256x1xi1>
    %720 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_282 = stablehlo.constant dense<12> : tensor<i32>
    %721 = stablehlo.broadcast_in_dim %c_282, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %722 = stablehlo.compare  LT, %720, %721,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_283 = stablehlo.constant dense<12> : tensor<i32>
    %723 = stablehlo.broadcast_in_dim %c_283, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %724 = stablehlo.subtract %720, %723 : tensor<24xi32>
    %c_284 = stablehlo.constant dense<257> : tensor<i32>
    %725 = stablehlo.broadcast_in_dim %c_284, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %726 = stablehlo.add %724, %725 : tensor<24xi32>
    %c_285 = stablehlo.constant dense<12> : tensor<i32>
    %727 = stablehlo.broadcast_in_dim %c_285, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %728 = stablehlo.subtract %726, %727 : tensor<24xi32>
    %729 = call @_where_13(%722, %720, %728) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_286 = stablehlo.constant dense<0> : tensor<i32>
    %730 = stablehlo.broadcast_in_dim %c_286, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %731 = stablehlo.compare  GE, %729, %730,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_287 = stablehlo.constant dense<257> : tensor<i32>
    %732 = stablehlo.broadcast_in_dim %c_287, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %733 = stablehlo.compare  LT, %729, %732,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %734 = stablehlo.and %731, %733 : tensor<24xi1>
    %735 = stablehlo.broadcast_in_dim %564, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %736 = stablehlo.compare  GE, %729, %735,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_288 = stablehlo.constant dense<257> : tensor<i32>
    %737 = stablehlo.add %564, %c_288 : tensor<i32>
    %738 = stablehlo.broadcast_in_dim %737, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %739 = stablehlo.compare  LT, %729, %738,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %740 = stablehlo.and %736, %739 : tensor<24xi1>
    %741 = stablehlo.and %734, %740 : tensor<24xi1>
    %742 = stablehlo.reshape %741 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %743 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2] : (tensor<257x256x1xi1>) -> tensor<257x256x24xi1>
    %744 = stablehlo.broadcast_in_dim %742, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<257x256x24xi1>
    %745 = stablehlo.and %743, %744 : tensor<257x256x24xi1>
    %cst_289 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %746 = stablehlo.broadcast_in_dim %cst_289, dims = [] : (tensor<f32>) -> tensor<257x256x24xf32>
    %747 = call @_where_92(%745, %572, %746) : (tensor<257x256x24xi1>, tensor<257x256x24xf32>, tensor<257x256x24xf32>) -> tensor<257x256x24xf32>
    %748 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_290 = stablehlo.constant dense<0> : tensor<i32>
    %749 = stablehlo.broadcast_in_dim %c_290, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %750 = stablehlo.compare  GE, %748, %749,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_291 = stablehlo.constant dense<256> : tensor<i32>
    %751 = stablehlo.broadcast_in_dim %c_291, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %752 = stablehlo.compare  LT, %748, %751,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %753 = stablehlo.and %750, %752 : tensor<256xi1>
    %754 = stablehlo.reshape %753 : (tensor<256xi1>) -> tensor<256x1x1xi1>
    %c_292 = stablehlo.constant dense<true> : tensor<i1>
    %755 = stablehlo.broadcast_in_dim %c_292, dims = [] : (tensor<i1>) -> tensor<256x1x1xi1>
    %756 = stablehlo.and %755, %754 : tensor<256x1x1xi1>
    %757 = stablehlo.iota dim = 0 : tensor<257xi32>
    %c_293 = stablehlo.constant dense<0> : tensor<i32>
    %758 = stablehlo.broadcast_in_dim %c_293, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %759 = stablehlo.compare  GE, %757, %758,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_294 = stablehlo.constant dense<257> : tensor<i32>
    %760 = stablehlo.broadcast_in_dim %c_294, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %761 = stablehlo.compare  LT, %757, %760,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %762 = stablehlo.and %759, %761 : tensor<257xi1>
    %763 = stablehlo.reshape %762 : (tensor<257xi1>) -> tensor<1x257x1xi1>
    %764 = stablehlo.broadcast_in_dim %756, dims = [0, 1, 2] : (tensor<256x1x1xi1>) -> tensor<256x257x1xi1>
    %765 = stablehlo.broadcast_in_dim %763, dims = [0, 1, 2] : (tensor<1x257x1xi1>) -> tensor<256x257x1xi1>
    %766 = stablehlo.and %764, %765 : tensor<256x257x1xi1>
    %767 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_295 = stablehlo.constant dense<12> : tensor<i32>
    %768 = stablehlo.broadcast_in_dim %c_295, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %769 = stablehlo.compare  LT, %767, %768,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_296 = stablehlo.constant dense<12> : tensor<i32>
    %770 = stablehlo.broadcast_in_dim %c_296, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %771 = stablehlo.subtract %767, %770 : tensor<24xi32>
    %c_297 = stablehlo.constant dense<257> : tensor<i32>
    %772 = stablehlo.broadcast_in_dim %c_297, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %773 = stablehlo.add %771, %772 : tensor<24xi32>
    %c_298 = stablehlo.constant dense<12> : tensor<i32>
    %774 = stablehlo.broadcast_in_dim %c_298, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %775 = stablehlo.subtract %773, %774 : tensor<24xi32>
    %776 = call @_where_13(%769, %767, %775) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_299 = stablehlo.constant dense<0> : tensor<i32>
    %777 = stablehlo.broadcast_in_dim %c_299, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %778 = stablehlo.compare  GE, %776, %777,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_300 = stablehlo.constant dense<257> : tensor<i32>
    %779 = stablehlo.broadcast_in_dim %c_300, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %780 = stablehlo.compare  LT, %776, %779,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %781 = stablehlo.and %778, %780 : tensor<24xi1>
    %782 = stablehlo.broadcast_in_dim %564, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %783 = stablehlo.compare  GE, %776, %782,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_301 = stablehlo.constant dense<257> : tensor<i32>
    %784 = stablehlo.add %564, %c_301 : tensor<i32>
    %785 = stablehlo.broadcast_in_dim %784, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %786 = stablehlo.compare  LT, %776, %785,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %787 = stablehlo.and %783, %786 : tensor<24xi1>
    %788 = stablehlo.and %781, %787 : tensor<24xi1>
    %789 = stablehlo.reshape %788 : (tensor<24xi1>) -> tensor<1x1x24xi1>
    %790 = stablehlo.broadcast_in_dim %766, dims = [0, 1, 2] : (tensor<256x257x1xi1>) -> tensor<256x257x24xi1>
    %791 = stablehlo.broadcast_in_dim %789, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<256x257x24xi1>
    %792 = stablehlo.and %790, %791 : tensor<256x257x24xi1>
    %cst_302 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %793 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<f32>) -> tensor<256x257x24xf32>
    %794 = call @_where_91(%792, %574, %793) : (tensor<256x257x24xi1>, tensor<256x257x24xf32>, tensor<256x257x24xf32>) -> tensor<256x257x24xf32>
    %795 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_303 = stablehlo.constant dense<0> : tensor<i32>
    %796 = stablehlo.broadcast_in_dim %c_303, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %797 = stablehlo.compare  GE, %795, %796,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %c_304 = stablehlo.constant dense<256> : tensor<i32>
    %798 = stablehlo.broadcast_in_dim %c_304, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %799 = stablehlo.compare  LT, %795, %798,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %800 = stablehlo.and %797, %799 : tensor<256xi1>
    %801 = stablehlo.reshape %800 : (tensor<256xi1>) -> tensor<256x1x1xi1>
    %c_305 = stablehlo.constant dense<true> : tensor<i1>
    %802 = stablehlo.broadcast_in_dim %c_305, dims = [] : (tensor<i1>) -> tensor<256x1x1xi1>
    %803 = stablehlo.and %802, %801 : tensor<256x1x1xi1>
    %804 = stablehlo.iota dim = 0 : tensor<24xi32>
    %c_306 = stablehlo.constant dense<12> : tensor<i32>
    %805 = stablehlo.broadcast_in_dim %c_306, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %806 = stablehlo.compare  LT, %804, %805,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_307 = stablehlo.constant dense<12> : tensor<i32>
    %807 = stablehlo.broadcast_in_dim %c_307, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %808 = stablehlo.subtract %804, %807 : tensor<24xi32>
    %c_308 = stablehlo.constant dense<257> : tensor<i32>
    %809 = stablehlo.broadcast_in_dim %c_308, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %810 = stablehlo.add %808, %809 : tensor<24xi32>
    %c_309 = stablehlo.constant dense<12> : tensor<i32>
    %811 = stablehlo.broadcast_in_dim %c_309, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %812 = stablehlo.subtract %810, %811 : tensor<24xi32>
    %813 = call @_where_13(%806, %804, %812) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
    %c_310 = stablehlo.constant dense<0> : tensor<i32>
    %814 = stablehlo.broadcast_in_dim %c_310, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %815 = stablehlo.compare  GE, %813, %814,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %c_311 = stablehlo.constant dense<257> : tensor<i32>
    %816 = stablehlo.broadcast_in_dim %c_311, dims = [] : (tensor<i32>) -> tensor<24xi32>
    %817 = stablehlo.compare  LT, %813, %816,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
    %818 = stablehlo.and %815, %817 : tensor<24xi1>
    %819 = stablehlo.reshape %818 : (tensor<24xi1>) -> tensor<1x24x1xi1>
    %820 = stablehlo.broadcast_in_dim %803, dims = [0, 1, 2] : (tensor<256x1x1xi1>) -> tensor<256x24x1xi1>
    %821 = stablehlo.broadcast_in_dim %819, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<256x24x1xi1>
    %822 = stablehlo.and %820, %821 : tensor<256x24x1xi1>
    %823 = stablehlo.iota dim = 0 : tensor<257xi32>
    %824 = stablehlo.broadcast_in_dim %564, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %825 = stablehlo.add %823, %824 : tensor<257xi32>
    %c_312 = stablehlo.constant dense<0> : tensor<i32>
    %826 = stablehlo.broadcast_in_dim %c_312, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %827 = stablehlo.compare  GE, %825, %826,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %c_313 = stablehlo.constant dense<257> : tensor<i32>
    %828 = stablehlo.broadcast_in_dim %c_313, dims = [] : (tensor<i32>) -> tensor<257xi32>
    %829 = stablehlo.compare  LT, %825, %828,  SIGNED : (tensor<257xi32>, tensor<257xi32>) -> tensor<257xi1>
    %830 = stablehlo.and %827, %829 : tensor<257xi1>
    %831 = stablehlo.reshape %830 : (tensor<257xi1>) -> tensor<1x1x257xi1>
    %832 = stablehlo.broadcast_in_dim %822, dims = [0, 1, 2] : (tensor<256x24x1xi1>) -> tensor<256x24x257xi1>
    %833 = stablehlo.broadcast_in_dim %831, dims = [0, 1, 2] : (tensor<1x1x257xi1>) -> tensor<256x24x257xi1>
    %834 = stablehlo.and %832, %833 : tensor<256x24x257xi1>
    %cst_314 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %835 = stablehlo.broadcast_in_dim %cst_314, dims = [] : (tensor<f32>) -> tensor<256x24x257xf32>
    %836 = call @_where_73(%834, %arg138, %835) : (tensor<256x24x257xi1>, tensor<256x24x257xf32>, tensor<256x24x257xf32>) -> tensor<256x24x257xf32>
    %cst_315 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %837 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<f32>) -> tensor<256x256x1xf32>
    %cst_316 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %838 = stablehlo.broadcast_in_dim %cst_316, dims = [] : (tensor<f32>) -> tensor<256x256x1xf32>
    %839 = stablehlo.slice %461 [0:256, 0:257, 256:257] : (tensor<256x257x257xf32>) -> tensor<256x257x1xf32>
    %840 = "stablehlo.collective_permute"(%839) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<256x257x1xf32>) -> tensor<256x257x1xf32>
    %cst_317 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %841 = stablehlo.broadcast_in_dim %cst_317, dims = [] : (tensor<f32>) -> tensor<256x257x1xf32>
    %842 = stablehlo.slice %559 [0:257, 0:256, 256:257] : (tensor<257x256x257xf32>) -> tensor<257x256x1xf32>
    %843 = "stablehlo.collective_permute"(%842) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<257x256x1xf32>) -> tensor<257x256x1xf32>
    %cst_318 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %844 = stablehlo.broadcast_in_dim %cst_318, dims = [] : (tensor<f32>) -> tensor<257x256x1xf32>
    %845 = stablehlo.transpose %arg49, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %846 = stablehlo.transpose %arg50, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %847 = stablehlo.transpose %arg51, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %848 = stablehlo.transpose %arg52, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %849 = stablehlo.transpose %arg53, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %850 = stablehlo.transpose %arg54, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %851 = stablehlo.transpose %arg55, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %852 = stablehlo.transpose %arg56, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %853 = stablehlo.transpose %arg57, dims = [2, 0, 1] : (tensor<1x1x24xf32>) -> tensor<24x1x1xf32>
    %854 = stablehlo.transpose %arg58, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %855 = stablehlo.transpose %arg59, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %856 = stablehlo.transpose %arg60, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %857 = stablehlo.transpose %arg43, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %858 = stablehlo.transpose %arg44, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %859 = stablehlo.transpose %arg45, dims = [2, 0, 1] : (tensor<1x24x1xf32>) -> tensor<1x1x24xf32>
    %860 = stablehlo.transpose %arg46, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %861 = stablehlo.transpose %arg47, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %862 = stablehlo.transpose %arg48, dims = [2, 0, 1] : (tensor<24x1x1xf32>) -> tensor<1x24x1xf32>
    %863 = stablehlo.slice %570 [0:1, 0:1] : (tensor<7x3xi32>) -> tensor<1x1xi32>
    %864 = stablehlo.reshape %863 : (tensor<1x1xi32>) -> tensor<i32>
    %c_319 = stablehlo.constant dense<0> : tensor<i32>
    %865 = stablehlo.compare  LT, %864, %c_319,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_320 = stablehlo.constant dense<3> : tensor<i32>
    %866 = stablehlo.add %864, %c_320 : tensor<i32>
    %867 = stablehlo.select %865, %866, %864 : tensor<i1>, tensor<i32>
    %868 = stablehlo.dynamic_slice %arg61, %867, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
    %869 = stablehlo.reshape %868 : (tensor<1xi32>) -> tensor<i32>
    %870 = stablehlo.slice %570 [0:1, 0:3] : (tensor<7x3xi32>) -> tensor<1x3xi32>
    %c_321 = stablehlo.constant dense<0> : tensor<i32>
    %871 = stablehlo.broadcast_in_dim %c_321, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_322 = stablehlo.constant dense<0> : tensor<i32>
    %872 = stablehlo.broadcast_in_dim %c_322, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %873 = stablehlo.concatenate %871, %872, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %874 = "stablehlo.scatter"(%870, %873, %869) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg148: tensor<i32>, %arg149: tensor<i32>):
      stablehlo.return %arg149 : tensor<i32>
    }) : (tensor<1x3xi32>, tensor<2xi32>, tensor<i32>) -> tensor<1x3xi32>
    %c_323 = stablehlo.constant dense<0> : tensor<i32>
    %875 = stablehlo.broadcast_in_dim %c_323, dims = [] : (tensor<i32>) -> tensor<6xi32>
    %876 = stablehlo.compare  LT, %arg62, %875,  SIGNED : (tensor<6xi32>, tensor<6xi32>) -> tensor<6xi1>
    %c_324 = stablehlo.constant dense<7> : tensor<i32>
    %877 = stablehlo.broadcast_in_dim %c_324, dims = [] : (tensor<i32>) -> tensor<6xi32>
    %878 = stablehlo.add %arg62, %877 : tensor<6xi32>
    %879 = stablehlo.select %876, %878, %arg62 : tensor<6xi1>, tensor<6xi32>
    %880 = stablehlo.broadcast_in_dim %879, dims = [0] : (tensor<6xi32>) -> tensor<6x1xi32>
    %881 = "stablehlo.gather"(%570, %880) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3>}> : (tensor<7x3xi32>, tensor<6x1xi32>) -> tensor<6x3xi32>
    %c_325 = stablehlo.constant dense<0> : tensor<i32>
    %882 = stablehlo.broadcast_in_dim %c_325, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %883 = stablehlo.compare  LT, %arg63, %882,  SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
    %c_326 = stablehlo.constant dense<3> : tensor<i32>
    %884 = stablehlo.broadcast_in_dim %c_326, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %885 = stablehlo.add %arg63, %884 : tensor<3xi32>
    %886 = stablehlo.select %883, %885, %arg63 : tensor<3xi1>, tensor<3xi32>
    %887 = stablehlo.broadcast_in_dim %886, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %888 = "stablehlo.gather"(%881, %887) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 6, 1>}> : (tensor<6x3xi32>, tensor<3x1xi32>) -> tensor<6x3xi32>
    %889 = stablehlo.concatenate %874, %888, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
    %890 = stablehlo.transpose %840, dims = [2, 0, 1] : (tensor<256x257x1xf32>) -> tensor<1x256x257xf32>
    %891 = stablehlo.transpose %841, dims = [2, 0, 1] : (tensor<256x257x1xf32>) -> tensor<1x256x257xf32>
    %892 = stablehlo.transpose %843, dims = [2, 0, 1] : (tensor<257x256x1xf32>) -> tensor<1x257x256xf32>
    %893 = stablehlo.transpose %844, dims = [2, 0, 1] : (tensor<257x256x1xf32>) -> tensor<1x257x256xf32>
    %894 = stablehlo.transpose %837, dims = [2, 0, 1] : (tensor<256x256x1xf32>) -> tensor<1x256x256xf32>
    %895 = stablehlo.transpose %838, dims = [2, 0, 1] : (tensor<256x256x1xf32>) -> tensor<1x256x256xf32>
    %896 = stablehlo.transpose %arg122, dims = [2, 0, 1] : (tensor<257x256x257xf32>) -> tensor<257x257x256xf32>
    %897 = stablehlo.transpose %arg123, dims = [2, 0, 1] : (tensor<256x257x257xf32>) -> tensor<257x256x257xf32>
    %898 = stablehlo.transpose %arg121, dims = [2, 0, 1] : (tensor<257x257x257xf32>) -> tensor<257x257x257xf32>
    %899 = stablehlo.transpose %461, dims = [2, 0, 1] : (tensor<256x257x257xf32>) -> tensor<257x256x257xf32>
    %900 = stablehlo.transpose %559, dims = [2, 0, 1] : (tensor<257x256x257xf32>) -> tensor<257x257x256xf32>
    %901 = stablehlo.transpose %353, dims = [2, 0, 1] : (tensor<256x256x257xf32>) -> tensor<257x256x256xf32>
    %902 = stablehlo.transpose %arg41, dims = [2, 0, 1] : (tensor<257x256x257xf32>) -> tensor<257x257x256xf32>
    %903 = stablehlo.transpose %arg42, dims = [2, 0, 1] : (tensor<256x257x257xf32>) -> tensor<257x256x257xf32>
    %904 = stablehlo.transpose %arg40, dims = [2, 0, 1] : (tensor<257x257x257xf32>) -> tensor<257x257x257xf32>
    %905 = stablehlo.transpose %700, dims = [2, 0, 1] : (tensor<24x256x257xf32>) -> tensor<257x24x256xf32>
    %906 = stablehlo.transpose %747, dims = [2, 0, 1] : (tensor<257x256x24xf32>) -> tensor<24x257x256xf32>
    %907 = stablehlo.transpose %794, dims = [2, 0, 1] : (tensor<256x257x24xf32>) -> tensor<24x256x257xf32>
    %908 = stablehlo.transpose %836, dims = [2, 0, 1] : (tensor<256x24x257xf32>) -> tensor<257x256x24xf32>
    %909 = stablehlo.transpose %616, dims = [2, 0, 1] : (tensor<257x24x257xf32>) -> tensor<257x257x24xf32>
    %910 = stablehlo.transpose %658, dims = [2, 0, 1] : (tensor<24x257x257xf32>) -> tensor<257x24x257xf32>
    %911:9 = stablehlo.custom_call @beamz_cuda_sharded(%896, %897, %898, %899, %900, %901, %arg38, %arg39, %arg37, %902, %903, %904, %arg64, %845, %846, %847, %848, %849, %850, %851, %852, %853, %854, %855, %856, %857, %858, %859, %860, %861, %862, %905, %906, %907, %908, %909, %910, %arg9, %arg9, %arg9, %889, %890, %891, %892, %893, %894, %895) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 1 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<257x257x256xf32>, tensor<257x256x257xf32>, tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<257x257x256xf32>, tensor<257x256x256xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<257x257x256xf32>, tensor<257x256x257xf32>, tensor<257x257x257xf32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<257x24x256xf32>, tensor<24x257x256xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<257x257x24xf32>, tensor<257x24x257xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<1x256x257xf32>, tensor<1x256x257xf32>, tensor<1x257x256xf32>, tensor<1x257x256xf32>, tensor<1x256x256xf32>, tensor<1x256x256xf32>) -> (tensor<257x257x256xf32>, tensor<257x256x257xf32>, tensor<257x257x257xf32>, tensor<257x24x256xf32>, tensor<24x257x256xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<257x257x24xf32>, tensor<257x24x257xf32>)
    %912 = stablehlo.transpose %911#2, dims = [1, 2, 0] : (tensor<257x257x257xf32>) -> tensor<257x257x257xf32>
    %913 = stablehlo.transpose %911#0, dims = [1, 2, 0] : (tensor<257x257x256xf32>) -> tensor<257x256x257xf32>
    %914 = stablehlo.transpose %911#1, dims = [1, 2, 0] : (tensor<257x256x257xf32>) -> tensor<256x257x257xf32>
    %915 = stablehlo.transpose %911#7, dims = [1, 2, 0] : (tensor<257x257x24xf32>) -> tensor<257x24x257xf32>
    %916 = stablehlo.transpose %911#8, dims = [1, 2, 0] : (tensor<257x24x257xf32>) -> tensor<24x257x257xf32>
    %917 = stablehlo.transpose %911#3, dims = [1, 2, 0] : (tensor<257x24x256xf32>) -> tensor<24x256x257xf32>
    %918 = stablehlo.transpose %911#4, dims = [1, 2, 0] : (tensor<24x257x256xf32>) -> tensor<257x256x24xf32>
    %919 = stablehlo.transpose %911#5, dims = [1, 2, 0] : (tensor<24x256x257xf32>) -> tensor<256x257x24xf32>
    %920 = stablehlo.transpose %911#6, dims = [1, 2, 0] : (tensor<257x256x24xf32>) -> tensor<256x24x257xf32>
    %921 = stablehlo.broadcast_in_dim %918, dims = [1, 2, 3] : (tensor<257x256x24xf32>) -> tensor<1x257x256x24xf32>
    %922 = stablehlo.broadcast_in_dim %919, dims = [1, 2, 3] : (tensor<256x257x24xf32>) -> tensor<1x256x257x24xf32>
    %c_327 = stablehlo.constant dense<0> : tensor<i32>
    %c_328 = stablehlo.constant dense<31> : tensor<i32>
    %923 = call @clip(%arg146, %c_327, %c_328) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_329 = stablehlo.constant dense<1> : tensor<ui32>
    %c_330 = stablehlo.constant dense<1> : tensor<ui32>
    %924 = stablehlo.partition_id : tensor<ui32>
    %925 = stablehlo.divide %924, %c_329 : tensor<ui32>
    %926 = stablehlo.remainder %925, %c_330 : tensor<ui32>
    %927 = stablehlo.convert %926 : (tensor<ui32>) -> tensor<i32>
    %c_331 = stablehlo.constant dense<257> : tensor<i32>
    %928 = stablehlo.multiply %927, %c_331 : tensor<i32>
    %c_332 = stablehlo.constant dense<75> : tensor<i32>
    %929 = stablehlo.subtract %c_332, %928 : tensor<i32>
    %c_333 = stablehlo.constant dense<0> : tensor<i32>
    %c_334 = stablehlo.constant dense<256> : tensor<i32>
    %930 = call @clip_170(%929, %c_333, %c_334) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %931 = stablehlo.add %928, %930 : tensor<i32>
    %932 = stablehlo.iota dim = 0 : tensor<1xi32>
    %933 = stablehlo.broadcast_in_dim %931, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %934 = stablehlo.add %933, %932 : tensor<1xi32>
    %c_335 = stablehlo.constant dense<75> : tensor<i32>
    %935 = stablehlo.broadcast_in_dim %c_335, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %936 = stablehlo.subtract %934, %935 : tensor<1xi32>
    %c_336 = stablehlo.constant dense<0> : tensor<i32>
    %937 = stablehlo.broadcast_in_dim %c_336, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %938 = stablehlo.compare  GE, %936, %937,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_337 = stablehlo.constant dense<1> : tensor<i32>
    %939 = stablehlo.broadcast_in_dim %c_337, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %940 = stablehlo.compare  LT, %936, %939,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %941 = stablehlo.and %938, %940 : tensor<1xi1>
    %942 = stablehlo.slice %arg65 [0:1, 0:15, 0:26, 0:1] : (tensor<2x15x26x1xf32>) -> tensor<1x15x26x1xf32>
    %943 = stablehlo.reshape %942 : (tensor<1x15x26x1xf32>) -> tensor<15x26x1xf32>
    %944 = call @_take_225(%943, %936) : (tensor<15x26x1xf32>, tensor<1xi32>) -> tensor<15x26x1xf32>
    %945 = stablehlo.reshape %941 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_338 = stablehlo.constant dense<0> : tensor<i32>
    %946 = stablehlo.compare  LT, %923, %c_338,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_339 = stablehlo.constant dense<32> : tensor<i32>
    %947 = stablehlo.add %923, %c_339 : tensor<i32>
    %948 = stablehlo.select %946, %947, %923 : tensor<i1>, tensor<i32>
    %c_340 = stablehlo.constant dense<0> : tensor<i32>
    %949 = stablehlo.dynamic_slice %arg66, %c_340, %948, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %950 = stablehlo.reshape %949 : (tensor<1x1xf32>) -> tensor<f32>
    %951 = stablehlo.broadcast_in_dim %950, dims = [] : (tensor<f32>) -> tensor<15x26x1xf32>
    %952 = stablehlo.multiply %944, %951 : tensor<15x26x1xf32>
    %c_341 = stablehlo.constant dense<0> : tensor<i32>
    %953 = call @_where_229(%945, %952, %c_341) : (tensor<1x1x1xi1>, tensor<15x26x1xf32>, tensor<i32>) -> tensor<15x26x1xf32>
    %c_342 = stablehlo.constant dense<121> : tensor<i32>
    %c_343 = stablehlo.constant dense<0> : tensor<i32>
    %954 = stablehlo.compare  LT, %c_342, %c_343,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_344 = stablehlo.constant dense<121> : tensor<i32>
    %c_345 = stablehlo.constant dense<257> : tensor<i32>
    %955 = stablehlo.add %c_344, %c_345 : tensor<i32>
    %c_346 = stablehlo.constant dense<121> : tensor<i32>
    %956 = stablehlo.select %954, %955, %c_346 : tensor<i1>, tensor<i32>
    %c_347 = stablehlo.constant dense<115> : tensor<i32>
    %c_348 = stablehlo.constant dense<0> : tensor<i32>
    %957 = stablehlo.compare  LT, %c_347, %c_348,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_349 = stablehlo.constant dense<115> : tensor<i32>
    %c_350 = stablehlo.constant dense<256> : tensor<i32>
    %958 = stablehlo.add %c_349, %c_350 : tensor<i32>
    %c_351 = stablehlo.constant dense<115> : tensor<i32>
    %959 = stablehlo.select %957, %958, %c_351 : tensor<i1>, tensor<i32>
    %c_352 = stablehlo.constant dense<0> : tensor<i32>
    %960 = stablehlo.compare  LT, %930, %c_352,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_353 = stablehlo.constant dense<257> : tensor<i32>
    %961 = stablehlo.add %930, %c_353 : tensor<i32>
    %962 = stablehlo.select %960, %961, %930 : tensor<i1>, tensor<i32>
    %963 = stablehlo.dynamic_slice %913, %956, %959, %962, sizes = [15, 26, 1] : (tensor<257x256x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<15x26x1xf32>
    %964 = stablehlo.add %963, %953 : tensor<15x26x1xf32>
    %c_354 = stablehlo.constant dense<121> : tensor<i32>
    %c_355 = stablehlo.constant dense<0> : tensor<i32>
    %965 = stablehlo.compare  LT, %c_354, %c_355,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_356 = stablehlo.constant dense<121> : tensor<i32>
    %c_357 = stablehlo.constant dense<257> : tensor<i32>
    %966 = stablehlo.add %c_356, %c_357 : tensor<i32>
    %c_358 = stablehlo.constant dense<121> : tensor<i32>
    %967 = stablehlo.select %965, %966, %c_358 : tensor<i1>, tensor<i32>
    %c_359 = stablehlo.constant dense<115> : tensor<i32>
    %c_360 = stablehlo.constant dense<0> : tensor<i32>
    %968 = stablehlo.compare  LT, %c_359, %c_360,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_361 = stablehlo.constant dense<115> : tensor<i32>
    %c_362 = stablehlo.constant dense<256> : tensor<i32>
    %969 = stablehlo.add %c_361, %c_362 : tensor<i32>
    %c_363 = stablehlo.constant dense<115> : tensor<i32>
    %970 = stablehlo.select %968, %969, %c_363 : tensor<i1>, tensor<i32>
    %c_364 = stablehlo.constant dense<0> : tensor<i32>
    %971 = stablehlo.compare  LT, %930, %c_364,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_365 = stablehlo.constant dense<257> : tensor<i32>
    %972 = stablehlo.add %930, %c_365 : tensor<i32>
    %973 = stablehlo.select %971, %972, %930 : tensor<i1>, tensor<i32>
    %974 = stablehlo.dynamic_update_slice %913, %964, %967, %970, %973 : (tensor<257x256x257xf32>, tensor<15x26x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<257x256x257xf32>
    %c_366 = stablehlo.constant dense<75> : tensor<i32>
    %975 = stablehlo.subtract %c_366, %928 : tensor<i32>
    %c_367 = stablehlo.constant dense<0> : tensor<i32>
    %c_368 = stablehlo.constant dense<256> : tensor<i32>
    %976 = call @clip_170(%975, %c_367, %c_368) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %977 = stablehlo.add %928, %976 : tensor<i32>
    %978 = stablehlo.iota dim = 0 : tensor<1xi32>
    %979 = stablehlo.broadcast_in_dim %977, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %980 = stablehlo.add %979, %978 : tensor<1xi32>
    %c_369 = stablehlo.constant dense<75> : tensor<i32>
    %981 = stablehlo.broadcast_in_dim %c_369, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %982 = stablehlo.subtract %980, %981 : tensor<1xi32>
    %c_370 = stablehlo.constant dense<0> : tensor<i32>
    %983 = stablehlo.broadcast_in_dim %c_370, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %984 = stablehlo.compare  GE, %982, %983,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_371 = stablehlo.constant dense<1> : tensor<i32>
    %985 = stablehlo.broadcast_in_dim %c_371, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %986 = stablehlo.compare  LT, %982, %985,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %987 = stablehlo.and %984, %986 : tensor<1xi1>
    %988 = stablehlo.slice %arg65 [1:2, 0:15, 0:26, 0:1] : (tensor<2x15x26x1xf32>) -> tensor<1x15x26x1xf32>
    %989 = stablehlo.reshape %988 : (tensor<1x15x26x1xf32>) -> tensor<15x26x1xf32>
    %990 = call @_take_225(%989, %982) : (tensor<15x26x1xf32>, tensor<1xi32>) -> tensor<15x26x1xf32>
    %991 = stablehlo.reshape %987 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_372 = stablehlo.constant dense<0> : tensor<i32>
    %992 = stablehlo.compare  LT, %923, %c_372,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_373 = stablehlo.constant dense<32> : tensor<i32>
    %993 = stablehlo.add %923, %c_373 : tensor<i32>
    %994 = stablehlo.select %992, %993, %923 : tensor<i1>, tensor<i32>
    %c_374 = stablehlo.constant dense<1> : tensor<i32>
    %995 = stablehlo.dynamic_slice %arg66, %c_374, %994, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %996 = stablehlo.reshape %995 : (tensor<1x1xf32>) -> tensor<f32>
    %997 = stablehlo.broadcast_in_dim %996, dims = [] : (tensor<f32>) -> tensor<15x26x1xf32>
    %998 = stablehlo.multiply %990, %997 : tensor<15x26x1xf32>
    %c_375 = stablehlo.constant dense<0> : tensor<i32>
    %999 = call @_where_229(%991, %998, %c_375) : (tensor<1x1x1xi1>, tensor<15x26x1xf32>, tensor<i32>) -> tensor<15x26x1xf32>
    %c_376 = stablehlo.constant dense<121> : tensor<i32>
    %c_377 = stablehlo.constant dense<0> : tensor<i32>
    %1000 = stablehlo.compare  LT, %c_376, %c_377,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_378 = stablehlo.constant dense<121> : tensor<i32>
    %c_379 = stablehlo.constant dense<257> : tensor<i32>
    %1001 = stablehlo.add %c_378, %c_379 : tensor<i32>
    %c_380 = stablehlo.constant dense<121> : tensor<i32>
    %1002 = stablehlo.select %1000, %1001, %c_380 : tensor<i1>, tensor<i32>
    %c_381 = stablehlo.constant dense<115> : tensor<i32>
    %c_382 = stablehlo.constant dense<0> : tensor<i32>
    %1003 = stablehlo.compare  LT, %c_381, %c_382,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_383 = stablehlo.constant dense<115> : tensor<i32>
    %c_384 = stablehlo.constant dense<256> : tensor<i32>
    %1004 = stablehlo.add %c_383, %c_384 : tensor<i32>
    %c_385 = stablehlo.constant dense<115> : tensor<i32>
    %1005 = stablehlo.select %1003, %1004, %c_385 : tensor<i1>, tensor<i32>
    %c_386 = stablehlo.constant dense<0> : tensor<i32>
    %1006 = stablehlo.compare  LT, %976, %c_386,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_387 = stablehlo.constant dense<257> : tensor<i32>
    %1007 = stablehlo.add %976, %c_387 : tensor<i32>
    %1008 = stablehlo.select %1006, %1007, %976 : tensor<i1>, tensor<i32>
    %1009 = stablehlo.dynamic_slice %974, %1002, %1005, %1008, sizes = [15, 26, 1] : (tensor<257x256x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<15x26x1xf32>
    %1010 = stablehlo.add %1009, %999 : tensor<15x26x1xf32>
    %c_388 = stablehlo.constant dense<121> : tensor<i32>
    %c_389 = stablehlo.constant dense<0> : tensor<i32>
    %1011 = stablehlo.compare  LT, %c_388, %c_389,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_390 = stablehlo.constant dense<121> : tensor<i32>
    %c_391 = stablehlo.constant dense<257> : tensor<i32>
    %1012 = stablehlo.add %c_390, %c_391 : tensor<i32>
    %c_392 = stablehlo.constant dense<121> : tensor<i32>
    %1013 = stablehlo.select %1011, %1012, %c_392 : tensor<i1>, tensor<i32>
    %c_393 = stablehlo.constant dense<115> : tensor<i32>
    %c_394 = stablehlo.constant dense<0> : tensor<i32>
    %1014 = stablehlo.compare  LT, %c_393, %c_394,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_395 = stablehlo.constant dense<115> : tensor<i32>
    %c_396 = stablehlo.constant dense<256> : tensor<i32>
    %1015 = stablehlo.add %c_395, %c_396 : tensor<i32>
    %c_397 = stablehlo.constant dense<115> : tensor<i32>
    %1016 = stablehlo.select %1014, %1015, %c_397 : tensor<i1>, tensor<i32>
    %c_398 = stablehlo.constant dense<0> : tensor<i32>
    %1017 = stablehlo.compare  LT, %976, %c_398,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_399 = stablehlo.constant dense<257> : tensor<i32>
    %1018 = stablehlo.add %976, %c_399 : tensor<i32>
    %1019 = stablehlo.select %1017, %1018, %976 : tensor<i1>, tensor<i32>
    %1020 = stablehlo.dynamic_update_slice %974, %1010, %1013, %1016, %1019 : (tensor<257x256x257xf32>, tensor<15x26x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<257x256x257xf32>
    %c_400 = stablehlo.constant dense<0> : tensor<i32>
    %c_401 = stablehlo.constant dense<31> : tensor<i32>
    %1021 = call @clip(%arg146, %c_400, %c_401) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_402 = stablehlo.constant dense<1> : tensor<ui32>
    %c_403 = stablehlo.constant dense<1> : tensor<ui32>
    %1022 = stablehlo.partition_id : tensor<ui32>
    %1023 = stablehlo.divide %1022, %c_402 : tensor<ui32>
    %1024 = stablehlo.remainder %1023, %c_403 : tensor<ui32>
    %1025 = stablehlo.convert %1024 : (tensor<ui32>) -> tensor<i32>
    %c_404 = stablehlo.constant dense<257> : tensor<i32>
    %1026 = stablehlo.multiply %1025, %c_404 : tensor<i32>
    %c_405 = stablehlo.constant dense<75> : tensor<i32>
    %1027 = stablehlo.subtract %c_405, %1026 : tensor<i32>
    %c_406 = stablehlo.constant dense<0> : tensor<i32>
    %c_407 = stablehlo.constant dense<256> : tensor<i32>
    %1028 = call @clip_170(%1027, %c_406, %c_407) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %1029 = stablehlo.add %1026, %1028 : tensor<i32>
    %1030 = stablehlo.iota dim = 0 : tensor<1xi32>
    %1031 = stablehlo.broadcast_in_dim %1029, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1032 = stablehlo.add %1031, %1030 : tensor<1xi32>
    %c_408 = stablehlo.constant dense<75> : tensor<i32>
    %1033 = stablehlo.broadcast_in_dim %c_408, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1034 = stablehlo.subtract %1032, %1033 : tensor<1xi32>
    %c_409 = stablehlo.constant dense<0> : tensor<i32>
    %1035 = stablehlo.broadcast_in_dim %c_409, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1036 = stablehlo.compare  GE, %1034, %1035,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_410 = stablehlo.constant dense<1> : tensor<i32>
    %1037 = stablehlo.broadcast_in_dim %c_410, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1038 = stablehlo.compare  LT, %1034, %1037,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %1039 = stablehlo.and %1036, %1038 : tensor<1xi1>
    %1040 = stablehlo.slice %arg67 [0:1, 0:16, 0:25, 0:1] : (tensor<2x16x25x1xf32>) -> tensor<1x16x25x1xf32>
    %1041 = stablehlo.reshape %1040 : (tensor<1x16x25x1xf32>) -> tensor<16x25x1xf32>
    %1042 = call @_take_240(%1041, %1034) : (tensor<16x25x1xf32>, tensor<1xi32>) -> tensor<16x25x1xf32>
    %1043 = stablehlo.reshape %1039 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_411 = stablehlo.constant dense<0> : tensor<i32>
    %1044 = stablehlo.compare  LT, %1021, %c_411,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_412 = stablehlo.constant dense<32> : tensor<i32>
    %1045 = stablehlo.add %1021, %c_412 : tensor<i32>
    %1046 = stablehlo.select %1044, %1045, %1021 : tensor<i1>, tensor<i32>
    %c_413 = stablehlo.constant dense<0> : tensor<i32>
    %1047 = stablehlo.dynamic_slice %arg68, %c_413, %1046, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %1048 = stablehlo.reshape %1047 : (tensor<1x1xf32>) -> tensor<f32>
    %1049 = stablehlo.broadcast_in_dim %1048, dims = [] : (tensor<f32>) -> tensor<16x25x1xf32>
    %1050 = stablehlo.multiply %1042, %1049 : tensor<16x25x1xf32>
    %c_414 = stablehlo.constant dense<0> : tensor<i32>
    %1051 = call @_where_244(%1043, %1050, %c_414) : (tensor<1x1x1xi1>, tensor<16x25x1xf32>, tensor<i32>) -> tensor<16x25x1xf32>
    %c_415 = stablehlo.constant dense<120> : tensor<i32>
    %c_416 = stablehlo.constant dense<0> : tensor<i32>
    %1052 = stablehlo.compare  LT, %c_415, %c_416,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_417 = stablehlo.constant dense<120> : tensor<i32>
    %c_418 = stablehlo.constant dense<256> : tensor<i32>
    %1053 = stablehlo.add %c_417, %c_418 : tensor<i32>
    %c_419 = stablehlo.constant dense<120> : tensor<i32>
    %1054 = stablehlo.select %1052, %1053, %c_419 : tensor<i1>, tensor<i32>
    %c_420 = stablehlo.constant dense<116> : tensor<i32>
    %c_421 = stablehlo.constant dense<0> : tensor<i32>
    %1055 = stablehlo.compare  LT, %c_420, %c_421,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_422 = stablehlo.constant dense<116> : tensor<i32>
    %c_423 = stablehlo.constant dense<257> : tensor<i32>
    %1056 = stablehlo.add %c_422, %c_423 : tensor<i32>
    %c_424 = stablehlo.constant dense<116> : tensor<i32>
    %1057 = stablehlo.select %1055, %1056, %c_424 : tensor<i1>, tensor<i32>
    %c_425 = stablehlo.constant dense<0> : tensor<i32>
    %1058 = stablehlo.compare  LT, %1028, %c_425,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_426 = stablehlo.constant dense<257> : tensor<i32>
    %1059 = stablehlo.add %1028, %c_426 : tensor<i32>
    %1060 = stablehlo.select %1058, %1059, %1028 : tensor<i1>, tensor<i32>
    %1061 = stablehlo.dynamic_slice %914, %1054, %1057, %1060, sizes = [16, 25, 1] : (tensor<256x257x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x25x1xf32>
    %1062 = stablehlo.add %1061, %1051 : tensor<16x25x1xf32>
    %c_427 = stablehlo.constant dense<120> : tensor<i32>
    %c_428 = stablehlo.constant dense<0> : tensor<i32>
    %1063 = stablehlo.compare  LT, %c_427, %c_428,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_429 = stablehlo.constant dense<120> : tensor<i32>
    %c_430 = stablehlo.constant dense<256> : tensor<i32>
    %1064 = stablehlo.add %c_429, %c_430 : tensor<i32>
    %c_431 = stablehlo.constant dense<120> : tensor<i32>
    %1065 = stablehlo.select %1063, %1064, %c_431 : tensor<i1>, tensor<i32>
    %c_432 = stablehlo.constant dense<116> : tensor<i32>
    %c_433 = stablehlo.constant dense<0> : tensor<i32>
    %1066 = stablehlo.compare  LT, %c_432, %c_433,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_434 = stablehlo.constant dense<116> : tensor<i32>
    %c_435 = stablehlo.constant dense<257> : tensor<i32>
    %1067 = stablehlo.add %c_434, %c_435 : tensor<i32>
    %c_436 = stablehlo.constant dense<116> : tensor<i32>
    %1068 = stablehlo.select %1066, %1067, %c_436 : tensor<i1>, tensor<i32>
    %c_437 = stablehlo.constant dense<0> : tensor<i32>
    %1069 = stablehlo.compare  LT, %1028, %c_437,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_438 = stablehlo.constant dense<257> : tensor<i32>
    %1070 = stablehlo.add %1028, %c_438 : tensor<i32>
    %1071 = stablehlo.select %1069, %1070, %1028 : tensor<i1>, tensor<i32>
    %1072 = stablehlo.dynamic_update_slice %914, %1062, %1065, %1068, %1071 : (tensor<256x257x257xf32>, tensor<16x25x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<256x257x257xf32>
    %c_439 = stablehlo.constant dense<75> : tensor<i32>
    %1073 = stablehlo.subtract %c_439, %1026 : tensor<i32>
    %c_440 = stablehlo.constant dense<0> : tensor<i32>
    %c_441 = stablehlo.constant dense<256> : tensor<i32>
    %1074 = call @clip_170(%1073, %c_440, %c_441) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %1075 = stablehlo.add %1026, %1074 : tensor<i32>
    %1076 = stablehlo.iota dim = 0 : tensor<1xi32>
    %1077 = stablehlo.broadcast_in_dim %1075, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1078 = stablehlo.add %1077, %1076 : tensor<1xi32>
    %c_442 = stablehlo.constant dense<75> : tensor<i32>
    %1079 = stablehlo.broadcast_in_dim %c_442, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1080 = stablehlo.subtract %1078, %1079 : tensor<1xi32>
    %c_443 = stablehlo.constant dense<0> : tensor<i32>
    %1081 = stablehlo.broadcast_in_dim %c_443, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1082 = stablehlo.compare  GE, %1080, %1081,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_444 = stablehlo.constant dense<1> : tensor<i32>
    %1083 = stablehlo.broadcast_in_dim %c_444, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1084 = stablehlo.compare  LT, %1080, %1083,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %1085 = stablehlo.and %1082, %1084 : tensor<1xi1>
    %1086 = stablehlo.slice %arg67 [1:2, 0:16, 0:25, 0:1] : (tensor<2x16x25x1xf32>) -> tensor<1x16x25x1xf32>
    %1087 = stablehlo.reshape %1086 : (tensor<1x16x25x1xf32>) -> tensor<16x25x1xf32>
    %1088 = call @_take_240(%1087, %1080) : (tensor<16x25x1xf32>, tensor<1xi32>) -> tensor<16x25x1xf32>
    %1089 = stablehlo.reshape %1085 : (tensor<1xi1>) -> tensor<1x1x1xi1>
    %c_445 = stablehlo.constant dense<0> : tensor<i32>
    %1090 = stablehlo.compare  LT, %1021, %c_445,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_446 = stablehlo.constant dense<32> : tensor<i32>
    %1091 = stablehlo.add %1021, %c_446 : tensor<i32>
    %1092 = stablehlo.select %1090, %1091, %1021 : tensor<i1>, tensor<i32>
    %c_447 = stablehlo.constant dense<1> : tensor<i32>
    %1093 = stablehlo.dynamic_slice %arg68, %c_447, %1092, sizes = [1, 1] : (tensor<2x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %1094 = stablehlo.reshape %1093 : (tensor<1x1xf32>) -> tensor<f32>
    %1095 = stablehlo.broadcast_in_dim %1094, dims = [] : (tensor<f32>) -> tensor<16x25x1xf32>
    %1096 = stablehlo.multiply %1088, %1095 : tensor<16x25x1xf32>
    %c_448 = stablehlo.constant dense<0> : tensor<i32>
    %1097 = call @_where_244(%1089, %1096, %c_448) : (tensor<1x1x1xi1>, tensor<16x25x1xf32>, tensor<i32>) -> tensor<16x25x1xf32>
    %c_449 = stablehlo.constant dense<120> : tensor<i32>
    %c_450 = stablehlo.constant dense<0> : tensor<i32>
    %1098 = stablehlo.compare  LT, %c_449, %c_450,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_451 = stablehlo.constant dense<120> : tensor<i32>
    %c_452 = stablehlo.constant dense<256> : tensor<i32>
    %1099 = stablehlo.add %c_451, %c_452 : tensor<i32>
    %c_453 = stablehlo.constant dense<120> : tensor<i32>
    %1100 = stablehlo.select %1098, %1099, %c_453 : tensor<i1>, tensor<i32>
    %c_454 = stablehlo.constant dense<116> : tensor<i32>
    %c_455 = stablehlo.constant dense<0> : tensor<i32>
    %1101 = stablehlo.compare  LT, %c_454, %c_455,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_456 = stablehlo.constant dense<116> : tensor<i32>
    %c_457 = stablehlo.constant dense<257> : tensor<i32>
    %1102 = stablehlo.add %c_456, %c_457 : tensor<i32>
    %c_458 = stablehlo.constant dense<116> : tensor<i32>
    %1103 = stablehlo.select %1101, %1102, %c_458 : tensor<i1>, tensor<i32>
    %c_459 = stablehlo.constant dense<0> : tensor<i32>
    %1104 = stablehlo.compare  LT, %1074, %c_459,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_460 = stablehlo.constant dense<257> : tensor<i32>
    %1105 = stablehlo.add %1074, %c_460 : tensor<i32>
    %1106 = stablehlo.select %1104, %1105, %1074 : tensor<i1>, tensor<i32>
    %1107 = stablehlo.dynamic_slice %1072, %1100, %1103, %1106, sizes = [16, 25, 1] : (tensor<256x257x257xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x25x1xf32>
    %1108 = stablehlo.add %1107, %1097 : tensor<16x25x1xf32>
    %c_461 = stablehlo.constant dense<120> : tensor<i32>
    %c_462 = stablehlo.constant dense<0> : tensor<i32>
    %1109 = stablehlo.compare  LT, %c_461, %c_462,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_463 = stablehlo.constant dense<120> : tensor<i32>
    %c_464 = stablehlo.constant dense<256> : tensor<i32>
    %1110 = stablehlo.add %c_463, %c_464 : tensor<i32>
    %c_465 = stablehlo.constant dense<120> : tensor<i32>
    %1111 = stablehlo.select %1109, %1110, %c_465 : tensor<i1>, tensor<i32>
    %c_466 = stablehlo.constant dense<116> : tensor<i32>
    %c_467 = stablehlo.constant dense<0> : tensor<i32>
    %1112 = stablehlo.compare  LT, %c_466, %c_467,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_468 = stablehlo.constant dense<116> : tensor<i32>
    %c_469 = stablehlo.constant dense<257> : tensor<i32>
    %1113 = stablehlo.add %c_468, %c_469 : tensor<i32>
    %c_470 = stablehlo.constant dense<116> : tensor<i32>
    %1114 = stablehlo.select %1112, %1113, %c_470 : tensor<i1>, tensor<i32>
    %c_471 = stablehlo.constant dense<0> : tensor<i32>
    %1115 = stablehlo.compare  LT, %1074, %c_471,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_472 = stablehlo.constant dense<257> : tensor<i32>
    %1116 = stablehlo.add %1074, %c_472 : tensor<i32>
    %1117 = stablehlo.select %1115, %1116, %1074 : tensor<i1>, tensor<i32>
    %1118 = stablehlo.dynamic_update_slice %1072, %1108, %1111, %1114, %1117 : (tensor<256x257x257xf32>, tensor<16x25x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<256x257x257xf32>
    %c_473 = stablehlo.constant dense<1> : tensor<i32>
    %1119 = stablehlo.add %arg146, %c_473 : tensor<i32>
    %c_474 = stablehlo.constant dense<1> : tensor<i32>
    %c_475 = stablehlo.constant dense<1> : tensor<i32>
    %1120 = stablehlo.maximum %c_474, %c_475 : tensor<i32>
    %1121 = call @remainder(%1119, %1120) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_476 = stablehlo.constant dense<0> : tensor<i32>
    %1122 = stablehlo.compare  EQ, %1121, %c_476,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1123 = stablehlo.slice %arg141 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %1124 = stablehlo.reshape %1123 : (tensor<1xi32>) -> tensor<i32>
    %c_477 = stablehlo.constant dense<1> : tensor<i32>
    %1125 = stablehlo.compare  LT, %1124, %c_477,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1126 = stablehlo.and %1122, %1125 : tensor<i1>
    %c_478 = stablehlo.constant dense<false> : tensor<i1>
    %1127 = stablehlo.and %1126, %c_478 : tensor<i1>
    %c_479 = stablehlo.constant dense<false> : tensor<i1>
    %1128 = stablehlo.or %1127, %c_479 : tensor<i1>
    %1129 = stablehlo.convert %1128 : (tensor<i1>) -> tensor<i32>
    %1130 = "stablehlo.case"(%1129) ({
      %cst_534 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      stablehlo.return %cst_534 : tensor<f32>
    }, {
      %c_534 = stablehlo.constant dense<257> : tensor<i32>
      %1238:2 = func.call @divmod(%arg69, %c_534) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_535 = stablehlo.constant dense<257> : tensor<i32>
      %1239:2 = func.call @divmod(%1238#0, %c_535) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_536 = stablehlo.constant dense<257> : tensor<i32>
      %1240:2 = func.call @divmod(%1239#0, %c_536) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_537 = stablehlo.constant dense<0> : tensor<i32>
      %1241 = stablehlo.broadcast_in_dim %c_537, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1242 = stablehlo.compare  GT, %1240#0, %1241,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_538 = stablehlo.constant dense<-1> : tensor<i32>
      %1243 = stablehlo.broadcast_in_dim %c_538, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1244 = stablehlo.compare  LT, %1240#0, %1243,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_539 = stablehlo.constant dense<0> : tensor<i32>
      %1245 = func.call @_where_279(%1244, %c_539, %1240#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_540 = stablehlo.constant dense<256> : tensor<i32>
      %1246 = func.call @_where_279(%1242, %c_540, %1245) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_541 = stablehlo.constant dense<0> : tensor<i32>
      %1247 = func.call @_where_279(%1244, %c_541, %1239#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_542 = stablehlo.constant dense<256> : tensor<i32>
      %1248 = func.call @_where_279(%1242, %c_542, %1247) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_543 = stablehlo.constant dense<0> : tensor<i32>
      %1249 = func.call @_where_279(%1244, %c_543, %1238#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_544 = stablehlo.constant dense<256> : tensor<i32>
      %1250 = func.call @_where_279(%1242, %c_544, %1249) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_545 = stablehlo.constant dense<0> : tensor<i32>
      %1251 = stablehlo.broadcast_in_dim %c_545, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1252 = stablehlo.compare  LT, %1246, %1251,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_546 = stablehlo.constant dense<257> : tensor<i32>
      %1253 = stablehlo.broadcast_in_dim %c_546, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1254 = stablehlo.add %1246, %1253 : tensor<200x1xi32>
      %1255 = stablehlo.select %1252, %1254, %1246 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_547 = stablehlo.constant dense<0> : tensor<i32>
      %1256 = stablehlo.broadcast_in_dim %c_547, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1257 = stablehlo.compare  LT, %1248, %1256,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_548 = stablehlo.constant dense<257> : tensor<i32>
      %1258 = stablehlo.broadcast_in_dim %c_548, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1259 = stablehlo.add %1248, %1258 : tensor<200x1xi32>
      %1260 = stablehlo.select %1257, %1259, %1248 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_549 = stablehlo.constant dense<0> : tensor<i32>
      %1261 = stablehlo.broadcast_in_dim %c_549, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1262 = stablehlo.compare  LT, %1250, %1261,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_550 = stablehlo.constant dense<257> : tensor<i32>
      %1263 = stablehlo.broadcast_in_dim %c_550, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1264 = stablehlo.add %1250, %1263 : tensor<200x1xi32>
      %1265 = stablehlo.select %1262, %1264, %1250 : tensor<200x1xi1>, tensor<200x1xi32>
      %1266 = stablehlo.broadcast_in_dim %1255, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1267 = stablehlo.broadcast_in_dim %1260, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1268 = stablehlo.broadcast_in_dim %1265, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1269 = stablehlo.concatenate %1266, %1267, %1268, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1270 = "stablehlo.gather"(%912, %1269) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1271 = stablehlo.multiply %1270, %arg70 : tensor<200x1xf32>
      %cst_551 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1272 = stablehlo.reduce(%1271 init: %cst_551) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_552 = stablehlo.constant dense<257> : tensor<i32>
      %1273:2 = func.call @divmod(%arg71, %c_552) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_553 = stablehlo.constant dense<256> : tensor<i32>
      %1274:2 = func.call @divmod(%1273#0, %c_553) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_554 = stablehlo.constant dense<257> : tensor<i32>
      %1275:2 = func.call @divmod(%1274#0, %c_554) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_555 = stablehlo.constant dense<0> : tensor<i32>
      %1276 = stablehlo.broadcast_in_dim %c_555, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1277 = stablehlo.compare  GT, %1275#0, %1276,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_556 = stablehlo.constant dense<-1> : tensor<i32>
      %1278 = stablehlo.broadcast_in_dim %c_556, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1279 = stablehlo.compare  LT, %1275#0, %1278,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_557 = stablehlo.constant dense<0> : tensor<i32>
      %1280 = func.call @_where_279(%1279, %c_557, %1275#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_558 = stablehlo.constant dense<256> : tensor<i32>
      %1281 = func.call @_where_279(%1277, %c_558, %1280) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_559 = stablehlo.constant dense<0> : tensor<i32>
      %1282 = func.call @_where_279(%1279, %c_559, %1274#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_560 = stablehlo.constant dense<255> : tensor<i32>
      %1283 = func.call @_where_279(%1277, %c_560, %1282) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_561 = stablehlo.constant dense<0> : tensor<i32>
      %1284 = func.call @_where_279(%1279, %c_561, %1273#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_562 = stablehlo.constant dense<256> : tensor<i32>
      %1285 = func.call @_where_279(%1277, %c_562, %1284) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_563 = stablehlo.constant dense<0> : tensor<i32>
      %1286 = stablehlo.broadcast_in_dim %c_563, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1287 = stablehlo.compare  LT, %1281, %1286,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_564 = stablehlo.constant dense<257> : tensor<i32>
      %1288 = stablehlo.broadcast_in_dim %c_564, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1289 = stablehlo.add %1281, %1288 : tensor<200x1xi32>
      %1290 = stablehlo.select %1287, %1289, %1281 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_565 = stablehlo.constant dense<0> : tensor<i32>
      %1291 = stablehlo.broadcast_in_dim %c_565, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1292 = stablehlo.compare  LT, %1283, %1291,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_566 = stablehlo.constant dense<256> : tensor<i32>
      %1293 = stablehlo.broadcast_in_dim %c_566, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1294 = stablehlo.add %1283, %1293 : tensor<200x1xi32>
      %1295 = stablehlo.select %1292, %1294, %1283 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_567 = stablehlo.constant dense<0> : tensor<i32>
      %1296 = stablehlo.broadcast_in_dim %c_567, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1297 = stablehlo.compare  LT, %1285, %1296,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_568 = stablehlo.constant dense<257> : tensor<i32>
      %1298 = stablehlo.broadcast_in_dim %c_568, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1299 = stablehlo.add %1285, %1298 : tensor<200x1xi32>
      %1300 = stablehlo.select %1297, %1299, %1285 : tensor<200x1xi1>, tensor<200x1xi32>
      %1301 = stablehlo.broadcast_in_dim %1290, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1302 = stablehlo.broadcast_in_dim %1295, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1303 = stablehlo.broadcast_in_dim %1300, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1304 = stablehlo.concatenate %1301, %1302, %1303, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1305 = "stablehlo.gather"(%1020, %1304) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1306 = stablehlo.multiply %1305, %arg72 : tensor<200x1xf32>
      %cst_569 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1307 = stablehlo.reduce(%1306 init: %cst_569) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_570 = stablehlo.constant dense<257> : tensor<i32>
      %1308:2 = func.call @divmod(%arg73, %c_570) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_571 = stablehlo.constant dense<257> : tensor<i32>
      %1309:2 = func.call @divmod(%1308#0, %c_571) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_572 = stablehlo.constant dense<256> : tensor<i32>
      %1310:2 = func.call @divmod(%1309#0, %c_572) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_573 = stablehlo.constant dense<0> : tensor<i32>
      %1311 = stablehlo.broadcast_in_dim %c_573, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1312 = stablehlo.compare  GT, %1310#0, %1311,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_574 = stablehlo.constant dense<-1> : tensor<i32>
      %1313 = stablehlo.broadcast_in_dim %c_574, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1314 = stablehlo.compare  LT, %1310#0, %1313,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_575 = stablehlo.constant dense<0> : tensor<i32>
      %1315 = func.call @_where_279(%1314, %c_575, %1310#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_576 = stablehlo.constant dense<255> : tensor<i32>
      %1316 = func.call @_where_279(%1312, %c_576, %1315) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_577 = stablehlo.constant dense<0> : tensor<i32>
      %1317 = func.call @_where_279(%1314, %c_577, %1309#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_578 = stablehlo.constant dense<256> : tensor<i32>
      %1318 = func.call @_where_279(%1312, %c_578, %1317) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_579 = stablehlo.constant dense<0> : tensor<i32>
      %1319 = func.call @_where_279(%1314, %c_579, %1308#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_580 = stablehlo.constant dense<256> : tensor<i32>
      %1320 = func.call @_where_279(%1312, %c_580, %1319) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_581 = stablehlo.constant dense<0> : tensor<i32>
      %1321 = stablehlo.broadcast_in_dim %c_581, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1322 = stablehlo.compare  LT, %1316, %1321,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_582 = stablehlo.constant dense<256> : tensor<i32>
      %1323 = stablehlo.broadcast_in_dim %c_582, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1324 = stablehlo.add %1316, %1323 : tensor<200x1xi32>
      %1325 = stablehlo.select %1322, %1324, %1316 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_583 = stablehlo.constant dense<0> : tensor<i32>
      %1326 = stablehlo.broadcast_in_dim %c_583, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1327 = stablehlo.compare  LT, %1318, %1326,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_584 = stablehlo.constant dense<257> : tensor<i32>
      %1328 = stablehlo.broadcast_in_dim %c_584, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1329 = stablehlo.add %1318, %1328 : tensor<200x1xi32>
      %1330 = stablehlo.select %1327, %1329, %1318 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_585 = stablehlo.constant dense<0> : tensor<i32>
      %1331 = stablehlo.broadcast_in_dim %c_585, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1332 = stablehlo.compare  LT, %1320, %1331,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_586 = stablehlo.constant dense<257> : tensor<i32>
      %1333 = stablehlo.broadcast_in_dim %c_586, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1334 = stablehlo.add %1320, %1333 : tensor<200x1xi32>
      %1335 = stablehlo.select %1332, %1334, %1320 : tensor<200x1xi1>, tensor<200x1xi32>
      %1336 = stablehlo.broadcast_in_dim %1325, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1337 = stablehlo.broadcast_in_dim %1330, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1338 = stablehlo.broadcast_in_dim %1335, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1339 = stablehlo.concatenate %1336, %1337, %1338, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1340 = "stablehlo.gather"(%1118, %1339) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1341 = stablehlo.multiply %1340, %arg74 : tensor<200x1xf32>
      %cst_587 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1342 = stablehlo.reduce(%1341 init: %cst_587) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_588 = stablehlo.constant dense<257> : tensor<i32>
      %1343:2 = func.call @divmod(%arg75, %c_588) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_589 = stablehlo.constant dense<256> : tensor<i32>
      %1344:2 = func.call @divmod(%1343#0, %c_589) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_590 = stablehlo.constant dense<256> : tensor<i32>
      %1345:2 = func.call @divmod(%1344#0, %c_590) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_591 = stablehlo.constant dense<0> : tensor<i32>
      %1346 = stablehlo.broadcast_in_dim %c_591, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1347 = stablehlo.compare  GT, %1345#0, %1346,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_592 = stablehlo.constant dense<-1> : tensor<i32>
      %1348 = stablehlo.broadcast_in_dim %c_592, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1349 = stablehlo.compare  LT, %1345#0, %1348,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_593 = stablehlo.constant dense<0> : tensor<i32>
      %1350 = func.call @_where_279(%1349, %c_593, %1345#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_594 = stablehlo.constant dense<255> : tensor<i32>
      %1351 = func.call @_where_279(%1347, %c_594, %1350) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_595 = stablehlo.constant dense<0> : tensor<i32>
      %1352 = func.call @_where_279(%1349, %c_595, %1344#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_596 = stablehlo.constant dense<255> : tensor<i32>
      %1353 = func.call @_where_279(%1347, %c_596, %1352) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_597 = stablehlo.constant dense<0> : tensor<i32>
      %1354 = func.call @_where_279(%1349, %c_597, %1343#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_598 = stablehlo.constant dense<256> : tensor<i32>
      %1355 = func.call @_where_279(%1347, %c_598, %1354) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_599 = stablehlo.constant dense<0> : tensor<i32>
      %1356 = stablehlo.broadcast_in_dim %c_599, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1357 = stablehlo.compare  LT, %1351, %1356,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_600 = stablehlo.constant dense<256> : tensor<i32>
      %1358 = stablehlo.broadcast_in_dim %c_600, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1359 = stablehlo.add %1351, %1358 : tensor<200x1xi32>
      %1360 = stablehlo.select %1357, %1359, %1351 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_601 = stablehlo.constant dense<0> : tensor<i32>
      %1361 = stablehlo.broadcast_in_dim %c_601, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1362 = stablehlo.compare  LT, %1353, %1361,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_602 = stablehlo.constant dense<256> : tensor<i32>
      %1363 = stablehlo.broadcast_in_dim %c_602, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1364 = stablehlo.add %1353, %1363 : tensor<200x1xi32>
      %1365 = stablehlo.select %1362, %1364, %1353 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_603 = stablehlo.constant dense<0> : tensor<i32>
      %1366 = stablehlo.broadcast_in_dim %c_603, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1367 = stablehlo.compare  LT, %1355, %1366,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_604 = stablehlo.constant dense<257> : tensor<i32>
      %1368 = stablehlo.broadcast_in_dim %c_604, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1369 = stablehlo.add %1355, %1368 : tensor<200x1xi32>
      %1370 = stablehlo.select %1367, %1369, %1355 : tensor<200x1xi1>, tensor<200x1xi32>
      %1371 = stablehlo.broadcast_in_dim %1360, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1372 = stablehlo.broadcast_in_dim %1365, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1373 = stablehlo.broadcast_in_dim %1370, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1374 = stablehlo.concatenate %1371, %1372, %1373, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1375 = "stablehlo.gather"(%353, %1374) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1376 = stablehlo.multiply %1375, %arg76 : tensor<200x1xf32>
      %cst_605 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1377 = stablehlo.reduce(%1376 init: %cst_605) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_606 = stablehlo.constant dense<257> : tensor<i32>
      %1378:2 = func.call @divmod(%arg77, %c_606) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_607 = stablehlo.constant dense<257> : tensor<i32>
      %1379:2 = func.call @divmod(%1378#0, %c_607) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_608 = stablehlo.constant dense<256> : tensor<i32>
      %1380:2 = func.call @divmod(%1379#0, %c_608) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_609 = stablehlo.constant dense<0> : tensor<i32>
      %1381 = stablehlo.broadcast_in_dim %c_609, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1382 = stablehlo.compare  GT, %1380#0, %1381,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_610 = stablehlo.constant dense<-1> : tensor<i32>
      %1383 = stablehlo.broadcast_in_dim %c_610, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1384 = stablehlo.compare  LT, %1380#0, %1383,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_611 = stablehlo.constant dense<0> : tensor<i32>
      %1385 = func.call @_where_279(%1384, %c_611, %1380#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_612 = stablehlo.constant dense<255> : tensor<i32>
      %1386 = func.call @_where_279(%1382, %c_612, %1385) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_613 = stablehlo.constant dense<0> : tensor<i32>
      %1387 = func.call @_where_279(%1384, %c_613, %1379#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_614 = stablehlo.constant dense<256> : tensor<i32>
      %1388 = func.call @_where_279(%1382, %c_614, %1387) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_615 = stablehlo.constant dense<0> : tensor<i32>
      %1389 = func.call @_where_279(%1384, %c_615, %1378#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_616 = stablehlo.constant dense<256> : tensor<i32>
      %1390 = func.call @_where_279(%1382, %c_616, %1389) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_617 = stablehlo.constant dense<0> : tensor<i32>
      %1391 = stablehlo.broadcast_in_dim %c_617, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1392 = stablehlo.compare  LT, %1386, %1391,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_618 = stablehlo.constant dense<256> : tensor<i32>
      %1393 = stablehlo.broadcast_in_dim %c_618, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1394 = stablehlo.add %1386, %1393 : tensor<200x1xi32>
      %1395 = stablehlo.select %1392, %1394, %1386 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_619 = stablehlo.constant dense<0> : tensor<i32>
      %1396 = stablehlo.broadcast_in_dim %c_619, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1397 = stablehlo.compare  LT, %1388, %1396,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_620 = stablehlo.constant dense<257> : tensor<i32>
      %1398 = stablehlo.broadcast_in_dim %c_620, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1399 = stablehlo.add %1388, %1398 : tensor<200x1xi32>
      %1400 = stablehlo.select %1397, %1399, %1388 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_621 = stablehlo.constant dense<0> : tensor<i32>
      %1401 = stablehlo.broadcast_in_dim %c_621, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1402 = stablehlo.compare  LT, %1390, %1401,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_622 = stablehlo.constant dense<257> : tensor<i32>
      %1403 = stablehlo.broadcast_in_dim %c_622, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1404 = stablehlo.add %1390, %1403 : tensor<200x1xi32>
      %1405 = stablehlo.select %1402, %1404, %1390 : tensor<200x1xi1>, tensor<200x1xi32>
      %1406 = stablehlo.broadcast_in_dim %1395, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1407 = stablehlo.broadcast_in_dim %1400, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1408 = stablehlo.broadcast_in_dim %1405, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1409 = stablehlo.concatenate %1406, %1407, %1408, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1410 = "stablehlo.gather"(%461, %1409) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1411 = stablehlo.multiply %1410, %arg78 : tensor<200x1xf32>
      %cst_623 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1412 = stablehlo.reduce(%1411 init: %cst_623) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_624 = stablehlo.constant dense<257> : tensor<i32>
      %1413:2 = func.call @divmod(%arg79, %c_624) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_625 = stablehlo.constant dense<256> : tensor<i32>
      %1414:2 = func.call @divmod(%1413#0, %c_625) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_626 = stablehlo.constant dense<257> : tensor<i32>
      %1415:2 = func.call @divmod(%1414#0, %c_626) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_627 = stablehlo.constant dense<0> : tensor<i32>
      %1416 = stablehlo.broadcast_in_dim %c_627, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1417 = stablehlo.compare  GT, %1415#0, %1416,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_628 = stablehlo.constant dense<-1> : tensor<i32>
      %1418 = stablehlo.broadcast_in_dim %c_628, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1419 = stablehlo.compare  LT, %1415#0, %1418,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_629 = stablehlo.constant dense<0> : tensor<i32>
      %1420 = func.call @_where_279(%1419, %c_629, %1415#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_630 = stablehlo.constant dense<256> : tensor<i32>
      %1421 = func.call @_where_279(%1417, %c_630, %1420) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_631 = stablehlo.constant dense<0> : tensor<i32>
      %1422 = func.call @_where_279(%1419, %c_631, %1414#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_632 = stablehlo.constant dense<255> : tensor<i32>
      %1423 = func.call @_where_279(%1417, %c_632, %1422) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_633 = stablehlo.constant dense<0> : tensor<i32>
      %1424 = func.call @_where_279(%1419, %c_633, %1413#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_634 = stablehlo.constant dense<256> : tensor<i32>
      %1425 = func.call @_where_279(%1417, %c_634, %1424) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_635 = stablehlo.constant dense<0> : tensor<i32>
      %1426 = stablehlo.broadcast_in_dim %c_635, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1427 = stablehlo.compare  LT, %1421, %1426,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_636 = stablehlo.constant dense<257> : tensor<i32>
      %1428 = stablehlo.broadcast_in_dim %c_636, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1429 = stablehlo.add %1421, %1428 : tensor<200x1xi32>
      %1430 = stablehlo.select %1427, %1429, %1421 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_637 = stablehlo.constant dense<0> : tensor<i32>
      %1431 = stablehlo.broadcast_in_dim %c_637, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1432 = stablehlo.compare  LT, %1423, %1431,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_638 = stablehlo.constant dense<256> : tensor<i32>
      %1433 = stablehlo.broadcast_in_dim %c_638, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1434 = stablehlo.add %1423, %1433 : tensor<200x1xi32>
      %1435 = stablehlo.select %1432, %1434, %1423 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_639 = stablehlo.constant dense<0> : tensor<i32>
      %1436 = stablehlo.broadcast_in_dim %c_639, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1437 = stablehlo.compare  LT, %1425, %1436,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_640 = stablehlo.constant dense<257> : tensor<i32>
      %1438 = stablehlo.broadcast_in_dim %c_640, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1439 = stablehlo.add %1425, %1438 : tensor<200x1xi32>
      %1440 = stablehlo.select %1437, %1439, %1425 : tensor<200x1xi1>, tensor<200x1xi32>
      %1441 = stablehlo.broadcast_in_dim %1430, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1442 = stablehlo.broadcast_in_dim %1435, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1443 = stablehlo.broadcast_in_dim %1440, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1444 = stablehlo.concatenate %1441, %1442, %1443, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1445 = "stablehlo.gather"(%559, %1444) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1446 = stablehlo.multiply %1445, %arg80 : tensor<200x1xf32>
      %cst_641 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1447 = stablehlo.reduce(%1446 init: %cst_641) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %1448 = stablehlo.broadcast_in_dim %1272, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1449 = stablehlo.broadcast_in_dim %1307, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1450 = stablehlo.broadcast_in_dim %1342, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1451 = stablehlo.broadcast_in_dim %1377, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1452 = stablehlo.broadcast_in_dim %1412, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1453 = stablehlo.broadcast_in_dim %1447, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1454 = stablehlo.concatenate %1448, %1449, %1450, %1451, %1452, %1453, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %1455 = stablehlo.slice %1454 [1:2, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1456 = stablehlo.reshape %1455 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1457 = stablehlo.slice %1454 [2:3, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1458 = stablehlo.reshape %1457 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1459 = stablehlo.slice %1454 [4:5, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1460 = stablehlo.reshape %1459 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1461 = stablehlo.slice %1454 [5:6, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1462 = stablehlo.reshape %1461 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1463 = stablehlo.multiply %1456, %1462 : tensor<200xf32>
      %1464 = stablehlo.multiply %1458, %1460 : tensor<200xf32>
      %1465 = stablehlo.subtract %1463, %1464 : tensor<200xf32>
      %cst_642 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1466 = stablehlo.broadcast_in_dim %cst_642, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %1467 = stablehlo.multiply %1465, %1466 : tensor<200xf32>
      %cst_643 = stablehlo.constant dense<7.485380e-15> : tensor<f32>
      %1468 = stablehlo.broadcast_in_dim %cst_643, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %1469 = stablehlo.multiply %1467, %1468 : tensor<200xf32>
      %cst_644 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1470 = stablehlo.reduce(%1469 init: %cst_644) applies stablehlo.add across dimensions = [0] : (tensor<200xf32>, tensor<f32>) -> tensor<f32>
      stablehlo.return %1470 : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %cst_480 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1131 = call @_where_299(%1127, %1130, %cst_480) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %1132 = stablehlo.slice %arg141 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %1133 = stablehlo.reshape %1132 : (tensor<1xi32>) -> tensor<i32>
    %c_481 = stablehlo.constant dense<0> : tensor<i32>
    %1134 = stablehlo.minimum %1133, %c_481 : tensor<i32>
    %c_482 = stablehlo.constant dense<0> : tensor<i32>
    %1135 = stablehlo.compare  LT, %1134, %c_482,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_483 = stablehlo.constant dense<1> : tensor<i32>
    %1136 = stablehlo.add %1134, %c_483 : tensor<i32>
    %1137 = stablehlo.select %1135, %1136, %1134 : tensor<i1>, tensor<i32>
    %c_484 = stablehlo.constant dense<0> : tensor<i32>
    %1138 = stablehlo.dynamic_slice %arg139, %c_484, %1137, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %1139 = stablehlo.reshape %1138 : (tensor<1x1xf32>) -> tensor<f32>
    %c_485 = stablehlo.constant dense<0> : tensor<i32>
    %1140 = stablehlo.compare  LT, %1134, %c_485,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_486 = stablehlo.constant dense<1> : tensor<i32>
    %1141 = stablehlo.add %1134, %c_486 : tensor<i32>
    %1142 = stablehlo.select %1140, %1141, %1134 : tensor<i1>, tensor<i32>
    %c_487 = stablehlo.constant dense<0> : tensor<i32>
    %1143 = stablehlo.dynamic_slice %arg140, %c_487, %1142, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %1144 = stablehlo.reshape %1143 : (tensor<1x1xf32>) -> tensor<f32>
    %1145 = call @_where_304(%1127, %1131, %1139) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_488 = stablehlo.constant dense<0> : tensor<i32>
    %1146 = stablehlo.compare  LT, %1134, %c_488,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_489 = stablehlo.constant dense<1> : tensor<i32>
    %1147 = stablehlo.add %1134, %c_489 : tensor<i32>
    %1148 = stablehlo.select %1146, %1147, %1134 : tensor<i1>, tensor<i32>
    %c_490 = stablehlo.constant dense<0> : tensor<i32>
    %1149 = stablehlo.broadcast_in_dim %c_490, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1150 = stablehlo.broadcast_in_dim %1148, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1151 = stablehlo.concatenate %1149, %1150, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %1152 = "stablehlo.scatter"(%arg139, %1151, %1145) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg148: tensor<f32>, %arg149: tensor<f32>):
      stablehlo.return %arg149 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %1153 = call @_where_304(%1127, %3, %1144) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_491 = stablehlo.constant dense<0> : tensor<i32>
    %1154 = stablehlo.compare  LT, %1134, %c_491,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_492 = stablehlo.constant dense<1> : tensor<i32>
    %1155 = stablehlo.add %1134, %c_492 : tensor<i32>
    %1156 = stablehlo.select %1154, %1155, %1134 : tensor<i1>, tensor<i32>
    %c_493 = stablehlo.constant dense<0> : tensor<i32>
    %1157 = stablehlo.broadcast_in_dim %c_493, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1158 = stablehlo.broadcast_in_dim %1156, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1159 = stablehlo.concatenate %1157, %1158, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %1160 = "stablehlo.scatter"(%arg140, %1159, %1153) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg148: tensor<f32>, %arg149: tensor<f32>):
      stablehlo.return %arg149 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %1161 = stablehlo.slice %arg141 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %1162 = stablehlo.reshape %1161 : (tensor<1xi32>) -> tensor<i32>
    %c_494 = stablehlo.constant dense<1> : tensor<i32>
    %c_495 = stablehlo.constant dense<0> : tensor<i32>
    %1163 = call @_where_310(%1127, %c_494, %c_495) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %1164 = stablehlo.convert %1163 : tensor<i32>
    %1165 = stablehlo.add %1162, %1164 : tensor<i32>
    %c_496 = stablehlo.constant dense<0> : tensor<i32>
    %1166 = stablehlo.broadcast_in_dim %c_496, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1167 = "stablehlo.scatter"(%arg141, %1166, %1165) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg148: tensor<i32>, %arg149: tensor<i32>):
      stablehlo.return %arg149 : tensor<i32>
    }) : (tensor<2xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
    %cst_497 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1168 = stablehlo.compare  GE, %3, %cst_497,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %c_498 = stablehlo.constant dense<true> : tensor<i1>
    %1169 = stablehlo.and %c_498, %1168 : tensor<i1>
    %cst_499 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %1170 = stablehlo.compare  LE, %3, %cst_499,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %1171 = stablehlo.and %1169, %1170 : tensor<i1>
    %c_500 = stablehlo.constant dense<1> : tensor<i32>
    %c_501 = stablehlo.constant dense<1> : tensor<i32>
    %1172 = stablehlo.maximum %c_500, %c_501 : tensor<i32>
    %1173 = call @remainder(%arg146, %1172) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_502 = stablehlo.constant dense<0> : tensor<i32>
    %1174 = stablehlo.compare  EQ, %1173, %c_502,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1175 = stablehlo.and %1171, %1174 : tensor<i1>
    %1176 = stablehlo.convert %1175 : (tensor<i1>) -> tensor<i32>
    %1177:3 = "stablehlo.case"(%1176) ({
      stablehlo.return %arg142, %arg143, %arg144 : tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>
    }, {
      %c_534 = stablehlo.constant dense<257> : tensor<i32>
      %1238:2 = func.call @divmod_319(%arg81, %c_534) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_535 = stablehlo.constant dense<257> : tensor<i32>
      %1239:2 = func.call @divmod_336(%1238#0, %c_535) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_536 = stablehlo.constant dense<257> : tensor<i32>
      %1240:2 = func.call @divmod_336(%1239#0, %c_536) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_537 = stablehlo.constant dense<0> : tensor<i32>
      %1241 = stablehlo.broadcast_in_dim %c_537, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1242 = stablehlo.compare  GT, %1240#0, %1241,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_538 = stablehlo.constant dense<-1> : tensor<i32>
      %1243 = stablehlo.broadcast_in_dim %c_538, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1244 = stablehlo.compare  LT, %1240#0, %1243,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_539 = stablehlo.constant dense<0> : tensor<i32>
      %1245 = func.call @_where_343(%1244, %c_539, %1240#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_540 = stablehlo.constant dense<256> : tensor<i32>
      %1246 = func.call @_where_343(%1242, %c_540, %1245) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_541 = stablehlo.constant dense<0> : tensor<i32>
      %1247 = func.call @_where_343(%1244, %c_541, %1239#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_542 = stablehlo.constant dense<256> : tensor<i32>
      %1248 = func.call @_where_343(%1242, %c_542, %1247) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_543 = stablehlo.constant dense<0> : tensor<i32>
      %1249 = func.call @_where_343(%1244, %c_543, %1238#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_544 = stablehlo.constant dense<256> : tensor<i32>
      %1250 = func.call @_where_343(%1242, %c_544, %1249) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_545 = stablehlo.constant dense<1> : tensor<ui32>
      %c_546 = stablehlo.constant dense<1> : tensor<ui32>
      %1251 = stablehlo.partition_id : tensor<ui32>
      %1252 = stablehlo.divide %1251, %c_545 : tensor<ui32>
      %1253 = stablehlo.remainder %1252, %c_546 : tensor<ui32>
      %1254 = stablehlo.convert %1253 : (tensor<ui32>) -> tensor<i32>
      %c_547 = stablehlo.constant dense<257> : tensor<i32>
      %1255 = stablehlo.multiply %1254, %c_547 : tensor<i32>
      %1256 = stablehlo.broadcast_in_dim %1255, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1257 = stablehlo.subtract %1250, %1256 : tensor<200x8xi32>
      %c_548 = stablehlo.constant dense<0> : tensor<i32>
      %1258 = stablehlo.broadcast_in_dim %c_548, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1259 = stablehlo.compare  GE, %1257, %1258,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_549 = stablehlo.constant dense<257> : tensor<i32>
      %1260 = stablehlo.broadcast_in_dim %c_549, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1261 = stablehlo.compare  LT, %1257, %1260,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %1262 = stablehlo.and %1259, %1261 : tensor<200x8xi1>
      %c_550 = stablehlo.constant dense<0> : tensor<i32>
      %c_551 = stablehlo.constant dense<256> : tensor<i32>
      %1263 = func.call @clip_350(%1257, %c_550, %c_551) : (tensor<200x8xi32>, tensor<i32>, tensor<i32>) -> tensor<200x8xi32>
      %c_552 = stablehlo.constant dense<0> : tensor<i32>
      %1264 = stablehlo.broadcast_in_dim %c_552, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1265 = stablehlo.compare  LT, %1246, %1264,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_553 = stablehlo.constant dense<257> : tensor<i32>
      %1266 = stablehlo.broadcast_in_dim %c_553, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1267 = stablehlo.add %1246, %1266 : tensor<200x8xi32>
      %1268 = stablehlo.select %1265, %1267, %1246 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_554 = stablehlo.constant dense<0> : tensor<i32>
      %1269 = stablehlo.broadcast_in_dim %c_554, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1270 = stablehlo.compare  LT, %1248, %1269,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_555 = stablehlo.constant dense<257> : tensor<i32>
      %1271 = stablehlo.broadcast_in_dim %c_555, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1272 = stablehlo.add %1248, %1271 : tensor<200x8xi32>
      %1273 = stablehlo.select %1270, %1272, %1248 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_556 = stablehlo.constant dense<0> : tensor<i32>
      %1274 = stablehlo.broadcast_in_dim %c_556, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1275 = stablehlo.compare  LT, %1263, %1274,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_557 = stablehlo.constant dense<257> : tensor<i32>
      %1276 = stablehlo.broadcast_in_dim %c_557, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1277 = stablehlo.add %1263, %1276 : tensor<200x8xi32>
      %1278 = stablehlo.select %1275, %1277, %1263 : tensor<200x8xi1>, tensor<200x8xi32>
      %1279 = stablehlo.broadcast_in_dim %1268, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1280 = stablehlo.broadcast_in_dim %1273, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1281 = stablehlo.broadcast_in_dim %1278, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1282 = stablehlo.concatenate %1279, %1280, %1281, dim = 2 : (tensor<200x8x1xi32>, tensor<200x8x1xi32>, tensor<200x8x1xi32>) -> tensor<200x8x3xi32>
      %1283 = "stablehlo.gather"(%912, %1282) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x257xf32>, tensor<200x8x3xi32>) -> tensor<200x8xf32>
      %c_558 = stablehlo.constant dense<0> : tensor<i32>
      %1284 = func.call @_where_362(%1262, %1283, %c_558) : (tensor<200x8xi1>, tensor<200x8xf32>, tensor<i32>) -> tensor<200x8xf32>
      %1285 = stablehlo.multiply %1284, %arg82 : tensor<200x8xf32>
      %cst_559 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1286 = stablehlo.reduce(%1285 init: %cst_559) applies stablehlo.add across dimensions = [1] : (tensor<200x8xf32>, tensor<f32>) -> tensor<200xf32>
      %c_560 = stablehlo.constant dense<257> : tensor<i32>
      %1287:2 = func.call @divmod_369(%arg83, %c_560) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_561 = stablehlo.constant dense<256> : tensor<i32>
      %1288:2 = func.call @divmod_383(%1287#0, %c_561) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_562 = stablehlo.constant dense<257> : tensor<i32>
      %1289:2 = func.call @divmod_383(%1288#0, %c_562) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_563 = stablehlo.constant dense<0> : tensor<i32>
      %1290 = stablehlo.broadcast_in_dim %c_563, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1291 = stablehlo.compare  GT, %1289#0, %1290,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_564 = stablehlo.constant dense<-1> : tensor<i32>
      %1292 = stablehlo.broadcast_in_dim %c_564, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1293 = stablehlo.compare  LT, %1289#0, %1292,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_565 = stablehlo.constant dense<0> : tensor<i32>
      %1294 = func.call @_where_390(%1293, %c_565, %1289#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_566 = stablehlo.constant dense<256> : tensor<i32>
      %1295 = func.call @_where_390(%1291, %c_566, %1294) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_567 = stablehlo.constant dense<0> : tensor<i32>
      %1296 = func.call @_where_390(%1293, %c_567, %1288#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_568 = stablehlo.constant dense<255> : tensor<i32>
      %1297 = func.call @_where_390(%1291, %c_568, %1296) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_569 = stablehlo.constant dense<0> : tensor<i32>
      %1298 = func.call @_where_390(%1293, %c_569, %1287#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_570 = stablehlo.constant dense<256> : tensor<i32>
      %1299 = func.call @_where_390(%1291, %c_570, %1298) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_571 = stablehlo.constant dense<1> : tensor<ui32>
      %c_572 = stablehlo.constant dense<1> : tensor<ui32>
      %1300 = stablehlo.partition_id : tensor<ui32>
      %1301 = stablehlo.divide %1300, %c_571 : tensor<ui32>
      %1302 = stablehlo.remainder %1301, %c_572 : tensor<ui32>
      %1303 = stablehlo.convert %1302 : (tensor<ui32>) -> tensor<i32>
      %c_573 = stablehlo.constant dense<257> : tensor<i32>
      %1304 = stablehlo.multiply %1303, %c_573 : tensor<i32>
      %1305 = stablehlo.broadcast_in_dim %1304, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1306 = stablehlo.subtract %1299, %1305 : tensor<200x4xi32>
      %c_574 = stablehlo.constant dense<0> : tensor<i32>
      %1307 = stablehlo.broadcast_in_dim %c_574, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1308 = stablehlo.compare  GE, %1306, %1307,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_575 = stablehlo.constant dense<257> : tensor<i32>
      %1309 = stablehlo.broadcast_in_dim %c_575, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1310 = stablehlo.compare  LT, %1306, %1309,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1311 = stablehlo.and %1308, %1310 : tensor<200x4xi1>
      %c_576 = stablehlo.constant dense<0> : tensor<i32>
      %c_577 = stablehlo.constant dense<256> : tensor<i32>
      %1312 = func.call @clip_397(%1306, %c_576, %c_577) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_578 = stablehlo.constant dense<0> : tensor<i32>
      %1313 = stablehlo.broadcast_in_dim %c_578, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1314 = stablehlo.compare  LT, %1295, %1313,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_579 = stablehlo.constant dense<257> : tensor<i32>
      %1315 = stablehlo.broadcast_in_dim %c_579, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1316 = stablehlo.add %1295, %1315 : tensor<200x4xi32>
      %1317 = stablehlo.select %1314, %1316, %1295 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_580 = stablehlo.constant dense<0> : tensor<i32>
      %1318 = stablehlo.broadcast_in_dim %c_580, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1319 = stablehlo.compare  LT, %1297, %1318,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_581 = stablehlo.constant dense<256> : tensor<i32>
      %1320 = stablehlo.broadcast_in_dim %c_581, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1321 = stablehlo.add %1297, %1320 : tensor<200x4xi32>
      %1322 = stablehlo.select %1319, %1321, %1297 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_582 = stablehlo.constant dense<0> : tensor<i32>
      %1323 = stablehlo.broadcast_in_dim %c_582, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1324 = stablehlo.compare  LT, %1312, %1323,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_583 = stablehlo.constant dense<257> : tensor<i32>
      %1325 = stablehlo.broadcast_in_dim %c_583, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1326 = stablehlo.add %1312, %1325 : tensor<200x4xi32>
      %1327 = stablehlo.select %1324, %1326, %1312 : tensor<200x4xi1>, tensor<200x4xi32>
      %1328 = stablehlo.broadcast_in_dim %1317, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1329 = stablehlo.broadcast_in_dim %1322, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1330 = stablehlo.broadcast_in_dim %1327, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1331 = stablehlo.concatenate %1328, %1329, %1330, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1332 = "stablehlo.gather"(%1020, %1331) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_584 = stablehlo.constant dense<0> : tensor<i32>
      %1333 = func.call @_where_408(%1311, %1332, %c_584) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1334 = stablehlo.multiply %1333, %arg84 : tensor<200x4xf32>
      %cst_585 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1335 = stablehlo.reduce(%1334 init: %cst_585) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_586 = stablehlo.constant dense<257> : tensor<i32>
      %1336:2 = func.call @divmod_369(%arg85, %c_586) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_587 = stablehlo.constant dense<257> : tensor<i32>
      %1337:2 = func.call @divmod_383(%1336#0, %c_587) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_588 = stablehlo.constant dense<256> : tensor<i32>
      %1338:2 = func.call @divmod_383(%1337#0, %c_588) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_589 = stablehlo.constant dense<0> : tensor<i32>
      %1339 = stablehlo.broadcast_in_dim %c_589, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1340 = stablehlo.compare  GT, %1338#0, %1339,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_590 = stablehlo.constant dense<-1> : tensor<i32>
      %1341 = stablehlo.broadcast_in_dim %c_590, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1342 = stablehlo.compare  LT, %1338#0, %1341,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_591 = stablehlo.constant dense<0> : tensor<i32>
      %1343 = func.call @_where_390(%1342, %c_591, %1338#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_592 = stablehlo.constant dense<255> : tensor<i32>
      %1344 = func.call @_where_390(%1340, %c_592, %1343) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_593 = stablehlo.constant dense<0> : tensor<i32>
      %1345 = func.call @_where_390(%1342, %c_593, %1337#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_594 = stablehlo.constant dense<256> : tensor<i32>
      %1346 = func.call @_where_390(%1340, %c_594, %1345) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_595 = stablehlo.constant dense<0> : tensor<i32>
      %1347 = func.call @_where_390(%1342, %c_595, %1336#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_596 = stablehlo.constant dense<256> : tensor<i32>
      %1348 = func.call @_where_390(%1340, %c_596, %1347) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_597 = stablehlo.constant dense<1> : tensor<ui32>
      %c_598 = stablehlo.constant dense<1> : tensor<ui32>
      %1349 = stablehlo.partition_id : tensor<ui32>
      %1350 = stablehlo.divide %1349, %c_597 : tensor<ui32>
      %1351 = stablehlo.remainder %1350, %c_598 : tensor<ui32>
      %1352 = stablehlo.convert %1351 : (tensor<ui32>) -> tensor<i32>
      %c_599 = stablehlo.constant dense<257> : tensor<i32>
      %1353 = stablehlo.multiply %1352, %c_599 : tensor<i32>
      %1354 = stablehlo.broadcast_in_dim %1353, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1355 = stablehlo.subtract %1348, %1354 : tensor<200x4xi32>
      %c_600 = stablehlo.constant dense<0> : tensor<i32>
      %1356 = stablehlo.broadcast_in_dim %c_600, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1357 = stablehlo.compare  GE, %1355, %1356,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_601 = stablehlo.constant dense<257> : tensor<i32>
      %1358 = stablehlo.broadcast_in_dim %c_601, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1359 = stablehlo.compare  LT, %1355, %1358,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1360 = stablehlo.and %1357, %1359 : tensor<200x4xi1>
      %c_602 = stablehlo.constant dense<0> : tensor<i32>
      %c_603 = stablehlo.constant dense<256> : tensor<i32>
      %1361 = func.call @clip_397(%1355, %c_602, %c_603) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_604 = stablehlo.constant dense<0> : tensor<i32>
      %1362 = stablehlo.broadcast_in_dim %c_604, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1363 = stablehlo.compare  LT, %1344, %1362,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_605 = stablehlo.constant dense<256> : tensor<i32>
      %1364 = stablehlo.broadcast_in_dim %c_605, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1365 = stablehlo.add %1344, %1364 : tensor<200x4xi32>
      %1366 = stablehlo.select %1363, %1365, %1344 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_606 = stablehlo.constant dense<0> : tensor<i32>
      %1367 = stablehlo.broadcast_in_dim %c_606, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1368 = stablehlo.compare  LT, %1346, %1367,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_607 = stablehlo.constant dense<257> : tensor<i32>
      %1369 = stablehlo.broadcast_in_dim %c_607, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1370 = stablehlo.add %1346, %1369 : tensor<200x4xi32>
      %1371 = stablehlo.select %1368, %1370, %1346 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_608 = stablehlo.constant dense<0> : tensor<i32>
      %1372 = stablehlo.broadcast_in_dim %c_608, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1373 = stablehlo.compare  LT, %1361, %1372,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_609 = stablehlo.constant dense<257> : tensor<i32>
      %1374 = stablehlo.broadcast_in_dim %c_609, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1375 = stablehlo.add %1361, %1374 : tensor<200x4xi32>
      %1376 = stablehlo.select %1373, %1375, %1361 : tensor<200x4xi1>, tensor<200x4xi32>
      %1377 = stablehlo.broadcast_in_dim %1366, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1378 = stablehlo.broadcast_in_dim %1371, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1379 = stablehlo.broadcast_in_dim %1376, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1380 = stablehlo.concatenate %1377, %1378, %1379, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1381 = "stablehlo.gather"(%1118, %1380) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_610 = stablehlo.constant dense<0> : tensor<i32>
      %1382 = func.call @_where_408(%1360, %1381, %c_610) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1383 = stablehlo.multiply %1382, %arg86 : tensor<200x4xf32>
      %cst_611 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1384 = stablehlo.reduce(%1383 init: %cst_611) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_612 = stablehlo.constant dense<257> : tensor<i32>
      %1385:2 = func.call @divmod_416(%arg87, %c_612) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_613 = stablehlo.constant dense<256> : tensor<i32>
      %1386:2 = func.call @divmod_430(%1385#0, %c_613) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_614 = stablehlo.constant dense<256> : tensor<i32>
      %1387:2 = func.call @divmod_430(%1386#0, %c_614) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_615 = stablehlo.constant dense<0> : tensor<i32>
      %1388 = stablehlo.broadcast_in_dim %c_615, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1389 = stablehlo.compare  GT, %1387#0, %1388,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_616 = stablehlo.constant dense<-1> : tensor<i32>
      %1390 = stablehlo.broadcast_in_dim %c_616, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1391 = stablehlo.compare  LT, %1387#0, %1390,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_617 = stablehlo.constant dense<0> : tensor<i32>
      %1392 = func.call @_where_437(%1391, %c_617, %1387#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_618 = stablehlo.constant dense<255> : tensor<i32>
      %1393 = func.call @_where_437(%1389, %c_618, %1392) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_619 = stablehlo.constant dense<0> : tensor<i32>
      %1394 = func.call @_where_437(%1391, %c_619, %1386#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_620 = stablehlo.constant dense<255> : tensor<i32>
      %1395 = func.call @_where_437(%1389, %c_620, %1394) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_621 = stablehlo.constant dense<0> : tensor<i32>
      %1396 = func.call @_where_437(%1391, %c_621, %1385#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_622 = stablehlo.constant dense<256> : tensor<i32>
      %1397 = func.call @_where_437(%1389, %c_622, %1396) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_623 = stablehlo.constant dense<1> : tensor<ui32>
      %c_624 = stablehlo.constant dense<1> : tensor<ui32>
      %1398 = stablehlo.partition_id : tensor<ui32>
      %1399 = stablehlo.divide %1398, %c_623 : tensor<ui32>
      %1400 = stablehlo.remainder %1399, %c_624 : tensor<ui32>
      %1401 = stablehlo.convert %1400 : (tensor<ui32>) -> tensor<i32>
      %c_625 = stablehlo.constant dense<257> : tensor<i32>
      %1402 = stablehlo.multiply %1401, %c_625 : tensor<i32>
      %1403 = stablehlo.broadcast_in_dim %1402, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1404 = stablehlo.subtract %1397, %1403 : tensor<200x2xi32>
      %c_626 = stablehlo.constant dense<0> : tensor<i32>
      %1405 = stablehlo.broadcast_in_dim %c_626, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1406 = stablehlo.compare  GE, %1404, %1405,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_627 = stablehlo.constant dense<257> : tensor<i32>
      %1407 = stablehlo.broadcast_in_dim %c_627, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1408 = stablehlo.compare  LT, %1404, %1407,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %1409 = stablehlo.and %1406, %1408 : tensor<200x2xi1>
      %c_628 = stablehlo.constant dense<0> : tensor<i32>
      %c_629 = stablehlo.constant dense<256> : tensor<i32>
      %1410 = func.call @clip_444(%1404, %c_628, %c_629) : (tensor<200x2xi32>, tensor<i32>, tensor<i32>) -> tensor<200x2xi32>
      %c_630 = stablehlo.constant dense<0> : tensor<i32>
      %1411 = stablehlo.broadcast_in_dim %c_630, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1412 = stablehlo.compare  LT, %1393, %1411,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_631 = stablehlo.constant dense<256> : tensor<i32>
      %1413 = stablehlo.broadcast_in_dim %c_631, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1414 = stablehlo.add %1393, %1413 : tensor<200x2xi32>
      %1415 = stablehlo.select %1412, %1414, %1393 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_632 = stablehlo.constant dense<0> : tensor<i32>
      %1416 = stablehlo.broadcast_in_dim %c_632, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1417 = stablehlo.compare  LT, %1395, %1416,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_633 = stablehlo.constant dense<256> : tensor<i32>
      %1418 = stablehlo.broadcast_in_dim %c_633, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1419 = stablehlo.add %1395, %1418 : tensor<200x2xi32>
      %1420 = stablehlo.select %1417, %1419, %1395 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_634 = stablehlo.constant dense<0> : tensor<i32>
      %1421 = stablehlo.broadcast_in_dim %c_634, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1422 = stablehlo.compare  LT, %1410, %1421,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_635 = stablehlo.constant dense<257> : tensor<i32>
      %1423 = stablehlo.broadcast_in_dim %c_635, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1424 = stablehlo.add %1410, %1423 : tensor<200x2xi32>
      %1425 = stablehlo.select %1422, %1424, %1410 : tensor<200x2xi1>, tensor<200x2xi32>
      %1426 = stablehlo.broadcast_in_dim %1415, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1427 = stablehlo.broadcast_in_dim %1420, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1428 = stablehlo.broadcast_in_dim %1425, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1429 = stablehlo.concatenate %1426, %1427, %1428, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
      %1430 = "stablehlo.gather"(%353, %1429) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
      %c_636 = stablehlo.constant dense<0> : tensor<i32>
      %1431 = func.call @_where_455(%1409, %1430, %c_636) : (tensor<200x2xi1>, tensor<200x2xf32>, tensor<i32>) -> tensor<200x2xf32>
      %1432 = stablehlo.multiply %1431, %arg88 : tensor<200x2xf32>
      %cst_637 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1433 = stablehlo.reduce(%1432 init: %cst_637) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
      %c_638 = stablehlo.constant dense<257> : tensor<i32>
      %1434:2 = func.call @divmod_369(%arg89, %c_638) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_639 = stablehlo.constant dense<257> : tensor<i32>
      %1435:2 = func.call @divmod_383(%1434#0, %c_639) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_640 = stablehlo.constant dense<256> : tensor<i32>
      %1436:2 = func.call @divmod_383(%1435#0, %c_640) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_641 = stablehlo.constant dense<0> : tensor<i32>
      %1437 = stablehlo.broadcast_in_dim %c_641, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1438 = stablehlo.compare  GT, %1436#0, %1437,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_642 = stablehlo.constant dense<-1> : tensor<i32>
      %1439 = stablehlo.broadcast_in_dim %c_642, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1440 = stablehlo.compare  LT, %1436#0, %1439,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_643 = stablehlo.constant dense<0> : tensor<i32>
      %1441 = func.call @_where_390(%1440, %c_643, %1436#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_644 = stablehlo.constant dense<255> : tensor<i32>
      %1442 = func.call @_where_390(%1438, %c_644, %1441) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_645 = stablehlo.constant dense<0> : tensor<i32>
      %1443 = func.call @_where_390(%1440, %c_645, %1435#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_646 = stablehlo.constant dense<256> : tensor<i32>
      %1444 = func.call @_where_390(%1438, %c_646, %1443) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_647 = stablehlo.constant dense<0> : tensor<i32>
      %1445 = func.call @_where_390(%1440, %c_647, %1434#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_648 = stablehlo.constant dense<256> : tensor<i32>
      %1446 = func.call @_where_390(%1438, %c_648, %1445) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_649 = stablehlo.constant dense<1> : tensor<ui32>
      %c_650 = stablehlo.constant dense<1> : tensor<ui32>
      %1447 = stablehlo.partition_id : tensor<ui32>
      %1448 = stablehlo.divide %1447, %c_649 : tensor<ui32>
      %1449 = stablehlo.remainder %1448, %c_650 : tensor<ui32>
      %1450 = stablehlo.convert %1449 : (tensor<ui32>) -> tensor<i32>
      %c_651 = stablehlo.constant dense<257> : tensor<i32>
      %1451 = stablehlo.multiply %1450, %c_651 : tensor<i32>
      %1452 = stablehlo.broadcast_in_dim %1451, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1453 = stablehlo.subtract %1446, %1452 : tensor<200x4xi32>
      %c_652 = stablehlo.constant dense<0> : tensor<i32>
      %1454 = stablehlo.broadcast_in_dim %c_652, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1455 = stablehlo.compare  GE, %1453, %1454,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_653 = stablehlo.constant dense<257> : tensor<i32>
      %1456 = stablehlo.broadcast_in_dim %c_653, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1457 = stablehlo.compare  LT, %1453, %1456,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1458 = stablehlo.and %1455, %1457 : tensor<200x4xi1>
      %c_654 = stablehlo.constant dense<0> : tensor<i32>
      %c_655 = stablehlo.constant dense<256> : tensor<i32>
      %1459 = func.call @clip_397(%1453, %c_654, %c_655) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_656 = stablehlo.constant dense<0> : tensor<i32>
      %1460 = stablehlo.broadcast_in_dim %c_656, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1461 = stablehlo.compare  LT, %1442, %1460,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_657 = stablehlo.constant dense<256> : tensor<i32>
      %1462 = stablehlo.broadcast_in_dim %c_657, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1463 = stablehlo.add %1442, %1462 : tensor<200x4xi32>
      %1464 = stablehlo.select %1461, %1463, %1442 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_658 = stablehlo.constant dense<0> : tensor<i32>
      %1465 = stablehlo.broadcast_in_dim %c_658, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1466 = stablehlo.compare  LT, %1444, %1465,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_659 = stablehlo.constant dense<257> : tensor<i32>
      %1467 = stablehlo.broadcast_in_dim %c_659, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1468 = stablehlo.add %1444, %1467 : tensor<200x4xi32>
      %1469 = stablehlo.select %1466, %1468, %1444 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_660 = stablehlo.constant dense<0> : tensor<i32>
      %1470 = stablehlo.broadcast_in_dim %c_660, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1471 = stablehlo.compare  LT, %1459, %1470,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_661 = stablehlo.constant dense<257> : tensor<i32>
      %1472 = stablehlo.broadcast_in_dim %c_661, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1473 = stablehlo.add %1459, %1472 : tensor<200x4xi32>
      %1474 = stablehlo.select %1471, %1473, %1459 : tensor<200x4xi1>, tensor<200x4xi32>
      %1475 = stablehlo.broadcast_in_dim %1464, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1476 = stablehlo.broadcast_in_dim %1469, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1477 = stablehlo.broadcast_in_dim %1474, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1478 = stablehlo.concatenate %1475, %1476, %1477, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1479 = "stablehlo.gather"(%461, %1478) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_662 = stablehlo.constant dense<0> : tensor<i32>
      %1480 = func.call @_where_408(%1458, %1479, %c_662) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1481 = stablehlo.multiply %1480, %arg90 : tensor<200x4xf32>
      %cst_663 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1482 = stablehlo.reduce(%1481 init: %cst_663) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_664 = stablehlo.constant dense<257> : tensor<i32>
      %1483:2 = func.call @divmod_369(%arg91, %c_664) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_665 = stablehlo.constant dense<256> : tensor<i32>
      %1484:2 = func.call @divmod_383(%1483#0, %c_665) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_666 = stablehlo.constant dense<257> : tensor<i32>
      %1485:2 = func.call @divmod_383(%1484#0, %c_666) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_667 = stablehlo.constant dense<0> : tensor<i32>
      %1486 = stablehlo.broadcast_in_dim %c_667, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1487 = stablehlo.compare  GT, %1485#0, %1486,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_668 = stablehlo.constant dense<-1> : tensor<i32>
      %1488 = stablehlo.broadcast_in_dim %c_668, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1489 = stablehlo.compare  LT, %1485#0, %1488,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_669 = stablehlo.constant dense<0> : tensor<i32>
      %1490 = func.call @_where_390(%1489, %c_669, %1485#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_670 = stablehlo.constant dense<256> : tensor<i32>
      %1491 = func.call @_where_390(%1487, %c_670, %1490) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_671 = stablehlo.constant dense<0> : tensor<i32>
      %1492 = func.call @_where_390(%1489, %c_671, %1484#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_672 = stablehlo.constant dense<255> : tensor<i32>
      %1493 = func.call @_where_390(%1487, %c_672, %1492) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_673 = stablehlo.constant dense<0> : tensor<i32>
      %1494 = func.call @_where_390(%1489, %c_673, %1483#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_674 = stablehlo.constant dense<256> : tensor<i32>
      %1495 = func.call @_where_390(%1487, %c_674, %1494) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_675 = stablehlo.constant dense<1> : tensor<ui32>
      %c_676 = stablehlo.constant dense<1> : tensor<ui32>
      %1496 = stablehlo.partition_id : tensor<ui32>
      %1497 = stablehlo.divide %1496, %c_675 : tensor<ui32>
      %1498 = stablehlo.remainder %1497, %c_676 : tensor<ui32>
      %1499 = stablehlo.convert %1498 : (tensor<ui32>) -> tensor<i32>
      %c_677 = stablehlo.constant dense<257> : tensor<i32>
      %1500 = stablehlo.multiply %1499, %c_677 : tensor<i32>
      %1501 = stablehlo.broadcast_in_dim %1500, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1502 = stablehlo.subtract %1495, %1501 : tensor<200x4xi32>
      %c_678 = stablehlo.constant dense<0> : tensor<i32>
      %1503 = stablehlo.broadcast_in_dim %c_678, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1504 = stablehlo.compare  GE, %1502, %1503,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_679 = stablehlo.constant dense<257> : tensor<i32>
      %1505 = stablehlo.broadcast_in_dim %c_679, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1506 = stablehlo.compare  LT, %1502, %1505,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1507 = stablehlo.and %1504, %1506 : tensor<200x4xi1>
      %c_680 = stablehlo.constant dense<0> : tensor<i32>
      %c_681 = stablehlo.constant dense<256> : tensor<i32>
      %1508 = func.call @clip_397(%1502, %c_680, %c_681) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_682 = stablehlo.constant dense<0> : tensor<i32>
      %1509 = stablehlo.broadcast_in_dim %c_682, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1510 = stablehlo.compare  LT, %1491, %1509,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_683 = stablehlo.constant dense<257> : tensor<i32>
      %1511 = stablehlo.broadcast_in_dim %c_683, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1512 = stablehlo.add %1491, %1511 : tensor<200x4xi32>
      %1513 = stablehlo.select %1510, %1512, %1491 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_684 = stablehlo.constant dense<0> : tensor<i32>
      %1514 = stablehlo.broadcast_in_dim %c_684, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1515 = stablehlo.compare  LT, %1493, %1514,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_685 = stablehlo.constant dense<256> : tensor<i32>
      %1516 = stablehlo.broadcast_in_dim %c_685, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1517 = stablehlo.add %1493, %1516 : tensor<200x4xi32>
      %1518 = stablehlo.select %1515, %1517, %1493 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_686 = stablehlo.constant dense<0> : tensor<i32>
      %1519 = stablehlo.broadcast_in_dim %c_686, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1520 = stablehlo.compare  LT, %1508, %1519,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_687 = stablehlo.constant dense<257> : tensor<i32>
      %1521 = stablehlo.broadcast_in_dim %c_687, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1522 = stablehlo.add %1508, %1521 : tensor<200x4xi32>
      %1523 = stablehlo.select %1520, %1522, %1508 : tensor<200x4xi1>, tensor<200x4xi32>
      %1524 = stablehlo.broadcast_in_dim %1513, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1525 = stablehlo.broadcast_in_dim %1518, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1526 = stablehlo.broadcast_in_dim %1523, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1527 = stablehlo.concatenate %1524, %1525, %1526, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1528 = "stablehlo.gather"(%559, %1527) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_688 = stablehlo.constant dense<0> : tensor<i32>
      %1529 = func.call @_where_408(%1507, %1528, %c_688) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1530 = stablehlo.multiply %1529, %arg92 : tensor<200x4xf32>
      %cst_689 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1531 = stablehlo.reduce(%1530 init: %cst_689) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %1532 = stablehlo.slice %arg142 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
      %1533 = stablehlo.reshape %1532 : (tensor<1x7200xf32>) -> tensor<7200xf32>
      %1534 = stablehlo.slice %arg143 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
      %1535 = stablehlo.reshape %1534 : (tensor<1x7200xf32>) -> tensor<7200xf32>
      %1536 = stablehlo.broadcast_in_dim %1286, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1537 = stablehlo.broadcast_in_dim %1335, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1538 = stablehlo.broadcast_in_dim %1384, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1539 = stablehlo.broadcast_in_dim %1433, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1540 = stablehlo.broadcast_in_dim %1482, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1541 = stablehlo.broadcast_in_dim %1531, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1542 = stablehlo.concatenate %1536, %1537, %1538, %1539, %1540, %1541, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %cst_690 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1543 = stablehlo.broadcast_in_dim %cst_690, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %1544 = stablehlo.multiply %1543, %arg93 : tensor<3xf32>
      %1545 = stablehlo.optimization_barrier %1544 : tensor<3xf32>
      %1546 = stablehlo.optimization_barrier %3 : tensor<f32>
      %1547 = stablehlo.broadcast_in_dim %1546, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %1548 = stablehlo.multiply %1545, %1547 : tensor<3xf32>
      %1549 = stablehlo.cosine %1548 : tensor<3xf32>
      %1550 = stablehlo.sine %1548 : tensor<3xf32>
      %cst_691 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_692 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1551 = stablehlo.subtract %cst_691, %cst_692 : tensor<f32>
      %cst_693 = stablehlo.constant dense<1.000000e-30> : tensor<f32>
      %1552 = stablehlo.maximum %1551, %cst_693 : tensor<f32>
      %cst_694 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1553 = stablehlo.subtract %3, %cst_694 : tensor<f32>
      %1554 = stablehlo.divide %1553, %1552 : tensor<f32>
      %cst_695 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %cst_696 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1555 = func.call @clip_472(%1554, %cst_695, %cst_696) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_697 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1556 = stablehlo.multiply %cst_697, %1555 : tensor<f32>
      %1557 = stablehlo.cosine %1556 : tensor<f32>
      %cst_698 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1558 = stablehlo.subtract %cst_698, %1557 : tensor<f32>
      %cst_699 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %1559 = stablehlo.multiply %cst_699, %1558 : tensor<f32>
      %cst_700 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %1560 = stablehlo.is_finite %cst_700 : (tensor<f32>) -> tensor<i1>
      %c_701 = stablehlo.constant dense<false> : tensor<i1>
      %1561 = stablehlo.and %c_701, %1560 : tensor<i1>
      %cst_702 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_703 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1562 = stablehlo.compare  GT, %cst_702, %cst_703,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %1563 = stablehlo.and %1561, %1562 : tensor<i1>
      %cst_704 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1564 = func.call @_where_479(%1563, %1559, %cst_704) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_705 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1565 = stablehlo.maximum %1564, %cst_705 : tensor<f32>
      %cst_706 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1566 = stablehlo.multiply %arg0, %cst_706 : tensor<f32>
      %cst_707 = stablehlo.constant dense<0x4D8EF3C2> : tensor<f32>
      %1567 = stablehlo.multiply %1566, %cst_707 : tensor<f32>
      %cst_708 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %1568 = stablehlo.divide %1567, %cst_708 : tensor<f32>
      %cst_709 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1569 = stablehlo.sqrt %cst_709 : tensor<f32>
      %1570 = stablehlo.divide %1568, %1569 : tensor<f32>
      %1571 = stablehlo.multiply %1565, %1570 : tensor<f32>
      %c_710 = stablehlo.constant dense<0> : tensor<i32>
      %c_711 = stablehlo.constant dense<1> : tensor<i32>
      %1572 = stablehlo.compare  EQ, %c_710, %c_711,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1573 = func.call @_where_482(%1572, %1571, %1565) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %1574 = stablehlo.broadcast_in_dim %arg94, dims = [0] : (tensor<6xf32>) -> tensor<6x1x1xf32>
      %1575 = stablehlo.broadcast_in_dim %1573, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %1576 = stablehlo.multiply %1575, %1574 : tensor<6x1x1xf32>
      %1577 = stablehlo.dot_general %1542, %1549, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %1578 = stablehlo.transpose %1577, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %1579 = stablehlo.broadcast_in_dim %1576, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %1580 = stablehlo.multiply %1579, %1578 : tensor<6x3x200xf32>
      %1581 = stablehlo.broadcast_in_dim %1573, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %1582 = stablehlo.multiply %1581, %1574 : tensor<6x1x1xf32>
      %1583 = stablehlo.dot_general %1542, %1550, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %1584 = stablehlo.transpose %1583, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %1585 = stablehlo.broadcast_in_dim %1582, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %1586 = stablehlo.multiply %1585, %1584 : tensor<6x3x200xf32>
      %1587 = stablehlo.reshape %1580 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_712 = stablehlo.constant dense<0> : tensor<i32>
      %1588 = stablehlo.broadcast_in_dim %c_712, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1589 = "stablehlo.scatter"(%1533, %1588, %1587) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg148: tensor<f32>, %arg149: tensor<f32>):
        %1598 = stablehlo.add %arg148, %arg149 : tensor<f32>
        stablehlo.return %1598 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %1590 = stablehlo.reshape %1586 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_713 = stablehlo.constant dense<0> : tensor<i32>
      %1591 = stablehlo.broadcast_in_dim %c_713, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1592 = "stablehlo.scatter"(%1535, %1591, %1590) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg148: tensor<f32>, %arg149: tensor<f32>):
        %1598 = stablehlo.add %arg148, %arg149 : tensor<f32>
        stablehlo.return %1598 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %1593 = stablehlo.broadcast_in_dim %1564, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %c_714 = stablehlo.constant dense<0> : tensor<i32>
      %1594 = stablehlo.broadcast_in_dim %c_714, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1595 = "stablehlo.scatter"(%arg144, %1594, %1593) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg148: tensor<f32>, %arg149: tensor<f32>):
        %1598 = stablehlo.add %arg148, %arg149 : tensor<f32>
        stablehlo.return %1598 : tensor<f32>
      }) : (tensor<6xf32>, tensor<1xi32>, tensor<3xf32>) -> tensor<6xf32>
      %1596 = stablehlo.broadcast_in_dim %1589, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
      %1597 = stablehlo.broadcast_in_dim %1592, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
      stablehlo.return %1596, %1597, %1595 : tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>
    }) : (tensor<i32>) -> (tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>)
    %c_503 = stablehlo.constant dense<1> : tensor<i32>
    %1178 = stablehlo.add %arg146, %c_503 : tensor<i32>
    %c_504 = stablehlo.constant dense<1> : tensor<i32>
    %c_505 = stablehlo.constant dense<1> : tensor<i32>
    %1179 = stablehlo.maximum %c_504, %c_505 : tensor<i32>
    %1180 = call @remainder(%1178, %1179) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_506 = stablehlo.constant dense<0> : tensor<i32>
    %1181 = stablehlo.compare  EQ, %1180, %c_506,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1182 = stablehlo.slice %1167 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %1183 = stablehlo.reshape %1182 : (tensor<1xi32>) -> tensor<i32>
    %c_507 = stablehlo.constant dense<1> : tensor<i32>
    %1184 = stablehlo.compare  LT, %1183, %c_507,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1185 = stablehlo.and %1181, %1184 : tensor<i1>
    %c_508 = stablehlo.constant dense<false> : tensor<i1>
    %1186 = stablehlo.and %1185, %c_508 : tensor<i1>
    %c_509 = stablehlo.constant dense<false> : tensor<i1>
    %1187 = stablehlo.or %1186, %c_509 : tensor<i1>
    %1188 = stablehlo.convert %1187 : (tensor<i1>) -> tensor<i32>
    %1189 = "stablehlo.case"(%1188) ({
      %cst_534 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      stablehlo.return %cst_534 : tensor<f32>
    }, {
      %c_534 = stablehlo.constant dense<257> : tensor<i32>
      %1238:2 = func.call @divmod(%arg95, %c_534) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_535 = stablehlo.constant dense<257> : tensor<i32>
      %1239:2 = func.call @divmod(%1238#0, %c_535) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_536 = stablehlo.constant dense<257> : tensor<i32>
      %1240:2 = func.call @divmod(%1239#0, %c_536) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_537 = stablehlo.constant dense<0> : tensor<i32>
      %1241 = stablehlo.broadcast_in_dim %c_537, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1242 = stablehlo.compare  GT, %1240#0, %1241,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_538 = stablehlo.constant dense<-1> : tensor<i32>
      %1243 = stablehlo.broadcast_in_dim %c_538, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1244 = stablehlo.compare  LT, %1240#0, %1243,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_539 = stablehlo.constant dense<0> : tensor<i32>
      %1245 = func.call @_where_279(%1244, %c_539, %1240#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_540 = stablehlo.constant dense<256> : tensor<i32>
      %1246 = func.call @_where_279(%1242, %c_540, %1245) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_541 = stablehlo.constant dense<0> : tensor<i32>
      %1247 = func.call @_where_279(%1244, %c_541, %1239#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_542 = stablehlo.constant dense<256> : tensor<i32>
      %1248 = func.call @_where_279(%1242, %c_542, %1247) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_543 = stablehlo.constant dense<0> : tensor<i32>
      %1249 = func.call @_where_279(%1244, %c_543, %1238#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_544 = stablehlo.constant dense<256> : tensor<i32>
      %1250 = func.call @_where_279(%1242, %c_544, %1249) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_545 = stablehlo.constant dense<0> : tensor<i32>
      %1251 = stablehlo.broadcast_in_dim %c_545, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1252 = stablehlo.compare  LT, %1246, %1251,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_546 = stablehlo.constant dense<257> : tensor<i32>
      %1253 = stablehlo.broadcast_in_dim %c_546, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1254 = stablehlo.add %1246, %1253 : tensor<200x1xi32>
      %1255 = stablehlo.select %1252, %1254, %1246 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_547 = stablehlo.constant dense<0> : tensor<i32>
      %1256 = stablehlo.broadcast_in_dim %c_547, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1257 = stablehlo.compare  LT, %1248, %1256,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_548 = stablehlo.constant dense<257> : tensor<i32>
      %1258 = stablehlo.broadcast_in_dim %c_548, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1259 = stablehlo.add %1248, %1258 : tensor<200x1xi32>
      %1260 = stablehlo.select %1257, %1259, %1248 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_549 = stablehlo.constant dense<0> : tensor<i32>
      %1261 = stablehlo.broadcast_in_dim %c_549, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1262 = stablehlo.compare  LT, %1250, %1261,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_550 = stablehlo.constant dense<257> : tensor<i32>
      %1263 = stablehlo.broadcast_in_dim %c_550, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1264 = stablehlo.add %1250, %1263 : tensor<200x1xi32>
      %1265 = stablehlo.select %1262, %1264, %1250 : tensor<200x1xi1>, tensor<200x1xi32>
      %1266 = stablehlo.broadcast_in_dim %1255, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1267 = stablehlo.broadcast_in_dim %1260, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1268 = stablehlo.broadcast_in_dim %1265, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1269 = stablehlo.concatenate %1266, %1267, %1268, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1270 = "stablehlo.gather"(%912, %1269) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1271 = stablehlo.multiply %1270, %arg96 : tensor<200x1xf32>
      %cst_551 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1272 = stablehlo.reduce(%1271 init: %cst_551) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_552 = stablehlo.constant dense<257> : tensor<i32>
      %1273:2 = func.call @divmod(%arg97, %c_552) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_553 = stablehlo.constant dense<256> : tensor<i32>
      %1274:2 = func.call @divmod(%1273#0, %c_553) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_554 = stablehlo.constant dense<257> : tensor<i32>
      %1275:2 = func.call @divmod(%1274#0, %c_554) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_555 = stablehlo.constant dense<0> : tensor<i32>
      %1276 = stablehlo.broadcast_in_dim %c_555, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1277 = stablehlo.compare  GT, %1275#0, %1276,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_556 = stablehlo.constant dense<-1> : tensor<i32>
      %1278 = stablehlo.broadcast_in_dim %c_556, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1279 = stablehlo.compare  LT, %1275#0, %1278,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_557 = stablehlo.constant dense<0> : tensor<i32>
      %1280 = func.call @_where_279(%1279, %c_557, %1275#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_558 = stablehlo.constant dense<256> : tensor<i32>
      %1281 = func.call @_where_279(%1277, %c_558, %1280) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_559 = stablehlo.constant dense<0> : tensor<i32>
      %1282 = func.call @_where_279(%1279, %c_559, %1274#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_560 = stablehlo.constant dense<255> : tensor<i32>
      %1283 = func.call @_where_279(%1277, %c_560, %1282) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_561 = stablehlo.constant dense<0> : tensor<i32>
      %1284 = func.call @_where_279(%1279, %c_561, %1273#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_562 = stablehlo.constant dense<256> : tensor<i32>
      %1285 = func.call @_where_279(%1277, %c_562, %1284) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_563 = stablehlo.constant dense<0> : tensor<i32>
      %1286 = stablehlo.broadcast_in_dim %c_563, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1287 = stablehlo.compare  LT, %1281, %1286,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_564 = stablehlo.constant dense<257> : tensor<i32>
      %1288 = stablehlo.broadcast_in_dim %c_564, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1289 = stablehlo.add %1281, %1288 : tensor<200x1xi32>
      %1290 = stablehlo.select %1287, %1289, %1281 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_565 = stablehlo.constant dense<0> : tensor<i32>
      %1291 = stablehlo.broadcast_in_dim %c_565, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1292 = stablehlo.compare  LT, %1283, %1291,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_566 = stablehlo.constant dense<256> : tensor<i32>
      %1293 = stablehlo.broadcast_in_dim %c_566, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1294 = stablehlo.add %1283, %1293 : tensor<200x1xi32>
      %1295 = stablehlo.select %1292, %1294, %1283 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_567 = stablehlo.constant dense<0> : tensor<i32>
      %1296 = stablehlo.broadcast_in_dim %c_567, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1297 = stablehlo.compare  LT, %1285, %1296,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_568 = stablehlo.constant dense<257> : tensor<i32>
      %1298 = stablehlo.broadcast_in_dim %c_568, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1299 = stablehlo.add %1285, %1298 : tensor<200x1xi32>
      %1300 = stablehlo.select %1297, %1299, %1285 : tensor<200x1xi1>, tensor<200x1xi32>
      %1301 = stablehlo.broadcast_in_dim %1290, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1302 = stablehlo.broadcast_in_dim %1295, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1303 = stablehlo.broadcast_in_dim %1300, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1304 = stablehlo.concatenate %1301, %1302, %1303, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1305 = "stablehlo.gather"(%1020, %1304) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1306 = stablehlo.multiply %1305, %arg98 : tensor<200x1xf32>
      %cst_569 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1307 = stablehlo.reduce(%1306 init: %cst_569) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_570 = stablehlo.constant dense<257> : tensor<i32>
      %1308:2 = func.call @divmod(%arg99, %c_570) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_571 = stablehlo.constant dense<257> : tensor<i32>
      %1309:2 = func.call @divmod(%1308#0, %c_571) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_572 = stablehlo.constant dense<256> : tensor<i32>
      %1310:2 = func.call @divmod(%1309#0, %c_572) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_573 = stablehlo.constant dense<0> : tensor<i32>
      %1311 = stablehlo.broadcast_in_dim %c_573, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1312 = stablehlo.compare  GT, %1310#0, %1311,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_574 = stablehlo.constant dense<-1> : tensor<i32>
      %1313 = stablehlo.broadcast_in_dim %c_574, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1314 = stablehlo.compare  LT, %1310#0, %1313,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_575 = stablehlo.constant dense<0> : tensor<i32>
      %1315 = func.call @_where_279(%1314, %c_575, %1310#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_576 = stablehlo.constant dense<255> : tensor<i32>
      %1316 = func.call @_where_279(%1312, %c_576, %1315) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_577 = stablehlo.constant dense<0> : tensor<i32>
      %1317 = func.call @_where_279(%1314, %c_577, %1309#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_578 = stablehlo.constant dense<256> : tensor<i32>
      %1318 = func.call @_where_279(%1312, %c_578, %1317) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_579 = stablehlo.constant dense<0> : tensor<i32>
      %1319 = func.call @_where_279(%1314, %c_579, %1308#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_580 = stablehlo.constant dense<256> : tensor<i32>
      %1320 = func.call @_where_279(%1312, %c_580, %1319) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_581 = stablehlo.constant dense<0> : tensor<i32>
      %1321 = stablehlo.broadcast_in_dim %c_581, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1322 = stablehlo.compare  LT, %1316, %1321,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_582 = stablehlo.constant dense<256> : tensor<i32>
      %1323 = stablehlo.broadcast_in_dim %c_582, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1324 = stablehlo.add %1316, %1323 : tensor<200x1xi32>
      %1325 = stablehlo.select %1322, %1324, %1316 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_583 = stablehlo.constant dense<0> : tensor<i32>
      %1326 = stablehlo.broadcast_in_dim %c_583, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1327 = stablehlo.compare  LT, %1318, %1326,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_584 = stablehlo.constant dense<257> : tensor<i32>
      %1328 = stablehlo.broadcast_in_dim %c_584, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1329 = stablehlo.add %1318, %1328 : tensor<200x1xi32>
      %1330 = stablehlo.select %1327, %1329, %1318 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_585 = stablehlo.constant dense<0> : tensor<i32>
      %1331 = stablehlo.broadcast_in_dim %c_585, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1332 = stablehlo.compare  LT, %1320, %1331,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_586 = stablehlo.constant dense<257> : tensor<i32>
      %1333 = stablehlo.broadcast_in_dim %c_586, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1334 = stablehlo.add %1320, %1333 : tensor<200x1xi32>
      %1335 = stablehlo.select %1332, %1334, %1320 : tensor<200x1xi1>, tensor<200x1xi32>
      %1336 = stablehlo.broadcast_in_dim %1325, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1337 = stablehlo.broadcast_in_dim %1330, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1338 = stablehlo.broadcast_in_dim %1335, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1339 = stablehlo.concatenate %1336, %1337, %1338, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1340 = "stablehlo.gather"(%1118, %1339) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1341 = stablehlo.multiply %1340, %arg100 : tensor<200x1xf32>
      %cst_587 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1342 = stablehlo.reduce(%1341 init: %cst_587) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_588 = stablehlo.constant dense<257> : tensor<i32>
      %1343:2 = func.call @divmod(%arg101, %c_588) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_589 = stablehlo.constant dense<256> : tensor<i32>
      %1344:2 = func.call @divmod(%1343#0, %c_589) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_590 = stablehlo.constant dense<256> : tensor<i32>
      %1345:2 = func.call @divmod(%1344#0, %c_590) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_591 = stablehlo.constant dense<0> : tensor<i32>
      %1346 = stablehlo.broadcast_in_dim %c_591, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1347 = stablehlo.compare  GT, %1345#0, %1346,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_592 = stablehlo.constant dense<-1> : tensor<i32>
      %1348 = stablehlo.broadcast_in_dim %c_592, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1349 = stablehlo.compare  LT, %1345#0, %1348,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_593 = stablehlo.constant dense<0> : tensor<i32>
      %1350 = func.call @_where_279(%1349, %c_593, %1345#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_594 = stablehlo.constant dense<255> : tensor<i32>
      %1351 = func.call @_where_279(%1347, %c_594, %1350) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_595 = stablehlo.constant dense<0> : tensor<i32>
      %1352 = func.call @_where_279(%1349, %c_595, %1344#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_596 = stablehlo.constant dense<255> : tensor<i32>
      %1353 = func.call @_where_279(%1347, %c_596, %1352) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_597 = stablehlo.constant dense<0> : tensor<i32>
      %1354 = func.call @_where_279(%1349, %c_597, %1343#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_598 = stablehlo.constant dense<256> : tensor<i32>
      %1355 = func.call @_where_279(%1347, %c_598, %1354) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_599 = stablehlo.constant dense<0> : tensor<i32>
      %1356 = stablehlo.broadcast_in_dim %c_599, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1357 = stablehlo.compare  LT, %1351, %1356,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_600 = stablehlo.constant dense<256> : tensor<i32>
      %1358 = stablehlo.broadcast_in_dim %c_600, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1359 = stablehlo.add %1351, %1358 : tensor<200x1xi32>
      %1360 = stablehlo.select %1357, %1359, %1351 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_601 = stablehlo.constant dense<0> : tensor<i32>
      %1361 = stablehlo.broadcast_in_dim %c_601, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1362 = stablehlo.compare  LT, %1353, %1361,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_602 = stablehlo.constant dense<256> : tensor<i32>
      %1363 = stablehlo.broadcast_in_dim %c_602, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1364 = stablehlo.add %1353, %1363 : tensor<200x1xi32>
      %1365 = stablehlo.select %1362, %1364, %1353 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_603 = stablehlo.constant dense<0> : tensor<i32>
      %1366 = stablehlo.broadcast_in_dim %c_603, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1367 = stablehlo.compare  LT, %1355, %1366,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_604 = stablehlo.constant dense<257> : tensor<i32>
      %1368 = stablehlo.broadcast_in_dim %c_604, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1369 = stablehlo.add %1355, %1368 : tensor<200x1xi32>
      %1370 = stablehlo.select %1367, %1369, %1355 : tensor<200x1xi1>, tensor<200x1xi32>
      %1371 = stablehlo.broadcast_in_dim %1360, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1372 = stablehlo.broadcast_in_dim %1365, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1373 = stablehlo.broadcast_in_dim %1370, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1374 = stablehlo.concatenate %1371, %1372, %1373, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1375 = "stablehlo.gather"(%353, %1374) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1376 = stablehlo.multiply %1375, %arg102 : tensor<200x1xf32>
      %cst_605 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1377 = stablehlo.reduce(%1376 init: %cst_605) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_606 = stablehlo.constant dense<257> : tensor<i32>
      %1378:2 = func.call @divmod(%arg103, %c_606) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_607 = stablehlo.constant dense<257> : tensor<i32>
      %1379:2 = func.call @divmod(%1378#0, %c_607) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_608 = stablehlo.constant dense<256> : tensor<i32>
      %1380:2 = func.call @divmod(%1379#0, %c_608) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_609 = stablehlo.constant dense<0> : tensor<i32>
      %1381 = stablehlo.broadcast_in_dim %c_609, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1382 = stablehlo.compare  GT, %1380#0, %1381,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_610 = stablehlo.constant dense<-1> : tensor<i32>
      %1383 = stablehlo.broadcast_in_dim %c_610, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1384 = stablehlo.compare  LT, %1380#0, %1383,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_611 = stablehlo.constant dense<0> : tensor<i32>
      %1385 = func.call @_where_279(%1384, %c_611, %1380#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_612 = stablehlo.constant dense<255> : tensor<i32>
      %1386 = func.call @_where_279(%1382, %c_612, %1385) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_613 = stablehlo.constant dense<0> : tensor<i32>
      %1387 = func.call @_where_279(%1384, %c_613, %1379#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_614 = stablehlo.constant dense<256> : tensor<i32>
      %1388 = func.call @_where_279(%1382, %c_614, %1387) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_615 = stablehlo.constant dense<0> : tensor<i32>
      %1389 = func.call @_where_279(%1384, %c_615, %1378#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_616 = stablehlo.constant dense<256> : tensor<i32>
      %1390 = func.call @_where_279(%1382, %c_616, %1389) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_617 = stablehlo.constant dense<0> : tensor<i32>
      %1391 = stablehlo.broadcast_in_dim %c_617, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1392 = stablehlo.compare  LT, %1386, %1391,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_618 = stablehlo.constant dense<256> : tensor<i32>
      %1393 = stablehlo.broadcast_in_dim %c_618, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1394 = stablehlo.add %1386, %1393 : tensor<200x1xi32>
      %1395 = stablehlo.select %1392, %1394, %1386 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_619 = stablehlo.constant dense<0> : tensor<i32>
      %1396 = stablehlo.broadcast_in_dim %c_619, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1397 = stablehlo.compare  LT, %1388, %1396,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_620 = stablehlo.constant dense<257> : tensor<i32>
      %1398 = stablehlo.broadcast_in_dim %c_620, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1399 = stablehlo.add %1388, %1398 : tensor<200x1xi32>
      %1400 = stablehlo.select %1397, %1399, %1388 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_621 = stablehlo.constant dense<0> : tensor<i32>
      %1401 = stablehlo.broadcast_in_dim %c_621, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1402 = stablehlo.compare  LT, %1390, %1401,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_622 = stablehlo.constant dense<257> : tensor<i32>
      %1403 = stablehlo.broadcast_in_dim %c_622, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1404 = stablehlo.add %1390, %1403 : tensor<200x1xi32>
      %1405 = stablehlo.select %1402, %1404, %1390 : tensor<200x1xi1>, tensor<200x1xi32>
      %1406 = stablehlo.broadcast_in_dim %1395, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1407 = stablehlo.broadcast_in_dim %1400, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1408 = stablehlo.broadcast_in_dim %1405, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1409 = stablehlo.concatenate %1406, %1407, %1408, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1410 = "stablehlo.gather"(%461, %1409) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1411 = stablehlo.multiply %1410, %arg104 : tensor<200x1xf32>
      %cst_623 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1412 = stablehlo.reduce(%1411 init: %cst_623) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_624 = stablehlo.constant dense<257> : tensor<i32>
      %1413:2 = func.call @divmod(%arg105, %c_624) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_625 = stablehlo.constant dense<256> : tensor<i32>
      %1414:2 = func.call @divmod(%1413#0, %c_625) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_626 = stablehlo.constant dense<257> : tensor<i32>
      %1415:2 = func.call @divmod(%1414#0, %c_626) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_627 = stablehlo.constant dense<0> : tensor<i32>
      %1416 = stablehlo.broadcast_in_dim %c_627, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1417 = stablehlo.compare  GT, %1415#0, %1416,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_628 = stablehlo.constant dense<-1> : tensor<i32>
      %1418 = stablehlo.broadcast_in_dim %c_628, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1419 = stablehlo.compare  LT, %1415#0, %1418,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_629 = stablehlo.constant dense<0> : tensor<i32>
      %1420 = func.call @_where_279(%1419, %c_629, %1415#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_630 = stablehlo.constant dense<256> : tensor<i32>
      %1421 = func.call @_where_279(%1417, %c_630, %1420) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_631 = stablehlo.constant dense<0> : tensor<i32>
      %1422 = func.call @_where_279(%1419, %c_631, %1414#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_632 = stablehlo.constant dense<255> : tensor<i32>
      %1423 = func.call @_where_279(%1417, %c_632, %1422) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_633 = stablehlo.constant dense<0> : tensor<i32>
      %1424 = func.call @_where_279(%1419, %c_633, %1413#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_634 = stablehlo.constant dense<256> : tensor<i32>
      %1425 = func.call @_where_279(%1417, %c_634, %1424) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_635 = stablehlo.constant dense<0> : tensor<i32>
      %1426 = stablehlo.broadcast_in_dim %c_635, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1427 = stablehlo.compare  LT, %1421, %1426,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_636 = stablehlo.constant dense<257> : tensor<i32>
      %1428 = stablehlo.broadcast_in_dim %c_636, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1429 = stablehlo.add %1421, %1428 : tensor<200x1xi32>
      %1430 = stablehlo.select %1427, %1429, %1421 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_637 = stablehlo.constant dense<0> : tensor<i32>
      %1431 = stablehlo.broadcast_in_dim %c_637, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1432 = stablehlo.compare  LT, %1423, %1431,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_638 = stablehlo.constant dense<256> : tensor<i32>
      %1433 = stablehlo.broadcast_in_dim %c_638, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1434 = stablehlo.add %1423, %1433 : tensor<200x1xi32>
      %1435 = stablehlo.select %1432, %1434, %1423 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_639 = stablehlo.constant dense<0> : tensor<i32>
      %1436 = stablehlo.broadcast_in_dim %c_639, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1437 = stablehlo.compare  LT, %1425, %1436,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_640 = stablehlo.constant dense<257> : tensor<i32>
      %1438 = stablehlo.broadcast_in_dim %c_640, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1439 = stablehlo.add %1425, %1438 : tensor<200x1xi32>
      %1440 = stablehlo.select %1437, %1439, %1425 : tensor<200x1xi1>, tensor<200x1xi32>
      %1441 = stablehlo.broadcast_in_dim %1430, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1442 = stablehlo.broadcast_in_dim %1435, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1443 = stablehlo.broadcast_in_dim %1440, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1444 = stablehlo.concatenate %1441, %1442, %1443, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1445 = "stablehlo.gather"(%559, %1444) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %1446 = stablehlo.multiply %1445, %arg106 : tensor<200x1xf32>
      %cst_641 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1447 = stablehlo.reduce(%1446 init: %cst_641) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %1448 = stablehlo.broadcast_in_dim %1272, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1449 = stablehlo.broadcast_in_dim %1307, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1450 = stablehlo.broadcast_in_dim %1342, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1451 = stablehlo.broadcast_in_dim %1377, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1452 = stablehlo.broadcast_in_dim %1412, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1453 = stablehlo.broadcast_in_dim %1447, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1454 = stablehlo.concatenate %1448, %1449, %1450, %1451, %1452, %1453, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %1455 = stablehlo.slice %1454 [1:2, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1456 = stablehlo.reshape %1455 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1457 = stablehlo.slice %1454 [2:3, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1458 = stablehlo.reshape %1457 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1459 = stablehlo.slice %1454 [4:5, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1460 = stablehlo.reshape %1459 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1461 = stablehlo.slice %1454 [5:6, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %1462 = stablehlo.reshape %1461 : (tensor<1x200xf32>) -> tensor<200xf32>
      %1463 = stablehlo.multiply %1456, %1462 : tensor<200xf32>
      %1464 = stablehlo.multiply %1458, %1460 : tensor<200xf32>
      %1465 = stablehlo.subtract %1463, %1464 : tensor<200xf32>
      %cst_642 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1466 = stablehlo.broadcast_in_dim %cst_642, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %1467 = stablehlo.multiply %1465, %1466 : tensor<200xf32>
      %cst_643 = stablehlo.constant dense<7.485380e-15> : tensor<f32>
      %1468 = stablehlo.broadcast_in_dim %cst_643, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %1469 = stablehlo.multiply %1467, %1468 : tensor<200xf32>
      %cst_644 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1470 = stablehlo.reduce(%1469 init: %cst_644) applies stablehlo.add across dimensions = [0] : (tensor<200xf32>, tensor<f32>) -> tensor<f32>
      stablehlo.return %1470 : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %cst_510 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1190 = call @_where_299(%1186, %1189, %cst_510) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %1191 = stablehlo.slice %1167 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %1192 = stablehlo.reshape %1191 : (tensor<1xi32>) -> tensor<i32>
    %c_511 = stablehlo.constant dense<0> : tensor<i32>
    %1193 = stablehlo.minimum %1192, %c_511 : tensor<i32>
    %c_512 = stablehlo.constant dense<0> : tensor<i32>
    %1194 = stablehlo.compare  LT, %1193, %c_512,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_513 = stablehlo.constant dense<1> : tensor<i32>
    %1195 = stablehlo.add %1193, %c_513 : tensor<i32>
    %1196 = stablehlo.select %1194, %1195, %1193 : tensor<i1>, tensor<i32>
    %c_514 = stablehlo.constant dense<1> : tensor<i32>
    %1197 = stablehlo.dynamic_slice %1152, %c_514, %1196, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %1198 = stablehlo.reshape %1197 : (tensor<1x1xf32>) -> tensor<f32>
    %c_515 = stablehlo.constant dense<0> : tensor<i32>
    %1199 = stablehlo.compare  LT, %1193, %c_515,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_516 = stablehlo.constant dense<1> : tensor<i32>
    %1200 = stablehlo.add %1193, %c_516 : tensor<i32>
    %1201 = stablehlo.select %1199, %1200, %1193 : tensor<i1>, tensor<i32>
    %c_517 = stablehlo.constant dense<1> : tensor<i32>
    %1202 = stablehlo.dynamic_slice %1160, %c_517, %1201, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %1203 = stablehlo.reshape %1202 : (tensor<1x1xf32>) -> tensor<f32>
    %1204 = call @_where_304(%1186, %1190, %1198) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_518 = stablehlo.constant dense<0> : tensor<i32>
    %1205 = stablehlo.compare  LT, %1193, %c_518,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_519 = stablehlo.constant dense<1> : tensor<i32>
    %1206 = stablehlo.add %1193, %c_519 : tensor<i32>
    %1207 = stablehlo.select %1205, %1206, %1193 : tensor<i1>, tensor<i32>
    %c_520 = stablehlo.constant dense<1> : tensor<i32>
    %1208 = stablehlo.broadcast_in_dim %c_520, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1209 = stablehlo.broadcast_in_dim %1207, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1210 = stablehlo.concatenate %1208, %1209, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %1211 = "stablehlo.scatter"(%1152, %1210, %1204) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg148: tensor<f32>, %arg149: tensor<f32>):
      stablehlo.return %arg149 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %1212 = call @_where_304(%1186, %3, %1203) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_521 = stablehlo.constant dense<0> : tensor<i32>
    %1213 = stablehlo.compare  LT, %1193, %c_521,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_522 = stablehlo.constant dense<1> : tensor<i32>
    %1214 = stablehlo.add %1193, %c_522 : tensor<i32>
    %1215 = stablehlo.select %1213, %1214, %1193 : tensor<i1>, tensor<i32>
    %c_523 = stablehlo.constant dense<1> : tensor<i32>
    %1216 = stablehlo.broadcast_in_dim %c_523, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1217 = stablehlo.broadcast_in_dim %1215, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1218 = stablehlo.concatenate %1216, %1217, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %1219 = "stablehlo.scatter"(%1160, %1218, %1212) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg148: tensor<f32>, %arg149: tensor<f32>):
      stablehlo.return %arg149 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %1220 = stablehlo.slice %1167 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %1221 = stablehlo.reshape %1220 : (tensor<1xi32>) -> tensor<i32>
    %c_524 = stablehlo.constant dense<1> : tensor<i32>
    %c_525 = stablehlo.constant dense<0> : tensor<i32>
    %1222 = call @_where_310(%1186, %c_524, %c_525) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %1223 = stablehlo.convert %1222 : tensor<i32>
    %1224 = stablehlo.add %1221, %1223 : tensor<i32>
    %c_526 = stablehlo.constant dense<1> : tensor<i32>
    %1225 = stablehlo.broadcast_in_dim %c_526, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1226 = "stablehlo.scatter"(%1167, %1225, %1224) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg148: tensor<i32>, %arg149: tensor<i32>):
      stablehlo.return %arg149 : tensor<i32>
    }) : (tensor<2xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
    %cst_527 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1227 = stablehlo.compare  GE, %3, %cst_527,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %c_528 = stablehlo.constant dense<true> : tensor<i1>
    %1228 = stablehlo.and %c_528, %1227 : tensor<i1>
    %cst_529 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %1229 = stablehlo.compare  LE, %3, %cst_529,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %1230 = stablehlo.and %1228, %1229 : tensor<i1>
    %c_530 = stablehlo.constant dense<1> : tensor<i32>
    %c_531 = stablehlo.constant dense<1> : tensor<i32>
    %1231 = stablehlo.maximum %c_530, %c_531 : tensor<i32>
    %1232 = call @remainder(%arg146, %1231) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_532 = stablehlo.constant dense<0> : tensor<i32>
    %1233 = stablehlo.compare  EQ, %1232, %c_532,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1234 = stablehlo.and %1230, %1233 : tensor<i1>
    %1235 = stablehlo.convert %1234 : (tensor<i1>) -> tensor<i32>
    %1236:3 = "stablehlo.case"(%1235) ({
      stablehlo.return %1177#0, %1177#1, %1177#2 : tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>
    }, {
      %c_534 = stablehlo.constant dense<257> : tensor<i32>
      %1238:2 = func.call @divmod_319(%arg107, %c_534) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_535 = stablehlo.constant dense<257> : tensor<i32>
      %1239:2 = func.call @divmod_336(%1238#0, %c_535) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_536 = stablehlo.constant dense<257> : tensor<i32>
      %1240:2 = func.call @divmod_336(%1239#0, %c_536) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_537 = stablehlo.constant dense<0> : tensor<i32>
      %1241 = stablehlo.broadcast_in_dim %c_537, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1242 = stablehlo.compare  GT, %1240#0, %1241,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_538 = stablehlo.constant dense<-1> : tensor<i32>
      %1243 = stablehlo.broadcast_in_dim %c_538, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1244 = stablehlo.compare  LT, %1240#0, %1243,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_539 = stablehlo.constant dense<0> : tensor<i32>
      %1245 = func.call @_where_343(%1244, %c_539, %1240#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_540 = stablehlo.constant dense<256> : tensor<i32>
      %1246 = func.call @_where_343(%1242, %c_540, %1245) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_541 = stablehlo.constant dense<0> : tensor<i32>
      %1247 = func.call @_where_343(%1244, %c_541, %1239#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_542 = stablehlo.constant dense<256> : tensor<i32>
      %1248 = func.call @_where_343(%1242, %c_542, %1247) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_543 = stablehlo.constant dense<0> : tensor<i32>
      %1249 = func.call @_where_343(%1244, %c_543, %1238#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_544 = stablehlo.constant dense<256> : tensor<i32>
      %1250 = func.call @_where_343(%1242, %c_544, %1249) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_545 = stablehlo.constant dense<1> : tensor<ui32>
      %c_546 = stablehlo.constant dense<1> : tensor<ui32>
      %1251 = stablehlo.partition_id : tensor<ui32>
      %1252 = stablehlo.divide %1251, %c_545 : tensor<ui32>
      %1253 = stablehlo.remainder %1252, %c_546 : tensor<ui32>
      %1254 = stablehlo.convert %1253 : (tensor<ui32>) -> tensor<i32>
      %c_547 = stablehlo.constant dense<257> : tensor<i32>
      %1255 = stablehlo.multiply %1254, %c_547 : tensor<i32>
      %1256 = stablehlo.broadcast_in_dim %1255, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1257 = stablehlo.subtract %1250, %1256 : tensor<200x8xi32>
      %c_548 = stablehlo.constant dense<0> : tensor<i32>
      %1258 = stablehlo.broadcast_in_dim %c_548, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1259 = stablehlo.compare  GE, %1257, %1258,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_549 = stablehlo.constant dense<257> : tensor<i32>
      %1260 = stablehlo.broadcast_in_dim %c_549, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1261 = stablehlo.compare  LT, %1257, %1260,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %1262 = stablehlo.and %1259, %1261 : tensor<200x8xi1>
      %c_550 = stablehlo.constant dense<0> : tensor<i32>
      %c_551 = stablehlo.constant dense<256> : tensor<i32>
      %1263 = func.call @clip_350(%1257, %c_550, %c_551) : (tensor<200x8xi32>, tensor<i32>, tensor<i32>) -> tensor<200x8xi32>
      %c_552 = stablehlo.constant dense<0> : tensor<i32>
      %1264 = stablehlo.broadcast_in_dim %c_552, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1265 = stablehlo.compare  LT, %1246, %1264,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_553 = stablehlo.constant dense<257> : tensor<i32>
      %1266 = stablehlo.broadcast_in_dim %c_553, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1267 = stablehlo.add %1246, %1266 : tensor<200x8xi32>
      %1268 = stablehlo.select %1265, %1267, %1246 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_554 = stablehlo.constant dense<0> : tensor<i32>
      %1269 = stablehlo.broadcast_in_dim %c_554, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1270 = stablehlo.compare  LT, %1248, %1269,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_555 = stablehlo.constant dense<257> : tensor<i32>
      %1271 = stablehlo.broadcast_in_dim %c_555, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1272 = stablehlo.add %1248, %1271 : tensor<200x8xi32>
      %1273 = stablehlo.select %1270, %1272, %1248 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_556 = stablehlo.constant dense<0> : tensor<i32>
      %1274 = stablehlo.broadcast_in_dim %c_556, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1275 = stablehlo.compare  LT, %1263, %1274,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_557 = stablehlo.constant dense<257> : tensor<i32>
      %1276 = stablehlo.broadcast_in_dim %c_557, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %1277 = stablehlo.add %1263, %1276 : tensor<200x8xi32>
      %1278 = stablehlo.select %1275, %1277, %1263 : tensor<200x8xi1>, tensor<200x8xi32>
      %1279 = stablehlo.broadcast_in_dim %1268, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1280 = stablehlo.broadcast_in_dim %1273, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1281 = stablehlo.broadcast_in_dim %1278, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %1282 = stablehlo.concatenate %1279, %1280, %1281, dim = 2 : (tensor<200x8x1xi32>, tensor<200x8x1xi32>, tensor<200x8x1xi32>) -> tensor<200x8x3xi32>
      %1283 = "stablehlo.gather"(%912, %1282) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x257xf32>, tensor<200x8x3xi32>) -> tensor<200x8xf32>
      %c_558 = stablehlo.constant dense<0> : tensor<i32>
      %1284 = func.call @_where_362(%1262, %1283, %c_558) : (tensor<200x8xi1>, tensor<200x8xf32>, tensor<i32>) -> tensor<200x8xf32>
      %1285 = stablehlo.multiply %1284, %arg108 : tensor<200x8xf32>
      %cst_559 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1286 = stablehlo.reduce(%1285 init: %cst_559) applies stablehlo.add across dimensions = [1] : (tensor<200x8xf32>, tensor<f32>) -> tensor<200xf32>
      %c_560 = stablehlo.constant dense<257> : tensor<i32>
      %1287:2 = func.call @divmod_416(%arg109, %c_560) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_561 = stablehlo.constant dense<256> : tensor<i32>
      %1288:2 = func.call @divmod_430(%1287#0, %c_561) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_562 = stablehlo.constant dense<257> : tensor<i32>
      %1289:2 = func.call @divmod_430(%1288#0, %c_562) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_563 = stablehlo.constant dense<0> : tensor<i32>
      %1290 = stablehlo.broadcast_in_dim %c_563, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1291 = stablehlo.compare  GT, %1289#0, %1290,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_564 = stablehlo.constant dense<-1> : tensor<i32>
      %1292 = stablehlo.broadcast_in_dim %c_564, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1293 = stablehlo.compare  LT, %1289#0, %1292,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_565 = stablehlo.constant dense<0> : tensor<i32>
      %1294 = func.call @_where_437(%1293, %c_565, %1289#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_566 = stablehlo.constant dense<256> : tensor<i32>
      %1295 = func.call @_where_437(%1291, %c_566, %1294) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_567 = stablehlo.constant dense<0> : tensor<i32>
      %1296 = func.call @_where_437(%1293, %c_567, %1288#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_568 = stablehlo.constant dense<255> : tensor<i32>
      %1297 = func.call @_where_437(%1291, %c_568, %1296) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_569 = stablehlo.constant dense<0> : tensor<i32>
      %1298 = func.call @_where_437(%1293, %c_569, %1287#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_570 = stablehlo.constant dense<256> : tensor<i32>
      %1299 = func.call @_where_437(%1291, %c_570, %1298) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_571 = stablehlo.constant dense<1> : tensor<ui32>
      %c_572 = stablehlo.constant dense<1> : tensor<ui32>
      %1300 = stablehlo.partition_id : tensor<ui32>
      %1301 = stablehlo.divide %1300, %c_571 : tensor<ui32>
      %1302 = stablehlo.remainder %1301, %c_572 : tensor<ui32>
      %1303 = stablehlo.convert %1302 : (tensor<ui32>) -> tensor<i32>
      %c_573 = stablehlo.constant dense<257> : tensor<i32>
      %1304 = stablehlo.multiply %1303, %c_573 : tensor<i32>
      %1305 = stablehlo.broadcast_in_dim %1304, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1306 = stablehlo.subtract %1299, %1305 : tensor<200x2xi32>
      %c_574 = stablehlo.constant dense<0> : tensor<i32>
      %1307 = stablehlo.broadcast_in_dim %c_574, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1308 = stablehlo.compare  GE, %1306, %1307,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_575 = stablehlo.constant dense<257> : tensor<i32>
      %1309 = stablehlo.broadcast_in_dim %c_575, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1310 = stablehlo.compare  LT, %1306, %1309,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %1311 = stablehlo.and %1308, %1310 : tensor<200x2xi1>
      %c_576 = stablehlo.constant dense<0> : tensor<i32>
      %c_577 = stablehlo.constant dense<256> : tensor<i32>
      %1312 = func.call @clip_444(%1306, %c_576, %c_577) : (tensor<200x2xi32>, tensor<i32>, tensor<i32>) -> tensor<200x2xi32>
      %c_578 = stablehlo.constant dense<0> : tensor<i32>
      %1313 = stablehlo.broadcast_in_dim %c_578, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1314 = stablehlo.compare  LT, %1295, %1313,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_579 = stablehlo.constant dense<257> : tensor<i32>
      %1315 = stablehlo.broadcast_in_dim %c_579, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1316 = stablehlo.add %1295, %1315 : tensor<200x2xi32>
      %1317 = stablehlo.select %1314, %1316, %1295 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_580 = stablehlo.constant dense<0> : tensor<i32>
      %1318 = stablehlo.broadcast_in_dim %c_580, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1319 = stablehlo.compare  LT, %1297, %1318,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_581 = stablehlo.constant dense<256> : tensor<i32>
      %1320 = stablehlo.broadcast_in_dim %c_581, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1321 = stablehlo.add %1297, %1320 : tensor<200x2xi32>
      %1322 = stablehlo.select %1319, %1321, %1297 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_582 = stablehlo.constant dense<0> : tensor<i32>
      %1323 = stablehlo.broadcast_in_dim %c_582, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1324 = stablehlo.compare  LT, %1312, %1323,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_583 = stablehlo.constant dense<257> : tensor<i32>
      %1325 = stablehlo.broadcast_in_dim %c_583, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1326 = stablehlo.add %1312, %1325 : tensor<200x2xi32>
      %1327 = stablehlo.select %1324, %1326, %1312 : tensor<200x2xi1>, tensor<200x2xi32>
      %1328 = stablehlo.broadcast_in_dim %1317, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1329 = stablehlo.broadcast_in_dim %1322, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1330 = stablehlo.broadcast_in_dim %1327, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1331 = stablehlo.concatenate %1328, %1329, %1330, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
      %1332 = "stablehlo.gather"(%1020, %1331) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
      %c_584 = stablehlo.constant dense<0> : tensor<i32>
      %1333 = func.call @_where_455(%1311, %1332, %c_584) : (tensor<200x2xi1>, tensor<200x2xf32>, tensor<i32>) -> tensor<200x2xf32>
      %1334 = stablehlo.multiply %1333, %arg110 : tensor<200x2xf32>
      %cst_585 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1335 = stablehlo.reduce(%1334 init: %cst_585) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
      %c_586 = stablehlo.constant dense<257> : tensor<i32>
      %1336:2 = func.call @divmod_416(%arg111, %c_586) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_587 = stablehlo.constant dense<257> : tensor<i32>
      %1337:2 = func.call @divmod_430(%1336#0, %c_587) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_588 = stablehlo.constant dense<256> : tensor<i32>
      %1338:2 = func.call @divmod_430(%1337#0, %c_588) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_589 = stablehlo.constant dense<0> : tensor<i32>
      %1339 = stablehlo.broadcast_in_dim %c_589, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1340 = stablehlo.compare  GT, %1338#0, %1339,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_590 = stablehlo.constant dense<-1> : tensor<i32>
      %1341 = stablehlo.broadcast_in_dim %c_590, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1342 = stablehlo.compare  LT, %1338#0, %1341,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_591 = stablehlo.constant dense<0> : tensor<i32>
      %1343 = func.call @_where_437(%1342, %c_591, %1338#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_592 = stablehlo.constant dense<255> : tensor<i32>
      %1344 = func.call @_where_437(%1340, %c_592, %1343) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_593 = stablehlo.constant dense<0> : tensor<i32>
      %1345 = func.call @_where_437(%1342, %c_593, %1337#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_594 = stablehlo.constant dense<256> : tensor<i32>
      %1346 = func.call @_where_437(%1340, %c_594, %1345) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_595 = stablehlo.constant dense<0> : tensor<i32>
      %1347 = func.call @_where_437(%1342, %c_595, %1336#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_596 = stablehlo.constant dense<256> : tensor<i32>
      %1348 = func.call @_where_437(%1340, %c_596, %1347) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_597 = stablehlo.constant dense<1> : tensor<ui32>
      %c_598 = stablehlo.constant dense<1> : tensor<ui32>
      %1349 = stablehlo.partition_id : tensor<ui32>
      %1350 = stablehlo.divide %1349, %c_597 : tensor<ui32>
      %1351 = stablehlo.remainder %1350, %c_598 : tensor<ui32>
      %1352 = stablehlo.convert %1351 : (tensor<ui32>) -> tensor<i32>
      %c_599 = stablehlo.constant dense<257> : tensor<i32>
      %1353 = stablehlo.multiply %1352, %c_599 : tensor<i32>
      %1354 = stablehlo.broadcast_in_dim %1353, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1355 = stablehlo.subtract %1348, %1354 : tensor<200x2xi32>
      %c_600 = stablehlo.constant dense<0> : tensor<i32>
      %1356 = stablehlo.broadcast_in_dim %c_600, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1357 = stablehlo.compare  GE, %1355, %1356,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_601 = stablehlo.constant dense<257> : tensor<i32>
      %1358 = stablehlo.broadcast_in_dim %c_601, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1359 = stablehlo.compare  LT, %1355, %1358,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %1360 = stablehlo.and %1357, %1359 : tensor<200x2xi1>
      %c_602 = stablehlo.constant dense<0> : tensor<i32>
      %c_603 = stablehlo.constant dense<256> : tensor<i32>
      %1361 = func.call @clip_444(%1355, %c_602, %c_603) : (tensor<200x2xi32>, tensor<i32>, tensor<i32>) -> tensor<200x2xi32>
      %c_604 = stablehlo.constant dense<0> : tensor<i32>
      %1362 = stablehlo.broadcast_in_dim %c_604, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1363 = stablehlo.compare  LT, %1344, %1362,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_605 = stablehlo.constant dense<256> : tensor<i32>
      %1364 = stablehlo.broadcast_in_dim %c_605, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1365 = stablehlo.add %1344, %1364 : tensor<200x2xi32>
      %1366 = stablehlo.select %1363, %1365, %1344 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_606 = stablehlo.constant dense<0> : tensor<i32>
      %1367 = stablehlo.broadcast_in_dim %c_606, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1368 = stablehlo.compare  LT, %1346, %1367,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_607 = stablehlo.constant dense<257> : tensor<i32>
      %1369 = stablehlo.broadcast_in_dim %c_607, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1370 = stablehlo.add %1346, %1369 : tensor<200x2xi32>
      %1371 = stablehlo.select %1368, %1370, %1346 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_608 = stablehlo.constant dense<0> : tensor<i32>
      %1372 = stablehlo.broadcast_in_dim %c_608, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1373 = stablehlo.compare  LT, %1361, %1372,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_609 = stablehlo.constant dense<257> : tensor<i32>
      %1374 = stablehlo.broadcast_in_dim %c_609, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %1375 = stablehlo.add %1361, %1374 : tensor<200x2xi32>
      %1376 = stablehlo.select %1373, %1375, %1361 : tensor<200x2xi1>, tensor<200x2xi32>
      %1377 = stablehlo.broadcast_in_dim %1366, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1378 = stablehlo.broadcast_in_dim %1371, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1379 = stablehlo.broadcast_in_dim %1376, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %1380 = stablehlo.concatenate %1377, %1378, %1379, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
      %1381 = "stablehlo.gather"(%1118, %1380) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
      %c_610 = stablehlo.constant dense<0> : tensor<i32>
      %1382 = func.call @_where_455(%1360, %1381, %c_610) : (tensor<200x2xi1>, tensor<200x2xf32>, tensor<i32>) -> tensor<200x2xf32>
      %1383 = stablehlo.multiply %1382, %arg112 : tensor<200x2xf32>
      %cst_611 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1384 = stablehlo.reduce(%1383 init: %cst_611) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
      %c_612 = stablehlo.constant dense<257> : tensor<i32>
      %1385:2 = func.call @divmod_500(%arg113, %c_612) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_613 = stablehlo.constant dense<256> : tensor<i32>
      %1386:2 = func.call @divmod_514(%1385#0, %c_613) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_614 = stablehlo.constant dense<256> : tensor<i32>
      %1387:2 = func.call @divmod_514(%1386#0, %c_614) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_615 = stablehlo.constant dense<0> : tensor<i32>
      %1388 = stablehlo.broadcast_in_dim %c_615, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1389 = stablehlo.compare  GT, %1387#0, %1388,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_616 = stablehlo.constant dense<-1> : tensor<i32>
      %1390 = stablehlo.broadcast_in_dim %c_616, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1391 = stablehlo.compare  LT, %1387#0, %1390,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_617 = stablehlo.constant dense<0> : tensor<i32>
      %1392 = func.call @_where_521(%1391, %c_617, %1387#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_618 = stablehlo.constant dense<255> : tensor<i32>
      %1393 = func.call @_where_521(%1389, %c_618, %1392) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_619 = stablehlo.constant dense<0> : tensor<i32>
      %1394 = func.call @_where_521(%1391, %c_619, %1386#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_620 = stablehlo.constant dense<255> : tensor<i32>
      %1395 = func.call @_where_521(%1389, %c_620, %1394) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_621 = stablehlo.constant dense<0> : tensor<i32>
      %1396 = func.call @_where_521(%1391, %c_621, %1385#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_622 = stablehlo.constant dense<256> : tensor<i32>
      %1397 = func.call @_where_521(%1389, %c_622, %1396) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_623 = stablehlo.constant dense<1> : tensor<ui32>
      %c_624 = stablehlo.constant dense<1> : tensor<ui32>
      %1398 = stablehlo.partition_id : tensor<ui32>
      %1399 = stablehlo.divide %1398, %c_623 : tensor<ui32>
      %1400 = stablehlo.remainder %1399, %c_624 : tensor<ui32>
      %1401 = stablehlo.convert %1400 : (tensor<ui32>) -> tensor<i32>
      %c_625 = stablehlo.constant dense<257> : tensor<i32>
      %1402 = stablehlo.multiply %1401, %c_625 : tensor<i32>
      %1403 = stablehlo.broadcast_in_dim %1402, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1404 = stablehlo.subtract %1397, %1403 : tensor<200x1xi32>
      %c_626 = stablehlo.constant dense<0> : tensor<i32>
      %1405 = stablehlo.broadcast_in_dim %c_626, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1406 = stablehlo.compare  GE, %1404, %1405,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_627 = stablehlo.constant dense<257> : tensor<i32>
      %1407 = stablehlo.broadcast_in_dim %c_627, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1408 = stablehlo.compare  LT, %1404, %1407,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %1409 = stablehlo.and %1406, %1408 : tensor<200x1xi1>
      %c_628 = stablehlo.constant dense<0> : tensor<i32>
      %c_629 = stablehlo.constant dense<256> : tensor<i32>
      %1410 = func.call @clip_528(%1404, %c_628, %c_629) : (tensor<200x1xi32>, tensor<i32>, tensor<i32>) -> tensor<200x1xi32>
      %c_630 = stablehlo.constant dense<0> : tensor<i32>
      %1411 = stablehlo.broadcast_in_dim %c_630, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1412 = stablehlo.compare  LT, %1393, %1411,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_631 = stablehlo.constant dense<256> : tensor<i32>
      %1413 = stablehlo.broadcast_in_dim %c_631, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1414 = stablehlo.add %1393, %1413 : tensor<200x1xi32>
      %1415 = stablehlo.select %1412, %1414, %1393 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_632 = stablehlo.constant dense<0> : tensor<i32>
      %1416 = stablehlo.broadcast_in_dim %c_632, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1417 = stablehlo.compare  LT, %1395, %1416,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_633 = stablehlo.constant dense<256> : tensor<i32>
      %1418 = stablehlo.broadcast_in_dim %c_633, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1419 = stablehlo.add %1395, %1418 : tensor<200x1xi32>
      %1420 = stablehlo.select %1417, %1419, %1395 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_634 = stablehlo.constant dense<0> : tensor<i32>
      %1421 = stablehlo.broadcast_in_dim %c_634, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1422 = stablehlo.compare  LT, %1410, %1421,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_635 = stablehlo.constant dense<257> : tensor<i32>
      %1423 = stablehlo.broadcast_in_dim %c_635, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %1424 = stablehlo.add %1410, %1423 : tensor<200x1xi32>
      %1425 = stablehlo.select %1422, %1424, %1410 : tensor<200x1xi1>, tensor<200x1xi32>
      %1426 = stablehlo.broadcast_in_dim %1415, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1427 = stablehlo.broadcast_in_dim %1420, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1428 = stablehlo.broadcast_in_dim %1425, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %1429 = stablehlo.concatenate %1426, %1427, %1428, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %1430 = "stablehlo.gather"(%353, %1429) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %c_636 = stablehlo.constant dense<0> : tensor<i32>
      %1431 = func.call @_where_539(%1409, %1430, %c_636) : (tensor<200x1xi1>, tensor<200x1xf32>, tensor<i32>) -> tensor<200x1xf32>
      %1432 = stablehlo.multiply %1431, %arg114 : tensor<200x1xf32>
      %cst_637 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1433 = stablehlo.reduce(%1432 init: %cst_637) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_638 = stablehlo.constant dense<257> : tensor<i32>
      %1434:2 = func.call @divmod_369(%arg115, %c_638) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_639 = stablehlo.constant dense<257> : tensor<i32>
      %1435:2 = func.call @divmod_383(%1434#0, %c_639) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_640 = stablehlo.constant dense<256> : tensor<i32>
      %1436:2 = func.call @divmod_383(%1435#0, %c_640) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_641 = stablehlo.constant dense<0> : tensor<i32>
      %1437 = stablehlo.broadcast_in_dim %c_641, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1438 = stablehlo.compare  GT, %1436#0, %1437,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_642 = stablehlo.constant dense<-1> : tensor<i32>
      %1439 = stablehlo.broadcast_in_dim %c_642, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1440 = stablehlo.compare  LT, %1436#0, %1439,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_643 = stablehlo.constant dense<0> : tensor<i32>
      %1441 = func.call @_where_390(%1440, %c_643, %1436#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_644 = stablehlo.constant dense<255> : tensor<i32>
      %1442 = func.call @_where_390(%1438, %c_644, %1441) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_645 = stablehlo.constant dense<0> : tensor<i32>
      %1443 = func.call @_where_390(%1440, %c_645, %1435#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_646 = stablehlo.constant dense<256> : tensor<i32>
      %1444 = func.call @_where_390(%1438, %c_646, %1443) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_647 = stablehlo.constant dense<0> : tensor<i32>
      %1445 = func.call @_where_390(%1440, %c_647, %1434#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_648 = stablehlo.constant dense<256> : tensor<i32>
      %1446 = func.call @_where_390(%1438, %c_648, %1445) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_649 = stablehlo.constant dense<1> : tensor<ui32>
      %c_650 = stablehlo.constant dense<1> : tensor<ui32>
      %1447 = stablehlo.partition_id : tensor<ui32>
      %1448 = stablehlo.divide %1447, %c_649 : tensor<ui32>
      %1449 = stablehlo.remainder %1448, %c_650 : tensor<ui32>
      %1450 = stablehlo.convert %1449 : (tensor<ui32>) -> tensor<i32>
      %c_651 = stablehlo.constant dense<257> : tensor<i32>
      %1451 = stablehlo.multiply %1450, %c_651 : tensor<i32>
      %1452 = stablehlo.broadcast_in_dim %1451, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1453 = stablehlo.subtract %1446, %1452 : tensor<200x4xi32>
      %c_652 = stablehlo.constant dense<0> : tensor<i32>
      %1454 = stablehlo.broadcast_in_dim %c_652, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1455 = stablehlo.compare  GE, %1453, %1454,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_653 = stablehlo.constant dense<257> : tensor<i32>
      %1456 = stablehlo.broadcast_in_dim %c_653, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1457 = stablehlo.compare  LT, %1453, %1456,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1458 = stablehlo.and %1455, %1457 : tensor<200x4xi1>
      %c_654 = stablehlo.constant dense<0> : tensor<i32>
      %c_655 = stablehlo.constant dense<256> : tensor<i32>
      %1459 = func.call @clip_397(%1453, %c_654, %c_655) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_656 = stablehlo.constant dense<0> : tensor<i32>
      %1460 = stablehlo.broadcast_in_dim %c_656, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1461 = stablehlo.compare  LT, %1442, %1460,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_657 = stablehlo.constant dense<256> : tensor<i32>
      %1462 = stablehlo.broadcast_in_dim %c_657, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1463 = stablehlo.add %1442, %1462 : tensor<200x4xi32>
      %1464 = stablehlo.select %1461, %1463, %1442 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_658 = stablehlo.constant dense<0> : tensor<i32>
      %1465 = stablehlo.broadcast_in_dim %c_658, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1466 = stablehlo.compare  LT, %1444, %1465,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_659 = stablehlo.constant dense<257> : tensor<i32>
      %1467 = stablehlo.broadcast_in_dim %c_659, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1468 = stablehlo.add %1444, %1467 : tensor<200x4xi32>
      %1469 = stablehlo.select %1466, %1468, %1444 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_660 = stablehlo.constant dense<0> : tensor<i32>
      %1470 = stablehlo.broadcast_in_dim %c_660, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1471 = stablehlo.compare  LT, %1459, %1470,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_661 = stablehlo.constant dense<257> : tensor<i32>
      %1472 = stablehlo.broadcast_in_dim %c_661, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1473 = stablehlo.add %1459, %1472 : tensor<200x4xi32>
      %1474 = stablehlo.select %1471, %1473, %1459 : tensor<200x4xi1>, tensor<200x4xi32>
      %1475 = stablehlo.broadcast_in_dim %1464, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1476 = stablehlo.broadcast_in_dim %1469, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1477 = stablehlo.broadcast_in_dim %1474, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1478 = stablehlo.concatenate %1475, %1476, %1477, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1479 = "stablehlo.gather"(%461, %1478) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_662 = stablehlo.constant dense<0> : tensor<i32>
      %1480 = func.call @_where_408(%1458, %1479, %c_662) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1481 = stablehlo.multiply %1480, %arg116 : tensor<200x4xf32>
      %cst_663 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1482 = stablehlo.reduce(%1481 init: %cst_663) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_664 = stablehlo.constant dense<257> : tensor<i32>
      %1483:2 = func.call @divmod_369(%arg117, %c_664) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_665 = stablehlo.constant dense<256> : tensor<i32>
      %1484:2 = func.call @divmod_383(%1483#0, %c_665) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_666 = stablehlo.constant dense<257> : tensor<i32>
      %1485:2 = func.call @divmod_383(%1484#0, %c_666) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_667 = stablehlo.constant dense<0> : tensor<i32>
      %1486 = stablehlo.broadcast_in_dim %c_667, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1487 = stablehlo.compare  GT, %1485#0, %1486,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_668 = stablehlo.constant dense<-1> : tensor<i32>
      %1488 = stablehlo.broadcast_in_dim %c_668, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1489 = stablehlo.compare  LT, %1485#0, %1488,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_669 = stablehlo.constant dense<0> : tensor<i32>
      %1490 = func.call @_where_390(%1489, %c_669, %1485#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_670 = stablehlo.constant dense<256> : tensor<i32>
      %1491 = func.call @_where_390(%1487, %c_670, %1490) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_671 = stablehlo.constant dense<0> : tensor<i32>
      %1492 = func.call @_where_390(%1489, %c_671, %1484#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_672 = stablehlo.constant dense<255> : tensor<i32>
      %1493 = func.call @_where_390(%1487, %c_672, %1492) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_673 = stablehlo.constant dense<0> : tensor<i32>
      %1494 = func.call @_where_390(%1489, %c_673, %1483#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_674 = stablehlo.constant dense<256> : tensor<i32>
      %1495 = func.call @_where_390(%1487, %c_674, %1494) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_675 = stablehlo.constant dense<1> : tensor<ui32>
      %c_676 = stablehlo.constant dense<1> : tensor<ui32>
      %1496 = stablehlo.partition_id : tensor<ui32>
      %1497 = stablehlo.divide %1496, %c_675 : tensor<ui32>
      %1498 = stablehlo.remainder %1497, %c_676 : tensor<ui32>
      %1499 = stablehlo.convert %1498 : (tensor<ui32>) -> tensor<i32>
      %c_677 = stablehlo.constant dense<257> : tensor<i32>
      %1500 = stablehlo.multiply %1499, %c_677 : tensor<i32>
      %1501 = stablehlo.broadcast_in_dim %1500, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1502 = stablehlo.subtract %1495, %1501 : tensor<200x4xi32>
      %c_678 = stablehlo.constant dense<0> : tensor<i32>
      %1503 = stablehlo.broadcast_in_dim %c_678, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1504 = stablehlo.compare  GE, %1502, %1503,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_679 = stablehlo.constant dense<257> : tensor<i32>
      %1505 = stablehlo.broadcast_in_dim %c_679, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1506 = stablehlo.compare  LT, %1502, %1505,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %1507 = stablehlo.and %1504, %1506 : tensor<200x4xi1>
      %c_680 = stablehlo.constant dense<0> : tensor<i32>
      %c_681 = stablehlo.constant dense<256> : tensor<i32>
      %1508 = func.call @clip_397(%1502, %c_680, %c_681) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
      %c_682 = stablehlo.constant dense<0> : tensor<i32>
      %1509 = stablehlo.broadcast_in_dim %c_682, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1510 = stablehlo.compare  LT, %1491, %1509,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_683 = stablehlo.constant dense<257> : tensor<i32>
      %1511 = stablehlo.broadcast_in_dim %c_683, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1512 = stablehlo.add %1491, %1511 : tensor<200x4xi32>
      %1513 = stablehlo.select %1510, %1512, %1491 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_684 = stablehlo.constant dense<0> : tensor<i32>
      %1514 = stablehlo.broadcast_in_dim %c_684, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1515 = stablehlo.compare  LT, %1493, %1514,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_685 = stablehlo.constant dense<256> : tensor<i32>
      %1516 = stablehlo.broadcast_in_dim %c_685, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1517 = stablehlo.add %1493, %1516 : tensor<200x4xi32>
      %1518 = stablehlo.select %1515, %1517, %1493 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_686 = stablehlo.constant dense<0> : tensor<i32>
      %1519 = stablehlo.broadcast_in_dim %c_686, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1520 = stablehlo.compare  LT, %1508, %1519,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_687 = stablehlo.constant dense<257> : tensor<i32>
      %1521 = stablehlo.broadcast_in_dim %c_687, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %1522 = stablehlo.add %1508, %1521 : tensor<200x4xi32>
      %1523 = stablehlo.select %1520, %1522, %1508 : tensor<200x4xi1>, tensor<200x4xi32>
      %1524 = stablehlo.broadcast_in_dim %1513, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1525 = stablehlo.broadcast_in_dim %1518, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1526 = stablehlo.broadcast_in_dim %1523, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %1527 = stablehlo.concatenate %1524, %1525, %1526, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %1528 = "stablehlo.gather"(%559, %1527) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %c_688 = stablehlo.constant dense<0> : tensor<i32>
      %1529 = func.call @_where_408(%1507, %1528, %c_688) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
      %1530 = stablehlo.multiply %1529, %arg118 : tensor<200x4xf32>
      %cst_689 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1531 = stablehlo.reduce(%1530 init: %cst_689) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %1532 = stablehlo.slice %1177#0 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
      %1533 = stablehlo.reshape %1532 : (tensor<1x7200xf32>) -> tensor<7200xf32>
      %1534 = stablehlo.slice %1177#1 [0:1, 0:7200] : (tensor<1x7200xf32>) -> tensor<1x7200xf32>
      %1535 = stablehlo.reshape %1534 : (tensor<1x7200xf32>) -> tensor<7200xf32>
      %1536 = stablehlo.broadcast_in_dim %1286, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1537 = stablehlo.broadcast_in_dim %1335, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1538 = stablehlo.broadcast_in_dim %1384, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1539 = stablehlo.broadcast_in_dim %1433, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1540 = stablehlo.broadcast_in_dim %1482, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1541 = stablehlo.broadcast_in_dim %1531, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %1542 = stablehlo.concatenate %1536, %1537, %1538, %1539, %1540, %1541, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %cst_690 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1543 = stablehlo.broadcast_in_dim %cst_690, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %1544 = stablehlo.multiply %1543, %arg119 : tensor<3xf32>
      %1545 = stablehlo.optimization_barrier %1544 : tensor<3xf32>
      %1546 = stablehlo.optimization_barrier %3 : tensor<f32>
      %1547 = stablehlo.broadcast_in_dim %1546, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %1548 = stablehlo.multiply %1545, %1547 : tensor<3xf32>
      %1549 = stablehlo.cosine %1548 : tensor<3xf32>
      %1550 = stablehlo.sine %1548 : tensor<3xf32>
      %cst_691 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_692 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1551 = stablehlo.subtract %cst_691, %cst_692 : tensor<f32>
      %cst_693 = stablehlo.constant dense<1.000000e-30> : tensor<f32>
      %1552 = stablehlo.maximum %1551, %cst_693 : tensor<f32>
      %cst_694 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1553 = stablehlo.subtract %3, %cst_694 : tensor<f32>
      %1554 = stablehlo.divide %1553, %1552 : tensor<f32>
      %cst_695 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %cst_696 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1555 = func.call @clip_472(%1554, %cst_695, %cst_696) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_697 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1556 = stablehlo.multiply %cst_697, %1555 : tensor<f32>
      %1557 = stablehlo.cosine %1556 : tensor<f32>
      %cst_698 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1558 = stablehlo.subtract %cst_698, %1557 : tensor<f32>
      %cst_699 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %1559 = stablehlo.multiply %cst_699, %1558 : tensor<f32>
      %cst_700 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %1560 = stablehlo.is_finite %cst_700 : (tensor<f32>) -> tensor<i1>
      %c_701 = stablehlo.constant dense<false> : tensor<i1>
      %1561 = stablehlo.and %c_701, %1560 : tensor<i1>
      %cst_702 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_703 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1562 = stablehlo.compare  GT, %cst_702, %cst_703,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %1563 = stablehlo.and %1561, %1562 : tensor<i1>
      %cst_704 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1564 = func.call @_where_479(%1563, %1559, %cst_704) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_705 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1565 = stablehlo.maximum %1564, %cst_705 : tensor<f32>
      %cst_706 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1566 = stablehlo.multiply %arg0, %cst_706 : tensor<f32>
      %cst_707 = stablehlo.constant dense<0x4D8EF3C2> : tensor<f32>
      %1567 = stablehlo.multiply %1566, %cst_707 : tensor<f32>
      %cst_708 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %1568 = stablehlo.divide %1567, %cst_708 : tensor<f32>
      %cst_709 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %1569 = stablehlo.sqrt %cst_709 : tensor<f32>
      %1570 = stablehlo.divide %1568, %1569 : tensor<f32>
      %1571 = stablehlo.multiply %1565, %1570 : tensor<f32>
      %c_710 = stablehlo.constant dense<0> : tensor<i32>
      %c_711 = stablehlo.constant dense<1> : tensor<i32>
      %1572 = stablehlo.compare  EQ, %c_710, %c_711,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %1573 = func.call @_where_482(%1572, %1571, %1565) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %1574 = stablehlo.broadcast_in_dim %arg120, dims = [0] : (tensor<6xf32>) -> tensor<6x1x1xf32>
      %1575 = stablehlo.broadcast_in_dim %1573, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %1576 = stablehlo.multiply %1575, %1574 : tensor<6x1x1xf32>
      %1577 = stablehlo.dot_general %1542, %1549, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %1578 = stablehlo.transpose %1577, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %1579 = stablehlo.broadcast_in_dim %1576, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %1580 = stablehlo.multiply %1579, %1578 : tensor<6x3x200xf32>
      %1581 = stablehlo.broadcast_in_dim %1573, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %1582 = stablehlo.multiply %1581, %1574 : tensor<6x1x1xf32>
      %1583 = stablehlo.dot_general %1542, %1550, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %1584 = stablehlo.transpose %1583, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %1585 = stablehlo.broadcast_in_dim %1582, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %1586 = stablehlo.multiply %1585, %1584 : tensor<6x3x200xf32>
      %1587 = stablehlo.reshape %1580 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_712 = stablehlo.constant dense<3600> : tensor<i32>
      %1588 = stablehlo.broadcast_in_dim %c_712, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1589 = "stablehlo.scatter"(%1533, %1588, %1587) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg148: tensor<f32>, %arg149: tensor<f32>):
        %1598 = stablehlo.add %arg148, %arg149 : tensor<f32>
        stablehlo.return %1598 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %1590 = stablehlo.reshape %1586 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_713 = stablehlo.constant dense<3600> : tensor<i32>
      %1591 = stablehlo.broadcast_in_dim %c_713, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1592 = "stablehlo.scatter"(%1535, %1591, %1590) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg148: tensor<f32>, %arg149: tensor<f32>):
        %1598 = stablehlo.add %arg148, %arg149 : tensor<f32>
        stablehlo.return %1598 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %1593 = stablehlo.broadcast_in_dim %1564, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %c_714 = stablehlo.constant dense<3> : tensor<i32>
      %1594 = stablehlo.broadcast_in_dim %c_714, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %1595 = "stablehlo.scatter"(%1177#2, %1594, %1593) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg148: tensor<f32>, %arg149: tensor<f32>):
        %1598 = stablehlo.add %arg148, %arg149 : tensor<f32>
        stablehlo.return %1598 : tensor<f32>
      }) : (tensor<6xf32>, tensor<1xi32>, tensor<3xf32>) -> tensor<6xf32>
      %1596 = stablehlo.broadcast_in_dim %1589, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
      %1597 = stablehlo.broadcast_in_dim %1592, dims = [1] : (tensor<7200xf32>) -> tensor<1x7200xf32>
      stablehlo.return %1596, %1597, %1595 : tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>
    }) : (tensor<i32>) -> (tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>)
    %c_533 = stablehlo.constant dense<1> : tensor<i32>
    %1237 = stablehlo.add %arg146, %c_533 : tensor<i32>
    return %912, %1020, %1118, %353, %461, %559, %356, %357, %358, %362, %363, %361, %915, %916, %917, %921, %922, %920, %1211, %1219, %1226, %1236#0, %1236#1, %1236#2, %3, %1237 : tensor<257x257x257xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x257xf32>, tensor<257x256x257xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x257xf32>, tensor<1x256x257x24xf32>, tensor<1x257x256x24xf32>, tensor<257x24x257xf32>, tensor<257x24x257xf32>, tensor<24x257x257xf32>, tensor<24x256x257xf32>, tensor<1x257x256x24xf32>, tensor<1x256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<1x7200xf32>, tensor<1x7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
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
  func.func private @clip_170(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
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
  func.func private @_where_191(%arg0: tensor<1x1x1xi1>, %arg1: tensor<14x25x1xf32>, %arg2: tensor<i32>) -> tensor<14x25x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x1x1xi1>) -> tensor<14x25x1xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<14x25x1xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<14x25x1xi1>, tensor<14x25x1xf32>
    return %3 : tensor<14x25x1xf32>
  }
  func.func private @_take_201(%arg0: tensor<15x24x1xf32>, %arg1: tensor<1xi32>) -> tensor<15x24x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 15, 24, 1>}> : (tensor<15x24x1xf32>, tensor<1x1xi32>) -> tensor<15x24x1xf32>
    return %1 : tensor<15x24x1xf32>
  }
  func.func private @_where_205(%arg0: tensor<1x1x1xi1>, %arg1: tensor<15x24x1xf32>, %arg2: tensor<i32>) -> tensor<15x24x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x1x1xi1>) -> tensor<15x24x1xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<15x24x1xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<15x24x1xi1>, tensor<15x24x1xf32>
    return %3 : tensor<15x24x1xf32>
  }
  func.func private @_take_225(%arg0: tensor<15x26x1xf32>, %arg1: tensor<1xi32>) -> tensor<15x26x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 15, 26, 1>}> : (tensor<15x26x1xf32>, tensor<1x1xi32>) -> tensor<15x26x1xf32>
    return %1 : tensor<15x26x1xf32>
  }
  func.func private @_where_229(%arg0: tensor<1x1x1xi1>, %arg1: tensor<15x26x1xf32>, %arg2: tensor<i32>) -> tensor<15x26x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x1x1xi1>) -> tensor<15x26x1xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<15x26x1xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<15x26x1xi1>, tensor<15x26x1xf32>
    return %3 : tensor<15x26x1xf32>
  }
  func.func private @_take_240(%arg0: tensor<16x25x1xf32>, %arg1: tensor<1xi32>) -> tensor<16x25x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 25, 1>}> : (tensor<16x25x1xf32>, tensor<1x1xi32>) -> tensor<16x25x1xf32>
    return %1 : tensor<16x25x1xf32>
  }
  func.func private @_where_244(%arg0: tensor<1x1x1xi1>, %arg1: tensor<16x25x1xf32>, %arg2: tensor<i32>) -> tensor<16x25x1xf32> {
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
    %1 = call @_where_255(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_255(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @divmod(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    %2 = call @remainder_275(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
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
    %13 = call @_where_273(%10, %12, %1) : (tensor<200x1xi1>, tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi32>
    return %13 : tensor<200x1xi32>
  }
  func.func private @_where_273(%arg0: tensor<200x1xi1>, %arg1: tensor<200x1xi32>, %arg2: tensor<200x1xi32>) -> tensor<200x1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x1xi1>, tensor<200x1xi32>
    return %0 : tensor<200x1xi32>
  }
  func.func private @remainder_275(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_255(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_279(%arg0: tensor<200x1xi1>, %arg1: tensor<i32>, %arg2: tensor<200x1xi32>) -> tensor<200x1xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x1xi1>, tensor<200x1xi32>
    return %2 : tensor<200x1xi32>
  }
  func.func private @_where_299(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_304(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_310(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @divmod_319(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_320(%arg0, %0) : (tensor<200x8xi32>, tensor<i32>) -> tensor<200x8xi32>
    %2 = call @remainder_330(%arg0, %0) : (tensor<200x8xi32>, tensor<i32>) -> tensor<200x8xi32>
    return %1, %2 : tensor<200x8xi32>, tensor<200x8xi32>
  }
  func.func private @floor_divide_320(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> tensor<200x8xi32> {
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
    %13 = call @_where_328(%10, %12, %1) : (tensor<200x8xi1>, tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi32>
    return %13 : tensor<200x8xi32>
  }
  func.func private @_where_328(%arg0: tensor<200x8xi1>, %arg1: tensor<200x8xi32>, %arg2: tensor<200x8xi32>) -> tensor<200x8xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x8xi1>, tensor<200x8xi32>
    return %0 : tensor<200x8xi32>
  }
  func.func private @remainder_330(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> tensor<200x8xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_332(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_332(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @divmod_336(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_337(%arg0, %0) : (tensor<200x8xi32>, tensor<i32>) -> tensor<200x8xi32>
    %2 = call @remainder_341(%arg0, %0) : (tensor<200x8xi32>, tensor<i32>) -> tensor<200x8xi32>
    return %1, %2 : tensor<200x8xi32>, tensor<200x8xi32>
  }
  func.func private @floor_divide_337(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> tensor<200x8xi32> {
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
    %13 = call @_where_328(%10, %12, %1) : (tensor<200x8xi1>, tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi32>
    return %13 : tensor<200x8xi32>
  }
  func.func private @remainder_341(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> tensor<200x8xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_332(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_343(%arg0: tensor<200x8xi1>, %arg1: tensor<i32>, %arg2: tensor<200x8xi32>) -> tensor<200x8xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x8xi1>, tensor<200x8xi32>
    return %2 : tensor<200x8xi32>
  }
  func.func private @clip_350(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x8xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x8xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x8xi32>
    return %5 : tensor<200x8xi32>
  }
  func.func private @_where_362(%arg0: tensor<200x8xi1>, %arg1: tensor<200x8xf32>, %arg2: tensor<i32>) -> tensor<200x8xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x8xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x8xi1>, tensor<200x8xf32>
    return %2 : tensor<200x8xf32>
  }
  func.func private @divmod_369(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_370(%arg0, %0) : (tensor<200x4xi32>, tensor<i32>) -> tensor<200x4xi32>
    %2 = call @remainder_379(%arg0, %0) : (tensor<200x4xi32>, tensor<i32>) -> tensor<200x4xi32>
    return %1, %2 : tensor<200x4xi32>, tensor<200x4xi32>
  }
  func.func private @floor_divide_370(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> tensor<200x4xi32> {
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
    %13 = call @_where_377(%10, %12, %1) : (tensor<200x4xi1>, tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi32>
    return %13 : tensor<200x4xi32>
  }
  func.func private @_where_377(%arg0: tensor<200x4xi1>, %arg1: tensor<200x4xi32>, %arg2: tensor<200x4xi32>) -> tensor<200x4xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x4xi1>, tensor<200x4xi32>
    return %0 : tensor<200x4xi32>
  }
  func.func private @remainder_379(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> tensor<200x4xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_332(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @divmod_383(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_384(%arg0, %0) : (tensor<200x4xi32>, tensor<i32>) -> tensor<200x4xi32>
    %2 = call @remainder_388(%arg0, %0) : (tensor<200x4xi32>, tensor<i32>) -> tensor<200x4xi32>
    return %1, %2 : tensor<200x4xi32>, tensor<200x4xi32>
  }
  func.func private @floor_divide_384(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> tensor<200x4xi32> {
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
    %13 = call @_where_377(%10, %12, %1) : (tensor<200x4xi1>, tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi32>
    return %13 : tensor<200x4xi32>
  }
  func.func private @remainder_388(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> tensor<200x4xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_332(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_390(%arg0: tensor<200x4xi1>, %arg1: tensor<i32>, %arg2: tensor<200x4xi32>) -> tensor<200x4xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x4xi1>, tensor<200x4xi32>
    return %2 : tensor<200x4xi32>
  }
  func.func private @clip_397(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x4xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x4xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x4xi32>
    return %5 : tensor<200x4xi32>
  }
  func.func private @_where_408(%arg0: tensor<200x4xi1>, %arg1: tensor<200x4xf32>, %arg2: tensor<i32>) -> tensor<200x4xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x4xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x4xi1>, tensor<200x4xf32>
    return %2 : tensor<200x4xf32>
  }
  func.func private @divmod_416(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_417(%arg0, %0) : (tensor<200x2xi32>, tensor<i32>) -> tensor<200x2xi32>
    %2 = call @remainder_426(%arg0, %0) : (tensor<200x2xi32>, tensor<i32>) -> tensor<200x2xi32>
    return %1, %2 : tensor<200x2xi32>, tensor<200x2xi32>
  }
  func.func private @floor_divide_417(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> tensor<200x2xi32> {
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
    %13 = call @_where_424(%10, %12, %1) : (tensor<200x2xi1>, tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi32>
    return %13 : tensor<200x2xi32>
  }
  func.func private @_where_424(%arg0: tensor<200x2xi1>, %arg1: tensor<200x2xi32>, %arg2: tensor<200x2xi32>) -> tensor<200x2xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x2xi1>, tensor<200x2xi32>
    return %0 : tensor<200x2xi32>
  }
  func.func private @remainder_426(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> tensor<200x2xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_332(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @divmod_430(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_431(%arg0, %0) : (tensor<200x2xi32>, tensor<i32>) -> tensor<200x2xi32>
    %2 = call @remainder_435(%arg0, %0) : (tensor<200x2xi32>, tensor<i32>) -> tensor<200x2xi32>
    return %1, %2 : tensor<200x2xi32>, tensor<200x2xi32>
  }
  func.func private @floor_divide_431(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> tensor<200x2xi32> {
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
    %13 = call @_where_424(%10, %12, %1) : (tensor<200x2xi1>, tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi32>
    return %13 : tensor<200x2xi32>
  }
  func.func private @remainder_435(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> tensor<200x2xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_332(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_437(%arg0: tensor<200x2xi1>, %arg1: tensor<i32>, %arg2: tensor<200x2xi32>) -> tensor<200x2xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x2xi1>, tensor<200x2xi32>
    return %2 : tensor<200x2xi32>
  }
  func.func private @clip_444(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x2xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x2xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x2xi32>
    return %5 : tensor<200x2xi32>
  }
  func.func private @_where_455(%arg0: tensor<200x2xi1>, %arg1: tensor<200x2xf32>, %arg2: tensor<i32>) -> tensor<200x2xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x2xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x2xi1>, tensor<200x2xf32>
    return %2 : tensor<200x2xf32>
  }
  func.func private @clip_472(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg1 : tensor<f32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<f32>
    %2 = stablehlo.convert %arg2 : tensor<f32>
    %3 = stablehlo.minimum %2, %1 : tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @_where_479(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @_where_482(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @divmod_500(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_501(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    %2 = call @remainder_510(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    return %1, %2 : tensor<200x1xi32>, tensor<200x1xi32>
  }
  func.func private @floor_divide_501(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
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
    %13 = call @_where_508(%10, %12, %1) : (tensor<200x1xi1>, tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi32>
    return %13 : tensor<200x1xi32>
  }
  func.func private @_where_508(%arg0: tensor<200x1xi1>, %arg1: tensor<200x1xi32>, %arg2: tensor<200x1xi32>) -> tensor<200x1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x1xi1>, tensor<200x1xi32>
    return %0 : tensor<200x1xi32>
  }
  func.func private @remainder_510(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_332(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @divmod_514(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_515(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    %2 = call @remainder_519(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    return %1, %2 : tensor<200x1xi32>, tensor<200x1xi32>
  }
  func.func private @floor_divide_515(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
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
    %13 = call @_where_508(%10, %12, %1) : (tensor<200x1xi1>, tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi32>
    return %13 : tensor<200x1xi32>
  }
  func.func private @remainder_519(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_332(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_521(%arg0: tensor<200x1xi1>, %arg1: tensor<i32>, %arg2: tensor<200x1xi32>) -> tensor<200x1xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x1xi1>, tensor<200x1xi32>
    return %2 : tensor<200x1xi32>
  }
  func.func private @clip_528(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x1xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x1xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x1xi32>
    return %5 : tensor<200x1xi32>
  }
  func.func private @_where_539(%arg0: tensor<200x1xi1>, %arg1: tensor<200x1xf32>, %arg2: tensor<i32>) -> tensor<200x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x1xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x1xi1>, tensor<200x1xf32>
    return %2 : tensor<200x1xf32>
  }
}
