module @jit_run_scan attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<257x257x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 0 : i32}, %arg1: tensor<257x256x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 1 : i32}, %arg2: tensor<256x257x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 2 : i32}, %arg3: tensor<256x256x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 3 : i32}, %arg4: tensor<256x257x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 4 : i32}, %arg5: tensor<257x256x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 5 : i32}, %arg6: tensor<256x24x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 6 : i32}, %arg7: tensor<24x256x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 7 : i32}, %arg8: tensor<24x257x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 8 : i32}, %arg9: tensor<256x257x24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 9 : i32}, %arg10: tensor<257x256x24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 10 : i32}, %arg11: tensor<257x24x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 11 : i32}, %arg12: tensor<257x24x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 12 : i32}, %arg13: tensor<24x257x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 13 : i32}, %arg14: tensor<24x256x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 14 : i32}, %arg15: tensor<257x256x24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 15 : i32}, %arg16: tensor<256x257x24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 16 : i32}, %arg17: tensor<256x24x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>, tf.aliasing_output = 17 : i32}, %arg18: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 18 : i32}, %arg19: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 19 : i32}, %arg20: tensor<2xi32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 20 : i32}, %arg21: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 21 : i32}, %arg22: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 22 : i32}, %arg23: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 23 : i32}, %arg24: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 24 : i32}, %arg25: tensor<7200xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 25 : i32}, %arg26: tensor<7200xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 26 : i32}, %arg27: tensor<6xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 27 : i32}, %arg28: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>, tf.aliasing_output = 28 : i32}, %arg29: tensor<i32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>, tf.aliasing_output = 29 : i32}, %arg30: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg31: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg32: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg33: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg34: tensor<257x257x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg35: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg36: tensor<257x256x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg37: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg38: tensor<256x257x257xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}) -> (tensor<257x257x256xf32> {jax.result_info = "result.ex"}, tensor<257x256x257xf32> {jax.result_info = "result.ey"}, tensor<256x257x257xf32> {jax.result_info = "result.ez"}, tensor<256x256x257xf32> {jax.result_info = "result.hx"}, tensor<256x257x256xf32> {jax.result_info = "result.hy"}, tensor<257x256x256xf32> {jax.result_info = "result.hz"}, tensor<256x24x257xf32> {jax.result_info = "result.cpml_psi_h_terms[0]"}, tensor<24x256x257xf32> {jax.result_info = "result.cpml_psi_h_terms[1]"}, tensor<24x257x256xf32> {jax.result_info = "result.cpml_psi_h_terms[2]"}, tensor<256x257x24xf32> {jax.result_info = "result.cpml_psi_h_terms[3]"}, tensor<257x256x24xf32> {jax.result_info = "result.cpml_psi_h_terms[4]"}, tensor<257x24x256xf32> {jax.result_info = "result.cpml_psi_h_terms[5]"}, tensor<257x24x256xf32> {jax.result_info = "result.cpml_psi_e_terms[0]"}, tensor<24x257x256xf32> {jax.result_info = "result.cpml_psi_e_terms[1]"}, tensor<24x256x257xf32> {jax.result_info = "result.cpml_psi_e_terms[2]"}, tensor<257x256x24xf32> {jax.result_info = "result.cpml_psi_e_terms[3]"}, tensor<256x257x24xf32> {jax.result_info = "result.cpml_psi_e_terms[4]"}, tensor<256x24x257xf32> {jax.result_info = "result.cpml_psi_e_terms[5]"}, tensor<2x1xf32> {jax.result_info = "result.powers"}, tensor<2x1xf32> {jax.result_info = "result.timestamps"}, tensor<2xi32> {jax.result_info = "result.counts"}, tensor<2x3xf32> {jax.result_info = "result.freq_flux_re"}, tensor<2x3xf32> {jax.result_info = "result.freq_flux_im"}, tensor<2x3xf32> {jax.result_info = "result.freq_phase_re"}, tensor<2x3xf32> {jax.result_info = "result.freq_phase_im"}, tensor<7200xf32> {jax.result_info = "result.dft_vec_re"}, tensor<7200xf32> {jax.result_info = "result.dft_vec_im"}, tensor<6xf32> {jax.result_info = "result.dft_weight_sum"}, tensor<f32> {jax.result_info = "result.t"}, tensor<i32> {jax.result_info = "result.current_step"}) {
    %cst = stablehlo.constant dense<1.46363323E-16> : tensor<f32>
    %cst_0 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_2 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_3 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_4 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_5 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_6 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_7 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_8 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_9 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_10 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_11 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_12 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_13 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_14 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_15 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_16 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_17 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_18 = stablehlo.constant dense_resource<__elided__> : tensor<2x14x25x1xf32>
    %cst_19 = stablehlo.constant dense_resource<__elided__> : tensor<2x128xf32>
    %cst_20 = stablehlo.constant dense_resource<__elided__> : tensor<2x15x24x1xf32>
    %cst_21 = stablehlo.constant dense_resource<__elided__> : tensor<2x128xf32>
    %c = stablehlo.constant dense<0> : tensor<1xi32>
    %c_22 = stablehlo.constant dense<255> : tensor<1xi32>
    %c_23 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_24 = stablehlo.constant dense<255> : tensor<1xi32>
    %c_25 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_26 = stablehlo.constant dense<255> : tensor<1xi32>
    %c_27 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_28 = stablehlo.constant dense<255> : tensor<1xi32>
    %c_29 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_30 = stablehlo.constant dense<255> : tensor<1xi32>
    %c_31 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_32 = stablehlo.constant dense<255> : tensor<1xi32>
    %cst_33 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_34 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_35 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_36 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_37 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_38 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_39 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_40 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_41 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_42 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_43 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_44 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_45 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_46 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_47 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_48 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_49 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_50 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_51 = stablehlo.constant dense_resource<__elided__> : tensor<2x15x26x1xf32>
    %cst_52 = stablehlo.constant dense_resource<__elided__> : tensor<2x128xf32>
    %cst_53 = stablehlo.constant dense_resource<__elided__> : tensor<2x16x25x1xf32>
    %cst_54 = stablehlo.constant dense_resource<__elided__> : tensor<2x128xf32>
    %c_55 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_57 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_59 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_61 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_62 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_63 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_64 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_65 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %cst_67 = stablehlo.constant dense<[1.87612057E+14, 1.9341449E+14, 1.99216924E+14]> : tensor<3xf32>
    %c_68 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %cst_69 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xf32>
    %c_70 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_71 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %c_72 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_73 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %c_74 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %cst_75 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xf32>
    %c_76 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_77 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %c_78 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_79 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xf32>
    %cst_80 = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
    %c_81 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_83 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_85 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_86 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_87 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_88 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_89 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_90 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_91 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_92 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %cst_93 = stablehlo.constant dense<[1.87612057E+14, 1.9341449E+14, 1.99216924E+14]> : tensor<3xf32>
    %c_94 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %cst_95 = stablehlo.constant dense<1.250000e-01> : tensor<200x8xf32>
    %c_96 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %cst_97 = stablehlo.constant dense<5.000000e-01> : tensor<200x2xf32>
    %c_98 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %cst_99 = stablehlo.constant dense<5.000000e-01> : tensor<200x2xf32>
    %c_100 = stablehlo.constant dense_resource<__elided__> : tensor<200x1xi32>
    %cst_101 = stablehlo.constant dense<1.000000e+00> : tensor<200x1xf32>
    %c_102 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_103 = stablehlo.constant dense<2.500000e-01> : tensor<200x4xf32>
    %c_104 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_105 = stablehlo.constant dense<2.500000e-01> : tensor<200x4xf32>
    %cst_106 = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
    %cst_107 = stablehlo.constant dense<1.25663701E-6> : tensor<f32>
    %0 = stablehlo.divide %cst, %cst_107 : tensor<f32>
    %cst_108 = stablehlo.constant dense<8.85418821E-12> : tensor<f32>
    %1 = stablehlo.divide %cst, %cst_108 : tensor<f32>
    %cst_109 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %2 = stablehlo.multiply %cst_109, %0 : tensor<f32>
    %3 = stablehlo.multiply %2, %arg30 : tensor<f32>
    %cst_110 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.add %cst_110, %3 : tensor<f32>
    %5 = stablehlo.divide %0, %4 : tensor<f32>
    %cst_111 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %6 = stablehlo.multiply %cst_111, %1 : tensor<f32>
    %7 = stablehlo.multiply %6, %arg33 : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<257x257x256xf32>
    %9 = stablehlo.add %arg34, %8 : tensor<257x257x256xf32>
    %10 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<257x257x256xf32>
    %11 = stablehlo.divide %10, %9 : tensor<257x257x256xf32>
    %cst_112 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %12 = stablehlo.multiply %cst_112, %0 : tensor<f32>
    %13 = stablehlo.multiply %12, %arg31 : tensor<f32>
    %cst_113 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %14 = stablehlo.add %cst_113, %13 : tensor<f32>
    %15 = stablehlo.divide %0, %14 : tensor<f32>
    %cst_114 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %16 = stablehlo.multiply %cst_114, %1 : tensor<f32>
    %17 = stablehlo.multiply %16, %arg35 : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<257x256x257xf32>
    %19 = stablehlo.add %arg36, %18 : tensor<257x256x257xf32>
    %20 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<257x256x257xf32>
    %21 = stablehlo.divide %20, %19 : tensor<257x256x257xf32>
    %cst_115 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %22 = stablehlo.multiply %cst_115, %0 : tensor<f32>
    %23 = stablehlo.multiply %22, %arg32 : tensor<f32>
    %cst_116 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %24 = stablehlo.add %cst_116, %23 : tensor<f32>
    %25 = stablehlo.divide %0, %24 : tensor<f32>
    %cst_117 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %26 = stablehlo.multiply %cst_117, %1 : tensor<f32>
    %27 = stablehlo.multiply %26, %arg37 : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [] : (tensor<f32>) -> tensor<256x257x257xf32>
    %29 = stablehlo.add %arg38, %28 : tensor<256x257x257xf32>
    %30 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<256x257x257xf32>
    %31 = stablehlo.divide %30, %29 : tensor<256x257x257xf32>
    %32 = stablehlo.iota dim = 0 : tensor<128xi32>
    %c_118 = stablehlo.constant dense<0> : tensor<i32>
    %33:150 = stablehlo.while(%iterArg = %32, %iterArg_119 = %cst, %iterArg_120 = %arg28, %iterArg_121 = %cst_0, %iterArg_122 = %cst_1, %iterArg_123 = %cst_2, %iterArg_124 = %cst_3, %iterArg_125 = %cst_4, %iterArg_126 = %cst_5, %iterArg_127 = %cst_6, %iterArg_128 = %cst_7, %iterArg_129 = %cst_8, %iterArg_130 = %cst_9, %iterArg_131 = %cst_10, %iterArg_132 = %cst_11, %iterArg_133 = %cst_12, %iterArg_134 = %cst_13, %iterArg_135 = %cst_14, %iterArg_136 = %cst_15, %iterArg_137 = %cst_16, %iterArg_138 = %cst_17, %iterArg_139 = %arg30, %iterArg_140 = %5, %iterArg_141 = %arg31, %iterArg_142 = %15, %iterArg_143 = %arg32, %iterArg_144 = %25, %iterArg_145 = %cst_18, %iterArg_146 = %cst_19, %iterArg_147 = %cst_20, %iterArg_148 = %cst_21, %iterArg_149 = %c, %iterArg_150 = %c_22, %iterArg_151 = %c_23, %iterArg_152 = %c_24, %iterArg_153 = %c_25, %iterArg_154 = %c_26, %iterArg_155 = %c_27, %iterArg_156 = %c_28, %iterArg_157 = %c_29, %iterArg_158 = %c_30, %iterArg_159 = %c_31, %iterArg_160 = %c_32, %iterArg_161 = %cst_33, %iterArg_162 = %cst_34, %iterArg_163 = %cst_35, %iterArg_164 = %cst_36, %iterArg_165 = %cst_37, %iterArg_166 = %cst_38, %iterArg_167 = %cst_39, %iterArg_168 = %cst_40, %iterArg_169 = %cst_41, %iterArg_170 = %cst_42, %iterArg_171 = %cst_43, %iterArg_172 = %cst_44, %iterArg_173 = %cst_45, %iterArg_174 = %cst_46, %iterArg_175 = %cst_47, %iterArg_176 = %cst_48, %iterArg_177 = %cst_49, %iterArg_178 = %cst_50, %iterArg_179 = %arg33, %iterArg_180 = %11, %iterArg_181 = %arg35, %iterArg_182 = %21, %iterArg_183 = %arg37, %iterArg_184 = %31, %iterArg_185 = %cst_51, %iterArg_186 = %cst_52, %iterArg_187 = %cst_53, %iterArg_188 = %cst_54, %iterArg_189 = %c_55, %iterArg_190 = %cst_56, %iterArg_191 = %c_57, %iterArg_192 = %cst_58, %iterArg_193 = %c_59, %iterArg_194 = %cst_60, %iterArg_195 = %c_61, %iterArg_196 = %cst_62, %iterArg_197 = %c_63, %iterArg_198 = %cst_64, %iterArg_199 = %c_65, %iterArg_200 = %cst_66, %iterArg_201 = %cst_67, %iterArg_202 = %c_68, %iterArg_203 = %cst_69, %iterArg_204 = %c_70, %iterArg_205 = %cst_71, %iterArg_206 = %c_72, %iterArg_207 = %cst_73, %iterArg_208 = %c_74, %iterArg_209 = %cst_75, %iterArg_210 = %c_76, %iterArg_211 = %cst_77, %iterArg_212 = %c_78, %iterArg_213 = %cst_79, %iterArg_214 = %cst_80, %iterArg_215 = %c_81, %iterArg_216 = %cst_82, %iterArg_217 = %c_83, %iterArg_218 = %cst_84, %iterArg_219 = %c_85, %iterArg_220 = %cst_86, %iterArg_221 = %c_87, %iterArg_222 = %cst_88, %iterArg_223 = %c_89, %iterArg_224 = %cst_90, %iterArg_225 = %c_91, %iterArg_226 = %cst_92, %iterArg_227 = %cst_93, %iterArg_228 = %c_94, %iterArg_229 = %cst_95, %iterArg_230 = %c_96, %iterArg_231 = %cst_97, %iterArg_232 = %c_98, %iterArg_233 = %cst_99, %iterArg_234 = %c_100, %iterArg_235 = %cst_101, %iterArg_236 = %c_102, %iterArg_237 = %cst_103, %iterArg_238 = %c_104, %iterArg_239 = %cst_105, %iterArg_240 = %cst_106, %iterArg_241 = %c_118, %iterArg_242 = %arg0, %iterArg_243 = %arg1, %iterArg_244 = %arg2, %iterArg_245 = %arg3, %iterArg_246 = %arg4, %iterArg_247 = %arg5, %iterArg_248 = %arg6, %iterArg_249 = %arg7, %iterArg_250 = %arg8, %iterArg_251 = %arg9, %iterArg_252 = %arg10, %iterArg_253 = %arg11, %iterArg_254 = %arg12, %iterArg_255 = %arg13, %iterArg_256 = %arg14, %iterArg_257 = %arg15, %iterArg_258 = %arg16, %iterArg_259 = %arg17, %iterArg_260 = %arg18, %iterArg_261 = %arg19, %iterArg_262 = %arg20, %iterArg_263 = %arg25, %iterArg_264 = %arg26, %iterArg_265 = %arg27, %iterArg_266 = %arg28, %iterArg_267 = %arg29) : tensor<128xi32>, tensor<f32>, tensor<f32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<2x14x25x1xf32>, tensor<2x128xf32>, tensor<2x15x24x1xf32>, tensor<2x128xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<257x257x256xf32>, tensor<f32>, tensor<257x256x257xf32>, tensor<f32>, tensor<256x257x257xf32>, tensor<2x15x26x1xf32>, tensor<2x128xf32>, tensor<2x16x25x1xf32>, tensor<2x128xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<3xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<3xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<6xf32>, tensor<i32>, tensor<257x257x256xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x256xf32>, tensor<257x256x256xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x256xf32>, tensor<256x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x256xf32>, tensor<257x24x256xf32>, tensor<24x257x256xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
    cond {
      %c_268 = stablehlo.constant dense<128> : tensor<i32>
      %34 = stablehlo.compare  LT, %iterArg_241, %c_268,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %34 : tensor<i1>
    } do {
      %34 = stablehlo.dynamic_slice %iterArg, %iterArg_241, sizes = [1] : (tensor<128xi32>, tensor<i32>) -> tensor<1xi32>
      %35 = stablehlo.reshape %34 : (tensor<1xi32>) -> tensor<i32>
      %36:26 = func.call @closed_call(%iterArg_119, %iterArg_120, %iterArg_121, %iterArg_122, %iterArg_123, %iterArg_124, %iterArg_125, %iterArg_126, %iterArg_127, %iterArg_128, %iterArg_129, %iterArg_130, %iterArg_131, %iterArg_132, %iterArg_133, %iterArg_134, %iterArg_135, %iterArg_136, %iterArg_137, %iterArg_138, %iterArg_139, %iterArg_140, %iterArg_141, %iterArg_142, %iterArg_143, %iterArg_144, %iterArg_145, %iterArg_146, %iterArg_147, %iterArg_148, %iterArg_149, %iterArg_150, %iterArg_151, %iterArg_152, %iterArg_153, %iterArg_154, %iterArg_155, %iterArg_156, %iterArg_157, %iterArg_158, %iterArg_159, %iterArg_160, %iterArg_161, %iterArg_162, %iterArg_163, %iterArg_164, %iterArg_165, %iterArg_166, %iterArg_167, %iterArg_168, %iterArg_169, %iterArg_170, %iterArg_171, %iterArg_172, %iterArg_173, %iterArg_174, %iterArg_175, %iterArg_176, %iterArg_177, %iterArg_178, %iterArg_179, %iterArg_180, %iterArg_181, %iterArg_182, %iterArg_183, %iterArg_184, %iterArg_185, %iterArg_186, %iterArg_187, %iterArg_188, %iterArg_189, %iterArg_190, %iterArg_191, %iterArg_192, %iterArg_193, %iterArg_194, %iterArg_195, %iterArg_196, %iterArg_197, %iterArg_198, %iterArg_199, %iterArg_200, %iterArg_201, %iterArg_202, %iterArg_203, %iterArg_204, %iterArg_205, %iterArg_206, %iterArg_207, %iterArg_208, %iterArg_209, %iterArg_210, %iterArg_211, %iterArg_212, %iterArg_213, %iterArg_214, %iterArg_215, %iterArg_216, %iterArg_217, %iterArg_218, %iterArg_219, %iterArg_220, %iterArg_221, %iterArg_222, %iterArg_223, %iterArg_224, %iterArg_225, %iterArg_226, %iterArg_227, %iterArg_228, %iterArg_229, %iterArg_230, %iterArg_231, %iterArg_232, %iterArg_233, %iterArg_234, %iterArg_235, %iterArg_236, %iterArg_237, %iterArg_238, %iterArg_239, %iterArg_240, %iterArg_242, %iterArg_243, %iterArg_244, %iterArg_245, %iterArg_246, %iterArg_247, %iterArg_248, %iterArg_249, %iterArg_250, %iterArg_251, %iterArg_252, %iterArg_253, %iterArg_254, %iterArg_255, %iterArg_256, %iterArg_257, %iterArg_258, %iterArg_259, %iterArg_260, %iterArg_261, %iterArg_262, %iterArg_263, %iterArg_264, %iterArg_265, %iterArg_266, %iterArg_267, %35) : (tensor<f32>, tensor<f32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<2x14x25x1xf32>, tensor<2x128xf32>, tensor<2x15x24x1xf32>, tensor<2x128xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<257x257x256xf32>, tensor<f32>, tensor<257x256x257xf32>, tensor<f32>, tensor<256x257x257xf32>, tensor<2x15x26x1xf32>, tensor<2x128xf32>, tensor<2x16x25x1xf32>, tensor<2x128xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<3xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<3xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<6xf32>, tensor<257x257x256xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x256xf32>, tensor<257x256x256xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x256xf32>, tensor<256x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x256xf32>, tensor<257x24x256xf32>, tensor<24x257x256xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>, tensor<i32>) -> (tensor<257x257x256xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x256xf32>, tensor<257x256x256xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x256xf32>, tensor<256x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x256xf32>, tensor<257x24x256xf32>, tensor<24x257x256xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>)
      %c_268 = stablehlo.constant dense<1> : tensor<i32>
      %37 = stablehlo.add %iterArg_241, %c_268 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_119, %iterArg_120, %iterArg_121, %iterArg_122, %iterArg_123, %iterArg_124, %iterArg_125, %iterArg_126, %iterArg_127, %iterArg_128, %iterArg_129, %iterArg_130, %iterArg_131, %iterArg_132, %iterArg_133, %iterArg_134, %iterArg_135, %iterArg_136, %iterArg_137, %iterArg_138, %iterArg_139, %iterArg_140, %iterArg_141, %iterArg_142, %iterArg_143, %iterArg_144, %iterArg_145, %iterArg_146, %iterArg_147, %iterArg_148, %iterArg_149, %iterArg_150, %iterArg_151, %iterArg_152, %iterArg_153, %iterArg_154, %iterArg_155, %iterArg_156, %iterArg_157, %iterArg_158, %iterArg_159, %iterArg_160, %iterArg_161, %iterArg_162, %iterArg_163, %iterArg_164, %iterArg_165, %iterArg_166, %iterArg_167, %iterArg_168, %iterArg_169, %iterArg_170, %iterArg_171, %iterArg_172, %iterArg_173, %iterArg_174, %iterArg_175, %iterArg_176, %iterArg_177, %iterArg_178, %iterArg_179, %iterArg_180, %iterArg_181, %iterArg_182, %iterArg_183, %iterArg_184, %iterArg_185, %iterArg_186, %iterArg_187, %iterArg_188, %iterArg_189, %iterArg_190, %iterArg_191, %iterArg_192, %iterArg_193, %iterArg_194, %iterArg_195, %iterArg_196, %iterArg_197, %iterArg_198, %iterArg_199, %iterArg_200, %iterArg_201, %iterArg_202, %iterArg_203, %iterArg_204, %iterArg_205, %iterArg_206, %iterArg_207, %iterArg_208, %iterArg_209, %iterArg_210, %iterArg_211, %iterArg_212, %iterArg_213, %iterArg_214, %iterArg_215, %iterArg_216, %iterArg_217, %iterArg_218, %iterArg_219, %iterArg_220, %iterArg_221, %iterArg_222, %iterArg_223, %iterArg_224, %iterArg_225, %iterArg_226, %iterArg_227, %iterArg_228, %iterArg_229, %iterArg_230, %iterArg_231, %iterArg_232, %iterArg_233, %iterArg_234, %iterArg_235, %iterArg_236, %iterArg_237, %iterArg_238, %iterArg_239, %iterArg_240, %37, %36#0, %36#1, %36#2, %36#3, %36#4, %36#5, %36#6, %36#7, %36#8, %36#9, %36#10, %36#11, %36#12, %36#13, %36#14, %36#15, %36#16, %36#17, %36#18, %36#19, %36#20, %36#21, %36#22, %36#23, %36#24, %36#25 : tensor<128xi32>, tensor<f32>, tensor<f32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<2x14x25x1xf32>, tensor<2x128xf32>, tensor<2x15x24x1xf32>, tensor<2x128xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<257x257x256xf32>, tensor<f32>, tensor<257x256x257xf32>, tensor<f32>, tensor<256x257x257xf32>, tensor<2x15x26x1xf32>, tensor<2x128xf32>, tensor<2x16x25x1xf32>, tensor<2x128xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<3xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<3xf32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<6xf32>, tensor<i32>, tensor<257x257x256xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x256xf32>, tensor<257x256x256xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x256xf32>, tensor<256x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x256xf32>, tensor<257x24x256xf32>, tensor<24x257x256xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
    }
    return %33#124, %33#125, %33#126, %33#127, %33#128, %33#129, %33#130, %33#131, %33#132, %33#133, %33#134, %33#135, %33#136, %33#137, %33#138, %33#139, %33#140, %33#141, %33#142, %33#143, %33#144, %arg21, %arg22, %arg23, %arg24, %33#145, %33#146, %33#147, %33#148, %33#149 : tensor<257x257x256xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x256xf32>, tensor<257x256x256xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x256xf32>, tensor<256x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x256xf32>, tensor<257x24x256xf32>, tensor<24x257x256xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @closed_call(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<1x24x1xf32>, %arg3: tensor<1x24x1xf32>, %arg4: tensor<1x24x1xf32>, %arg5: tensor<24x1x1xf32>, %arg6: tensor<24x1x1xf32>, %arg7: tensor<24x1x1xf32>, %arg8: tensor<24x1x1xf32>, %arg9: tensor<24x1x1xf32>, %arg10: tensor<24x1x1xf32>, %arg11: tensor<1x1x24xf32>, %arg12: tensor<1x1x24xf32>, %arg13: tensor<1x1x24xf32>, %arg14: tensor<1x1x24xf32>, %arg15: tensor<1x1x24xf32>, %arg16: tensor<1x1x24xf32>, %arg17: tensor<1x24x1xf32>, %arg18: tensor<1x24x1xf32>, %arg19: tensor<1x24x1xf32>, %arg20: tensor<f32>, %arg21: tensor<f32>, %arg22: tensor<f32>, %arg23: tensor<f32>, %arg24: tensor<f32>, %arg25: tensor<f32>, %arg26: tensor<2x14x25x1xf32>, %arg27: tensor<2x128xf32>, %arg28: tensor<2x15x24x1xf32>, %arg29: tensor<2x128xf32>, %arg30: tensor<1xi32>, %arg31: tensor<1xi32>, %arg32: tensor<1xi32>, %arg33: tensor<1xi32>, %arg34: tensor<1xi32>, %arg35: tensor<1xi32>, %arg36: tensor<1xi32>, %arg37: tensor<1xi32>, %arg38: tensor<1xi32>, %arg39: tensor<1xi32>, %arg40: tensor<1xi32>, %arg41: tensor<1xi32>, %arg42: tensor<1x24x1xf32>, %arg43: tensor<1x24x1xf32>, %arg44: tensor<1x24x1xf32>, %arg45: tensor<24x1x1xf32>, %arg46: tensor<24x1x1xf32>, %arg47: tensor<24x1x1xf32>, %arg48: tensor<24x1x1xf32>, %arg49: tensor<24x1x1xf32>, %arg50: tensor<24x1x1xf32>, %arg51: tensor<1x1x24xf32>, %arg52: tensor<1x1x24xf32>, %arg53: tensor<1x1x24xf32>, %arg54: tensor<1x1x24xf32>, %arg55: tensor<1x1x24xf32>, %arg56: tensor<1x1x24xf32>, %arg57: tensor<1x24x1xf32>, %arg58: tensor<1x24x1xf32>, %arg59: tensor<1x24x1xf32>, %arg60: tensor<f32>, %arg61: tensor<257x257x256xf32>, %arg62: tensor<f32>, %arg63: tensor<257x256x257xf32>, %arg64: tensor<f32>, %arg65: tensor<256x257x257xf32>, %arg66: tensor<2x15x26x1xf32>, %arg67: tensor<2x128xf32>, %arg68: tensor<2x16x25x1xf32>, %arg69: tensor<2x128xf32>, %arg70: tensor<200x1xi32>, %arg71: tensor<200x1xf32>, %arg72: tensor<200x1xi32>, %arg73: tensor<200x1xf32>, %arg74: tensor<200x1xi32>, %arg75: tensor<200x1xf32>, %arg76: tensor<200x1xi32>, %arg77: tensor<200x1xf32>, %arg78: tensor<200x1xi32>, %arg79: tensor<200x1xf32>, %arg80: tensor<200x1xi32>, %arg81: tensor<200x1xf32>, %arg82: tensor<3xf32>, %arg83: tensor<200x8xi32>, %arg84: tensor<200x8xf32>, %arg85: tensor<200x4xi32>, %arg86: tensor<200x4xf32>, %arg87: tensor<200x4xi32>, %arg88: tensor<200x4xf32>, %arg89: tensor<200x2xi32>, %arg90: tensor<200x2xf32>, %arg91: tensor<200x4xi32>, %arg92: tensor<200x4xf32>, %arg93: tensor<200x4xi32>, %arg94: tensor<200x4xf32>, %arg95: tensor<6xf32>, %arg96: tensor<200x1xi32>, %arg97: tensor<200x1xf32>, %arg98: tensor<200x1xi32>, %arg99: tensor<200x1xf32>, %arg100: tensor<200x1xi32>, %arg101: tensor<200x1xf32>, %arg102: tensor<200x1xi32>, %arg103: tensor<200x1xf32>, %arg104: tensor<200x1xi32>, %arg105: tensor<200x1xf32>, %arg106: tensor<200x1xi32>, %arg107: tensor<200x1xf32>, %arg108: tensor<3xf32>, %arg109: tensor<200x8xi32>, %arg110: tensor<200x8xf32>, %arg111: tensor<200x2xi32>, %arg112: tensor<200x2xf32>, %arg113: tensor<200x2xi32>, %arg114: tensor<200x2xf32>, %arg115: tensor<200x1xi32>, %arg116: tensor<200x1xf32>, %arg117: tensor<200x4xi32>, %arg118: tensor<200x4xf32>, %arg119: tensor<200x4xi32>, %arg120: tensor<200x4xf32>, %arg121: tensor<6xf32>, %arg122: tensor<257x257x256xf32>, %arg123: tensor<257x256x257xf32>, %arg124: tensor<256x257x257xf32>, %arg125: tensor<256x256x257xf32>, %arg126: tensor<256x257x256xf32>, %arg127: tensor<257x256x256xf32>, %arg128: tensor<256x24x257xf32>, %arg129: tensor<24x256x257xf32>, %arg130: tensor<24x257x256xf32>, %arg131: tensor<256x257x24xf32>, %arg132: tensor<257x256x24xf32>, %arg133: tensor<257x24x256xf32>, %arg134: tensor<257x24x256xf32>, %arg135: tensor<24x257x256xf32>, %arg136: tensor<24x256x257xf32>, %arg137: tensor<257x256x24xf32>, %arg138: tensor<256x257x24xf32>, %arg139: tensor<256x24x257xf32>, %arg140: tensor<2x1xf32>, %arg141: tensor<2x1xf32>, %arg142: tensor<2xi32>, %arg143: tensor<7200xf32>, %arg144: tensor<7200xf32>, %arg145: tensor<6xf32>, %arg146: tensor<f32>, %arg147: tensor<i32>, %arg148: tensor<i32>) -> (tensor<257x257x256xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x256xf32>, tensor<257x256x256xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x256xf32>, tensor<256x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x256xf32>, tensor<257x24x256xf32>, tensor<24x257x256xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg148, %c : tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<f32>
    %3 = stablehlo.add %arg1, %2 : tensor<f32>
    %4 = call @diff(%arg124) : (tensor<256x257x257xf32>) -> tensor<256x256x257xf32>
    %cst = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x256x257xf32>
    %6 = stablehlo.divide %4, %5 : tensor<256x256x257xf32>
    %7 = stablehlo.slice %6 [0:256, 0:12, 0:257] : (tensor<256x256x257xf32>) -> tensor<256x12x257xf32>
    %8 = stablehlo.slice %6 [0:256, 244:256, 0:257] : (tensor<256x256x257xf32>) -> tensor<256x12x257xf32>
    %9 = stablehlo.concatenate %7, %8, dim = 1 : (tensor<256x12x257xf32>, tensor<256x12x257xf32>) -> tensor<256x24x257xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<256x24x257xf32>
    %11 = stablehlo.multiply %10, %arg128 : tensor<256x24x257xf32>
    %12 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<256x24x257xf32>
    %13 = stablehlo.multiply %12, %9 : tensor<256x24x257xf32>
    %14 = stablehlo.add %11, %13 : tensor<256x24x257xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<256x24x257xf32>
    %16 = stablehlo.multiply %9, %15 : tensor<256x24x257xf32>
    %17 = stablehlo.add %16, %14 : tensor<256x24x257xf32>
    %18 = stablehlo.slice %17 [0:256, 0:12, 0:257] : (tensor<256x24x257xf32>) -> tensor<256x12x257xf32>
    %19 = stablehlo.slice %6 [0:256, 12:244, 0:257] : (tensor<256x256x257xf32>) -> tensor<256x232x257xf32>
    %20 = stablehlo.slice %17 [0:256, 12:24, 0:257] : (tensor<256x24x257xf32>) -> tensor<256x12x257xf32>
    %21 = stablehlo.concatenate %18, %19, %20, dim = 1 : (tensor<256x12x257xf32>, tensor<256x232x257xf32>, tensor<256x12x257xf32>) -> tensor<256x256x257xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x256x257xf32>
    %23 = stablehlo.multiply %22, %21 : tensor<256x256x257xf32>
    %24 = call @diff_19(%arg123) : (tensor<257x256x257xf32>) -> tensor<256x256x257xf32>
    %cst_1 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256x256x257xf32>
    %26 = stablehlo.divide %24, %25 : tensor<256x256x257xf32>
    %27 = stablehlo.slice %26 [0:12, 0:256, 0:257] : (tensor<256x256x257xf32>) -> tensor<12x256x257xf32>
    %28 = stablehlo.slice %26 [244:256, 0:256, 0:257] : (tensor<256x256x257xf32>) -> tensor<12x256x257xf32>
    %29 = stablehlo.concatenate %27, %28, dim = 0 : (tensor<12x256x257xf32>, tensor<12x256x257xf32>) -> tensor<24x256x257xf32>
    %30 = stablehlo.broadcast_in_dim %arg5, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x256x257xf32>
    %31 = stablehlo.multiply %30, %arg129 : tensor<24x256x257xf32>
    %32 = stablehlo.broadcast_in_dim %arg6, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x256x257xf32>
    %33 = stablehlo.multiply %32, %29 : tensor<24x256x257xf32>
    %34 = stablehlo.add %31, %33 : tensor<24x256x257xf32>
    %35 = stablehlo.broadcast_in_dim %arg7, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x256x257xf32>
    %36 = stablehlo.multiply %29, %35 : tensor<24x256x257xf32>
    %37 = stablehlo.add %36, %34 : tensor<24x256x257xf32>
    %38 = stablehlo.slice %37 [0:12, 0:256, 0:257] : (tensor<24x256x257xf32>) -> tensor<12x256x257xf32>
    %39 = stablehlo.slice %26 [12:244, 0:256, 0:257] : (tensor<256x256x257xf32>) -> tensor<232x256x257xf32>
    %40 = stablehlo.slice %37 [12:24, 0:256, 0:257] : (tensor<24x256x257xf32>) -> tensor<12x256x257xf32>
    %41 = stablehlo.concatenate %38, %39, %40, dim = 0 : (tensor<12x256x257xf32>, tensor<232x256x257xf32>, tensor<12x256x257xf32>) -> tensor<256x256x257xf32>
    %cst_2 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x256x257xf32>
    %43 = stablehlo.multiply %42, %41 : tensor<256x256x257xf32>
    %44 = call @diff_32(%arg122) : (tensor<257x257x256xf32>) -> tensor<256x257x256xf32>
    %cst_3 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<256x257x256xf32>
    %46 = stablehlo.divide %44, %45 : tensor<256x257x256xf32>
    %47 = stablehlo.slice %46 [0:12, 0:257, 0:256] : (tensor<256x257x256xf32>) -> tensor<12x257x256xf32>
    %48 = stablehlo.slice %46 [244:256, 0:257, 0:256] : (tensor<256x257x256xf32>) -> tensor<12x257x256xf32>
    %49 = stablehlo.concatenate %47, %48, dim = 0 : (tensor<12x257x256xf32>, tensor<12x257x256xf32>) -> tensor<24x257x256xf32>
    %50 = stablehlo.broadcast_in_dim %arg8, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x257x256xf32>
    %51 = stablehlo.multiply %50, %arg130 : tensor<24x257x256xf32>
    %52 = stablehlo.broadcast_in_dim %arg9, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x257x256xf32>
    %53 = stablehlo.multiply %52, %49 : tensor<24x257x256xf32>
    %54 = stablehlo.add %51, %53 : tensor<24x257x256xf32>
    %55 = stablehlo.broadcast_in_dim %arg10, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x257x256xf32>
    %56 = stablehlo.multiply %49, %55 : tensor<24x257x256xf32>
    %57 = stablehlo.add %56, %54 : tensor<24x257x256xf32>
    %58 = stablehlo.slice %57 [0:12, 0:257, 0:256] : (tensor<24x257x256xf32>) -> tensor<12x257x256xf32>
    %59 = stablehlo.slice %46 [12:244, 0:257, 0:256] : (tensor<256x257x256xf32>) -> tensor<232x257x256xf32>
    %60 = stablehlo.slice %57 [12:24, 0:257, 0:256] : (tensor<24x257x256xf32>) -> tensor<12x257x256xf32>
    %61 = stablehlo.concatenate %58, %59, %60, dim = 0 : (tensor<12x257x256xf32>, tensor<232x257x256xf32>, tensor<12x257x256xf32>) -> tensor<256x257x256xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<256x257x256xf32>
    %63 = stablehlo.multiply %62, %61 : tensor<256x257x256xf32>
    %64 = call @diff_48(%arg124) : (tensor<256x257x257xf32>) -> tensor<256x257x256xf32>
    %cst_5 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<256x257x256xf32>
    %66 = stablehlo.divide %64, %65 : tensor<256x257x256xf32>
    %67 = stablehlo.slice %66 [0:256, 0:257, 0:12] : (tensor<256x257x256xf32>) -> tensor<256x257x12xf32>
    %68 = stablehlo.slice %66 [0:256, 0:257, 244:256] : (tensor<256x257x256xf32>) -> tensor<256x257x12xf32>
    %69 = stablehlo.concatenate %67, %68, dim = 2 : (tensor<256x257x12xf32>, tensor<256x257x12xf32>) -> tensor<256x257x24xf32>
    %70 = stablehlo.broadcast_in_dim %arg11, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<256x257x24xf32>
    %71 = stablehlo.multiply %70, %arg131 : tensor<256x257x24xf32>
    %72 = stablehlo.broadcast_in_dim %arg12, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<256x257x24xf32>
    %73 = stablehlo.multiply %72, %69 : tensor<256x257x24xf32>
    %74 = stablehlo.add %71, %73 : tensor<256x257x24xf32>
    %75 = stablehlo.broadcast_in_dim %arg13, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<256x257x24xf32>
    %76 = stablehlo.multiply %69, %75 : tensor<256x257x24xf32>
    %77 = stablehlo.add %76, %74 : tensor<256x257x24xf32>
    %78 = stablehlo.slice %77 [0:256, 0:257, 0:12] : (tensor<256x257x24xf32>) -> tensor<256x257x12xf32>
    %79 = stablehlo.slice %66 [0:256, 0:257, 12:244] : (tensor<256x257x256xf32>) -> tensor<256x257x232xf32>
    %80 = stablehlo.slice %77 [0:256, 0:257, 12:24] : (tensor<256x257x24xf32>) -> tensor<256x257x12xf32>
    %81 = stablehlo.concatenate %78, %79, %80, dim = 2 : (tensor<256x257x12xf32>, tensor<256x257x232xf32>, tensor<256x257x12xf32>) -> tensor<256x257x256xf32>
    %cst_6 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %82 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x257x256xf32>
    %83 = stablehlo.multiply %82, %81 : tensor<256x257x256xf32>
    %84 = call @diff_61(%arg123) : (tensor<257x256x257xf32>) -> tensor<257x256x256xf32>
    %cst_7 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %85 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<257x256x256xf32>
    %86 = stablehlo.divide %84, %85 : tensor<257x256x256xf32>
    %87 = stablehlo.slice %86 [0:257, 0:256, 0:12] : (tensor<257x256x256xf32>) -> tensor<257x256x12xf32>
    %88 = stablehlo.slice %86 [0:257, 0:256, 244:256] : (tensor<257x256x256xf32>) -> tensor<257x256x12xf32>
    %89 = stablehlo.concatenate %87, %88, dim = 2 : (tensor<257x256x12xf32>, tensor<257x256x12xf32>) -> tensor<257x256x24xf32>
    %90 = stablehlo.broadcast_in_dim %arg14, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<257x256x24xf32>
    %91 = stablehlo.multiply %90, %arg132 : tensor<257x256x24xf32>
    %92 = stablehlo.broadcast_in_dim %arg15, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<257x256x24xf32>
    %93 = stablehlo.multiply %92, %89 : tensor<257x256x24xf32>
    %94 = stablehlo.add %91, %93 : tensor<257x256x24xf32>
    %95 = stablehlo.broadcast_in_dim %arg16, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<257x256x24xf32>
    %96 = stablehlo.multiply %89, %95 : tensor<257x256x24xf32>
    %97 = stablehlo.add %96, %94 : tensor<257x256x24xf32>
    %98 = stablehlo.slice %97 [0:257, 0:256, 0:12] : (tensor<257x256x24xf32>) -> tensor<257x256x12xf32>
    %99 = stablehlo.slice %86 [0:257, 0:256, 12:244] : (tensor<257x256x256xf32>) -> tensor<257x256x232xf32>
    %100 = stablehlo.slice %97 [0:257, 0:256, 12:24] : (tensor<257x256x24xf32>) -> tensor<257x256x12xf32>
    %101 = stablehlo.concatenate %98, %99, %100, dim = 2 : (tensor<257x256x12xf32>, tensor<257x256x232xf32>, tensor<257x256x12xf32>) -> tensor<257x256x256xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %102 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<257x256x256xf32>
    %103 = stablehlo.multiply %102, %101 : tensor<257x256x256xf32>
    %104 = call @diff_77(%arg122) : (tensor<257x257x256xf32>) -> tensor<257x256x256xf32>
    %cst_9 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %105 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<257x256x256xf32>
    %106 = stablehlo.divide %104, %105 : tensor<257x256x256xf32>
    %107 = stablehlo.slice %106 [0:257, 0:12, 0:256] : (tensor<257x256x256xf32>) -> tensor<257x12x256xf32>
    %108 = stablehlo.slice %106 [0:257, 244:256, 0:256] : (tensor<257x256x256xf32>) -> tensor<257x12x256xf32>
    %109 = stablehlo.concatenate %107, %108, dim = 1 : (tensor<257x12x256xf32>, tensor<257x12x256xf32>) -> tensor<257x24x256xf32>
    %110 = stablehlo.broadcast_in_dim %arg17, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<257x24x256xf32>
    %111 = stablehlo.multiply %110, %arg133 : tensor<257x24x256xf32>
    %112 = stablehlo.broadcast_in_dim %arg18, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<257x24x256xf32>
    %113 = stablehlo.multiply %112, %109 : tensor<257x24x256xf32>
    %114 = stablehlo.add %111, %113 : tensor<257x24x256xf32>
    %115 = stablehlo.broadcast_in_dim %arg19, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<257x24x256xf32>
    %116 = stablehlo.multiply %109, %115 : tensor<257x24x256xf32>
    %117 = stablehlo.add %116, %114 : tensor<257x24x256xf32>
    %118 = stablehlo.slice %117 [0:257, 0:12, 0:256] : (tensor<257x24x256xf32>) -> tensor<257x12x256xf32>
    %119 = stablehlo.slice %106 [0:257, 12:244, 0:256] : (tensor<257x256x256xf32>) -> tensor<257x232x256xf32>
    %120 = stablehlo.slice %117 [0:257, 12:24, 0:256] : (tensor<257x24x256xf32>) -> tensor<257x12x256xf32>
    %121 = stablehlo.concatenate %118, %119, %120, dim = 1 : (tensor<257x12x256xf32>, tensor<257x232x256xf32>, tensor<257x12x256xf32>) -> tensor<257x256x256xf32>
    %cst_10 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %122 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<257x256x256xf32>
    %123 = stablehlo.multiply %122, %121 : tensor<257x256x256xf32>
    %124 = stablehlo.add %23, %43 : tensor<256x256x257xf32>
    %125 = stablehlo.add %63, %83 : tensor<256x257x256xf32>
    %126 = stablehlo.add %103, %123 : tensor<257x256x256xf32>
    %127 = stablehlo.broadcast_in_dim %arg20, dims = [] : (tensor<f32>) -> tensor<256x256x257xf32>
    %128 = stablehlo.multiply %127, %arg125 : tensor<256x256x257xf32>
    %129 = stablehlo.add %124, %128 : tensor<256x256x257xf32>
    %130 = stablehlo.broadcast_in_dim %arg21, dims = [] : (tensor<f32>) -> tensor<256x256x257xf32>
    %131 = stablehlo.multiply %130, %129 : tensor<256x256x257xf32>
    %132 = stablehlo.subtract %arg125, %131 : tensor<256x256x257xf32>
    %133 = stablehlo.broadcast_in_dim %arg22, dims = [] : (tensor<f32>) -> tensor<256x257x256xf32>
    %134 = stablehlo.multiply %133, %arg126 : tensor<256x257x256xf32>
    %135 = stablehlo.add %125, %134 : tensor<256x257x256xf32>
    %136 = stablehlo.broadcast_in_dim %arg23, dims = [] : (tensor<f32>) -> tensor<256x257x256xf32>
    %137 = stablehlo.multiply %136, %135 : tensor<256x257x256xf32>
    %138 = stablehlo.subtract %arg126, %137 : tensor<256x257x256xf32>
    %139 = stablehlo.broadcast_in_dim %arg24, dims = [] : (tensor<f32>) -> tensor<257x256x256xf32>
    %140 = stablehlo.multiply %139, %arg127 : tensor<257x256x256xf32>
    %141 = stablehlo.add %126, %140 : tensor<257x256x256xf32>
    %142 = stablehlo.broadcast_in_dim %arg25, dims = [] : (tensor<f32>) -> tensor<257x256x256xf32>
    %143 = stablehlo.multiply %142, %141 : tensor<257x256x256xf32>
    %144 = stablehlo.subtract %arg127, %143 : tensor<257x256x256xf32>
    %c_11 = stablehlo.constant dense<0> : tensor<i32>
    %c_12 = stablehlo.constant dense<127> : tensor<i32>
    %145 = call @clip(%arg147, %c_11, %c_12) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %146 = stablehlo.slice %arg26 [0:1, 0:14, 0:25, 0:1] : (tensor<2x14x25x1xf32>) -> tensor<1x14x25x1xf32>
    %147 = stablehlo.reshape %146 : (tensor<1x14x25x1xf32>) -> tensor<14x25x1xf32>
    %c_13 = stablehlo.constant dense<0> : tensor<i32>
    %148 = stablehlo.compare  LT, %145, %c_13,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_14 = stablehlo.constant dense<128> : tensor<i32>
    %149 = stablehlo.add %145, %c_14 : tensor<i32>
    %150 = stablehlo.select %148, %149, %145 : tensor<i1>, tensor<i32>
    %c_15 = stablehlo.constant dense<0> : tensor<i32>
    %151 = stablehlo.dynamic_slice %arg27, %c_15, %150, sizes = [1, 1] : (tensor<2x128xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %152 = stablehlo.reshape %151 : (tensor<1x1xf32>) -> tensor<f32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [] : (tensor<f32>) -> tensor<14x25x1xf32>
    %154 = stablehlo.multiply %147, %153 : tensor<14x25x1xf32>
    %c_16 = stablehlo.constant dense<121> : tensor<i32>
    %155 = stablehlo.broadcast_in_dim %c_16, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_17 = stablehlo.constant dense<116> : tensor<i32>
    %156 = stablehlo.broadcast_in_dim %c_17, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_18 = stablehlo.constant dense<75> : tensor<i32>
    %157 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %158 = stablehlo.concatenate %155, %156, %157, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %159 = "stablehlo.scatter"(%138, %158, %154) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      %568 = stablehlo.add %arg149, %arg150 : tensor<f32>
      stablehlo.return %568 : tensor<f32>
    }) : (tensor<256x257x256xf32>, tensor<3xi32>, tensor<14x25x1xf32>) -> tensor<256x257x256xf32>
    %160 = stablehlo.slice %arg26 [1:2, 0:14, 0:25, 0:1] : (tensor<2x14x25x1xf32>) -> tensor<1x14x25x1xf32>
    %161 = stablehlo.reshape %160 : (tensor<1x14x25x1xf32>) -> tensor<14x25x1xf32>
    %c_19 = stablehlo.constant dense<0> : tensor<i32>
    %162 = stablehlo.compare  LT, %145, %c_19,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_20 = stablehlo.constant dense<128> : tensor<i32>
    %163 = stablehlo.add %145, %c_20 : tensor<i32>
    %164 = stablehlo.select %162, %163, %145 : tensor<i1>, tensor<i32>
    %c_21 = stablehlo.constant dense<1> : tensor<i32>
    %165 = stablehlo.dynamic_slice %arg27, %c_21, %164, sizes = [1, 1] : (tensor<2x128xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %166 = stablehlo.reshape %165 : (tensor<1x1xf32>) -> tensor<f32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<f32>) -> tensor<14x25x1xf32>
    %168 = stablehlo.multiply %161, %167 : tensor<14x25x1xf32>
    %c_22 = stablehlo.constant dense<121> : tensor<i32>
    %169 = stablehlo.broadcast_in_dim %c_22, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_23 = stablehlo.constant dense<116> : tensor<i32>
    %170 = stablehlo.broadcast_in_dim %c_23, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_24 = stablehlo.constant dense<75> : tensor<i32>
    %171 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %172 = stablehlo.concatenate %169, %170, %171, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %173 = "stablehlo.scatter"(%159, %172, %168) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      %568 = stablehlo.add %arg149, %arg150 : tensor<f32>
      stablehlo.return %568 : tensor<f32>
    }) : (tensor<256x257x256xf32>, tensor<3xi32>, tensor<14x25x1xf32>) -> tensor<256x257x256xf32>
    %c_25 = stablehlo.constant dense<0> : tensor<i32>
    %c_26 = stablehlo.constant dense<127> : tensor<i32>
    %174 = call @clip(%arg147, %c_25, %c_26) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %175 = stablehlo.slice %arg28 [0:1, 0:15, 0:24, 0:1] : (tensor<2x15x24x1xf32>) -> tensor<1x15x24x1xf32>
    %176 = stablehlo.reshape %175 : (tensor<1x15x24x1xf32>) -> tensor<15x24x1xf32>
    %c_27 = stablehlo.constant dense<0> : tensor<i32>
    %177 = stablehlo.compare  LT, %174, %c_27,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_28 = stablehlo.constant dense<128> : tensor<i32>
    %178 = stablehlo.add %174, %c_28 : tensor<i32>
    %179 = stablehlo.select %177, %178, %174 : tensor<i1>, tensor<i32>
    %c_29 = stablehlo.constant dense<0> : tensor<i32>
    %180 = stablehlo.dynamic_slice %arg29, %c_29, %179, sizes = [1, 1] : (tensor<2x128xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %181 = stablehlo.reshape %180 : (tensor<1x1xf32>) -> tensor<f32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [] : (tensor<f32>) -> tensor<15x24x1xf32>
    %183 = stablehlo.multiply %176, %182 : tensor<15x24x1xf32>
    %c_30 = stablehlo.constant dense<121> : tensor<i32>
    %184 = stablehlo.broadcast_in_dim %c_30, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_31 = stablehlo.constant dense<116> : tensor<i32>
    %185 = stablehlo.broadcast_in_dim %c_31, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_32 = stablehlo.constant dense<75> : tensor<i32>
    %186 = stablehlo.broadcast_in_dim %c_32, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %187 = stablehlo.concatenate %184, %185, %186, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %188 = "stablehlo.scatter"(%144, %187, %183) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      %568 = stablehlo.add %arg149, %arg150 : tensor<f32>
      stablehlo.return %568 : tensor<f32>
    }) : (tensor<257x256x256xf32>, tensor<3xi32>, tensor<15x24x1xf32>) -> tensor<257x256x256xf32>
    %189 = stablehlo.slice %arg28 [1:2, 0:15, 0:24, 0:1] : (tensor<2x15x24x1xf32>) -> tensor<1x15x24x1xf32>
    %190 = stablehlo.reshape %189 : (tensor<1x15x24x1xf32>) -> tensor<15x24x1xf32>
    %c_33 = stablehlo.constant dense<0> : tensor<i32>
    %191 = stablehlo.compare  LT, %174, %c_33,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_34 = stablehlo.constant dense<128> : tensor<i32>
    %192 = stablehlo.add %174, %c_34 : tensor<i32>
    %193 = stablehlo.select %191, %192, %174 : tensor<i1>, tensor<i32>
    %c_35 = stablehlo.constant dense<1> : tensor<i32>
    %194 = stablehlo.dynamic_slice %arg29, %c_35, %193, sizes = [1, 1] : (tensor<2x128xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %195 = stablehlo.reshape %194 : (tensor<1x1xf32>) -> tensor<f32>
    %196 = stablehlo.broadcast_in_dim %195, dims = [] : (tensor<f32>) -> tensor<15x24x1xf32>
    %197 = stablehlo.multiply %190, %196 : tensor<15x24x1xf32>
    %c_36 = stablehlo.constant dense<121> : tensor<i32>
    %198 = stablehlo.broadcast_in_dim %c_36, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_37 = stablehlo.constant dense<116> : tensor<i32>
    %199 = stablehlo.broadcast_in_dim %c_37, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_38 = stablehlo.constant dense<75> : tensor<i32>
    %200 = stablehlo.broadcast_in_dim %c_38, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %201 = stablehlo.concatenate %198, %199, %200, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %202 = "stablehlo.scatter"(%188, %201, %197) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      %568 = stablehlo.add %arg149, %arg150 : tensor<f32>
      stablehlo.return %568 : tensor<f32>
    }) : (tensor<257x256x256xf32>, tensor<3xi32>, tensor<15x24x1xf32>) -> tensor<257x256x256xf32>
    %203 = stablehlo.slice %202 [0:257, 256:256, 0:256] : (tensor<257x256x256xf32>) -> tensor<257x0x256xf32>
    %204 = call @_take(%202, %arg30) : (tensor<257x256x256xf32>, tensor<1xi32>) -> tensor<257x1x256xf32>
    %205 = call @_take(%202, %arg31) : (tensor<257x256x256xf32>, tensor<1xi32>) -> tensor<257x1x256xf32>
    %206 = stablehlo.concatenate %204, %202, %205, %203, dim = 1 : (tensor<257x1x256xf32>, tensor<257x256x256xf32>, tensor<257x1x256xf32>, tensor<257x0x256xf32>) -> tensor<257x258x256xf32>
    %207 = stablehlo.slice %173 [256:256, 0:257, 0:256] : (tensor<256x257x256xf32>) -> tensor<0x257x256xf32>
    %208 = call @_take_119(%173, %arg32) : (tensor<256x257x256xf32>, tensor<1xi32>) -> tensor<1x257x256xf32>
    %209 = call @_take_119(%173, %arg33) : (tensor<256x257x256xf32>, tensor<1xi32>) -> tensor<1x257x256xf32>
    %210 = stablehlo.concatenate %208, %173, %209, %207, dim = 0 : (tensor<1x257x256xf32>, tensor<256x257x256xf32>, tensor<1x257x256xf32>, tensor<0x257x256xf32>) -> tensor<258x257x256xf32>
    %211 = stablehlo.slice %132 [256:256, 0:256, 0:257] : (tensor<256x256x257xf32>) -> tensor<0x256x257xf32>
    %212 = call @_take_127(%132, %arg34) : (tensor<256x256x257xf32>, tensor<1xi32>) -> tensor<1x256x257xf32>
    %213 = call @_take_127(%132, %arg35) : (tensor<256x256x257xf32>, tensor<1xi32>) -> tensor<1x256x257xf32>
    %214 = stablehlo.concatenate %212, %132, %213, %211, dim = 0 : (tensor<1x256x257xf32>, tensor<256x256x257xf32>, tensor<1x256x257xf32>, tensor<0x256x257xf32>) -> tensor<258x256x257xf32>
    %215 = stablehlo.slice %202 [0:257, 0:256, 256:256] : (tensor<257x256x256xf32>) -> tensor<257x256x0xf32>
    %216 = call @_take_135(%202, %arg36) : (tensor<257x256x256xf32>, tensor<1xi32>) -> tensor<257x256x1xf32>
    %217 = call @_take_135(%202, %arg37) : (tensor<257x256x256xf32>, tensor<1xi32>) -> tensor<257x256x1xf32>
    %218 = stablehlo.concatenate %216, %202, %217, %215, dim = 2 : (tensor<257x256x1xf32>, tensor<257x256x256xf32>, tensor<257x256x1xf32>, tensor<257x256x0xf32>) -> tensor<257x256x258xf32>
    %219 = stablehlo.slice %173 [0:256, 0:257, 256:256] : (tensor<256x257x256xf32>) -> tensor<256x257x0xf32>
    %220 = call @_take_143(%173, %arg38) : (tensor<256x257x256xf32>, tensor<1xi32>) -> tensor<256x257x1xf32>
    %221 = call @_take_143(%173, %arg39) : (tensor<256x257x256xf32>, tensor<1xi32>) -> tensor<256x257x1xf32>
    %222 = stablehlo.concatenate %220, %173, %221, %219, dim = 2 : (tensor<256x257x1xf32>, tensor<256x257x256xf32>, tensor<256x257x1xf32>, tensor<256x257x0xf32>) -> tensor<256x257x258xf32>
    %223 = stablehlo.slice %132 [0:256, 256:256, 0:257] : (tensor<256x256x257xf32>) -> tensor<256x0x257xf32>
    %224 = call @_take_151(%132, %arg40) : (tensor<256x256x257xf32>, tensor<1xi32>) -> tensor<256x1x257xf32>
    %225 = call @_take_151(%132, %arg41) : (tensor<256x256x257xf32>, tensor<1xi32>) -> tensor<256x1x257xf32>
    %226 = stablehlo.concatenate %224, %132, %225, %223, dim = 1 : (tensor<256x1x257xf32>, tensor<256x256x257xf32>, tensor<256x1x257xf32>, tensor<256x0x257xf32>) -> tensor<256x258x257xf32>
    %227 = stablehlo.transpose %206, dims = [1, 0, 2] : (tensor<257x258x256xf32>) -> tensor<258x257x256xf32>
    %228 = stablehlo.slice %227 [1:258, 0:257, 0:256] : (tensor<258x257x256xf32>) -> tensor<257x257x256xf32>
    %229 = stablehlo.slice %227 [0:257, 0:257, 0:256] : (tensor<258x257x256xf32>) -> tensor<257x257x256xf32>
    %230 = stablehlo.subtract %228, %229 : tensor<257x257x256xf32>
    %cst_39 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %231 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<257x257x256xf32>
    %232 = stablehlo.divide %230, %231 : tensor<257x257x256xf32>
    %233 = stablehlo.transpose %232, dims = [1, 0, 2] : (tensor<257x257x256xf32>) -> tensor<257x257x256xf32>
    %234 = stablehlo.slice %233 [0:257, 0:12, 0:256] : (tensor<257x257x256xf32>) -> tensor<257x12x256xf32>
    %235 = stablehlo.slice %233 [0:257, 245:257, 0:256] : (tensor<257x257x256xf32>) -> tensor<257x12x256xf32>
    %236 = stablehlo.concatenate %234, %235, dim = 1 : (tensor<257x12x256xf32>, tensor<257x12x256xf32>) -> tensor<257x24x256xf32>
    %237 = stablehlo.broadcast_in_dim %arg42, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<257x24x256xf32>
    %238 = stablehlo.multiply %237, %arg134 : tensor<257x24x256xf32>
    %239 = stablehlo.broadcast_in_dim %arg43, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<257x24x256xf32>
    %240 = stablehlo.multiply %239, %236 : tensor<257x24x256xf32>
    %241 = stablehlo.add %238, %240 : tensor<257x24x256xf32>
    %242 = stablehlo.broadcast_in_dim %arg44, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<257x24x256xf32>
    %243 = stablehlo.multiply %236, %242 : tensor<257x24x256xf32>
    %244 = stablehlo.add %243, %241 : tensor<257x24x256xf32>
    %245 = stablehlo.slice %244 [0:257, 0:12, 0:256] : (tensor<257x24x256xf32>) -> tensor<257x12x256xf32>
    %246 = stablehlo.slice %233 [0:257, 12:245, 0:256] : (tensor<257x257x256xf32>) -> tensor<257x233x256xf32>
    %247 = stablehlo.slice %244 [0:257, 12:24, 0:256] : (tensor<257x24x256xf32>) -> tensor<257x12x256xf32>
    %248 = stablehlo.concatenate %245, %246, %247, dim = 1 : (tensor<257x12x256xf32>, tensor<257x233x256xf32>, tensor<257x12x256xf32>) -> tensor<257x257x256xf32>
    %cst_40 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %249 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<257x257x256xf32>
    %250 = stablehlo.multiply %249, %248 : tensor<257x257x256xf32>
    %251 = stablehlo.slice %210 [1:258, 0:257, 0:256] : (tensor<258x257x256xf32>) -> tensor<257x257x256xf32>
    %252 = stablehlo.slice %210 [0:257, 0:257, 0:256] : (tensor<258x257x256xf32>) -> tensor<257x257x256xf32>
    %253 = stablehlo.subtract %251, %252 : tensor<257x257x256xf32>
    %cst_41 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %254 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<257x257x256xf32>
    %255 = stablehlo.divide %253, %254 : tensor<257x257x256xf32>
    %256 = stablehlo.slice %255 [0:12, 0:257, 0:256] : (tensor<257x257x256xf32>) -> tensor<12x257x256xf32>
    %257 = stablehlo.slice %255 [245:257, 0:257, 0:256] : (tensor<257x257x256xf32>) -> tensor<12x257x256xf32>
    %258 = stablehlo.concatenate %256, %257, dim = 0 : (tensor<12x257x256xf32>, tensor<12x257x256xf32>) -> tensor<24x257x256xf32>
    %259 = stablehlo.broadcast_in_dim %arg45, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x257x256xf32>
    %260 = stablehlo.multiply %259, %arg135 : tensor<24x257x256xf32>
    %261 = stablehlo.broadcast_in_dim %arg46, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x257x256xf32>
    %262 = stablehlo.multiply %261, %258 : tensor<24x257x256xf32>
    %263 = stablehlo.add %260, %262 : tensor<24x257x256xf32>
    %264 = stablehlo.broadcast_in_dim %arg47, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x257x256xf32>
    %265 = stablehlo.multiply %258, %264 : tensor<24x257x256xf32>
    %266 = stablehlo.add %265, %263 : tensor<24x257x256xf32>
    %267 = stablehlo.slice %266 [0:12, 0:257, 0:256] : (tensor<24x257x256xf32>) -> tensor<12x257x256xf32>
    %268 = stablehlo.slice %255 [12:245, 0:257, 0:256] : (tensor<257x257x256xf32>) -> tensor<233x257x256xf32>
    %269 = stablehlo.slice %266 [12:24, 0:257, 0:256] : (tensor<24x257x256xf32>) -> tensor<12x257x256xf32>
    %270 = stablehlo.concatenate %267, %268, %269, dim = 0 : (tensor<12x257x256xf32>, tensor<233x257x256xf32>, tensor<12x257x256xf32>) -> tensor<257x257x256xf32>
    %cst_42 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %271 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<257x257x256xf32>
    %272 = stablehlo.multiply %271, %270 : tensor<257x257x256xf32>
    %273 = stablehlo.slice %214 [1:258, 0:256, 0:257] : (tensor<258x256x257xf32>) -> tensor<257x256x257xf32>
    %274 = stablehlo.slice %214 [0:257, 0:256, 0:257] : (tensor<258x256x257xf32>) -> tensor<257x256x257xf32>
    %275 = stablehlo.subtract %273, %274 : tensor<257x256x257xf32>
    %cst_43 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %276 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<257x256x257xf32>
    %277 = stablehlo.divide %275, %276 : tensor<257x256x257xf32>
    %278 = stablehlo.slice %277 [0:12, 0:256, 0:257] : (tensor<257x256x257xf32>) -> tensor<12x256x257xf32>
    %279 = stablehlo.slice %277 [245:257, 0:256, 0:257] : (tensor<257x256x257xf32>) -> tensor<12x256x257xf32>
    %280 = stablehlo.concatenate %278, %279, dim = 0 : (tensor<12x256x257xf32>, tensor<12x256x257xf32>) -> tensor<24x256x257xf32>
    %281 = stablehlo.broadcast_in_dim %arg48, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x256x257xf32>
    %282 = stablehlo.multiply %281, %arg136 : tensor<24x256x257xf32>
    %283 = stablehlo.broadcast_in_dim %arg49, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x256x257xf32>
    %284 = stablehlo.multiply %283, %280 : tensor<24x256x257xf32>
    %285 = stablehlo.add %282, %284 : tensor<24x256x257xf32>
    %286 = stablehlo.broadcast_in_dim %arg50, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x256x257xf32>
    %287 = stablehlo.multiply %280, %286 : tensor<24x256x257xf32>
    %288 = stablehlo.add %287, %285 : tensor<24x256x257xf32>
    %289 = stablehlo.slice %288 [0:12, 0:256, 0:257] : (tensor<24x256x257xf32>) -> tensor<12x256x257xf32>
    %290 = stablehlo.slice %277 [12:245, 0:256, 0:257] : (tensor<257x256x257xf32>) -> tensor<233x256x257xf32>
    %291 = stablehlo.slice %288 [12:24, 0:256, 0:257] : (tensor<24x256x257xf32>) -> tensor<12x256x257xf32>
    %292 = stablehlo.concatenate %289, %290, %291, dim = 0 : (tensor<12x256x257xf32>, tensor<233x256x257xf32>, tensor<12x256x257xf32>) -> tensor<257x256x257xf32>
    %cst_44 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %293 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<257x256x257xf32>
    %294 = stablehlo.multiply %293, %292 : tensor<257x256x257xf32>
    %295 = stablehlo.transpose %218, dims = [2, 0, 1] : (tensor<257x256x258xf32>) -> tensor<258x257x256xf32>
    %296 = stablehlo.slice %295 [1:258, 0:257, 0:256] : (tensor<258x257x256xf32>) -> tensor<257x257x256xf32>
    %297 = stablehlo.slice %295 [0:257, 0:257, 0:256] : (tensor<258x257x256xf32>) -> tensor<257x257x256xf32>
    %298 = stablehlo.subtract %296, %297 : tensor<257x257x256xf32>
    %cst_45 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %299 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<257x257x256xf32>
    %300 = stablehlo.divide %298, %299 : tensor<257x257x256xf32>
    %301 = stablehlo.transpose %300, dims = [1, 2, 0] : (tensor<257x257x256xf32>) -> tensor<257x256x257xf32>
    %302 = stablehlo.slice %301 [0:257, 0:256, 0:12] : (tensor<257x256x257xf32>) -> tensor<257x256x12xf32>
    %303 = stablehlo.slice %301 [0:257, 0:256, 245:257] : (tensor<257x256x257xf32>) -> tensor<257x256x12xf32>
    %304 = stablehlo.concatenate %302, %303, dim = 2 : (tensor<257x256x12xf32>, tensor<257x256x12xf32>) -> tensor<257x256x24xf32>
    %305 = stablehlo.broadcast_in_dim %arg51, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<257x256x24xf32>
    %306 = stablehlo.multiply %305, %arg137 : tensor<257x256x24xf32>
    %307 = stablehlo.broadcast_in_dim %arg52, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<257x256x24xf32>
    %308 = stablehlo.multiply %307, %304 : tensor<257x256x24xf32>
    %309 = stablehlo.add %306, %308 : tensor<257x256x24xf32>
    %310 = stablehlo.broadcast_in_dim %arg53, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<257x256x24xf32>
    %311 = stablehlo.multiply %304, %310 : tensor<257x256x24xf32>
    %312 = stablehlo.add %311, %309 : tensor<257x256x24xf32>
    %313 = stablehlo.slice %312 [0:257, 0:256, 0:12] : (tensor<257x256x24xf32>) -> tensor<257x256x12xf32>
    %314 = stablehlo.slice %301 [0:257, 0:256, 12:245] : (tensor<257x256x257xf32>) -> tensor<257x256x233xf32>
    %315 = stablehlo.slice %312 [0:257, 0:256, 12:24] : (tensor<257x256x24xf32>) -> tensor<257x256x12xf32>
    %316 = stablehlo.concatenate %313, %314, %315, dim = 2 : (tensor<257x256x12xf32>, tensor<257x256x233xf32>, tensor<257x256x12xf32>) -> tensor<257x256x257xf32>
    %cst_46 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %317 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<257x256x257xf32>
    %318 = stablehlo.multiply %317, %316 : tensor<257x256x257xf32>
    %319 = stablehlo.transpose %222, dims = [2, 0, 1] : (tensor<256x257x258xf32>) -> tensor<258x256x257xf32>
    %320 = stablehlo.slice %319 [1:258, 0:256, 0:257] : (tensor<258x256x257xf32>) -> tensor<257x256x257xf32>
    %321 = stablehlo.slice %319 [0:257, 0:256, 0:257] : (tensor<258x256x257xf32>) -> tensor<257x256x257xf32>
    %322 = stablehlo.subtract %320, %321 : tensor<257x256x257xf32>
    %cst_47 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %323 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<257x256x257xf32>
    %324 = stablehlo.divide %322, %323 : tensor<257x256x257xf32>
    %325 = stablehlo.transpose %324, dims = [1, 2, 0] : (tensor<257x256x257xf32>) -> tensor<256x257x257xf32>
    %326 = stablehlo.slice %325 [0:256, 0:257, 0:12] : (tensor<256x257x257xf32>) -> tensor<256x257x12xf32>
    %327 = stablehlo.slice %325 [0:256, 0:257, 245:257] : (tensor<256x257x257xf32>) -> tensor<256x257x12xf32>
    %328 = stablehlo.concatenate %326, %327, dim = 2 : (tensor<256x257x12xf32>, tensor<256x257x12xf32>) -> tensor<256x257x24xf32>
    %329 = stablehlo.broadcast_in_dim %arg54, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<256x257x24xf32>
    %330 = stablehlo.multiply %329, %arg138 : tensor<256x257x24xf32>
    %331 = stablehlo.broadcast_in_dim %arg55, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<256x257x24xf32>
    %332 = stablehlo.multiply %331, %328 : tensor<256x257x24xf32>
    %333 = stablehlo.add %330, %332 : tensor<256x257x24xf32>
    %334 = stablehlo.broadcast_in_dim %arg56, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<256x257x24xf32>
    %335 = stablehlo.multiply %328, %334 : tensor<256x257x24xf32>
    %336 = stablehlo.add %335, %333 : tensor<256x257x24xf32>
    %337 = stablehlo.slice %336 [0:256, 0:257, 0:12] : (tensor<256x257x24xf32>) -> tensor<256x257x12xf32>
    %338 = stablehlo.slice %325 [0:256, 0:257, 12:245] : (tensor<256x257x257xf32>) -> tensor<256x257x233xf32>
    %339 = stablehlo.slice %336 [0:256, 0:257, 12:24] : (tensor<256x257x24xf32>) -> tensor<256x257x12xf32>
    %340 = stablehlo.concatenate %337, %338, %339, dim = 2 : (tensor<256x257x12xf32>, tensor<256x257x233xf32>, tensor<256x257x12xf32>) -> tensor<256x257x257xf32>
    %cst_48 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %341 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<256x257x257xf32>
    %342 = stablehlo.multiply %341, %340 : tensor<256x257x257xf32>
    %343 = stablehlo.transpose %226, dims = [1, 0, 2] : (tensor<256x258x257xf32>) -> tensor<258x256x257xf32>
    %344 = stablehlo.slice %343 [1:258, 0:256, 0:257] : (tensor<258x256x257xf32>) -> tensor<257x256x257xf32>
    %345 = stablehlo.slice %343 [0:257, 0:256, 0:257] : (tensor<258x256x257xf32>) -> tensor<257x256x257xf32>
    %346 = stablehlo.subtract %344, %345 : tensor<257x256x257xf32>
    %cst_49 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %347 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<257x256x257xf32>
    %348 = stablehlo.divide %346, %347 : tensor<257x256x257xf32>
    %349 = stablehlo.transpose %348, dims = [1, 0, 2] : (tensor<257x256x257xf32>) -> tensor<256x257x257xf32>
    %350 = stablehlo.slice %349 [0:256, 0:12, 0:257] : (tensor<256x257x257xf32>) -> tensor<256x12x257xf32>
    %351 = stablehlo.slice %349 [0:256, 245:257, 0:257] : (tensor<256x257x257xf32>) -> tensor<256x12x257xf32>
    %352 = stablehlo.concatenate %350, %351, dim = 1 : (tensor<256x12x257xf32>, tensor<256x12x257xf32>) -> tensor<256x24x257xf32>
    %353 = stablehlo.broadcast_in_dim %arg57, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<256x24x257xf32>
    %354 = stablehlo.multiply %353, %arg139 : tensor<256x24x257xf32>
    %355 = stablehlo.broadcast_in_dim %arg58, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<256x24x257xf32>
    %356 = stablehlo.multiply %355, %352 : tensor<256x24x257xf32>
    %357 = stablehlo.add %354, %356 : tensor<256x24x257xf32>
    %358 = stablehlo.broadcast_in_dim %arg59, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<256x24x257xf32>
    %359 = stablehlo.multiply %352, %358 : tensor<256x24x257xf32>
    %360 = stablehlo.add %359, %357 : tensor<256x24x257xf32>
    %361 = stablehlo.slice %360 [0:256, 0:12, 0:257] : (tensor<256x24x257xf32>) -> tensor<256x12x257xf32>
    %362 = stablehlo.slice %349 [0:256, 12:245, 0:257] : (tensor<256x257x257xf32>) -> tensor<256x233x257xf32>
    %363 = stablehlo.slice %360 [0:256, 12:24, 0:257] : (tensor<256x24x257xf32>) -> tensor<256x12x257xf32>
    %364 = stablehlo.concatenate %361, %362, %363, dim = 1 : (tensor<256x12x257xf32>, tensor<256x233x257xf32>, tensor<256x12x257xf32>) -> tensor<256x257x257xf32>
    %cst_50 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %365 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<256x257x257xf32>
    %366 = stablehlo.multiply %365, %364 : tensor<256x257x257xf32>
    %367 = stablehlo.add %250, %272 : tensor<257x257x256xf32>
    %368 = stablehlo.add %294, %318 : tensor<257x256x257xf32>
    %369 = stablehlo.add %342, %366 : tensor<256x257x257xf32>
    %370 = stablehlo.broadcast_in_dim %arg60, dims = [] : (tensor<f32>) -> tensor<257x257x256xf32>
    %371 = stablehlo.multiply %370, %arg122 : tensor<257x257x256xf32>
    %372 = stablehlo.subtract %367, %371 : tensor<257x257x256xf32>
    %373 = stablehlo.multiply %arg61, %372 : tensor<257x257x256xf32>
    %374 = stablehlo.add %arg122, %373 : tensor<257x257x256xf32>
    %375 = stablehlo.broadcast_in_dim %arg62, dims = [] : (tensor<f32>) -> tensor<257x256x257xf32>
    %376 = stablehlo.multiply %375, %arg123 : tensor<257x256x257xf32>
    %377 = stablehlo.subtract %368, %376 : tensor<257x256x257xf32>
    %378 = stablehlo.multiply %arg63, %377 : tensor<257x256x257xf32>
    %379 = stablehlo.add %arg123, %378 : tensor<257x256x257xf32>
    %380 = stablehlo.broadcast_in_dim %arg64, dims = [] : (tensor<f32>) -> tensor<256x257x257xf32>
    %381 = stablehlo.multiply %380, %arg124 : tensor<256x257x257xf32>
    %382 = stablehlo.subtract %369, %381 : tensor<256x257x257xf32>
    %383 = stablehlo.multiply %arg65, %382 : tensor<256x257x257xf32>
    %384 = stablehlo.add %arg124, %383 : tensor<256x257x257xf32>
    %c_51 = stablehlo.constant dense<0> : tensor<i32>
    %c_52 = stablehlo.constant dense<127> : tensor<i32>
    %385 = call @clip(%arg147, %c_51, %c_52) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %386 = stablehlo.slice %arg66 [0:1, 0:15, 0:26, 0:1] : (tensor<2x15x26x1xf32>) -> tensor<1x15x26x1xf32>
    %387 = stablehlo.reshape %386 : (tensor<1x15x26x1xf32>) -> tensor<15x26x1xf32>
    %c_53 = stablehlo.constant dense<0> : tensor<i32>
    %388 = stablehlo.compare  LT, %385, %c_53,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_54 = stablehlo.constant dense<128> : tensor<i32>
    %389 = stablehlo.add %385, %c_54 : tensor<i32>
    %390 = stablehlo.select %388, %389, %385 : tensor<i1>, tensor<i32>
    %c_55 = stablehlo.constant dense<0> : tensor<i32>
    %391 = stablehlo.dynamic_slice %arg67, %c_55, %390, sizes = [1, 1] : (tensor<2x128xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %392 = stablehlo.reshape %391 : (tensor<1x1xf32>) -> tensor<f32>
    %393 = stablehlo.broadcast_in_dim %392, dims = [] : (tensor<f32>) -> tensor<15x26x1xf32>
    %394 = stablehlo.multiply %387, %393 : tensor<15x26x1xf32>
    %c_56 = stablehlo.constant dense<121> : tensor<i32>
    %395 = stablehlo.broadcast_in_dim %c_56, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_57 = stablehlo.constant dense<115> : tensor<i32>
    %396 = stablehlo.broadcast_in_dim %c_57, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_58 = stablehlo.constant dense<75> : tensor<i32>
    %397 = stablehlo.broadcast_in_dim %c_58, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %398 = stablehlo.concatenate %395, %396, %397, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %399 = "stablehlo.scatter"(%379, %398, %394) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      %568 = stablehlo.add %arg149, %arg150 : tensor<f32>
      stablehlo.return %568 : tensor<f32>
    }) : (tensor<257x256x257xf32>, tensor<3xi32>, tensor<15x26x1xf32>) -> tensor<257x256x257xf32>
    %400 = stablehlo.slice %arg66 [1:2, 0:15, 0:26, 0:1] : (tensor<2x15x26x1xf32>) -> tensor<1x15x26x1xf32>
    %401 = stablehlo.reshape %400 : (tensor<1x15x26x1xf32>) -> tensor<15x26x1xf32>
    %c_59 = stablehlo.constant dense<0> : tensor<i32>
    %402 = stablehlo.compare  LT, %385, %c_59,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_60 = stablehlo.constant dense<128> : tensor<i32>
    %403 = stablehlo.add %385, %c_60 : tensor<i32>
    %404 = stablehlo.select %402, %403, %385 : tensor<i1>, tensor<i32>
    %c_61 = stablehlo.constant dense<1> : tensor<i32>
    %405 = stablehlo.dynamic_slice %arg67, %c_61, %404, sizes = [1, 1] : (tensor<2x128xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %406 = stablehlo.reshape %405 : (tensor<1x1xf32>) -> tensor<f32>
    %407 = stablehlo.broadcast_in_dim %406, dims = [] : (tensor<f32>) -> tensor<15x26x1xf32>
    %408 = stablehlo.multiply %401, %407 : tensor<15x26x1xf32>
    %c_62 = stablehlo.constant dense<121> : tensor<i32>
    %409 = stablehlo.broadcast_in_dim %c_62, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_63 = stablehlo.constant dense<115> : tensor<i32>
    %410 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_64 = stablehlo.constant dense<75> : tensor<i32>
    %411 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %412 = stablehlo.concatenate %409, %410, %411, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %413 = "stablehlo.scatter"(%399, %412, %408) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      %568 = stablehlo.add %arg149, %arg150 : tensor<f32>
      stablehlo.return %568 : tensor<f32>
    }) : (tensor<257x256x257xf32>, tensor<3xi32>, tensor<15x26x1xf32>) -> tensor<257x256x257xf32>
    %c_65 = stablehlo.constant dense<0> : tensor<i32>
    %c_66 = stablehlo.constant dense<127> : tensor<i32>
    %414 = call @clip(%arg147, %c_65, %c_66) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %415 = stablehlo.slice %arg68 [0:1, 0:16, 0:25, 0:1] : (tensor<2x16x25x1xf32>) -> tensor<1x16x25x1xf32>
    %416 = stablehlo.reshape %415 : (tensor<1x16x25x1xf32>) -> tensor<16x25x1xf32>
    %c_67 = stablehlo.constant dense<0> : tensor<i32>
    %417 = stablehlo.compare  LT, %414, %c_67,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_68 = stablehlo.constant dense<128> : tensor<i32>
    %418 = stablehlo.add %414, %c_68 : tensor<i32>
    %419 = stablehlo.select %417, %418, %414 : tensor<i1>, tensor<i32>
    %c_69 = stablehlo.constant dense<0> : tensor<i32>
    %420 = stablehlo.dynamic_slice %arg69, %c_69, %419, sizes = [1, 1] : (tensor<2x128xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %421 = stablehlo.reshape %420 : (tensor<1x1xf32>) -> tensor<f32>
    %422 = stablehlo.broadcast_in_dim %421, dims = [] : (tensor<f32>) -> tensor<16x25x1xf32>
    %423 = stablehlo.multiply %416, %422 : tensor<16x25x1xf32>
    %c_70 = stablehlo.constant dense<120> : tensor<i32>
    %424 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_71 = stablehlo.constant dense<116> : tensor<i32>
    %425 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_72 = stablehlo.constant dense<75> : tensor<i32>
    %426 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %427 = stablehlo.concatenate %424, %425, %426, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %428 = "stablehlo.scatter"(%384, %427, %423) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      %568 = stablehlo.add %arg149, %arg150 : tensor<f32>
      stablehlo.return %568 : tensor<f32>
    }) : (tensor<256x257x257xf32>, tensor<3xi32>, tensor<16x25x1xf32>) -> tensor<256x257x257xf32>
    %429 = stablehlo.slice %arg68 [1:2, 0:16, 0:25, 0:1] : (tensor<2x16x25x1xf32>) -> tensor<1x16x25x1xf32>
    %430 = stablehlo.reshape %429 : (tensor<1x16x25x1xf32>) -> tensor<16x25x1xf32>
    %c_73 = stablehlo.constant dense<0> : tensor<i32>
    %431 = stablehlo.compare  LT, %414, %c_73,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_74 = stablehlo.constant dense<128> : tensor<i32>
    %432 = stablehlo.add %414, %c_74 : tensor<i32>
    %433 = stablehlo.select %431, %432, %414 : tensor<i1>, tensor<i32>
    %c_75 = stablehlo.constant dense<1> : tensor<i32>
    %434 = stablehlo.dynamic_slice %arg69, %c_75, %433, sizes = [1, 1] : (tensor<2x128xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %435 = stablehlo.reshape %434 : (tensor<1x1xf32>) -> tensor<f32>
    %436 = stablehlo.broadcast_in_dim %435, dims = [] : (tensor<f32>) -> tensor<16x25x1xf32>
    %437 = stablehlo.multiply %430, %436 : tensor<16x25x1xf32>
    %c_76 = stablehlo.constant dense<120> : tensor<i32>
    %438 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_77 = stablehlo.constant dense<116> : tensor<i32>
    %439 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_78 = stablehlo.constant dense<75> : tensor<i32>
    %440 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %441 = stablehlo.concatenate %438, %439, %440, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %442 = "stablehlo.scatter"(%428, %441, %437) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2], scatter_dims_to_operand_dims = [0, 1, 2]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      %568 = stablehlo.add %arg149, %arg150 : tensor<f32>
      stablehlo.return %568 : tensor<f32>
    }) : (tensor<256x257x257xf32>, tensor<3xi32>, tensor<16x25x1xf32>) -> tensor<256x257x257xf32>
    %c_79 = stablehlo.constant dense<1> : tensor<i32>
    %443 = stablehlo.add %arg147, %c_79 : tensor<i32>
    %c_80 = stablehlo.constant dense<1> : tensor<i32>
    %c_81 = stablehlo.constant dense<1> : tensor<i32>
    %444 = stablehlo.maximum %c_80, %c_81 : tensor<i32>
    %445 = call @remainder(%443, %444) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_82 = stablehlo.constant dense<0> : tensor<i32>
    %446 = stablehlo.compare  EQ, %445, %c_82,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %447 = stablehlo.slice %arg142 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %448 = stablehlo.reshape %447 : (tensor<1xi32>) -> tensor<i32>
    %c_83 = stablehlo.constant dense<1> : tensor<i32>
    %449 = stablehlo.compare  LT, %448, %c_83,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %450 = stablehlo.and %446, %449 : tensor<i1>
    %c_84 = stablehlo.constant dense<false> : tensor<i1>
    %451 = stablehlo.and %450, %c_84 : tensor<i1>
    %c_85 = stablehlo.constant dense<false> : tensor<i1>
    %452 = stablehlo.or %451, %c_85 : tensor<i1>
    %453 = stablehlo.convert %452 : (tensor<i1>) -> tensor<i32>
    %454 = "stablehlo.case"(%453) ({
      %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      stablehlo.return %cst_140 : tensor<f32>
    }, {
      %c_140 = stablehlo.constant dense<256> : tensor<i32>
      %568:2 = func.call @divmod(%arg70, %c_140) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_141 = stablehlo.constant dense<257> : tensor<i32>
      %569:2 = func.call @divmod(%568#0, %c_141) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_142 = stablehlo.constant dense<257> : tensor<i32>
      %570:2 = func.call @divmod(%569#0, %c_142) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_143 = stablehlo.constant dense<0> : tensor<i32>
      %571 = stablehlo.broadcast_in_dim %c_143, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %572 = stablehlo.compare  GT, %570#0, %571,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_144 = stablehlo.constant dense<-1> : tensor<i32>
      %573 = stablehlo.broadcast_in_dim %c_144, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %574 = stablehlo.compare  LT, %570#0, %573,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_145 = stablehlo.constant dense<0> : tensor<i32>
      %575 = func.call @_where_234(%574, %c_145, %570#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_146 = stablehlo.constant dense<256> : tensor<i32>
      %576 = func.call @_where_234(%572, %c_146, %575) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_147 = stablehlo.constant dense<0> : tensor<i32>
      %577 = func.call @_where_234(%574, %c_147, %569#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_148 = stablehlo.constant dense<256> : tensor<i32>
      %578 = func.call @_where_234(%572, %c_148, %577) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_149 = stablehlo.constant dense<0> : tensor<i32>
      %579 = func.call @_where_234(%574, %c_149, %568#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_150 = stablehlo.constant dense<255> : tensor<i32>
      %580 = func.call @_where_234(%572, %c_150, %579) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_151 = stablehlo.constant dense<0> : tensor<i32>
      %581 = stablehlo.broadcast_in_dim %c_151, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %582 = stablehlo.compare  LT, %576, %581,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_152 = stablehlo.constant dense<257> : tensor<i32>
      %583 = stablehlo.broadcast_in_dim %c_152, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %584 = stablehlo.add %576, %583 : tensor<200x1xi32>
      %585 = stablehlo.select %582, %584, %576 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_153 = stablehlo.constant dense<0> : tensor<i32>
      %586 = stablehlo.broadcast_in_dim %c_153, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %587 = stablehlo.compare  LT, %578, %586,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_154 = stablehlo.constant dense<257> : tensor<i32>
      %588 = stablehlo.broadcast_in_dim %c_154, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %589 = stablehlo.add %578, %588 : tensor<200x1xi32>
      %590 = stablehlo.select %587, %589, %578 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_155 = stablehlo.constant dense<0> : tensor<i32>
      %591 = stablehlo.broadcast_in_dim %c_155, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %592 = stablehlo.compare  LT, %580, %591,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_156 = stablehlo.constant dense<256> : tensor<i32>
      %593 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %594 = stablehlo.add %580, %593 : tensor<200x1xi32>
      %595 = stablehlo.select %592, %594, %580 : tensor<200x1xi1>, tensor<200x1xi32>
      %596 = stablehlo.broadcast_in_dim %585, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %597 = stablehlo.broadcast_in_dim %590, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %598 = stablehlo.broadcast_in_dim %595, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %599 = stablehlo.concatenate %596, %597, %598, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %600 = "stablehlo.gather"(%374, %599) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x256xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %601 = stablehlo.multiply %600, %arg71 : tensor<200x1xf32>
      %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %602 = stablehlo.reduce(%601 init: %cst_157) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_158 = stablehlo.constant dense<257> : tensor<i32>
      %603:2 = func.call @divmod(%arg72, %c_158) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_159 = stablehlo.constant dense<256> : tensor<i32>
      %604:2 = func.call @divmod(%603#0, %c_159) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_160 = stablehlo.constant dense<257> : tensor<i32>
      %605:2 = func.call @divmod(%604#0, %c_160) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_161 = stablehlo.constant dense<0> : tensor<i32>
      %606 = stablehlo.broadcast_in_dim %c_161, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %607 = stablehlo.compare  GT, %605#0, %606,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_162 = stablehlo.constant dense<-1> : tensor<i32>
      %608 = stablehlo.broadcast_in_dim %c_162, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %609 = stablehlo.compare  LT, %605#0, %608,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_163 = stablehlo.constant dense<0> : tensor<i32>
      %610 = func.call @_where_234(%609, %c_163, %605#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_164 = stablehlo.constant dense<256> : tensor<i32>
      %611 = func.call @_where_234(%607, %c_164, %610) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_165 = stablehlo.constant dense<0> : tensor<i32>
      %612 = func.call @_where_234(%609, %c_165, %604#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_166 = stablehlo.constant dense<255> : tensor<i32>
      %613 = func.call @_where_234(%607, %c_166, %612) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_167 = stablehlo.constant dense<0> : tensor<i32>
      %614 = func.call @_where_234(%609, %c_167, %603#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_168 = stablehlo.constant dense<256> : tensor<i32>
      %615 = func.call @_where_234(%607, %c_168, %614) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_169 = stablehlo.constant dense<0> : tensor<i32>
      %616 = stablehlo.broadcast_in_dim %c_169, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %617 = stablehlo.compare  LT, %611, %616,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_170 = stablehlo.constant dense<257> : tensor<i32>
      %618 = stablehlo.broadcast_in_dim %c_170, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %619 = stablehlo.add %611, %618 : tensor<200x1xi32>
      %620 = stablehlo.select %617, %619, %611 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_171 = stablehlo.constant dense<0> : tensor<i32>
      %621 = stablehlo.broadcast_in_dim %c_171, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %622 = stablehlo.compare  LT, %613, %621,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_172 = stablehlo.constant dense<256> : tensor<i32>
      %623 = stablehlo.broadcast_in_dim %c_172, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %624 = stablehlo.add %613, %623 : tensor<200x1xi32>
      %625 = stablehlo.select %622, %624, %613 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_173 = stablehlo.constant dense<0> : tensor<i32>
      %626 = stablehlo.broadcast_in_dim %c_173, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %627 = stablehlo.compare  LT, %615, %626,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_174 = stablehlo.constant dense<257> : tensor<i32>
      %628 = stablehlo.broadcast_in_dim %c_174, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %629 = stablehlo.add %615, %628 : tensor<200x1xi32>
      %630 = stablehlo.select %627, %629, %615 : tensor<200x1xi1>, tensor<200x1xi32>
      %631 = stablehlo.broadcast_in_dim %620, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %632 = stablehlo.broadcast_in_dim %625, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %633 = stablehlo.broadcast_in_dim %630, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %634 = stablehlo.concatenate %631, %632, %633, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %635 = "stablehlo.gather"(%413, %634) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %636 = stablehlo.multiply %635, %arg73 : tensor<200x1xf32>
      %cst_175 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %637 = stablehlo.reduce(%636 init: %cst_175) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_176 = stablehlo.constant dense<257> : tensor<i32>
      %638:2 = func.call @divmod(%arg74, %c_176) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_177 = stablehlo.constant dense<257> : tensor<i32>
      %639:2 = func.call @divmod(%638#0, %c_177) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_178 = stablehlo.constant dense<256> : tensor<i32>
      %640:2 = func.call @divmod(%639#0, %c_178) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_179 = stablehlo.constant dense<0> : tensor<i32>
      %641 = stablehlo.broadcast_in_dim %c_179, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %642 = stablehlo.compare  GT, %640#0, %641,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_180 = stablehlo.constant dense<-1> : tensor<i32>
      %643 = stablehlo.broadcast_in_dim %c_180, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %644 = stablehlo.compare  LT, %640#0, %643,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_181 = stablehlo.constant dense<0> : tensor<i32>
      %645 = func.call @_where_234(%644, %c_181, %640#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_182 = stablehlo.constant dense<255> : tensor<i32>
      %646 = func.call @_where_234(%642, %c_182, %645) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_183 = stablehlo.constant dense<0> : tensor<i32>
      %647 = func.call @_where_234(%644, %c_183, %639#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_184 = stablehlo.constant dense<256> : tensor<i32>
      %648 = func.call @_where_234(%642, %c_184, %647) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_185 = stablehlo.constant dense<0> : tensor<i32>
      %649 = func.call @_where_234(%644, %c_185, %638#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_186 = stablehlo.constant dense<256> : tensor<i32>
      %650 = func.call @_where_234(%642, %c_186, %649) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_187 = stablehlo.constant dense<0> : tensor<i32>
      %651 = stablehlo.broadcast_in_dim %c_187, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %652 = stablehlo.compare  LT, %646, %651,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_188 = stablehlo.constant dense<256> : tensor<i32>
      %653 = stablehlo.broadcast_in_dim %c_188, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %654 = stablehlo.add %646, %653 : tensor<200x1xi32>
      %655 = stablehlo.select %652, %654, %646 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_189 = stablehlo.constant dense<0> : tensor<i32>
      %656 = stablehlo.broadcast_in_dim %c_189, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %657 = stablehlo.compare  LT, %648, %656,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_190 = stablehlo.constant dense<257> : tensor<i32>
      %658 = stablehlo.broadcast_in_dim %c_190, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %659 = stablehlo.add %648, %658 : tensor<200x1xi32>
      %660 = stablehlo.select %657, %659, %648 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_191 = stablehlo.constant dense<0> : tensor<i32>
      %661 = stablehlo.broadcast_in_dim %c_191, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %662 = stablehlo.compare  LT, %650, %661,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_192 = stablehlo.constant dense<257> : tensor<i32>
      %663 = stablehlo.broadcast_in_dim %c_192, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %664 = stablehlo.add %650, %663 : tensor<200x1xi32>
      %665 = stablehlo.select %662, %664, %650 : tensor<200x1xi1>, tensor<200x1xi32>
      %666 = stablehlo.broadcast_in_dim %655, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %667 = stablehlo.broadcast_in_dim %660, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %668 = stablehlo.broadcast_in_dim %665, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %669 = stablehlo.concatenate %666, %667, %668, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %670 = "stablehlo.gather"(%442, %669) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %671 = stablehlo.multiply %670, %arg75 : tensor<200x1xf32>
      %cst_193 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %672 = stablehlo.reduce(%671 init: %cst_193) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_194 = stablehlo.constant dense<257> : tensor<i32>
      %673:2 = func.call @divmod(%arg76, %c_194) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_195 = stablehlo.constant dense<256> : tensor<i32>
      %674:2 = func.call @divmod(%673#0, %c_195) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_196 = stablehlo.constant dense<256> : tensor<i32>
      %675:2 = func.call @divmod(%674#0, %c_196) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_197 = stablehlo.constant dense<0> : tensor<i32>
      %676 = stablehlo.broadcast_in_dim %c_197, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %677 = stablehlo.compare  GT, %675#0, %676,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_198 = stablehlo.constant dense<-1> : tensor<i32>
      %678 = stablehlo.broadcast_in_dim %c_198, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %679 = stablehlo.compare  LT, %675#0, %678,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_199 = stablehlo.constant dense<0> : tensor<i32>
      %680 = func.call @_where_234(%679, %c_199, %675#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_200 = stablehlo.constant dense<255> : tensor<i32>
      %681 = func.call @_where_234(%677, %c_200, %680) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_201 = stablehlo.constant dense<0> : tensor<i32>
      %682 = func.call @_where_234(%679, %c_201, %674#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_202 = stablehlo.constant dense<255> : tensor<i32>
      %683 = func.call @_where_234(%677, %c_202, %682) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_203 = stablehlo.constant dense<0> : tensor<i32>
      %684 = func.call @_where_234(%679, %c_203, %673#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_204 = stablehlo.constant dense<256> : tensor<i32>
      %685 = func.call @_where_234(%677, %c_204, %684) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_205 = stablehlo.constant dense<0> : tensor<i32>
      %686 = stablehlo.broadcast_in_dim %c_205, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %687 = stablehlo.compare  LT, %681, %686,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_206 = stablehlo.constant dense<256> : tensor<i32>
      %688 = stablehlo.broadcast_in_dim %c_206, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %689 = stablehlo.add %681, %688 : tensor<200x1xi32>
      %690 = stablehlo.select %687, %689, %681 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_207 = stablehlo.constant dense<0> : tensor<i32>
      %691 = stablehlo.broadcast_in_dim %c_207, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %692 = stablehlo.compare  LT, %683, %691,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_208 = stablehlo.constant dense<256> : tensor<i32>
      %693 = stablehlo.broadcast_in_dim %c_208, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %694 = stablehlo.add %683, %693 : tensor<200x1xi32>
      %695 = stablehlo.select %692, %694, %683 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_209 = stablehlo.constant dense<0> : tensor<i32>
      %696 = stablehlo.broadcast_in_dim %c_209, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %697 = stablehlo.compare  LT, %685, %696,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_210 = stablehlo.constant dense<257> : tensor<i32>
      %698 = stablehlo.broadcast_in_dim %c_210, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %699 = stablehlo.add %685, %698 : tensor<200x1xi32>
      %700 = stablehlo.select %697, %699, %685 : tensor<200x1xi1>, tensor<200x1xi32>
      %701 = stablehlo.broadcast_in_dim %690, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %702 = stablehlo.broadcast_in_dim %695, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %703 = stablehlo.broadcast_in_dim %700, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %704 = stablehlo.concatenate %701, %702, %703, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %705 = "stablehlo.gather"(%132, %704) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %706 = stablehlo.multiply %705, %arg77 : tensor<200x1xf32>
      %cst_211 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %707 = stablehlo.reduce(%706 init: %cst_211) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_212 = stablehlo.constant dense<256> : tensor<i32>
      %708:2 = func.call @divmod(%arg78, %c_212) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_213 = stablehlo.constant dense<257> : tensor<i32>
      %709:2 = func.call @divmod(%708#0, %c_213) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_214 = stablehlo.constant dense<256> : tensor<i32>
      %710:2 = func.call @divmod(%709#0, %c_214) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_215 = stablehlo.constant dense<0> : tensor<i32>
      %711 = stablehlo.broadcast_in_dim %c_215, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %712 = stablehlo.compare  GT, %710#0, %711,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_216 = stablehlo.constant dense<-1> : tensor<i32>
      %713 = stablehlo.broadcast_in_dim %c_216, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %714 = stablehlo.compare  LT, %710#0, %713,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_217 = stablehlo.constant dense<0> : tensor<i32>
      %715 = func.call @_where_234(%714, %c_217, %710#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_218 = stablehlo.constant dense<255> : tensor<i32>
      %716 = func.call @_where_234(%712, %c_218, %715) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_219 = stablehlo.constant dense<0> : tensor<i32>
      %717 = func.call @_where_234(%714, %c_219, %709#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_220 = stablehlo.constant dense<256> : tensor<i32>
      %718 = func.call @_where_234(%712, %c_220, %717) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_221 = stablehlo.constant dense<0> : tensor<i32>
      %719 = func.call @_where_234(%714, %c_221, %708#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_222 = stablehlo.constant dense<255> : tensor<i32>
      %720 = func.call @_where_234(%712, %c_222, %719) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_223 = stablehlo.constant dense<0> : tensor<i32>
      %721 = stablehlo.broadcast_in_dim %c_223, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %722 = stablehlo.compare  LT, %716, %721,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_224 = stablehlo.constant dense<256> : tensor<i32>
      %723 = stablehlo.broadcast_in_dim %c_224, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %724 = stablehlo.add %716, %723 : tensor<200x1xi32>
      %725 = stablehlo.select %722, %724, %716 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_225 = stablehlo.constant dense<0> : tensor<i32>
      %726 = stablehlo.broadcast_in_dim %c_225, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %727 = stablehlo.compare  LT, %718, %726,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_226 = stablehlo.constant dense<257> : tensor<i32>
      %728 = stablehlo.broadcast_in_dim %c_226, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %729 = stablehlo.add %718, %728 : tensor<200x1xi32>
      %730 = stablehlo.select %727, %729, %718 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_227 = stablehlo.constant dense<0> : tensor<i32>
      %731 = stablehlo.broadcast_in_dim %c_227, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %732 = stablehlo.compare  LT, %720, %731,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_228 = stablehlo.constant dense<256> : tensor<i32>
      %733 = stablehlo.broadcast_in_dim %c_228, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %734 = stablehlo.add %720, %733 : tensor<200x1xi32>
      %735 = stablehlo.select %732, %734, %720 : tensor<200x1xi1>, tensor<200x1xi32>
      %736 = stablehlo.broadcast_in_dim %725, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %737 = stablehlo.broadcast_in_dim %730, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %738 = stablehlo.broadcast_in_dim %735, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %739 = stablehlo.concatenate %736, %737, %738, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %740 = "stablehlo.gather"(%173, %739) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x256xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %741 = stablehlo.multiply %740, %arg79 : tensor<200x1xf32>
      %cst_229 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %742 = stablehlo.reduce(%741 init: %cst_229) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_230 = stablehlo.constant dense<256> : tensor<i32>
      %743:2 = func.call @divmod(%arg80, %c_230) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_231 = stablehlo.constant dense<256> : tensor<i32>
      %744:2 = func.call @divmod(%743#0, %c_231) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_232 = stablehlo.constant dense<257> : tensor<i32>
      %745:2 = func.call @divmod(%744#0, %c_232) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_233 = stablehlo.constant dense<0> : tensor<i32>
      %746 = stablehlo.broadcast_in_dim %c_233, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %747 = stablehlo.compare  GT, %745#0, %746,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_234 = stablehlo.constant dense<-1> : tensor<i32>
      %748 = stablehlo.broadcast_in_dim %c_234, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %749 = stablehlo.compare  LT, %745#0, %748,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_235 = stablehlo.constant dense<0> : tensor<i32>
      %750 = func.call @_where_234(%749, %c_235, %745#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_236 = stablehlo.constant dense<256> : tensor<i32>
      %751 = func.call @_where_234(%747, %c_236, %750) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_237 = stablehlo.constant dense<0> : tensor<i32>
      %752 = func.call @_where_234(%749, %c_237, %744#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_238 = stablehlo.constant dense<255> : tensor<i32>
      %753 = func.call @_where_234(%747, %c_238, %752) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_239 = stablehlo.constant dense<0> : tensor<i32>
      %754 = func.call @_where_234(%749, %c_239, %743#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_240 = stablehlo.constant dense<255> : tensor<i32>
      %755 = func.call @_where_234(%747, %c_240, %754) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_241 = stablehlo.constant dense<0> : tensor<i32>
      %756 = stablehlo.broadcast_in_dim %c_241, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %757 = stablehlo.compare  LT, %751, %756,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_242 = stablehlo.constant dense<257> : tensor<i32>
      %758 = stablehlo.broadcast_in_dim %c_242, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %759 = stablehlo.add %751, %758 : tensor<200x1xi32>
      %760 = stablehlo.select %757, %759, %751 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_243 = stablehlo.constant dense<0> : tensor<i32>
      %761 = stablehlo.broadcast_in_dim %c_243, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %762 = stablehlo.compare  LT, %753, %761,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_244 = stablehlo.constant dense<256> : tensor<i32>
      %763 = stablehlo.broadcast_in_dim %c_244, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %764 = stablehlo.add %753, %763 : tensor<200x1xi32>
      %765 = stablehlo.select %762, %764, %753 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_245 = stablehlo.constant dense<0> : tensor<i32>
      %766 = stablehlo.broadcast_in_dim %c_245, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %767 = stablehlo.compare  LT, %755, %766,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_246 = stablehlo.constant dense<256> : tensor<i32>
      %768 = stablehlo.broadcast_in_dim %c_246, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %769 = stablehlo.add %755, %768 : tensor<200x1xi32>
      %770 = stablehlo.select %767, %769, %755 : tensor<200x1xi1>, tensor<200x1xi32>
      %771 = stablehlo.broadcast_in_dim %760, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %772 = stablehlo.broadcast_in_dim %765, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %773 = stablehlo.broadcast_in_dim %770, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %774 = stablehlo.concatenate %771, %772, %773, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %775 = "stablehlo.gather"(%202, %774) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x256xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %776 = stablehlo.multiply %775, %arg81 : tensor<200x1xf32>
      %cst_247 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %777 = stablehlo.reduce(%776 init: %cst_247) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %778 = stablehlo.broadcast_in_dim %602, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %779 = stablehlo.broadcast_in_dim %637, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %780 = stablehlo.broadcast_in_dim %672, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %781 = stablehlo.broadcast_in_dim %707, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %782 = stablehlo.broadcast_in_dim %742, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %783 = stablehlo.broadcast_in_dim %777, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %784 = stablehlo.concatenate %778, %779, %780, %781, %782, %783, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %785 = stablehlo.slice %784 [1:2, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %786 = stablehlo.reshape %785 : (tensor<1x200xf32>) -> tensor<200xf32>
      %787 = stablehlo.slice %784 [2:3, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %788 = stablehlo.reshape %787 : (tensor<1x200xf32>) -> tensor<200xf32>
      %789 = stablehlo.slice %784 [4:5, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %790 = stablehlo.reshape %789 : (tensor<1x200xf32>) -> tensor<200xf32>
      %791 = stablehlo.slice %784 [5:6, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %792 = stablehlo.reshape %791 : (tensor<1x200xf32>) -> tensor<200xf32>
      %793 = stablehlo.multiply %786, %792 : tensor<200xf32>
      %794 = stablehlo.multiply %788, %790 : tensor<200xf32>
      %795 = stablehlo.subtract %793, %794 : tensor<200xf32>
      %cst_248 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %796 = stablehlo.broadcast_in_dim %cst_248, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %797 = stablehlo.multiply %795, %796 : tensor<200xf32>
      %cst_249 = stablehlo.constant dense<7.485380e-15> : tensor<f32>
      %798 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %799 = stablehlo.multiply %797, %798 : tensor<200xf32>
      %cst_250 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %800 = stablehlo.reduce(%799 init: %cst_250) applies stablehlo.add across dimensions = [0] : (tensor<200xf32>, tensor<f32>) -> tensor<f32>
      stablehlo.return %800 : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %cst_86 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %455 = call @_where_256(%451, %454, %cst_86) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %456 = stablehlo.slice %arg142 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %457 = stablehlo.reshape %456 : (tensor<1xi32>) -> tensor<i32>
    %c_87 = stablehlo.constant dense<0> : tensor<i32>
    %458 = stablehlo.minimum %457, %c_87 : tensor<i32>
    %c_88 = stablehlo.constant dense<0> : tensor<i32>
    %459 = stablehlo.compare  LT, %458, %c_88,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_89 = stablehlo.constant dense<1> : tensor<i32>
    %460 = stablehlo.add %458, %c_89 : tensor<i32>
    %461 = stablehlo.select %459, %460, %458 : tensor<i1>, tensor<i32>
    %c_90 = stablehlo.constant dense<0> : tensor<i32>
    %462 = stablehlo.dynamic_slice %arg140, %c_90, %461, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %463 = stablehlo.reshape %462 : (tensor<1x1xf32>) -> tensor<f32>
    %c_91 = stablehlo.constant dense<0> : tensor<i32>
    %464 = stablehlo.compare  LT, %458, %c_91,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_92 = stablehlo.constant dense<1> : tensor<i32>
    %465 = stablehlo.add %458, %c_92 : tensor<i32>
    %466 = stablehlo.select %464, %465, %458 : tensor<i1>, tensor<i32>
    %c_93 = stablehlo.constant dense<0> : tensor<i32>
    %467 = stablehlo.dynamic_slice %arg141, %c_93, %466, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %468 = stablehlo.reshape %467 : (tensor<1x1xf32>) -> tensor<f32>
    %469 = call @_where_256(%451, %455, %463) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_94 = stablehlo.constant dense<0> : tensor<i32>
    %470 = stablehlo.compare  LT, %458, %c_94,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_95 = stablehlo.constant dense<1> : tensor<i32>
    %471 = stablehlo.add %458, %c_95 : tensor<i32>
    %472 = stablehlo.select %470, %471, %458 : tensor<i1>, tensor<i32>
    %c_96 = stablehlo.constant dense<0> : tensor<i32>
    %473 = stablehlo.broadcast_in_dim %c_96, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %474 = stablehlo.broadcast_in_dim %472, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %475 = stablehlo.concatenate %473, %474, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %476 = "stablehlo.scatter"(%arg140, %475, %469) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      stablehlo.return %arg150 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %477 = call @_where_256(%451, %3, %468) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_97 = stablehlo.constant dense<0> : tensor<i32>
    %478 = stablehlo.compare  LT, %458, %c_97,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_98 = stablehlo.constant dense<1> : tensor<i32>
    %479 = stablehlo.add %458, %c_98 : tensor<i32>
    %480 = stablehlo.select %478, %479, %458 : tensor<i1>, tensor<i32>
    %c_99 = stablehlo.constant dense<0> : tensor<i32>
    %481 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %482 = stablehlo.broadcast_in_dim %480, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %483 = stablehlo.concatenate %481, %482, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %484 = "stablehlo.scatter"(%arg141, %483, %477) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      stablehlo.return %arg150 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %485 = stablehlo.slice %arg142 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %486 = stablehlo.reshape %485 : (tensor<1xi32>) -> tensor<i32>
    %c_100 = stablehlo.constant dense<1> : tensor<i32>
    %c_101 = stablehlo.constant dense<0> : tensor<i32>
    %487 = call @_where_260(%451, %c_100, %c_101) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %488 = stablehlo.convert %487 : tensor<i32>
    %489 = stablehlo.add %486, %488 : tensor<i32>
    %c_102 = stablehlo.constant dense<0> : tensor<i32>
    %490 = stablehlo.broadcast_in_dim %c_102, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %491 = "stablehlo.scatter"(%arg142, %490, %489) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<i32>, %arg150: tensor<i32>):
      stablehlo.return %arg150 : tensor<i32>
    }) : (tensor<2xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
    %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %492 = stablehlo.compare  GE, %3, %cst_103,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %c_104 = stablehlo.constant dense<true> : tensor<i1>
    %493 = stablehlo.and %c_104, %492 : tensor<i1>
    %cst_105 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %494 = stablehlo.compare  LE, %3, %cst_105,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %495 = stablehlo.and %493, %494 : tensor<i1>
    %c_106 = stablehlo.constant dense<1> : tensor<i32>
    %c_107 = stablehlo.constant dense<1> : tensor<i32>
    %496 = stablehlo.maximum %c_106, %c_107 : tensor<i32>
    %497 = call @remainder(%arg147, %496) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_108 = stablehlo.constant dense<0> : tensor<i32>
    %498 = stablehlo.compare  EQ, %497, %c_108,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %499 = stablehlo.and %495, %498 : tensor<i1>
    %500 = stablehlo.convert %499 : (tensor<i1>) -> tensor<i32>
    %501:3 = "stablehlo.case"(%500) ({
      stablehlo.return %arg143, %arg144, %arg145 : tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>
    }, {
      %cst_140 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %568 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %569 = stablehlo.multiply %568, %arg82 : tensor<3xf32>
      %570 = stablehlo.optimization_barrier %569 : tensor<3xf32>
      %571 = stablehlo.optimization_barrier %3 : tensor<f32>
      %572 = stablehlo.broadcast_in_dim %571, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %573 = stablehlo.multiply %570, %572 : tensor<3xf32>
      %574 = stablehlo.cosine %573 : tensor<3xf32>
      %575 = stablehlo.sine %573 : tensor<3xf32>
      %cst_141 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %576 = stablehlo.subtract %cst_141, %cst_142 : tensor<f32>
      %cst_143 = stablehlo.constant dense<1.000000e-30> : tensor<f32>
      %577 = stablehlo.maximum %576, %cst_143 : tensor<f32>
      %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %578 = stablehlo.subtract %3, %cst_144 : tensor<f32>
      %579 = stablehlo.divide %578, %577 : tensor<f32>
      %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %cst_146 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %580 = func.call @clip_271(%579, %cst_145, %cst_146) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_147 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %581 = stablehlo.multiply %cst_147, %580 : tensor<f32>
      %582 = stablehlo.cosine %581 : tensor<f32>
      %cst_148 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %583 = stablehlo.subtract %cst_148, %582 : tensor<f32>
      %cst_149 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %584 = stablehlo.multiply %cst_149, %583 : tensor<f32>
      %cst_150 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %585 = stablehlo.is_finite %cst_150 : (tensor<f32>) -> tensor<i1>
      %c_151 = stablehlo.constant dense<false> : tensor<i1>
      %586 = stablehlo.and %c_151, %585 : tensor<i1>
      %cst_152 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_153 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %587 = stablehlo.compare  GT, %cst_152, %cst_153,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %588 = stablehlo.and %586, %587 : tensor<i1>
      %cst_154 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %589 = func.call @_where_276(%588, %584, %cst_154) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %590 = stablehlo.maximum %589, %cst_155 : tensor<f32>
      %cst_156 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %591 = stablehlo.multiply %arg0, %cst_156 : tensor<f32>
      %cst_157 = stablehlo.constant dense<0x4D8EF3C2> : tensor<f32>
      %592 = stablehlo.multiply %591, %cst_157 : tensor<f32>
      %cst_158 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %593 = stablehlo.divide %592, %cst_158 : tensor<f32>
      %cst_159 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %594 = stablehlo.sqrt %cst_159 : tensor<f32>
      %595 = stablehlo.divide %593, %594 : tensor<f32>
      %596 = stablehlo.multiply %590, %595 : tensor<f32>
      %c_160 = stablehlo.constant dense<0> : tensor<i32>
      %c_161 = stablehlo.constant dense<1> : tensor<i32>
      %597 = stablehlo.compare  EQ, %c_160, %c_161,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %598 = func.call @_where_278(%597, %596, %590) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %c_162 = stablehlo.constant dense<256> : tensor<i32>
      %599:2 = func.call @divmod_280(%arg83, %c_162) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_163 = stablehlo.constant dense<257> : tensor<i32>
      %600:2 = func.call @divmod_280(%599#0, %c_163) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_164 = stablehlo.constant dense<257> : tensor<i32>
      %601:2 = func.call @divmod_280(%600#0, %c_164) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_165 = stablehlo.constant dense<0> : tensor<i32>
      %602 = stablehlo.broadcast_in_dim %c_165, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %603 = stablehlo.compare  GT, %601#0, %602,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_166 = stablehlo.constant dense<-1> : tensor<i32>
      %604 = stablehlo.broadcast_in_dim %c_166, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %605 = stablehlo.compare  LT, %601#0, %604,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_167 = stablehlo.constant dense<0> : tensor<i32>
      %606 = func.call @_where_295(%605, %c_167, %601#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_168 = stablehlo.constant dense<256> : tensor<i32>
      %607 = func.call @_where_295(%603, %c_168, %606) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_169 = stablehlo.constant dense<0> : tensor<i32>
      %608 = func.call @_where_295(%605, %c_169, %600#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_170 = stablehlo.constant dense<256> : tensor<i32>
      %609 = func.call @_where_295(%603, %c_170, %608) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_171 = stablehlo.constant dense<0> : tensor<i32>
      %610 = func.call @_where_295(%605, %c_171, %599#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_172 = stablehlo.constant dense<255> : tensor<i32>
      %611 = func.call @_where_295(%603, %c_172, %610) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_173 = stablehlo.constant dense<0> : tensor<i32>
      %612 = stablehlo.broadcast_in_dim %c_173, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %613 = stablehlo.compare  LT, %607, %612,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_174 = stablehlo.constant dense<257> : tensor<i32>
      %614 = stablehlo.broadcast_in_dim %c_174, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %615 = stablehlo.add %607, %614 : tensor<200x8xi32>
      %616 = stablehlo.select %613, %615, %607 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_175 = stablehlo.constant dense<0> : tensor<i32>
      %617 = stablehlo.broadcast_in_dim %c_175, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %618 = stablehlo.compare  LT, %609, %617,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_176 = stablehlo.constant dense<257> : tensor<i32>
      %619 = stablehlo.broadcast_in_dim %c_176, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %620 = stablehlo.add %609, %619 : tensor<200x8xi32>
      %621 = stablehlo.select %618, %620, %609 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_177 = stablehlo.constant dense<0> : tensor<i32>
      %622 = stablehlo.broadcast_in_dim %c_177, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %623 = stablehlo.compare  LT, %611, %622,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_178 = stablehlo.constant dense<256> : tensor<i32>
      %624 = stablehlo.broadcast_in_dim %c_178, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %625 = stablehlo.add %611, %624 : tensor<200x8xi32>
      %626 = stablehlo.select %623, %625, %611 : tensor<200x8xi1>, tensor<200x8xi32>
      %627 = stablehlo.broadcast_in_dim %616, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %628 = stablehlo.broadcast_in_dim %621, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %629 = stablehlo.broadcast_in_dim %626, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %630 = stablehlo.concatenate %627, %628, %629, dim = 2 : (tensor<200x8x1xi32>, tensor<200x8x1xi32>, tensor<200x8x1xi32>) -> tensor<200x8x3xi32>
      %631 = "stablehlo.gather"(%374, %630) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x256xf32>, tensor<200x8x3xi32>) -> tensor<200x8xf32>
      %632 = stablehlo.multiply %631, %arg84 : tensor<200x8xf32>
      %cst_179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %633 = stablehlo.reduce(%632 init: %cst_179) applies stablehlo.add across dimensions = [1] : (tensor<200x8xf32>, tensor<f32>) -> tensor<200xf32>
      %c_180 = stablehlo.constant dense<257> : tensor<i32>
      %634:2 = func.call @divmod_302(%arg85, %c_180) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_181 = stablehlo.constant dense<256> : tensor<i32>
      %635:2 = func.call @divmod_302(%634#0, %c_181) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_182 = stablehlo.constant dense<257> : tensor<i32>
      %636:2 = func.call @divmod_302(%635#0, %c_182) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_183 = stablehlo.constant dense<0> : tensor<i32>
      %637 = stablehlo.broadcast_in_dim %c_183, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %638 = stablehlo.compare  GT, %636#0, %637,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_184 = stablehlo.constant dense<-1> : tensor<i32>
      %639 = stablehlo.broadcast_in_dim %c_184, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %640 = stablehlo.compare  LT, %636#0, %639,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_185 = stablehlo.constant dense<0> : tensor<i32>
      %641 = func.call @_where_317(%640, %c_185, %636#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_186 = stablehlo.constant dense<256> : tensor<i32>
      %642 = func.call @_where_317(%638, %c_186, %641) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_187 = stablehlo.constant dense<0> : tensor<i32>
      %643 = func.call @_where_317(%640, %c_187, %635#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_188 = stablehlo.constant dense<255> : tensor<i32>
      %644 = func.call @_where_317(%638, %c_188, %643) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_189 = stablehlo.constant dense<0> : tensor<i32>
      %645 = func.call @_where_317(%640, %c_189, %634#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_190 = stablehlo.constant dense<256> : tensor<i32>
      %646 = func.call @_where_317(%638, %c_190, %645) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_191 = stablehlo.constant dense<0> : tensor<i32>
      %647 = stablehlo.broadcast_in_dim %c_191, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %648 = stablehlo.compare  LT, %642, %647,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_192 = stablehlo.constant dense<257> : tensor<i32>
      %649 = stablehlo.broadcast_in_dim %c_192, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %650 = stablehlo.add %642, %649 : tensor<200x4xi32>
      %651 = stablehlo.select %648, %650, %642 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_193 = stablehlo.constant dense<0> : tensor<i32>
      %652 = stablehlo.broadcast_in_dim %c_193, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %653 = stablehlo.compare  LT, %644, %652,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_194 = stablehlo.constant dense<256> : tensor<i32>
      %654 = stablehlo.broadcast_in_dim %c_194, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %655 = stablehlo.add %644, %654 : tensor<200x4xi32>
      %656 = stablehlo.select %653, %655, %644 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_195 = stablehlo.constant dense<0> : tensor<i32>
      %657 = stablehlo.broadcast_in_dim %c_195, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %658 = stablehlo.compare  LT, %646, %657,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_196 = stablehlo.constant dense<257> : tensor<i32>
      %659 = stablehlo.broadcast_in_dim %c_196, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %660 = stablehlo.add %646, %659 : tensor<200x4xi32>
      %661 = stablehlo.select %658, %660, %646 : tensor<200x4xi1>, tensor<200x4xi32>
      %662 = stablehlo.broadcast_in_dim %651, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %663 = stablehlo.broadcast_in_dim %656, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %664 = stablehlo.broadcast_in_dim %661, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %665 = stablehlo.concatenate %662, %663, %664, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %666 = "stablehlo.gather"(%413, %665) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %667 = stablehlo.multiply %666, %arg86 : tensor<200x4xf32>
      %cst_197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %668 = stablehlo.reduce(%667 init: %cst_197) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_198 = stablehlo.constant dense<257> : tensor<i32>
      %669:2 = func.call @divmod_302(%arg87, %c_198) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_199 = stablehlo.constant dense<257> : tensor<i32>
      %670:2 = func.call @divmod_302(%669#0, %c_199) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_200 = stablehlo.constant dense<256> : tensor<i32>
      %671:2 = func.call @divmod_302(%670#0, %c_200) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_201 = stablehlo.constant dense<0> : tensor<i32>
      %672 = stablehlo.broadcast_in_dim %c_201, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %673 = stablehlo.compare  GT, %671#0, %672,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_202 = stablehlo.constant dense<-1> : tensor<i32>
      %674 = stablehlo.broadcast_in_dim %c_202, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %675 = stablehlo.compare  LT, %671#0, %674,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_203 = stablehlo.constant dense<0> : tensor<i32>
      %676 = func.call @_where_317(%675, %c_203, %671#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_204 = stablehlo.constant dense<255> : tensor<i32>
      %677 = func.call @_where_317(%673, %c_204, %676) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_205 = stablehlo.constant dense<0> : tensor<i32>
      %678 = func.call @_where_317(%675, %c_205, %670#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_206 = stablehlo.constant dense<256> : tensor<i32>
      %679 = func.call @_where_317(%673, %c_206, %678) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_207 = stablehlo.constant dense<0> : tensor<i32>
      %680 = func.call @_where_317(%675, %c_207, %669#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_208 = stablehlo.constant dense<256> : tensor<i32>
      %681 = func.call @_where_317(%673, %c_208, %680) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_209 = stablehlo.constant dense<0> : tensor<i32>
      %682 = stablehlo.broadcast_in_dim %c_209, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %683 = stablehlo.compare  LT, %677, %682,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_210 = stablehlo.constant dense<256> : tensor<i32>
      %684 = stablehlo.broadcast_in_dim %c_210, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %685 = stablehlo.add %677, %684 : tensor<200x4xi32>
      %686 = stablehlo.select %683, %685, %677 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_211 = stablehlo.constant dense<0> : tensor<i32>
      %687 = stablehlo.broadcast_in_dim %c_211, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %688 = stablehlo.compare  LT, %679, %687,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_212 = stablehlo.constant dense<257> : tensor<i32>
      %689 = stablehlo.broadcast_in_dim %c_212, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %690 = stablehlo.add %679, %689 : tensor<200x4xi32>
      %691 = stablehlo.select %688, %690, %679 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_213 = stablehlo.constant dense<0> : tensor<i32>
      %692 = stablehlo.broadcast_in_dim %c_213, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %693 = stablehlo.compare  LT, %681, %692,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_214 = stablehlo.constant dense<257> : tensor<i32>
      %694 = stablehlo.broadcast_in_dim %c_214, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %695 = stablehlo.add %681, %694 : tensor<200x4xi32>
      %696 = stablehlo.select %693, %695, %681 : tensor<200x4xi1>, tensor<200x4xi32>
      %697 = stablehlo.broadcast_in_dim %686, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %698 = stablehlo.broadcast_in_dim %691, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %699 = stablehlo.broadcast_in_dim %696, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %700 = stablehlo.concatenate %697, %698, %699, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %701 = "stablehlo.gather"(%442, %700) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %702 = stablehlo.multiply %701, %arg88 : tensor<200x4xf32>
      %cst_215 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %703 = stablehlo.reduce(%702 init: %cst_215) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_216 = stablehlo.constant dense<257> : tensor<i32>
      %704:2 = func.call @divmod_325(%arg89, %c_216) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_217 = stablehlo.constant dense<256> : tensor<i32>
      %705:2 = func.call @divmod_325(%704#0, %c_217) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_218 = stablehlo.constant dense<256> : tensor<i32>
      %706:2 = func.call @divmod_325(%705#0, %c_218) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_219 = stablehlo.constant dense<0> : tensor<i32>
      %707 = stablehlo.broadcast_in_dim %c_219, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %708 = stablehlo.compare  GT, %706#0, %707,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_220 = stablehlo.constant dense<-1> : tensor<i32>
      %709 = stablehlo.broadcast_in_dim %c_220, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %710 = stablehlo.compare  LT, %706#0, %709,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_221 = stablehlo.constant dense<0> : tensor<i32>
      %711 = func.call @_where_340(%710, %c_221, %706#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_222 = stablehlo.constant dense<255> : tensor<i32>
      %712 = func.call @_where_340(%708, %c_222, %711) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_223 = stablehlo.constant dense<0> : tensor<i32>
      %713 = func.call @_where_340(%710, %c_223, %705#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_224 = stablehlo.constant dense<255> : tensor<i32>
      %714 = func.call @_where_340(%708, %c_224, %713) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_225 = stablehlo.constant dense<0> : tensor<i32>
      %715 = func.call @_where_340(%710, %c_225, %704#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_226 = stablehlo.constant dense<256> : tensor<i32>
      %716 = func.call @_where_340(%708, %c_226, %715) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_227 = stablehlo.constant dense<0> : tensor<i32>
      %717 = stablehlo.broadcast_in_dim %c_227, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %718 = stablehlo.compare  LT, %712, %717,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_228 = stablehlo.constant dense<256> : tensor<i32>
      %719 = stablehlo.broadcast_in_dim %c_228, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %720 = stablehlo.add %712, %719 : tensor<200x2xi32>
      %721 = stablehlo.select %718, %720, %712 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_229 = stablehlo.constant dense<0> : tensor<i32>
      %722 = stablehlo.broadcast_in_dim %c_229, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %723 = stablehlo.compare  LT, %714, %722,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_230 = stablehlo.constant dense<256> : tensor<i32>
      %724 = stablehlo.broadcast_in_dim %c_230, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %725 = stablehlo.add %714, %724 : tensor<200x2xi32>
      %726 = stablehlo.select %723, %725, %714 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_231 = stablehlo.constant dense<0> : tensor<i32>
      %727 = stablehlo.broadcast_in_dim %c_231, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %728 = stablehlo.compare  LT, %716, %727,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_232 = stablehlo.constant dense<257> : tensor<i32>
      %729 = stablehlo.broadcast_in_dim %c_232, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %730 = stablehlo.add %716, %729 : tensor<200x2xi32>
      %731 = stablehlo.select %728, %730, %716 : tensor<200x2xi1>, tensor<200x2xi32>
      %732 = stablehlo.broadcast_in_dim %721, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %733 = stablehlo.broadcast_in_dim %726, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %734 = stablehlo.broadcast_in_dim %731, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %735 = stablehlo.concatenate %732, %733, %734, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
      %736 = "stablehlo.gather"(%132, %735) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
      %737 = stablehlo.multiply %736, %arg90 : tensor<200x2xf32>
      %cst_233 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %738 = stablehlo.reduce(%737 init: %cst_233) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
      %c_234 = stablehlo.constant dense<256> : tensor<i32>
      %739:2 = func.call @divmod_302(%arg91, %c_234) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_235 = stablehlo.constant dense<257> : tensor<i32>
      %740:2 = func.call @divmod_302(%739#0, %c_235) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_236 = stablehlo.constant dense<256> : tensor<i32>
      %741:2 = func.call @divmod_302(%740#0, %c_236) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_237 = stablehlo.constant dense<0> : tensor<i32>
      %742 = stablehlo.broadcast_in_dim %c_237, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %743 = stablehlo.compare  GT, %741#0, %742,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_238 = stablehlo.constant dense<-1> : tensor<i32>
      %744 = stablehlo.broadcast_in_dim %c_238, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %745 = stablehlo.compare  LT, %741#0, %744,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_239 = stablehlo.constant dense<0> : tensor<i32>
      %746 = func.call @_where_317(%745, %c_239, %741#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_240 = stablehlo.constant dense<255> : tensor<i32>
      %747 = func.call @_where_317(%743, %c_240, %746) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_241 = stablehlo.constant dense<0> : tensor<i32>
      %748 = func.call @_where_317(%745, %c_241, %740#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_242 = stablehlo.constant dense<256> : tensor<i32>
      %749 = func.call @_where_317(%743, %c_242, %748) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_243 = stablehlo.constant dense<0> : tensor<i32>
      %750 = func.call @_where_317(%745, %c_243, %739#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_244 = stablehlo.constant dense<255> : tensor<i32>
      %751 = func.call @_where_317(%743, %c_244, %750) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_245 = stablehlo.constant dense<0> : tensor<i32>
      %752 = stablehlo.broadcast_in_dim %c_245, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %753 = stablehlo.compare  LT, %747, %752,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_246 = stablehlo.constant dense<256> : tensor<i32>
      %754 = stablehlo.broadcast_in_dim %c_246, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %755 = stablehlo.add %747, %754 : tensor<200x4xi32>
      %756 = stablehlo.select %753, %755, %747 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_247 = stablehlo.constant dense<0> : tensor<i32>
      %757 = stablehlo.broadcast_in_dim %c_247, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %758 = stablehlo.compare  LT, %749, %757,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_248 = stablehlo.constant dense<257> : tensor<i32>
      %759 = stablehlo.broadcast_in_dim %c_248, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %760 = stablehlo.add %749, %759 : tensor<200x4xi32>
      %761 = stablehlo.select %758, %760, %749 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_249 = stablehlo.constant dense<0> : tensor<i32>
      %762 = stablehlo.broadcast_in_dim %c_249, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %763 = stablehlo.compare  LT, %751, %762,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_250 = stablehlo.constant dense<256> : tensor<i32>
      %764 = stablehlo.broadcast_in_dim %c_250, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %765 = stablehlo.add %751, %764 : tensor<200x4xi32>
      %766 = stablehlo.select %763, %765, %751 : tensor<200x4xi1>, tensor<200x4xi32>
      %767 = stablehlo.broadcast_in_dim %756, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %768 = stablehlo.broadcast_in_dim %761, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %769 = stablehlo.broadcast_in_dim %766, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %770 = stablehlo.concatenate %767, %768, %769, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %771 = "stablehlo.gather"(%173, %770) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x256xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %772 = stablehlo.multiply %771, %arg92 : tensor<200x4xf32>
      %cst_251 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %773 = stablehlo.reduce(%772 init: %cst_251) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_252 = stablehlo.constant dense<256> : tensor<i32>
      %774:2 = func.call @divmod_302(%arg93, %c_252) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_253 = stablehlo.constant dense<256> : tensor<i32>
      %775:2 = func.call @divmod_302(%774#0, %c_253) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_254 = stablehlo.constant dense<257> : tensor<i32>
      %776:2 = func.call @divmod_302(%775#0, %c_254) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_255 = stablehlo.constant dense<0> : tensor<i32>
      %777 = stablehlo.broadcast_in_dim %c_255, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %778 = stablehlo.compare  GT, %776#0, %777,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_256 = stablehlo.constant dense<-1> : tensor<i32>
      %779 = stablehlo.broadcast_in_dim %c_256, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %780 = stablehlo.compare  LT, %776#0, %779,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_257 = stablehlo.constant dense<0> : tensor<i32>
      %781 = func.call @_where_317(%780, %c_257, %776#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_258 = stablehlo.constant dense<256> : tensor<i32>
      %782 = func.call @_where_317(%778, %c_258, %781) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_259 = stablehlo.constant dense<0> : tensor<i32>
      %783 = func.call @_where_317(%780, %c_259, %775#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_260 = stablehlo.constant dense<255> : tensor<i32>
      %784 = func.call @_where_317(%778, %c_260, %783) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_261 = stablehlo.constant dense<0> : tensor<i32>
      %785 = func.call @_where_317(%780, %c_261, %774#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_262 = stablehlo.constant dense<255> : tensor<i32>
      %786 = func.call @_where_317(%778, %c_262, %785) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_263 = stablehlo.constant dense<0> : tensor<i32>
      %787 = stablehlo.broadcast_in_dim %c_263, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %788 = stablehlo.compare  LT, %782, %787,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_264 = stablehlo.constant dense<257> : tensor<i32>
      %789 = stablehlo.broadcast_in_dim %c_264, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %790 = stablehlo.add %782, %789 : tensor<200x4xi32>
      %791 = stablehlo.select %788, %790, %782 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_265 = stablehlo.constant dense<0> : tensor<i32>
      %792 = stablehlo.broadcast_in_dim %c_265, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %793 = stablehlo.compare  LT, %784, %792,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_266 = stablehlo.constant dense<256> : tensor<i32>
      %794 = stablehlo.broadcast_in_dim %c_266, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %795 = stablehlo.add %784, %794 : tensor<200x4xi32>
      %796 = stablehlo.select %793, %795, %784 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_267 = stablehlo.constant dense<0> : tensor<i32>
      %797 = stablehlo.broadcast_in_dim %c_267, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %798 = stablehlo.compare  LT, %786, %797,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_268 = stablehlo.constant dense<256> : tensor<i32>
      %799 = stablehlo.broadcast_in_dim %c_268, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %800 = stablehlo.add %786, %799 : tensor<200x4xi32>
      %801 = stablehlo.select %798, %800, %786 : tensor<200x4xi1>, tensor<200x4xi32>
      %802 = stablehlo.broadcast_in_dim %791, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %803 = stablehlo.broadcast_in_dim %796, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %804 = stablehlo.broadcast_in_dim %801, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %805 = stablehlo.concatenate %802, %803, %804, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %806 = "stablehlo.gather"(%202, %805) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x256xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %807 = stablehlo.multiply %806, %arg94 : tensor<200x4xf32>
      %cst_269 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %808 = stablehlo.reduce(%807 init: %cst_269) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %809 = stablehlo.broadcast_in_dim %633, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %810 = stablehlo.broadcast_in_dim %668, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %811 = stablehlo.broadcast_in_dim %703, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %812 = stablehlo.broadcast_in_dim %738, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %813 = stablehlo.broadcast_in_dim %773, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %814 = stablehlo.broadcast_in_dim %808, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %815 = stablehlo.concatenate %809, %810, %811, %812, %813, %814, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %816 = stablehlo.broadcast_in_dim %arg95, dims = [0] : (tensor<6xf32>) -> tensor<6x1x1xf32>
      %817 = stablehlo.broadcast_in_dim %598, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %818 = stablehlo.multiply %817, %816 : tensor<6x1x1xf32>
      %819 = stablehlo.dot_general %815, %574, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %820 = stablehlo.transpose %819, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %821 = stablehlo.broadcast_in_dim %818, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %822 = stablehlo.multiply %821, %820 : tensor<6x3x200xf32>
      %823 = stablehlo.broadcast_in_dim %598, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %824 = stablehlo.multiply %823, %816 : tensor<6x1x1xf32>
      %825 = stablehlo.dot_general %815, %575, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %826 = stablehlo.transpose %825, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %827 = stablehlo.broadcast_in_dim %824, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %828 = stablehlo.multiply %827, %826 : tensor<6x3x200xf32>
      %829 = stablehlo.reshape %822 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_270 = stablehlo.constant dense<0> : tensor<i32>
      %830 = stablehlo.broadcast_in_dim %c_270, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %831 = "stablehlo.scatter"(%arg143, %830, %829) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
        %838 = stablehlo.add %arg149, %arg150 : tensor<f32>
        stablehlo.return %838 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %832 = stablehlo.reshape %828 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_271 = stablehlo.constant dense<0> : tensor<i32>
      %833 = stablehlo.broadcast_in_dim %c_271, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %834 = "stablehlo.scatter"(%arg144, %833, %832) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
        %838 = stablehlo.add %arg149, %arg150 : tensor<f32>
        stablehlo.return %838 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %835 = stablehlo.broadcast_in_dim %589, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %c_272 = stablehlo.constant dense<0> : tensor<i32>
      %836 = stablehlo.broadcast_in_dim %c_272, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %837 = "stablehlo.scatter"(%arg145, %836, %835) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
        %838 = stablehlo.add %arg149, %arg150 : tensor<f32>
        stablehlo.return %838 : tensor<f32>
      }) : (tensor<6xf32>, tensor<1xi32>, tensor<3xf32>) -> tensor<6xf32>
      stablehlo.return %831, %834, %837 : tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>
    }) : (tensor<i32>) -> (tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>)
    %c_109 = stablehlo.constant dense<1> : tensor<i32>
    %502 = stablehlo.add %arg147, %c_109 : tensor<i32>
    %c_110 = stablehlo.constant dense<1> : tensor<i32>
    %c_111 = stablehlo.constant dense<1> : tensor<i32>
    %503 = stablehlo.maximum %c_110, %c_111 : tensor<i32>
    %504 = call @remainder(%502, %503) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_112 = stablehlo.constant dense<0> : tensor<i32>
    %505 = stablehlo.compare  EQ, %504, %c_112,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %506 = stablehlo.slice %491 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %507 = stablehlo.reshape %506 : (tensor<1xi32>) -> tensor<i32>
    %c_113 = stablehlo.constant dense<1> : tensor<i32>
    %508 = stablehlo.compare  LT, %507, %c_113,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %509 = stablehlo.and %505, %508 : tensor<i1>
    %c_114 = stablehlo.constant dense<false> : tensor<i1>
    %510 = stablehlo.and %509, %c_114 : tensor<i1>
    %c_115 = stablehlo.constant dense<false> : tensor<i1>
    %511 = stablehlo.or %510, %c_115 : tensor<i1>
    %512 = stablehlo.convert %511 : (tensor<i1>) -> tensor<i32>
    %513 = "stablehlo.case"(%512) ({
      %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      stablehlo.return %cst_140 : tensor<f32>
    }, {
      %c_140 = stablehlo.constant dense<256> : tensor<i32>
      %568:2 = func.call @divmod(%arg96, %c_140) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_141 = stablehlo.constant dense<257> : tensor<i32>
      %569:2 = func.call @divmod(%568#0, %c_141) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_142 = stablehlo.constant dense<257> : tensor<i32>
      %570:2 = func.call @divmod(%569#0, %c_142) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_143 = stablehlo.constant dense<0> : tensor<i32>
      %571 = stablehlo.broadcast_in_dim %c_143, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %572 = stablehlo.compare  GT, %570#0, %571,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_144 = stablehlo.constant dense<-1> : tensor<i32>
      %573 = stablehlo.broadcast_in_dim %c_144, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %574 = stablehlo.compare  LT, %570#0, %573,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_145 = stablehlo.constant dense<0> : tensor<i32>
      %575 = func.call @_where_234(%574, %c_145, %570#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_146 = stablehlo.constant dense<256> : tensor<i32>
      %576 = func.call @_where_234(%572, %c_146, %575) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_147 = stablehlo.constant dense<0> : tensor<i32>
      %577 = func.call @_where_234(%574, %c_147, %569#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_148 = stablehlo.constant dense<256> : tensor<i32>
      %578 = func.call @_where_234(%572, %c_148, %577) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_149 = stablehlo.constant dense<0> : tensor<i32>
      %579 = func.call @_where_234(%574, %c_149, %568#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_150 = stablehlo.constant dense<255> : tensor<i32>
      %580 = func.call @_where_234(%572, %c_150, %579) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_151 = stablehlo.constant dense<0> : tensor<i32>
      %581 = stablehlo.broadcast_in_dim %c_151, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %582 = stablehlo.compare  LT, %576, %581,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_152 = stablehlo.constant dense<257> : tensor<i32>
      %583 = stablehlo.broadcast_in_dim %c_152, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %584 = stablehlo.add %576, %583 : tensor<200x1xi32>
      %585 = stablehlo.select %582, %584, %576 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_153 = stablehlo.constant dense<0> : tensor<i32>
      %586 = stablehlo.broadcast_in_dim %c_153, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %587 = stablehlo.compare  LT, %578, %586,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_154 = stablehlo.constant dense<257> : tensor<i32>
      %588 = stablehlo.broadcast_in_dim %c_154, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %589 = stablehlo.add %578, %588 : tensor<200x1xi32>
      %590 = stablehlo.select %587, %589, %578 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_155 = stablehlo.constant dense<0> : tensor<i32>
      %591 = stablehlo.broadcast_in_dim %c_155, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %592 = stablehlo.compare  LT, %580, %591,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_156 = stablehlo.constant dense<256> : tensor<i32>
      %593 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %594 = stablehlo.add %580, %593 : tensor<200x1xi32>
      %595 = stablehlo.select %592, %594, %580 : tensor<200x1xi1>, tensor<200x1xi32>
      %596 = stablehlo.broadcast_in_dim %585, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %597 = stablehlo.broadcast_in_dim %590, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %598 = stablehlo.broadcast_in_dim %595, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %599 = stablehlo.concatenate %596, %597, %598, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %600 = "stablehlo.gather"(%374, %599) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x256xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %601 = stablehlo.multiply %600, %arg97 : tensor<200x1xf32>
      %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %602 = stablehlo.reduce(%601 init: %cst_157) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_158 = stablehlo.constant dense<257> : tensor<i32>
      %603:2 = func.call @divmod(%arg98, %c_158) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_159 = stablehlo.constant dense<256> : tensor<i32>
      %604:2 = func.call @divmod(%603#0, %c_159) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_160 = stablehlo.constant dense<257> : tensor<i32>
      %605:2 = func.call @divmod(%604#0, %c_160) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_161 = stablehlo.constant dense<0> : tensor<i32>
      %606 = stablehlo.broadcast_in_dim %c_161, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %607 = stablehlo.compare  GT, %605#0, %606,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_162 = stablehlo.constant dense<-1> : tensor<i32>
      %608 = stablehlo.broadcast_in_dim %c_162, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %609 = stablehlo.compare  LT, %605#0, %608,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_163 = stablehlo.constant dense<0> : tensor<i32>
      %610 = func.call @_where_234(%609, %c_163, %605#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_164 = stablehlo.constant dense<256> : tensor<i32>
      %611 = func.call @_where_234(%607, %c_164, %610) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_165 = stablehlo.constant dense<0> : tensor<i32>
      %612 = func.call @_where_234(%609, %c_165, %604#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_166 = stablehlo.constant dense<255> : tensor<i32>
      %613 = func.call @_where_234(%607, %c_166, %612) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_167 = stablehlo.constant dense<0> : tensor<i32>
      %614 = func.call @_where_234(%609, %c_167, %603#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_168 = stablehlo.constant dense<256> : tensor<i32>
      %615 = func.call @_where_234(%607, %c_168, %614) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_169 = stablehlo.constant dense<0> : tensor<i32>
      %616 = stablehlo.broadcast_in_dim %c_169, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %617 = stablehlo.compare  LT, %611, %616,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_170 = stablehlo.constant dense<257> : tensor<i32>
      %618 = stablehlo.broadcast_in_dim %c_170, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %619 = stablehlo.add %611, %618 : tensor<200x1xi32>
      %620 = stablehlo.select %617, %619, %611 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_171 = stablehlo.constant dense<0> : tensor<i32>
      %621 = stablehlo.broadcast_in_dim %c_171, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %622 = stablehlo.compare  LT, %613, %621,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_172 = stablehlo.constant dense<256> : tensor<i32>
      %623 = stablehlo.broadcast_in_dim %c_172, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %624 = stablehlo.add %613, %623 : tensor<200x1xi32>
      %625 = stablehlo.select %622, %624, %613 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_173 = stablehlo.constant dense<0> : tensor<i32>
      %626 = stablehlo.broadcast_in_dim %c_173, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %627 = stablehlo.compare  LT, %615, %626,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_174 = stablehlo.constant dense<257> : tensor<i32>
      %628 = stablehlo.broadcast_in_dim %c_174, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %629 = stablehlo.add %615, %628 : tensor<200x1xi32>
      %630 = stablehlo.select %627, %629, %615 : tensor<200x1xi1>, tensor<200x1xi32>
      %631 = stablehlo.broadcast_in_dim %620, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %632 = stablehlo.broadcast_in_dim %625, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %633 = stablehlo.broadcast_in_dim %630, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %634 = stablehlo.concatenate %631, %632, %633, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %635 = "stablehlo.gather"(%413, %634) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %636 = stablehlo.multiply %635, %arg99 : tensor<200x1xf32>
      %cst_175 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %637 = stablehlo.reduce(%636 init: %cst_175) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_176 = stablehlo.constant dense<257> : tensor<i32>
      %638:2 = func.call @divmod(%arg100, %c_176) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_177 = stablehlo.constant dense<257> : tensor<i32>
      %639:2 = func.call @divmod(%638#0, %c_177) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_178 = stablehlo.constant dense<256> : tensor<i32>
      %640:2 = func.call @divmod(%639#0, %c_178) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_179 = stablehlo.constant dense<0> : tensor<i32>
      %641 = stablehlo.broadcast_in_dim %c_179, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %642 = stablehlo.compare  GT, %640#0, %641,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_180 = stablehlo.constant dense<-1> : tensor<i32>
      %643 = stablehlo.broadcast_in_dim %c_180, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %644 = stablehlo.compare  LT, %640#0, %643,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_181 = stablehlo.constant dense<0> : tensor<i32>
      %645 = func.call @_where_234(%644, %c_181, %640#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_182 = stablehlo.constant dense<255> : tensor<i32>
      %646 = func.call @_where_234(%642, %c_182, %645) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_183 = stablehlo.constant dense<0> : tensor<i32>
      %647 = func.call @_where_234(%644, %c_183, %639#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_184 = stablehlo.constant dense<256> : tensor<i32>
      %648 = func.call @_where_234(%642, %c_184, %647) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_185 = stablehlo.constant dense<0> : tensor<i32>
      %649 = func.call @_where_234(%644, %c_185, %638#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_186 = stablehlo.constant dense<256> : tensor<i32>
      %650 = func.call @_where_234(%642, %c_186, %649) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_187 = stablehlo.constant dense<0> : tensor<i32>
      %651 = stablehlo.broadcast_in_dim %c_187, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %652 = stablehlo.compare  LT, %646, %651,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_188 = stablehlo.constant dense<256> : tensor<i32>
      %653 = stablehlo.broadcast_in_dim %c_188, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %654 = stablehlo.add %646, %653 : tensor<200x1xi32>
      %655 = stablehlo.select %652, %654, %646 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_189 = stablehlo.constant dense<0> : tensor<i32>
      %656 = stablehlo.broadcast_in_dim %c_189, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %657 = stablehlo.compare  LT, %648, %656,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_190 = stablehlo.constant dense<257> : tensor<i32>
      %658 = stablehlo.broadcast_in_dim %c_190, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %659 = stablehlo.add %648, %658 : tensor<200x1xi32>
      %660 = stablehlo.select %657, %659, %648 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_191 = stablehlo.constant dense<0> : tensor<i32>
      %661 = stablehlo.broadcast_in_dim %c_191, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %662 = stablehlo.compare  LT, %650, %661,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_192 = stablehlo.constant dense<257> : tensor<i32>
      %663 = stablehlo.broadcast_in_dim %c_192, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %664 = stablehlo.add %650, %663 : tensor<200x1xi32>
      %665 = stablehlo.select %662, %664, %650 : tensor<200x1xi1>, tensor<200x1xi32>
      %666 = stablehlo.broadcast_in_dim %655, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %667 = stablehlo.broadcast_in_dim %660, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %668 = stablehlo.broadcast_in_dim %665, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %669 = stablehlo.concatenate %666, %667, %668, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %670 = "stablehlo.gather"(%442, %669) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %671 = stablehlo.multiply %670, %arg101 : tensor<200x1xf32>
      %cst_193 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %672 = stablehlo.reduce(%671 init: %cst_193) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_194 = stablehlo.constant dense<257> : tensor<i32>
      %673:2 = func.call @divmod(%arg102, %c_194) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_195 = stablehlo.constant dense<256> : tensor<i32>
      %674:2 = func.call @divmod(%673#0, %c_195) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_196 = stablehlo.constant dense<256> : tensor<i32>
      %675:2 = func.call @divmod(%674#0, %c_196) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_197 = stablehlo.constant dense<0> : tensor<i32>
      %676 = stablehlo.broadcast_in_dim %c_197, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %677 = stablehlo.compare  GT, %675#0, %676,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_198 = stablehlo.constant dense<-1> : tensor<i32>
      %678 = stablehlo.broadcast_in_dim %c_198, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %679 = stablehlo.compare  LT, %675#0, %678,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_199 = stablehlo.constant dense<0> : tensor<i32>
      %680 = func.call @_where_234(%679, %c_199, %675#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_200 = stablehlo.constant dense<255> : tensor<i32>
      %681 = func.call @_where_234(%677, %c_200, %680) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_201 = stablehlo.constant dense<0> : tensor<i32>
      %682 = func.call @_where_234(%679, %c_201, %674#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_202 = stablehlo.constant dense<255> : tensor<i32>
      %683 = func.call @_where_234(%677, %c_202, %682) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_203 = stablehlo.constant dense<0> : tensor<i32>
      %684 = func.call @_where_234(%679, %c_203, %673#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_204 = stablehlo.constant dense<256> : tensor<i32>
      %685 = func.call @_where_234(%677, %c_204, %684) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_205 = stablehlo.constant dense<0> : tensor<i32>
      %686 = stablehlo.broadcast_in_dim %c_205, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %687 = stablehlo.compare  LT, %681, %686,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_206 = stablehlo.constant dense<256> : tensor<i32>
      %688 = stablehlo.broadcast_in_dim %c_206, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %689 = stablehlo.add %681, %688 : tensor<200x1xi32>
      %690 = stablehlo.select %687, %689, %681 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_207 = stablehlo.constant dense<0> : tensor<i32>
      %691 = stablehlo.broadcast_in_dim %c_207, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %692 = stablehlo.compare  LT, %683, %691,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_208 = stablehlo.constant dense<256> : tensor<i32>
      %693 = stablehlo.broadcast_in_dim %c_208, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %694 = stablehlo.add %683, %693 : tensor<200x1xi32>
      %695 = stablehlo.select %692, %694, %683 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_209 = stablehlo.constant dense<0> : tensor<i32>
      %696 = stablehlo.broadcast_in_dim %c_209, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %697 = stablehlo.compare  LT, %685, %696,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_210 = stablehlo.constant dense<257> : tensor<i32>
      %698 = stablehlo.broadcast_in_dim %c_210, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %699 = stablehlo.add %685, %698 : tensor<200x1xi32>
      %700 = stablehlo.select %697, %699, %685 : tensor<200x1xi1>, tensor<200x1xi32>
      %701 = stablehlo.broadcast_in_dim %690, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %702 = stablehlo.broadcast_in_dim %695, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %703 = stablehlo.broadcast_in_dim %700, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %704 = stablehlo.concatenate %701, %702, %703, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %705 = "stablehlo.gather"(%132, %704) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %706 = stablehlo.multiply %705, %arg103 : tensor<200x1xf32>
      %cst_211 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %707 = stablehlo.reduce(%706 init: %cst_211) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_212 = stablehlo.constant dense<256> : tensor<i32>
      %708:2 = func.call @divmod(%arg104, %c_212) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_213 = stablehlo.constant dense<257> : tensor<i32>
      %709:2 = func.call @divmod(%708#0, %c_213) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_214 = stablehlo.constant dense<256> : tensor<i32>
      %710:2 = func.call @divmod(%709#0, %c_214) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_215 = stablehlo.constant dense<0> : tensor<i32>
      %711 = stablehlo.broadcast_in_dim %c_215, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %712 = stablehlo.compare  GT, %710#0, %711,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_216 = stablehlo.constant dense<-1> : tensor<i32>
      %713 = stablehlo.broadcast_in_dim %c_216, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %714 = stablehlo.compare  LT, %710#0, %713,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_217 = stablehlo.constant dense<0> : tensor<i32>
      %715 = func.call @_where_234(%714, %c_217, %710#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_218 = stablehlo.constant dense<255> : tensor<i32>
      %716 = func.call @_where_234(%712, %c_218, %715) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_219 = stablehlo.constant dense<0> : tensor<i32>
      %717 = func.call @_where_234(%714, %c_219, %709#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_220 = stablehlo.constant dense<256> : tensor<i32>
      %718 = func.call @_where_234(%712, %c_220, %717) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_221 = stablehlo.constant dense<0> : tensor<i32>
      %719 = func.call @_where_234(%714, %c_221, %708#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_222 = stablehlo.constant dense<255> : tensor<i32>
      %720 = func.call @_where_234(%712, %c_222, %719) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_223 = stablehlo.constant dense<0> : tensor<i32>
      %721 = stablehlo.broadcast_in_dim %c_223, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %722 = stablehlo.compare  LT, %716, %721,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_224 = stablehlo.constant dense<256> : tensor<i32>
      %723 = stablehlo.broadcast_in_dim %c_224, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %724 = stablehlo.add %716, %723 : tensor<200x1xi32>
      %725 = stablehlo.select %722, %724, %716 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_225 = stablehlo.constant dense<0> : tensor<i32>
      %726 = stablehlo.broadcast_in_dim %c_225, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %727 = stablehlo.compare  LT, %718, %726,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_226 = stablehlo.constant dense<257> : tensor<i32>
      %728 = stablehlo.broadcast_in_dim %c_226, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %729 = stablehlo.add %718, %728 : tensor<200x1xi32>
      %730 = stablehlo.select %727, %729, %718 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_227 = stablehlo.constant dense<0> : tensor<i32>
      %731 = stablehlo.broadcast_in_dim %c_227, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %732 = stablehlo.compare  LT, %720, %731,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_228 = stablehlo.constant dense<256> : tensor<i32>
      %733 = stablehlo.broadcast_in_dim %c_228, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %734 = stablehlo.add %720, %733 : tensor<200x1xi32>
      %735 = stablehlo.select %732, %734, %720 : tensor<200x1xi1>, tensor<200x1xi32>
      %736 = stablehlo.broadcast_in_dim %725, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %737 = stablehlo.broadcast_in_dim %730, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %738 = stablehlo.broadcast_in_dim %735, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %739 = stablehlo.concatenate %736, %737, %738, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %740 = "stablehlo.gather"(%173, %739) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x256xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %741 = stablehlo.multiply %740, %arg105 : tensor<200x1xf32>
      %cst_229 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %742 = stablehlo.reduce(%741 init: %cst_229) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_230 = stablehlo.constant dense<256> : tensor<i32>
      %743:2 = func.call @divmod(%arg106, %c_230) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_231 = stablehlo.constant dense<256> : tensor<i32>
      %744:2 = func.call @divmod(%743#0, %c_231) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_232 = stablehlo.constant dense<257> : tensor<i32>
      %745:2 = func.call @divmod(%744#0, %c_232) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_233 = stablehlo.constant dense<0> : tensor<i32>
      %746 = stablehlo.broadcast_in_dim %c_233, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %747 = stablehlo.compare  GT, %745#0, %746,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_234 = stablehlo.constant dense<-1> : tensor<i32>
      %748 = stablehlo.broadcast_in_dim %c_234, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %749 = stablehlo.compare  LT, %745#0, %748,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_235 = stablehlo.constant dense<0> : tensor<i32>
      %750 = func.call @_where_234(%749, %c_235, %745#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_236 = stablehlo.constant dense<256> : tensor<i32>
      %751 = func.call @_where_234(%747, %c_236, %750) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_237 = stablehlo.constant dense<0> : tensor<i32>
      %752 = func.call @_where_234(%749, %c_237, %744#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_238 = stablehlo.constant dense<255> : tensor<i32>
      %753 = func.call @_where_234(%747, %c_238, %752) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_239 = stablehlo.constant dense<0> : tensor<i32>
      %754 = func.call @_where_234(%749, %c_239, %743#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_240 = stablehlo.constant dense<255> : tensor<i32>
      %755 = func.call @_where_234(%747, %c_240, %754) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_241 = stablehlo.constant dense<0> : tensor<i32>
      %756 = stablehlo.broadcast_in_dim %c_241, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %757 = stablehlo.compare  LT, %751, %756,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_242 = stablehlo.constant dense<257> : tensor<i32>
      %758 = stablehlo.broadcast_in_dim %c_242, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %759 = stablehlo.add %751, %758 : tensor<200x1xi32>
      %760 = stablehlo.select %757, %759, %751 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_243 = stablehlo.constant dense<0> : tensor<i32>
      %761 = stablehlo.broadcast_in_dim %c_243, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %762 = stablehlo.compare  LT, %753, %761,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_244 = stablehlo.constant dense<256> : tensor<i32>
      %763 = stablehlo.broadcast_in_dim %c_244, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %764 = stablehlo.add %753, %763 : tensor<200x1xi32>
      %765 = stablehlo.select %762, %764, %753 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_245 = stablehlo.constant dense<0> : tensor<i32>
      %766 = stablehlo.broadcast_in_dim %c_245, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %767 = stablehlo.compare  LT, %755, %766,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_246 = stablehlo.constant dense<256> : tensor<i32>
      %768 = stablehlo.broadcast_in_dim %c_246, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %769 = stablehlo.add %755, %768 : tensor<200x1xi32>
      %770 = stablehlo.select %767, %769, %755 : tensor<200x1xi1>, tensor<200x1xi32>
      %771 = stablehlo.broadcast_in_dim %760, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %772 = stablehlo.broadcast_in_dim %765, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %773 = stablehlo.broadcast_in_dim %770, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %774 = stablehlo.concatenate %771, %772, %773, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %775 = "stablehlo.gather"(%202, %774) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x256xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %776 = stablehlo.multiply %775, %arg107 : tensor<200x1xf32>
      %cst_247 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %777 = stablehlo.reduce(%776 init: %cst_247) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %778 = stablehlo.broadcast_in_dim %602, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %779 = stablehlo.broadcast_in_dim %637, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %780 = stablehlo.broadcast_in_dim %672, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %781 = stablehlo.broadcast_in_dim %707, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %782 = stablehlo.broadcast_in_dim %742, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %783 = stablehlo.broadcast_in_dim %777, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %784 = stablehlo.concatenate %778, %779, %780, %781, %782, %783, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %785 = stablehlo.slice %784 [1:2, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %786 = stablehlo.reshape %785 : (tensor<1x200xf32>) -> tensor<200xf32>
      %787 = stablehlo.slice %784 [2:3, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %788 = stablehlo.reshape %787 : (tensor<1x200xf32>) -> tensor<200xf32>
      %789 = stablehlo.slice %784 [4:5, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %790 = stablehlo.reshape %789 : (tensor<1x200xf32>) -> tensor<200xf32>
      %791 = stablehlo.slice %784 [5:6, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %792 = stablehlo.reshape %791 : (tensor<1x200xf32>) -> tensor<200xf32>
      %793 = stablehlo.multiply %786, %792 : tensor<200xf32>
      %794 = stablehlo.multiply %788, %790 : tensor<200xf32>
      %795 = stablehlo.subtract %793, %794 : tensor<200xf32>
      %cst_248 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %796 = stablehlo.broadcast_in_dim %cst_248, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %797 = stablehlo.multiply %795, %796 : tensor<200xf32>
      %cst_249 = stablehlo.constant dense<7.485380e-15> : tensor<f32>
      %798 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %799 = stablehlo.multiply %797, %798 : tensor<200xf32>
      %cst_250 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %800 = stablehlo.reduce(%799 init: %cst_250) applies stablehlo.add across dimensions = [0] : (tensor<200xf32>, tensor<f32>) -> tensor<f32>
      stablehlo.return %800 : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %514 = call @_where_256(%510, %513, %cst_116) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %515 = stablehlo.slice %491 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %516 = stablehlo.reshape %515 : (tensor<1xi32>) -> tensor<i32>
    %c_117 = stablehlo.constant dense<0> : tensor<i32>
    %517 = stablehlo.minimum %516, %c_117 : tensor<i32>
    %c_118 = stablehlo.constant dense<0> : tensor<i32>
    %518 = stablehlo.compare  LT, %517, %c_118,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_119 = stablehlo.constant dense<1> : tensor<i32>
    %519 = stablehlo.add %517, %c_119 : tensor<i32>
    %520 = stablehlo.select %518, %519, %517 : tensor<i1>, tensor<i32>
    %c_120 = stablehlo.constant dense<1> : tensor<i32>
    %521 = stablehlo.dynamic_slice %476, %c_120, %520, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %522 = stablehlo.reshape %521 : (tensor<1x1xf32>) -> tensor<f32>
    %c_121 = stablehlo.constant dense<0> : tensor<i32>
    %523 = stablehlo.compare  LT, %517, %c_121,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_122 = stablehlo.constant dense<1> : tensor<i32>
    %524 = stablehlo.add %517, %c_122 : tensor<i32>
    %525 = stablehlo.select %523, %524, %517 : tensor<i1>, tensor<i32>
    %c_123 = stablehlo.constant dense<1> : tensor<i32>
    %526 = stablehlo.dynamic_slice %484, %c_123, %525, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %527 = stablehlo.reshape %526 : (tensor<1x1xf32>) -> tensor<f32>
    %528 = call @_where_256(%510, %514, %522) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_124 = stablehlo.constant dense<0> : tensor<i32>
    %529 = stablehlo.compare  LT, %517, %c_124,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_125 = stablehlo.constant dense<1> : tensor<i32>
    %530 = stablehlo.add %517, %c_125 : tensor<i32>
    %531 = stablehlo.select %529, %530, %517 : tensor<i1>, tensor<i32>
    %c_126 = stablehlo.constant dense<1> : tensor<i32>
    %532 = stablehlo.broadcast_in_dim %c_126, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %533 = stablehlo.broadcast_in_dim %531, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %534 = stablehlo.concatenate %532, %533, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %535 = "stablehlo.scatter"(%476, %534, %528) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      stablehlo.return %arg150 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %536 = call @_where_256(%510, %3, %527) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_127 = stablehlo.constant dense<0> : tensor<i32>
    %537 = stablehlo.compare  LT, %517, %c_127,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_128 = stablehlo.constant dense<1> : tensor<i32>
    %538 = stablehlo.add %517, %c_128 : tensor<i32>
    %539 = stablehlo.select %537, %538, %517 : tensor<i1>, tensor<i32>
    %c_129 = stablehlo.constant dense<1> : tensor<i32>
    %540 = stablehlo.broadcast_in_dim %c_129, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %541 = stablehlo.broadcast_in_dim %539, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %542 = stablehlo.concatenate %540, %541, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %543 = "stablehlo.scatter"(%484, %542, %536) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
      stablehlo.return %arg150 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %544 = stablehlo.slice %491 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %545 = stablehlo.reshape %544 : (tensor<1xi32>) -> tensor<i32>
    %c_130 = stablehlo.constant dense<1> : tensor<i32>
    %c_131 = stablehlo.constant dense<0> : tensor<i32>
    %546 = call @_where_260(%510, %c_130, %c_131) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %547 = stablehlo.convert %546 : tensor<i32>
    %548 = stablehlo.add %545, %547 : tensor<i32>
    %c_132 = stablehlo.constant dense<1> : tensor<i32>
    %549 = stablehlo.broadcast_in_dim %c_132, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %550 = "stablehlo.scatter"(%491, %549, %548) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg149: tensor<i32>, %arg150: tensor<i32>):
      stablehlo.return %arg150 : tensor<i32>
    }) : (tensor<2xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
    %cst_133 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %551 = stablehlo.compare  GE, %3, %cst_133,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %c_134 = stablehlo.constant dense<true> : tensor<i1>
    %552 = stablehlo.and %c_134, %551 : tensor<i1>
    %cst_135 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %553 = stablehlo.compare  LE, %3, %cst_135,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %554 = stablehlo.and %552, %553 : tensor<i1>
    %c_136 = stablehlo.constant dense<1> : tensor<i32>
    %c_137 = stablehlo.constant dense<1> : tensor<i32>
    %555 = stablehlo.maximum %c_136, %c_137 : tensor<i32>
    %556 = call @remainder(%arg147, %555) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_138 = stablehlo.constant dense<0> : tensor<i32>
    %557 = stablehlo.compare  EQ, %556, %c_138,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %558 = stablehlo.and %554, %557 : tensor<i1>
    %559 = stablehlo.convert %558 : (tensor<i1>) -> tensor<i32>
    %560:3 = "stablehlo.case"(%559) ({
      stablehlo.return %501#0, %501#1, %501#2 : tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>
    }, {
      %cst_140 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %568 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %569 = stablehlo.multiply %568, %arg108 : tensor<3xf32>
      %570 = stablehlo.optimization_barrier %569 : tensor<3xf32>
      %571 = stablehlo.optimization_barrier %3 : tensor<f32>
      %572 = stablehlo.broadcast_in_dim %571, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %573 = stablehlo.multiply %570, %572 : tensor<3xf32>
      %574 = stablehlo.cosine %573 : tensor<3xf32>
      %575 = stablehlo.sine %573 : tensor<3xf32>
      %cst_141 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %576 = stablehlo.subtract %cst_141, %cst_142 : tensor<f32>
      %cst_143 = stablehlo.constant dense<1.000000e-30> : tensor<f32>
      %577 = stablehlo.maximum %576, %cst_143 : tensor<f32>
      %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %578 = stablehlo.subtract %3, %cst_144 : tensor<f32>
      %579 = stablehlo.divide %578, %577 : tensor<f32>
      %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %cst_146 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %580 = func.call @clip_271(%579, %cst_145, %cst_146) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_147 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %581 = stablehlo.multiply %cst_147, %580 : tensor<f32>
      %582 = stablehlo.cosine %581 : tensor<f32>
      %cst_148 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %583 = stablehlo.subtract %cst_148, %582 : tensor<f32>
      %cst_149 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %584 = stablehlo.multiply %cst_149, %583 : tensor<f32>
      %cst_150 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %585 = stablehlo.is_finite %cst_150 : (tensor<f32>) -> tensor<i1>
      %c_151 = stablehlo.constant dense<false> : tensor<i1>
      %586 = stablehlo.and %c_151, %585 : tensor<i1>
      %cst_152 = stablehlo.constant dense<0x7F800000> : tensor<f32>
      %cst_153 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %587 = stablehlo.compare  GT, %cst_152, %cst_153,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %588 = stablehlo.and %586, %587 : tensor<i1>
      %cst_154 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %589 = func.call @_where_276(%588, %584, %cst_154) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %590 = stablehlo.maximum %589, %cst_155 : tensor<f32>
      %cst_156 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %591 = stablehlo.multiply %arg0, %cst_156 : tensor<f32>
      %cst_157 = stablehlo.constant dense<0x4D8EF3C2> : tensor<f32>
      %592 = stablehlo.multiply %591, %cst_157 : tensor<f32>
      %cst_158 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %593 = stablehlo.divide %592, %cst_158 : tensor<f32>
      %cst_159 = stablehlo.constant dense<6.28318548> : tensor<f32>
      %594 = stablehlo.sqrt %cst_159 : tensor<f32>
      %595 = stablehlo.divide %593, %594 : tensor<f32>
      %596 = stablehlo.multiply %590, %595 : tensor<f32>
      %c_160 = stablehlo.constant dense<0> : tensor<i32>
      %c_161 = stablehlo.constant dense<1> : tensor<i32>
      %597 = stablehlo.compare  EQ, %c_160, %c_161,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %598 = func.call @_where_278(%597, %596, %590) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %c_162 = stablehlo.constant dense<256> : tensor<i32>
      %599:2 = func.call @divmod_280(%arg109, %c_162) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_163 = stablehlo.constant dense<257> : tensor<i32>
      %600:2 = func.call @divmod_280(%599#0, %c_163) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_164 = stablehlo.constant dense<257> : tensor<i32>
      %601:2 = func.call @divmod_280(%600#0, %c_164) : (tensor<200x8xi32>, tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>)
      %c_165 = stablehlo.constant dense<0> : tensor<i32>
      %602 = stablehlo.broadcast_in_dim %c_165, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %603 = stablehlo.compare  GT, %601#0, %602,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_166 = stablehlo.constant dense<-1> : tensor<i32>
      %604 = stablehlo.broadcast_in_dim %c_166, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %605 = stablehlo.compare  LT, %601#0, %604,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_167 = stablehlo.constant dense<0> : tensor<i32>
      %606 = func.call @_where_295(%605, %c_167, %601#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_168 = stablehlo.constant dense<256> : tensor<i32>
      %607 = func.call @_where_295(%603, %c_168, %606) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_169 = stablehlo.constant dense<0> : tensor<i32>
      %608 = func.call @_where_295(%605, %c_169, %600#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_170 = stablehlo.constant dense<256> : tensor<i32>
      %609 = func.call @_where_295(%603, %c_170, %608) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_171 = stablehlo.constant dense<0> : tensor<i32>
      %610 = func.call @_where_295(%605, %c_171, %599#1) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_172 = stablehlo.constant dense<255> : tensor<i32>
      %611 = func.call @_where_295(%603, %c_172, %610) : (tensor<200x8xi1>, tensor<i32>, tensor<200x8xi32>) -> tensor<200x8xi32>
      %c_173 = stablehlo.constant dense<0> : tensor<i32>
      %612 = stablehlo.broadcast_in_dim %c_173, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %613 = stablehlo.compare  LT, %607, %612,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_174 = stablehlo.constant dense<257> : tensor<i32>
      %614 = stablehlo.broadcast_in_dim %c_174, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %615 = stablehlo.add %607, %614 : tensor<200x8xi32>
      %616 = stablehlo.select %613, %615, %607 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_175 = stablehlo.constant dense<0> : tensor<i32>
      %617 = stablehlo.broadcast_in_dim %c_175, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %618 = stablehlo.compare  LT, %609, %617,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_176 = stablehlo.constant dense<257> : tensor<i32>
      %619 = stablehlo.broadcast_in_dim %c_176, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %620 = stablehlo.add %609, %619 : tensor<200x8xi32>
      %621 = stablehlo.select %618, %620, %609 : tensor<200x8xi1>, tensor<200x8xi32>
      %c_177 = stablehlo.constant dense<0> : tensor<i32>
      %622 = stablehlo.broadcast_in_dim %c_177, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %623 = stablehlo.compare  LT, %611, %622,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
      %c_178 = stablehlo.constant dense<256> : tensor<i32>
      %624 = stablehlo.broadcast_in_dim %c_178, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
      %625 = stablehlo.add %611, %624 : tensor<200x8xi32>
      %626 = stablehlo.select %623, %625, %611 : tensor<200x8xi1>, tensor<200x8xi32>
      %627 = stablehlo.broadcast_in_dim %616, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %628 = stablehlo.broadcast_in_dim %621, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %629 = stablehlo.broadcast_in_dim %626, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
      %630 = stablehlo.concatenate %627, %628, %629, dim = 2 : (tensor<200x8x1xi32>, tensor<200x8x1xi32>, tensor<200x8x1xi32>) -> tensor<200x8x3xi32>
      %631 = "stablehlo.gather"(%374, %630) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x257x256xf32>, tensor<200x8x3xi32>) -> tensor<200x8xf32>
      %632 = stablehlo.multiply %631, %arg110 : tensor<200x8xf32>
      %cst_179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %633 = stablehlo.reduce(%632 init: %cst_179) applies stablehlo.add across dimensions = [1] : (tensor<200x8xf32>, tensor<f32>) -> tensor<200xf32>
      %c_180 = stablehlo.constant dense<257> : tensor<i32>
      %634:2 = func.call @divmod_325(%arg111, %c_180) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_181 = stablehlo.constant dense<256> : tensor<i32>
      %635:2 = func.call @divmod_325(%634#0, %c_181) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_182 = stablehlo.constant dense<257> : tensor<i32>
      %636:2 = func.call @divmod_325(%635#0, %c_182) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_183 = stablehlo.constant dense<0> : tensor<i32>
      %637 = stablehlo.broadcast_in_dim %c_183, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %638 = stablehlo.compare  GT, %636#0, %637,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_184 = stablehlo.constant dense<-1> : tensor<i32>
      %639 = stablehlo.broadcast_in_dim %c_184, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %640 = stablehlo.compare  LT, %636#0, %639,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_185 = stablehlo.constant dense<0> : tensor<i32>
      %641 = func.call @_where_340(%640, %c_185, %636#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_186 = stablehlo.constant dense<256> : tensor<i32>
      %642 = func.call @_where_340(%638, %c_186, %641) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_187 = stablehlo.constant dense<0> : tensor<i32>
      %643 = func.call @_where_340(%640, %c_187, %635#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_188 = stablehlo.constant dense<255> : tensor<i32>
      %644 = func.call @_where_340(%638, %c_188, %643) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_189 = stablehlo.constant dense<0> : tensor<i32>
      %645 = func.call @_where_340(%640, %c_189, %634#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_190 = stablehlo.constant dense<256> : tensor<i32>
      %646 = func.call @_where_340(%638, %c_190, %645) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_191 = stablehlo.constant dense<0> : tensor<i32>
      %647 = stablehlo.broadcast_in_dim %c_191, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %648 = stablehlo.compare  LT, %642, %647,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_192 = stablehlo.constant dense<257> : tensor<i32>
      %649 = stablehlo.broadcast_in_dim %c_192, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %650 = stablehlo.add %642, %649 : tensor<200x2xi32>
      %651 = stablehlo.select %648, %650, %642 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_193 = stablehlo.constant dense<0> : tensor<i32>
      %652 = stablehlo.broadcast_in_dim %c_193, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %653 = stablehlo.compare  LT, %644, %652,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_194 = stablehlo.constant dense<256> : tensor<i32>
      %654 = stablehlo.broadcast_in_dim %c_194, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %655 = stablehlo.add %644, %654 : tensor<200x2xi32>
      %656 = stablehlo.select %653, %655, %644 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_195 = stablehlo.constant dense<0> : tensor<i32>
      %657 = stablehlo.broadcast_in_dim %c_195, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %658 = stablehlo.compare  LT, %646, %657,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_196 = stablehlo.constant dense<257> : tensor<i32>
      %659 = stablehlo.broadcast_in_dim %c_196, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %660 = stablehlo.add %646, %659 : tensor<200x2xi32>
      %661 = stablehlo.select %658, %660, %646 : tensor<200x2xi1>, tensor<200x2xi32>
      %662 = stablehlo.broadcast_in_dim %651, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %663 = stablehlo.broadcast_in_dim %656, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %664 = stablehlo.broadcast_in_dim %661, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %665 = stablehlo.concatenate %662, %663, %664, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
      %666 = "stablehlo.gather"(%413, %665) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x257xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
      %667 = stablehlo.multiply %666, %arg112 : tensor<200x2xf32>
      %cst_197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %668 = stablehlo.reduce(%667 init: %cst_197) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
      %c_198 = stablehlo.constant dense<257> : tensor<i32>
      %669:2 = func.call @divmod_325(%arg113, %c_198) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_199 = stablehlo.constant dense<257> : tensor<i32>
      %670:2 = func.call @divmod_325(%669#0, %c_199) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_200 = stablehlo.constant dense<256> : tensor<i32>
      %671:2 = func.call @divmod_325(%670#0, %c_200) : (tensor<200x2xi32>, tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>)
      %c_201 = stablehlo.constant dense<0> : tensor<i32>
      %672 = stablehlo.broadcast_in_dim %c_201, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %673 = stablehlo.compare  GT, %671#0, %672,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_202 = stablehlo.constant dense<-1> : tensor<i32>
      %674 = stablehlo.broadcast_in_dim %c_202, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %675 = stablehlo.compare  LT, %671#0, %674,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_203 = stablehlo.constant dense<0> : tensor<i32>
      %676 = func.call @_where_340(%675, %c_203, %671#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_204 = stablehlo.constant dense<255> : tensor<i32>
      %677 = func.call @_where_340(%673, %c_204, %676) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_205 = stablehlo.constant dense<0> : tensor<i32>
      %678 = func.call @_where_340(%675, %c_205, %670#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_206 = stablehlo.constant dense<256> : tensor<i32>
      %679 = func.call @_where_340(%673, %c_206, %678) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_207 = stablehlo.constant dense<0> : tensor<i32>
      %680 = func.call @_where_340(%675, %c_207, %669#1) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_208 = stablehlo.constant dense<256> : tensor<i32>
      %681 = func.call @_where_340(%673, %c_208, %680) : (tensor<200x2xi1>, tensor<i32>, tensor<200x2xi32>) -> tensor<200x2xi32>
      %c_209 = stablehlo.constant dense<0> : tensor<i32>
      %682 = stablehlo.broadcast_in_dim %c_209, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %683 = stablehlo.compare  LT, %677, %682,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_210 = stablehlo.constant dense<256> : tensor<i32>
      %684 = stablehlo.broadcast_in_dim %c_210, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %685 = stablehlo.add %677, %684 : tensor<200x2xi32>
      %686 = stablehlo.select %683, %685, %677 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_211 = stablehlo.constant dense<0> : tensor<i32>
      %687 = stablehlo.broadcast_in_dim %c_211, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %688 = stablehlo.compare  LT, %679, %687,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_212 = stablehlo.constant dense<257> : tensor<i32>
      %689 = stablehlo.broadcast_in_dim %c_212, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %690 = stablehlo.add %679, %689 : tensor<200x2xi32>
      %691 = stablehlo.select %688, %690, %679 : tensor<200x2xi1>, tensor<200x2xi32>
      %c_213 = stablehlo.constant dense<0> : tensor<i32>
      %692 = stablehlo.broadcast_in_dim %c_213, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %693 = stablehlo.compare  LT, %681, %692,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
      %c_214 = stablehlo.constant dense<257> : tensor<i32>
      %694 = stablehlo.broadcast_in_dim %c_214, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
      %695 = stablehlo.add %681, %694 : tensor<200x2xi32>
      %696 = stablehlo.select %693, %695, %681 : tensor<200x2xi1>, tensor<200x2xi32>
      %697 = stablehlo.broadcast_in_dim %686, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %698 = stablehlo.broadcast_in_dim %691, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %699 = stablehlo.broadcast_in_dim %696, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
      %700 = stablehlo.concatenate %697, %698, %699, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
      %701 = "stablehlo.gather"(%442, %700) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x257xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
      %702 = stablehlo.multiply %701, %arg114 : tensor<200x2xf32>
      %cst_215 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %703 = stablehlo.reduce(%702 init: %cst_215) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
      %c_216 = stablehlo.constant dense<257> : tensor<i32>
      %704:2 = func.call @divmod(%arg115, %c_216) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_217 = stablehlo.constant dense<256> : tensor<i32>
      %705:2 = func.call @divmod(%704#0, %c_217) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_218 = stablehlo.constant dense<256> : tensor<i32>
      %706:2 = func.call @divmod(%705#0, %c_218) : (tensor<200x1xi32>, tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>)
      %c_219 = stablehlo.constant dense<0> : tensor<i32>
      %707 = stablehlo.broadcast_in_dim %c_219, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %708 = stablehlo.compare  GT, %706#0, %707,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_220 = stablehlo.constant dense<-1> : tensor<i32>
      %709 = stablehlo.broadcast_in_dim %c_220, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %710 = stablehlo.compare  LT, %706#0, %709,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_221 = stablehlo.constant dense<0> : tensor<i32>
      %711 = func.call @_where_234(%710, %c_221, %706#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_222 = stablehlo.constant dense<255> : tensor<i32>
      %712 = func.call @_where_234(%708, %c_222, %711) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_223 = stablehlo.constant dense<0> : tensor<i32>
      %713 = func.call @_where_234(%710, %c_223, %705#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_224 = stablehlo.constant dense<255> : tensor<i32>
      %714 = func.call @_where_234(%708, %c_224, %713) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_225 = stablehlo.constant dense<0> : tensor<i32>
      %715 = func.call @_where_234(%710, %c_225, %704#1) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_226 = stablehlo.constant dense<256> : tensor<i32>
      %716 = func.call @_where_234(%708, %c_226, %715) : (tensor<200x1xi1>, tensor<i32>, tensor<200x1xi32>) -> tensor<200x1xi32>
      %c_227 = stablehlo.constant dense<0> : tensor<i32>
      %717 = stablehlo.broadcast_in_dim %c_227, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %718 = stablehlo.compare  LT, %712, %717,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_228 = stablehlo.constant dense<256> : tensor<i32>
      %719 = stablehlo.broadcast_in_dim %c_228, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %720 = stablehlo.add %712, %719 : tensor<200x1xi32>
      %721 = stablehlo.select %718, %720, %712 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_229 = stablehlo.constant dense<0> : tensor<i32>
      %722 = stablehlo.broadcast_in_dim %c_229, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %723 = stablehlo.compare  LT, %714, %722,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_230 = stablehlo.constant dense<256> : tensor<i32>
      %724 = stablehlo.broadcast_in_dim %c_230, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %725 = stablehlo.add %714, %724 : tensor<200x1xi32>
      %726 = stablehlo.select %723, %725, %714 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_231 = stablehlo.constant dense<0> : tensor<i32>
      %727 = stablehlo.broadcast_in_dim %c_231, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %728 = stablehlo.compare  LT, %716, %727,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_232 = stablehlo.constant dense<257> : tensor<i32>
      %729 = stablehlo.broadcast_in_dim %c_232, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %730 = stablehlo.add %716, %729 : tensor<200x1xi32>
      %731 = stablehlo.select %728, %730, %716 : tensor<200x1xi1>, tensor<200x1xi32>
      %732 = stablehlo.broadcast_in_dim %721, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %733 = stablehlo.broadcast_in_dim %726, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %734 = stablehlo.broadcast_in_dim %731, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %735 = stablehlo.concatenate %732, %733, %734, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %736 = "stablehlo.gather"(%132, %735) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x256x257xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %737 = stablehlo.multiply %736, %arg116 : tensor<200x1xf32>
      %cst_233 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %738 = stablehlo.reduce(%737 init: %cst_233) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_234 = stablehlo.constant dense<256> : tensor<i32>
      %739:2 = func.call @divmod_302(%arg117, %c_234) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_235 = stablehlo.constant dense<257> : tensor<i32>
      %740:2 = func.call @divmod_302(%739#0, %c_235) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_236 = stablehlo.constant dense<256> : tensor<i32>
      %741:2 = func.call @divmod_302(%740#0, %c_236) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_237 = stablehlo.constant dense<0> : tensor<i32>
      %742 = stablehlo.broadcast_in_dim %c_237, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %743 = stablehlo.compare  GT, %741#0, %742,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_238 = stablehlo.constant dense<-1> : tensor<i32>
      %744 = stablehlo.broadcast_in_dim %c_238, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %745 = stablehlo.compare  LT, %741#0, %744,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_239 = stablehlo.constant dense<0> : tensor<i32>
      %746 = func.call @_where_317(%745, %c_239, %741#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_240 = stablehlo.constant dense<255> : tensor<i32>
      %747 = func.call @_where_317(%743, %c_240, %746) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_241 = stablehlo.constant dense<0> : tensor<i32>
      %748 = func.call @_where_317(%745, %c_241, %740#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_242 = stablehlo.constant dense<256> : tensor<i32>
      %749 = func.call @_where_317(%743, %c_242, %748) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_243 = stablehlo.constant dense<0> : tensor<i32>
      %750 = func.call @_where_317(%745, %c_243, %739#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_244 = stablehlo.constant dense<255> : tensor<i32>
      %751 = func.call @_where_317(%743, %c_244, %750) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_245 = stablehlo.constant dense<0> : tensor<i32>
      %752 = stablehlo.broadcast_in_dim %c_245, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %753 = stablehlo.compare  LT, %747, %752,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_246 = stablehlo.constant dense<256> : tensor<i32>
      %754 = stablehlo.broadcast_in_dim %c_246, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %755 = stablehlo.add %747, %754 : tensor<200x4xi32>
      %756 = stablehlo.select %753, %755, %747 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_247 = stablehlo.constant dense<0> : tensor<i32>
      %757 = stablehlo.broadcast_in_dim %c_247, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %758 = stablehlo.compare  LT, %749, %757,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_248 = stablehlo.constant dense<257> : tensor<i32>
      %759 = stablehlo.broadcast_in_dim %c_248, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %760 = stablehlo.add %749, %759 : tensor<200x4xi32>
      %761 = stablehlo.select %758, %760, %749 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_249 = stablehlo.constant dense<0> : tensor<i32>
      %762 = stablehlo.broadcast_in_dim %c_249, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %763 = stablehlo.compare  LT, %751, %762,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_250 = stablehlo.constant dense<256> : tensor<i32>
      %764 = stablehlo.broadcast_in_dim %c_250, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %765 = stablehlo.add %751, %764 : tensor<200x4xi32>
      %766 = stablehlo.select %763, %765, %751 : tensor<200x4xi1>, tensor<200x4xi32>
      %767 = stablehlo.broadcast_in_dim %756, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %768 = stablehlo.broadcast_in_dim %761, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %769 = stablehlo.broadcast_in_dim %766, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %770 = stablehlo.concatenate %767, %768, %769, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %771 = "stablehlo.gather"(%173, %770) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<256x257x256xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %772 = stablehlo.multiply %771, %arg118 : tensor<200x4xf32>
      %cst_251 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %773 = stablehlo.reduce(%772 init: %cst_251) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %c_252 = stablehlo.constant dense<256> : tensor<i32>
      %774:2 = func.call @divmod_302(%arg119, %c_252) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_253 = stablehlo.constant dense<256> : tensor<i32>
      %775:2 = func.call @divmod_302(%774#0, %c_253) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_254 = stablehlo.constant dense<257> : tensor<i32>
      %776:2 = func.call @divmod_302(%775#0, %c_254) : (tensor<200x4xi32>, tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>)
      %c_255 = stablehlo.constant dense<0> : tensor<i32>
      %777 = stablehlo.broadcast_in_dim %c_255, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %778 = stablehlo.compare  GT, %776#0, %777,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_256 = stablehlo.constant dense<-1> : tensor<i32>
      %779 = stablehlo.broadcast_in_dim %c_256, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %780 = stablehlo.compare  LT, %776#0, %779,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_257 = stablehlo.constant dense<0> : tensor<i32>
      %781 = func.call @_where_317(%780, %c_257, %776#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_258 = stablehlo.constant dense<256> : tensor<i32>
      %782 = func.call @_where_317(%778, %c_258, %781) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_259 = stablehlo.constant dense<0> : tensor<i32>
      %783 = func.call @_where_317(%780, %c_259, %775#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_260 = stablehlo.constant dense<255> : tensor<i32>
      %784 = func.call @_where_317(%778, %c_260, %783) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_261 = stablehlo.constant dense<0> : tensor<i32>
      %785 = func.call @_where_317(%780, %c_261, %774#1) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_262 = stablehlo.constant dense<255> : tensor<i32>
      %786 = func.call @_where_317(%778, %c_262, %785) : (tensor<200x4xi1>, tensor<i32>, tensor<200x4xi32>) -> tensor<200x4xi32>
      %c_263 = stablehlo.constant dense<0> : tensor<i32>
      %787 = stablehlo.broadcast_in_dim %c_263, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %788 = stablehlo.compare  LT, %782, %787,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_264 = stablehlo.constant dense<257> : tensor<i32>
      %789 = stablehlo.broadcast_in_dim %c_264, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %790 = stablehlo.add %782, %789 : tensor<200x4xi32>
      %791 = stablehlo.select %788, %790, %782 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_265 = stablehlo.constant dense<0> : tensor<i32>
      %792 = stablehlo.broadcast_in_dim %c_265, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %793 = stablehlo.compare  LT, %784, %792,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_266 = stablehlo.constant dense<256> : tensor<i32>
      %794 = stablehlo.broadcast_in_dim %c_266, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %795 = stablehlo.add %784, %794 : tensor<200x4xi32>
      %796 = stablehlo.select %793, %795, %784 : tensor<200x4xi1>, tensor<200x4xi32>
      %c_267 = stablehlo.constant dense<0> : tensor<i32>
      %797 = stablehlo.broadcast_in_dim %c_267, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %798 = stablehlo.compare  LT, %786, %797,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
      %c_268 = stablehlo.constant dense<256> : tensor<i32>
      %799 = stablehlo.broadcast_in_dim %c_268, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
      %800 = stablehlo.add %786, %799 : tensor<200x4xi32>
      %801 = stablehlo.select %798, %800, %786 : tensor<200x4xi1>, tensor<200x4xi32>
      %802 = stablehlo.broadcast_in_dim %791, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %803 = stablehlo.broadcast_in_dim %796, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %804 = stablehlo.broadcast_in_dim %801, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
      %805 = stablehlo.concatenate %802, %803, %804, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
      %806 = "stablehlo.gather"(%202, %805) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<257x256x256xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
      %807 = stablehlo.multiply %806, %arg120 : tensor<200x4xf32>
      %cst_269 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %808 = stablehlo.reduce(%807 init: %cst_269) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
      %809 = stablehlo.broadcast_in_dim %633, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %810 = stablehlo.broadcast_in_dim %668, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %811 = stablehlo.broadcast_in_dim %703, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %812 = stablehlo.broadcast_in_dim %738, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %813 = stablehlo.broadcast_in_dim %773, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %814 = stablehlo.broadcast_in_dim %808, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %815 = stablehlo.concatenate %809, %810, %811, %812, %813, %814, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %816 = stablehlo.broadcast_in_dim %arg121, dims = [0] : (tensor<6xf32>) -> tensor<6x1x1xf32>
      %817 = stablehlo.broadcast_in_dim %598, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %818 = stablehlo.multiply %817, %816 : tensor<6x1x1xf32>
      %819 = stablehlo.dot_general %815, %574, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %820 = stablehlo.transpose %819, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %821 = stablehlo.broadcast_in_dim %818, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %822 = stablehlo.multiply %821, %820 : tensor<6x3x200xf32>
      %823 = stablehlo.broadcast_in_dim %598, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
      %824 = stablehlo.multiply %823, %816 : tensor<6x1x1xf32>
      %825 = stablehlo.dot_general %815, %575, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<3xf32>) -> tensor<6x200x3xf32>
      %826 = stablehlo.transpose %825, dims = [0, 2, 1] : (tensor<6x200x3xf32>) -> tensor<6x3x200xf32>
      %827 = stablehlo.broadcast_in_dim %824, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x3x200xf32>
      %828 = stablehlo.multiply %827, %826 : tensor<6x3x200xf32>
      %829 = stablehlo.reshape %822 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_270 = stablehlo.constant dense<3600> : tensor<i32>
      %830 = stablehlo.broadcast_in_dim %c_270, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %831 = "stablehlo.scatter"(%501#0, %830, %829) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
        %838 = stablehlo.add %arg149, %arg150 : tensor<f32>
        stablehlo.return %838 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %832 = stablehlo.reshape %828 : (tensor<6x3x200xf32>) -> tensor<3600xf32>
      %c_271 = stablehlo.constant dense<3600> : tensor<i32>
      %833 = stablehlo.broadcast_in_dim %c_271, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %834 = "stablehlo.scatter"(%501#1, %833, %832) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
        %838 = stablehlo.add %arg149, %arg150 : tensor<f32>
        stablehlo.return %838 : tensor<f32>
      }) : (tensor<7200xf32>, tensor<1xi32>, tensor<3600xf32>) -> tensor<7200xf32>
      %835 = stablehlo.broadcast_in_dim %589, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %c_272 = stablehlo.constant dense<3> : tensor<i32>
      %836 = stablehlo.broadcast_in_dim %c_272, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %837 = "stablehlo.scatter"(%501#2, %836, %835) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
      ^bb0(%arg149: tensor<f32>, %arg150: tensor<f32>):
        %838 = stablehlo.add %arg149, %arg150 : tensor<f32>
        stablehlo.return %838 : tensor<f32>
      }) : (tensor<6xf32>, tensor<1xi32>, tensor<3xf32>) -> tensor<6xf32>
      stablehlo.return %831, %834, %837 : tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>
    }) : (tensor<i32>) -> (tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>)
    %561 = stablehlo.custom_call @LayoutConstraint(%374) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<257x257x256xf32>) -> tensor<257x257x256xf32>
    %562 = stablehlo.custom_call @LayoutConstraint(%413) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<257x256x257xf32>) -> tensor<257x256x257xf32>
    %563 = stablehlo.custom_call @LayoutConstraint(%442) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<256x257x257xf32>) -> tensor<256x257x257xf32>
    %564 = stablehlo.custom_call @LayoutConstraint(%132) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<256x256x257xf32>) -> tensor<256x256x257xf32>
    %565 = stablehlo.custom_call @LayoutConstraint(%173) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<256x257x256xf32>) -> tensor<256x257x256xf32>
    %566 = stablehlo.custom_call @LayoutConstraint(%202) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<257x256x256xf32>) -> tensor<257x256x256xf32>
    %c_139 = stablehlo.constant dense<1> : tensor<i32>
    %567 = stablehlo.add %arg147, %c_139 : tensor<i32>
    return %561, %562, %563, %564, %565, %566, %14, %34, %54, %74, %94, %114, %241, %263, %285, %309, %333, %357, %535, %543, %550, %560#0, %560#1, %560#2, %3, %567 : tensor<257x257x256xf32>, tensor<257x256x257xf32>, tensor<256x257x257xf32>, tensor<256x256x257xf32>, tensor<256x257x256xf32>, tensor<257x256x256xf32>, tensor<256x24x257xf32>, tensor<24x256x257xf32>, tensor<24x257x256xf32>, tensor<256x257x24xf32>, tensor<257x256x24xf32>, tensor<257x24x256xf32>, tensor<257x24x256xf32>, tensor<24x257x256xf32>, tensor<24x256x257xf32>, tensor<257x256x24xf32>, tensor<256x257x24xf32>, tensor<256x24x257xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<7200xf32>, tensor<7200xf32>, tensor<6xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @diff(%arg0: tensor<256x257x257xf32>) -> tensor<256x256x257xf32> {
    %0 = stablehlo.slice %arg0 [0:256, 1:257, 0:257] : (tensor<256x257x257xf32>) -> tensor<256x256x257xf32>
    %1 = stablehlo.slice %arg0 [0:256, 0:256, 0:257] : (tensor<256x257x257xf32>) -> tensor<256x256x257xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<256x256x257xf32>
    return %2 : tensor<256x256x257xf32>
  }
  func.func private @diff_19(%arg0: tensor<257x256x257xf32>) -> tensor<256x256x257xf32> {
    %0 = stablehlo.slice %arg0 [1:257, 0:256, 0:257] : (tensor<257x256x257xf32>) -> tensor<256x256x257xf32>
    %1 = stablehlo.slice %arg0 [0:256, 0:256, 0:257] : (tensor<257x256x257xf32>) -> tensor<256x256x257xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<256x256x257xf32>
    return %2 : tensor<256x256x257xf32>
  }
  func.func private @diff_32(%arg0: tensor<257x257x256xf32>) -> tensor<256x257x256xf32> {
    %0 = stablehlo.slice %arg0 [1:257, 0:257, 0:256] : (tensor<257x257x256xf32>) -> tensor<256x257x256xf32>
    %1 = stablehlo.slice %arg0 [0:256, 0:257, 0:256] : (tensor<257x257x256xf32>) -> tensor<256x257x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<256x257x256xf32>
    return %2 : tensor<256x257x256xf32>
  }
  func.func private @diff_48(%arg0: tensor<256x257x257xf32>) -> tensor<256x257x256xf32> {
    %0 = stablehlo.slice %arg0 [0:256, 0:257, 1:257] : (tensor<256x257x257xf32>) -> tensor<256x257x256xf32>
    %1 = stablehlo.slice %arg0 [0:256, 0:257, 0:256] : (tensor<256x257x257xf32>) -> tensor<256x257x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<256x257x256xf32>
    return %2 : tensor<256x257x256xf32>
  }
  func.func private @diff_61(%arg0: tensor<257x256x257xf32>) -> tensor<257x256x256xf32> {
    %0 = stablehlo.slice %arg0 [0:257, 0:256, 1:257] : (tensor<257x256x257xf32>) -> tensor<257x256x256xf32>
    %1 = stablehlo.slice %arg0 [0:257, 0:256, 0:256] : (tensor<257x256x257xf32>) -> tensor<257x256x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<257x256x256xf32>
    return %2 : tensor<257x256x256xf32>
  }
  func.func private @diff_77(%arg0: tensor<257x257x256xf32>) -> tensor<257x256x256xf32> {
    %0 = stablehlo.slice %arg0 [0:257, 1:257, 0:256] : (tensor<257x257x256xf32>) -> tensor<257x256x256xf32>
    %1 = stablehlo.slice %arg0 [0:257, 0:256, 0:256] : (tensor<257x257x256xf32>) -> tensor<257x256x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<257x256x256xf32>
    return %2 : tensor<257x256x256xf32>
  }
  func.func private @clip(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %arg2 : tensor<i32>
    %3 = stablehlo.minimum %2, %1 : tensor<i32>
    return %3 : tensor<i32>
  }
  func.func private @_take(%arg0: tensor<257x256x256xf32>, %arg1: tensor<1xi32>) -> tensor<257x1x256xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<256> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<255> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 257, 1, 256>}> : (tensor<257x256x256xf32>, tensor<1x1xi32>) -> tensor<257x1x256xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<1xi1>) -> tensor<257x1x256xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<257x1x256xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<257x1x256xi1>, tensor<257x1x256xf32>
    return %15 : tensor<257x1x256xf32>
  }
  func.func private @_where(%arg0: tensor<1xi1>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>) -> tensor<1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1xi1>, tensor<1xi32>
    return %0 : tensor<1xi32>
  }
  func.func private @_take_119(%arg0: tensor<256x257x256xf32>, %arg1: tensor<1xi32>) -> tensor<1x257x256xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<256> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<255> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 257, 256>}> : (tensor<256x257x256xf32>, tensor<1x1xi32>) -> tensor<1x257x256xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<1xi1>) -> tensor<1x257x256xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x257x256xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<1x257x256xi1>, tensor<1x257x256xf32>
    return %15 : tensor<1x257x256xf32>
  }
  func.func private @_take_127(%arg0: tensor<256x256x257xf32>, %arg1: tensor<1xi32>) -> tensor<1x256x257xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<256> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<255> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 256, 257>}> : (tensor<256x256x257xf32>, tensor<1x1xi32>) -> tensor<1x256x257xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<1xi1>) -> tensor<1x256x257xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x257xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<1x256x257xi1>, tensor<1x256x257xf32>
    return %15 : tensor<1x256x257xf32>
  }
  func.func private @_take_135(%arg0: tensor<257x256x256xf32>, %arg1: tensor<1xi32>) -> tensor<257x256x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<256> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<255> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 257, 256, 1>}> : (tensor<257x256x256xf32>, tensor<1x1xi32>) -> tensor<257x256x1xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<1xi1>) -> tensor<257x256x1xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<257x256x1xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<257x256x1xi1>, tensor<257x256x1xf32>
    return %15 : tensor<257x256x1xf32>
  }
  func.func private @_take_143(%arg0: tensor<256x257x256xf32>, %arg1: tensor<1xi32>) -> tensor<256x257x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<256> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<255> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 257, 1>}> : (tensor<256x257x256xf32>, tensor<1x1xi32>) -> tensor<256x257x1xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<1xi1>) -> tensor<256x257x1xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x257x1xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<256x257x1xi1>, tensor<256x257x1xf32>
    return %15 : tensor<256x257x1xf32>
  }
  func.func private @_take_151(%arg0: tensor<256x256x257xf32>, %arg1: tensor<1xi32>) -> tensor<256x1x257xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<256> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<255> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 256, 1, 257>}> : (tensor<256x256x257xf32>, tensor<1x1xi32>) -> tensor<256x1x257xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<1xi1>) -> tensor<256x1x257xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x1x257xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<256x1x257xi1>, tensor<256x1x257xf32>
    return %15 : tensor<256x1x257xf32>
  }
  func.func private @remainder(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_217(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_217(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @divmod(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> (tensor<200x1xi32>, tensor<200x1xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
    %2 = call @remainder_230(%arg0, %0) : (tensor<200x1xi32>, tensor<i32>) -> tensor<200x1xi32>
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
    %13 = call @_where_228(%10, %12, %1) : (tensor<200x1xi1>, tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi32>
    return %13 : tensor<200x1xi32>
  }
  func.func private @_where_228(%arg0: tensor<200x1xi1>, %arg1: tensor<200x1xi32>, %arg2: tensor<200x1xi32>) -> tensor<200x1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x1xi1>, tensor<200x1xi32>
    return %0 : tensor<200x1xi32>
  }
  func.func private @remainder_230(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>) -> tensor<200x1xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_217(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_234(%arg0: tensor<200x1xi1>, %arg1: tensor<i32>, %arg2: tensor<200x1xi32>) -> tensor<200x1xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x1xi1>, tensor<200x1xi32>
    return %2 : tensor<200x1xi32>
  }
  func.func private @_where_256(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_260(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @clip_271(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg1 : tensor<f32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<f32>
    %2 = stablehlo.convert %arg2 : tensor<f32>
    %3 = stablehlo.minimum %2, %1 : tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @_where_276(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @_where_278(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @divmod_280(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> (tensor<200x8xi32>, tensor<200x8xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_281(%arg0, %0) : (tensor<200x8xi32>, tensor<i32>) -> tensor<200x8xi32>
    %2 = call @remainder_290(%arg0, %0) : (tensor<200x8xi32>, tensor<i32>) -> tensor<200x8xi32>
    return %1, %2 : tensor<200x8xi32>, tensor<200x8xi32>
  }
  func.func private @floor_divide_281(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> tensor<200x8xi32> {
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
    %13 = call @_where_288(%10, %12, %1) : (tensor<200x8xi1>, tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi32>
    return %13 : tensor<200x8xi32>
  }
  func.func private @_where_288(%arg0: tensor<200x8xi1>, %arg1: tensor<200x8xi32>, %arg2: tensor<200x8xi32>) -> tensor<200x8xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x8xi1>, tensor<200x8xi32>
    return %0 : tensor<200x8xi32>
  }
  func.func private @remainder_290(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>) -> tensor<200x8xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_217(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_295(%arg0: tensor<200x8xi1>, %arg1: tensor<i32>, %arg2: tensor<200x8xi32>) -> tensor<200x8xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x8xi1>, tensor<200x8xi32>
    return %2 : tensor<200x8xi32>
  }
  func.func private @divmod_302(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> (tensor<200x4xi32>, tensor<200x4xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_303(%arg0, %0) : (tensor<200x4xi32>, tensor<i32>) -> tensor<200x4xi32>
    %2 = call @remainder_312(%arg0, %0) : (tensor<200x4xi32>, tensor<i32>) -> tensor<200x4xi32>
    return %1, %2 : tensor<200x4xi32>, tensor<200x4xi32>
  }
  func.func private @floor_divide_303(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> tensor<200x4xi32> {
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
    %13 = call @_where_310(%10, %12, %1) : (tensor<200x4xi1>, tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi32>
    return %13 : tensor<200x4xi32>
  }
  func.func private @_where_310(%arg0: tensor<200x4xi1>, %arg1: tensor<200x4xi32>, %arg2: tensor<200x4xi32>) -> tensor<200x4xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x4xi1>, tensor<200x4xi32>
    return %0 : tensor<200x4xi32>
  }
  func.func private @remainder_312(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>) -> tensor<200x4xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_217(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_317(%arg0: tensor<200x4xi1>, %arg1: tensor<i32>, %arg2: tensor<200x4xi32>) -> tensor<200x4xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x4xi1>, tensor<200x4xi32>
    return %2 : tensor<200x4xi32>
  }
  func.func private @divmod_325(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> (tensor<200x2xi32>, tensor<200x2xi32>) {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = call @floor_divide_326(%arg0, %0) : (tensor<200x2xi32>, tensor<i32>) -> tensor<200x2xi32>
    %2 = call @remainder_335(%arg0, %0) : (tensor<200x2xi32>, tensor<i32>) -> tensor<200x2xi32>
    return %1, %2 : tensor<200x2xi32>, tensor<200x2xi32>
  }
  func.func private @floor_divide_326(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> tensor<200x2xi32> {
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
    %13 = call @_where_333(%10, %12, %1) : (tensor<200x2xi1>, tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi32>
    return %13 : tensor<200x2xi32>
  }
  func.func private @_where_333(%arg0: tensor<200x2xi1>, %arg1: tensor<200x2xi32>, %arg2: tensor<200x2xi32>) -> tensor<200x2xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<200x2xi1>, tensor<200x2xi32>
    return %0 : tensor<200x2xi32>
  }
  func.func private @remainder_335(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>) -> tensor<200x2xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_217(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_340(%arg0: tensor<200x2xi1>, %arg1: tensor<i32>, %arg2: tensor<200x2xi32>) -> tensor<200x2xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<200x2xi1>, tensor<200x2xi32>
    return %2 : tensor<200x2xi32>
  }
}
