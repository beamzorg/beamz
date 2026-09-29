module @jit_run_scan attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<702x702x701xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<702x701x702xf32> {tf.aliasing_output = 1 : i32}, %arg2: tensor<701x702x702xf32> {tf.aliasing_output = 2 : i32}, %arg3: tensor<701x701x702xf32> {tf.aliasing_output = 3 : i32}, %arg4: tensor<701x702x701xf32> {tf.aliasing_output = 4 : i32}, %arg5: tensor<702x701x701xf32> {tf.aliasing_output = 5 : i32}, %arg6: tensor<701x24x702xf32> {tf.aliasing_output = 6 : i32}, %arg7: tensor<24x701x702xf32> {tf.aliasing_output = 7 : i32}, %arg8: tensor<24x702x701xf32> {tf.aliasing_output = 8 : i32}, %arg9: tensor<701x702x24xf32> {tf.aliasing_output = 9 : i32}, %arg10: tensor<702x701x24xf32> {tf.aliasing_output = 10 : i32}, %arg11: tensor<702x24x701xf32> {tf.aliasing_output = 11 : i32}, %arg12: tensor<702x24x701xf32> {tf.aliasing_output = 12 : i32}, %arg13: tensor<24x702x701xf32> {tf.aliasing_output = 13 : i32}, %arg14: tensor<24x701x702xf32> {tf.aliasing_output = 14 : i32}, %arg15: tensor<702x701x24xf32> {tf.aliasing_output = 15 : i32}, %arg16: tensor<701x702x24xf32> {tf.aliasing_output = 16 : i32}, %arg17: tensor<701x24x702xf32> {tf.aliasing_output = 17 : i32}, %arg18: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 18 : i32}, %arg19: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 19 : i32}, %arg20: tensor<0xi32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 20 : i32}, %arg21: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 21 : i32}, %arg22: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 22 : i32}, %arg23: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 23 : i32}, %arg24: tensor<0x0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>, tf.aliasing_output = 24 : i32}, %arg25: tensor<0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 25 : i32}, %arg26: tensor<0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 26 : i32}, %arg27: tensor<0xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>, tf.aliasing_output = 27 : i32}, %arg28: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>, tf.aliasing_output = 28 : i32}, %arg29: tensor<i32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>, tf.aliasing_output = 29 : i32}, %arg30: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg31: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg32: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg33: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg34: tensor<702x702x701xf32>, %arg35: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg36: tensor<702x701x702xf32>, %arg37: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg38: tensor<701x702x702xf32>) -> (tensor<702x702x701xf32> {jax.result_info = "result.ex"}, tensor<702x701x702xf32> {jax.result_info = "result.ey"}, tensor<701x702x702xf32> {jax.result_info = "result.ez"}, tensor<701x701x702xf32> {jax.result_info = "result.hx"}, tensor<701x702x701xf32> {jax.result_info = "result.hy"}, tensor<702x701x701xf32> {jax.result_info = "result.hz"}, tensor<701x24x702xf32> {jax.result_info = "result.cpml_psi_h_terms[0]"}, tensor<24x701x702xf32> {jax.result_info = "result.cpml_psi_h_terms[1]"}, tensor<24x702x701xf32> {jax.result_info = "result.cpml_psi_h_terms[2]"}, tensor<701x702x24xf32> {jax.result_info = "result.cpml_psi_h_terms[3]"}, tensor<702x701x24xf32> {jax.result_info = "result.cpml_psi_h_terms[4]"}, tensor<702x24x701xf32> {jax.result_info = "result.cpml_psi_h_terms[5]"}, tensor<702x24x701xf32> {jax.result_info = "result.cpml_psi_e_terms[0]"}, tensor<24x702x701xf32> {jax.result_info = "result.cpml_psi_e_terms[1]"}, tensor<24x701x702xf32> {jax.result_info = "result.cpml_psi_e_terms[2]"}, tensor<702x701x24xf32> {jax.result_info = "result.cpml_psi_e_terms[3]"}, tensor<701x702x24xf32> {jax.result_info = "result.cpml_psi_e_terms[4]"}, tensor<701x24x702xf32> {jax.result_info = "result.cpml_psi_e_terms[5]"}, tensor<0x0xf32> {jax.result_info = "result.powers"}, tensor<0x0xf32> {jax.result_info = "result.timestamps"}, tensor<0xi32> {jax.result_info = "result.counts"}, tensor<0x0xf32> {jax.result_info = "result.freq_flux_re"}, tensor<0x0xf32> {jax.result_info = "result.freq_flux_im"}, tensor<0x0xf32> {jax.result_info = "result.freq_phase_re"}, tensor<0x0xf32> {jax.result_info = "result.freq_phase_im"}, tensor<0xf32> {jax.result_info = "result.dft_vec_re"}, tensor<0xf32> {jax.result_info = "result.dft_vec_im"}, tensor<0xf32> {jax.result_info = "result.dft_weight_sum"}, tensor<f32> {jax.result_info = "result.t"}, tensor<i32> {jax.result_info = "result.current_step"}) {
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
    %c = stablehlo.constant dense<0> : tensor<1xi32>
    %c_18 = stablehlo.constant dense<700> : tensor<1xi32>
    %c_19 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_20 = stablehlo.constant dense<700> : tensor<1xi32>
    %c_21 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_22 = stablehlo.constant dense<700> : tensor<1xi32>
    %c_23 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_24 = stablehlo.constant dense<700> : tensor<1xi32>
    %c_25 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_26 = stablehlo.constant dense<700> : tensor<1xi32>
    %c_27 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_28 = stablehlo.constant dense<700> : tensor<1xi32>
    %cst_29 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_30 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_31 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_32 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_33 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_34 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_35 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_36 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_37 = stablehlo.constant dense_resource<__elided__> : tensor<24x1x1xf32>
    %cst_38 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_39 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_40 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_41 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_42 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_43 = stablehlo.constant dense_resource<__elided__> : tensor<1x1x24xf32>
    %cst_44 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_45 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_46 = stablehlo.constant dense_resource<__elided__> : tensor<1x24x1xf32>
    %cst_47 = stablehlo.constant dense<1.25663701E-6> : tensor<f32>
    %0 = stablehlo.divide %cst, %cst_47 : tensor<f32>
    %cst_48 = stablehlo.constant dense<8.85418821E-12> : tensor<f32>
    %1 = stablehlo.divide %cst, %cst_48 : tensor<f32>
    %cst_49 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %2 = stablehlo.multiply %cst_49, %0 : tensor<f32>
    %3 = stablehlo.multiply %2, %arg30 : tensor<f32>
    %cst_50 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.add %cst_50, %3 : tensor<f32>
    %5 = stablehlo.divide %0, %4 : tensor<f32>
    %cst_51 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %6 = stablehlo.multiply %cst_51, %1 : tensor<f32>
    %7 = stablehlo.multiply %6, %arg33 : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<702x702x701xf32>
    %9 = stablehlo.add %arg34, %8 : tensor<702x702x701xf32>
    %10 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<702x702x701xf32>
    %11 = stablehlo.divide %10, %9 : tensor<702x702x701xf32>
    %cst_52 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %12 = stablehlo.multiply %cst_52, %0 : tensor<f32>
    %13 = stablehlo.multiply %12, %arg31 : tensor<f32>
    %cst_53 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %14 = stablehlo.add %cst_53, %13 : tensor<f32>
    %15 = stablehlo.divide %0, %14 : tensor<f32>
    %cst_54 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %16 = stablehlo.multiply %cst_54, %1 : tensor<f32>
    %17 = stablehlo.multiply %16, %arg35 : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<702x701x702xf32>
    %19 = stablehlo.add %arg36, %18 : tensor<702x701x702xf32>
    %20 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<702x701x702xf32>
    %21 = stablehlo.divide %20, %19 : tensor<702x701x702xf32>
    %cst_55 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %22 = stablehlo.multiply %cst_55, %0 : tensor<f32>
    %23 = stablehlo.multiply %22, %arg32 : tensor<f32>
    %cst_56 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %24 = stablehlo.add %cst_56, %23 : tensor<f32>
    %25 = stablehlo.divide %0, %24 : tensor<f32>
    %cst_57 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %26 = stablehlo.multiply %cst_57, %1 : tensor<f32>
    %27 = stablehlo.multiply %26, %arg37 : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [] : (tensor<f32>) -> tensor<701x702x702xf32>
    %29 = stablehlo.add %arg38, %28 : tensor<701x702x702xf32>
    %30 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<701x702x702xf32>
    %31 = stablehlo.divide %30, %29 : tensor<701x702x702xf32>
    %32 = stablehlo.iota dim = 0 : tensor<32xi32>
    %c_58 = stablehlo.constant dense<0> : tensor<i32>
    %33:84 = stablehlo.while(%iterArg = %32, %iterArg_59 = %cst, %iterArg_60 = %arg28, %iterArg_61 = %cst_0, %iterArg_62 = %cst_1, %iterArg_63 = %cst_2, %iterArg_64 = %cst_3, %iterArg_65 = %cst_4, %iterArg_66 = %cst_5, %iterArg_67 = %cst_6, %iterArg_68 = %cst_7, %iterArg_69 = %cst_8, %iterArg_70 = %cst_9, %iterArg_71 = %cst_10, %iterArg_72 = %cst_11, %iterArg_73 = %cst_12, %iterArg_74 = %cst_13, %iterArg_75 = %cst_14, %iterArg_76 = %cst_15, %iterArg_77 = %cst_16, %iterArg_78 = %cst_17, %iterArg_79 = %arg30, %iterArg_80 = %5, %iterArg_81 = %arg31, %iterArg_82 = %15, %iterArg_83 = %arg32, %iterArg_84 = %25, %iterArg_85 = %c, %iterArg_86 = %c_18, %iterArg_87 = %c_19, %iterArg_88 = %c_20, %iterArg_89 = %c_21, %iterArg_90 = %c_22, %iterArg_91 = %c_23, %iterArg_92 = %c_24, %iterArg_93 = %c_25, %iterArg_94 = %c_26, %iterArg_95 = %c_27, %iterArg_96 = %c_28, %iterArg_97 = %cst_29, %iterArg_98 = %cst_30, %iterArg_99 = %cst_31, %iterArg_100 = %cst_32, %iterArg_101 = %cst_33, %iterArg_102 = %cst_34, %iterArg_103 = %cst_35, %iterArg_104 = %cst_36, %iterArg_105 = %cst_37, %iterArg_106 = %cst_38, %iterArg_107 = %cst_39, %iterArg_108 = %cst_40, %iterArg_109 = %cst_41, %iterArg_110 = %cst_42, %iterArg_111 = %cst_43, %iterArg_112 = %cst_44, %iterArg_113 = %cst_45, %iterArg_114 = %cst_46, %iterArg_115 = %arg33, %iterArg_116 = %11, %iterArg_117 = %arg35, %iterArg_118 = %21, %iterArg_119 = %arg37, %iterArg_120 = %31, %iterArg_121 = %c_58, %iterArg_122 = %arg0, %iterArg_123 = %arg1, %iterArg_124 = %arg2, %iterArg_125 = %arg3, %iterArg_126 = %arg4, %iterArg_127 = %arg5, %iterArg_128 = %arg6, %iterArg_129 = %arg7, %iterArg_130 = %arg8, %iterArg_131 = %arg9, %iterArg_132 = %arg10, %iterArg_133 = %arg11, %iterArg_134 = %arg12, %iterArg_135 = %arg13, %iterArg_136 = %arg14, %iterArg_137 = %arg15, %iterArg_138 = %arg16, %iterArg_139 = %arg17, %iterArg_140 = %arg28, %iterArg_141 = %arg29) : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<702x702x701xf32>, tensor<f32>, tensor<702x701x702xf32>, tensor<f32>, tensor<701x702x702xf32>, tensor<i32>, tensor<702x702x701xf32>, tensor<702x701x702xf32>, tensor<701x702x702xf32>, tensor<701x701x702xf32>, tensor<701x702x701xf32>, tensor<702x701x701xf32>, tensor<701x24x702xf32>, tensor<24x701x702xf32>, tensor<24x702x701xf32>, tensor<701x702x24xf32>, tensor<702x701x24xf32>, tensor<702x24x701xf32>, tensor<702x24x701xf32>, tensor<24x702x701xf32>, tensor<24x701x702xf32>, tensor<702x701x24xf32>, tensor<701x702x24xf32>, tensor<701x24x702xf32>, tensor<f32>, tensor<i32>
    cond {
      %c_142 = stablehlo.constant dense<32> : tensor<i32>
      %34 = stablehlo.compare  LT, %iterArg_121, %c_142,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %34 : tensor<i1>
    } do {
      %34 = stablehlo.dynamic_slice %iterArg, %iterArg_121, sizes = [1] : (tensor<32xi32>, tensor<i32>) -> tensor<1xi32>
      %35 = stablehlo.reshape %34 : (tensor<1xi32>) -> tensor<i32>
      %36:20 = func.call @closed_call(%iterArg_59, %iterArg_60, %iterArg_61, %iterArg_62, %iterArg_63, %iterArg_64, %iterArg_65, %iterArg_66, %iterArg_67, %iterArg_68, %iterArg_69, %iterArg_70, %iterArg_71, %iterArg_72, %iterArg_73, %iterArg_74, %iterArg_75, %iterArg_76, %iterArg_77, %iterArg_78, %iterArg_79, %iterArg_80, %iterArg_81, %iterArg_82, %iterArg_83, %iterArg_84, %iterArg_85, %iterArg_86, %iterArg_87, %iterArg_88, %iterArg_89, %iterArg_90, %iterArg_91, %iterArg_92, %iterArg_93, %iterArg_94, %iterArg_95, %iterArg_96, %iterArg_97, %iterArg_98, %iterArg_99, %iterArg_100, %iterArg_101, %iterArg_102, %iterArg_103, %iterArg_104, %iterArg_105, %iterArg_106, %iterArg_107, %iterArg_108, %iterArg_109, %iterArg_110, %iterArg_111, %iterArg_112, %iterArg_113, %iterArg_114, %iterArg_115, %iterArg_116, %iterArg_117, %iterArg_118, %iterArg_119, %iterArg_120, %iterArg_122, %iterArg_123, %iterArg_124, %iterArg_125, %iterArg_126, %iterArg_127, %iterArg_128, %iterArg_129, %iterArg_130, %iterArg_131, %iterArg_132, %iterArg_133, %iterArg_134, %iterArg_135, %iterArg_136, %iterArg_137, %iterArg_138, %iterArg_139, %iterArg_140, %iterArg_141, %35) : (tensor<f32>, tensor<f32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<702x702x701xf32>, tensor<f32>, tensor<702x701x702xf32>, tensor<f32>, tensor<701x702x702xf32>, tensor<702x702x701xf32>, tensor<702x701x702xf32>, tensor<701x702x702xf32>, tensor<701x701x702xf32>, tensor<701x702x701xf32>, tensor<702x701x701xf32>, tensor<701x24x702xf32>, tensor<24x701x702xf32>, tensor<24x702x701xf32>, tensor<701x702x24xf32>, tensor<702x701x24xf32>, tensor<702x24x701xf32>, tensor<702x24x701xf32>, tensor<24x702x701xf32>, tensor<24x701x702xf32>, tensor<702x701x24xf32>, tensor<701x702x24xf32>, tensor<701x24x702xf32>, tensor<f32>, tensor<i32>, tensor<i32>) -> (tensor<702x702x701xf32>, tensor<702x701x702xf32>, tensor<701x702x702xf32>, tensor<701x701x702xf32>, tensor<701x702x701xf32>, tensor<702x701x701xf32>, tensor<701x24x702xf32>, tensor<24x701x702xf32>, tensor<24x702x701xf32>, tensor<701x702x24xf32>, tensor<702x701x24xf32>, tensor<702x24x701xf32>, tensor<702x24x701xf32>, tensor<24x702x701xf32>, tensor<24x701x702xf32>, tensor<702x701x24xf32>, tensor<701x702x24xf32>, tensor<701x24x702xf32>, tensor<f32>, tensor<i32>)
      %c_142 = stablehlo.constant dense<1> : tensor<i32>
      %37 = stablehlo.add %iterArg_121, %c_142 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_59, %iterArg_60, %iterArg_61, %iterArg_62, %iterArg_63, %iterArg_64, %iterArg_65, %iterArg_66, %iterArg_67, %iterArg_68, %iterArg_69, %iterArg_70, %iterArg_71, %iterArg_72, %iterArg_73, %iterArg_74, %iterArg_75, %iterArg_76, %iterArg_77, %iterArg_78, %iterArg_79, %iterArg_80, %iterArg_81, %iterArg_82, %iterArg_83, %iterArg_84, %iterArg_85, %iterArg_86, %iterArg_87, %iterArg_88, %iterArg_89, %iterArg_90, %iterArg_91, %iterArg_92, %iterArg_93, %iterArg_94, %iterArg_95, %iterArg_96, %iterArg_97, %iterArg_98, %iterArg_99, %iterArg_100, %iterArg_101, %iterArg_102, %iterArg_103, %iterArg_104, %iterArg_105, %iterArg_106, %iterArg_107, %iterArg_108, %iterArg_109, %iterArg_110, %iterArg_111, %iterArg_112, %iterArg_113, %iterArg_114, %iterArg_115, %iterArg_116, %iterArg_117, %iterArg_118, %iterArg_119, %iterArg_120, %37, %36#0, %36#1, %36#2, %36#3, %36#4, %36#5, %36#6, %36#7, %36#8, %36#9, %36#10, %36#11, %36#12, %36#13, %36#14, %36#15, %36#16, %36#17, %36#18, %36#19 : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<f32>, tensor<702x702x701xf32>, tensor<f32>, tensor<702x701x702xf32>, tensor<f32>, tensor<701x702x702xf32>, tensor<i32>, tensor<702x702x701xf32>, tensor<702x701x702xf32>, tensor<701x702x702xf32>, tensor<701x701x702xf32>, tensor<701x702x701xf32>, tensor<702x701x701xf32>, tensor<701x24x702xf32>, tensor<24x701x702xf32>, tensor<24x702x701xf32>, tensor<701x702x24xf32>, tensor<702x701x24xf32>, tensor<702x24x701xf32>, tensor<702x24x701xf32>, tensor<24x702x701xf32>, tensor<24x701x702xf32>, tensor<702x701x24xf32>, tensor<701x702x24xf32>, tensor<701x24x702xf32>, tensor<f32>, tensor<i32>
    }
    return %33#64, %33#65, %33#66, %33#67, %33#68, %33#69, %33#70, %33#71, %33#72, %33#73, %33#74, %33#75, %33#76, %33#77, %33#78, %33#79, %33#80, %33#81, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %33#82, %33#83 : tensor<702x702x701xf32>, tensor<702x701x702xf32>, tensor<701x702x702xf32>, tensor<701x701x702xf32>, tensor<701x702x701xf32>, tensor<702x701x701xf32>, tensor<701x24x702xf32>, tensor<24x701x702xf32>, tensor<24x702x701xf32>, tensor<701x702x24xf32>, tensor<702x701x24xf32>, tensor<702x24x701xf32>, tensor<702x24x701xf32>, tensor<24x702x701xf32>, tensor<24x701x702xf32>, tensor<702x701x24xf32>, tensor<701x702x24xf32>, tensor<701x24x702xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0xi32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @closed_call(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<1x24x1xf32>, %arg3: tensor<1x24x1xf32>, %arg4: tensor<1x24x1xf32>, %arg5: tensor<24x1x1xf32>, %arg6: tensor<24x1x1xf32>, %arg7: tensor<24x1x1xf32>, %arg8: tensor<24x1x1xf32>, %arg9: tensor<24x1x1xf32>, %arg10: tensor<24x1x1xf32>, %arg11: tensor<1x1x24xf32>, %arg12: tensor<1x1x24xf32>, %arg13: tensor<1x1x24xf32>, %arg14: tensor<1x1x24xf32>, %arg15: tensor<1x1x24xf32>, %arg16: tensor<1x1x24xf32>, %arg17: tensor<1x24x1xf32>, %arg18: tensor<1x24x1xf32>, %arg19: tensor<1x24x1xf32>, %arg20: tensor<f32>, %arg21: tensor<f32>, %arg22: tensor<f32>, %arg23: tensor<f32>, %arg24: tensor<f32>, %arg25: tensor<f32>, %arg26: tensor<1xi32>, %arg27: tensor<1xi32>, %arg28: tensor<1xi32>, %arg29: tensor<1xi32>, %arg30: tensor<1xi32>, %arg31: tensor<1xi32>, %arg32: tensor<1xi32>, %arg33: tensor<1xi32>, %arg34: tensor<1xi32>, %arg35: tensor<1xi32>, %arg36: tensor<1xi32>, %arg37: tensor<1xi32>, %arg38: tensor<1x24x1xf32>, %arg39: tensor<1x24x1xf32>, %arg40: tensor<1x24x1xf32>, %arg41: tensor<24x1x1xf32>, %arg42: tensor<24x1x1xf32>, %arg43: tensor<24x1x1xf32>, %arg44: tensor<24x1x1xf32>, %arg45: tensor<24x1x1xf32>, %arg46: tensor<24x1x1xf32>, %arg47: tensor<1x1x24xf32>, %arg48: tensor<1x1x24xf32>, %arg49: tensor<1x1x24xf32>, %arg50: tensor<1x1x24xf32>, %arg51: tensor<1x1x24xf32>, %arg52: tensor<1x1x24xf32>, %arg53: tensor<1x24x1xf32>, %arg54: tensor<1x24x1xf32>, %arg55: tensor<1x24x1xf32>, %arg56: tensor<f32>, %arg57: tensor<702x702x701xf32>, %arg58: tensor<f32>, %arg59: tensor<702x701x702xf32>, %arg60: tensor<f32>, %arg61: tensor<701x702x702xf32>, %arg62: tensor<702x702x701xf32>, %arg63: tensor<702x701x702xf32>, %arg64: tensor<701x702x702xf32>, %arg65: tensor<701x701x702xf32>, %arg66: tensor<701x702x701xf32>, %arg67: tensor<702x701x701xf32>, %arg68: tensor<701x24x702xf32>, %arg69: tensor<24x701x702xf32>, %arg70: tensor<24x702x701xf32>, %arg71: tensor<701x702x24xf32>, %arg72: tensor<702x701x24xf32>, %arg73: tensor<702x24x701xf32>, %arg74: tensor<702x24x701xf32>, %arg75: tensor<24x702x701xf32>, %arg76: tensor<24x701x702xf32>, %arg77: tensor<702x701x24xf32>, %arg78: tensor<701x702x24xf32>, %arg79: tensor<701x24x702xf32>, %arg80: tensor<f32>, %arg81: tensor<i32>, %arg82: tensor<i32>) -> (tensor<702x702x701xf32>, tensor<702x701x702xf32>, tensor<701x702x702xf32>, tensor<701x701x702xf32>, tensor<701x702x701xf32>, tensor<702x701x701xf32>, tensor<701x24x702xf32>, tensor<24x701x702xf32>, tensor<24x702x701xf32>, tensor<701x702x24xf32>, tensor<702x701x24xf32>, tensor<702x24x701xf32>, tensor<702x24x701xf32>, tensor<24x702x701xf32>, tensor<24x701x702xf32>, tensor<702x701x24xf32>, tensor<701x702x24xf32>, tensor<701x24x702xf32>, tensor<f32>, tensor<i32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg82, %c : tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<f32>
    %3 = stablehlo.add %arg1, %2 : tensor<f32>
    %4 = call @diff(%arg64) : (tensor<701x702x702xf32>) -> tensor<701x701x702xf32>
    %cst = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<701x701x702xf32>
    %6 = stablehlo.divide %4, %5 : tensor<701x701x702xf32>
    %7 = stablehlo.slice %6 [0:701, 0:12, 0:702] : (tensor<701x701x702xf32>) -> tensor<701x12x702xf32>
    %8 = stablehlo.slice %6 [0:701, 689:701, 0:702] : (tensor<701x701x702xf32>) -> tensor<701x12x702xf32>
    %9 = stablehlo.concatenate %7, %8, dim = 1 : (tensor<701x12x702xf32>, tensor<701x12x702xf32>) -> tensor<701x24x702xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<701x24x702xf32>
    %11 = stablehlo.multiply %10, %arg68 : tensor<701x24x702xf32>
    %12 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<701x24x702xf32>
    %13 = stablehlo.multiply %12, %9 : tensor<701x24x702xf32>
    %14 = stablehlo.add %11, %13 : tensor<701x24x702xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<701x24x702xf32>
    %16 = stablehlo.multiply %9, %15 : tensor<701x24x702xf32>
    %17 = stablehlo.add %16, %14 : tensor<701x24x702xf32>
    %18 = stablehlo.slice %17 [0:701, 0:12, 0:702] : (tensor<701x24x702xf32>) -> tensor<701x12x702xf32>
    %19 = stablehlo.slice %6 [0:701, 12:689, 0:702] : (tensor<701x701x702xf32>) -> tensor<701x677x702xf32>
    %20 = stablehlo.slice %17 [0:701, 12:24, 0:702] : (tensor<701x24x702xf32>) -> tensor<701x12x702xf32>
    %21 = stablehlo.concatenate %18, %19, %20, dim = 1 : (tensor<701x12x702xf32>, tensor<701x677x702xf32>, tensor<701x12x702xf32>) -> tensor<701x701x702xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<701x701x702xf32>
    %23 = stablehlo.multiply %22, %21 : tensor<701x701x702xf32>
    %24 = call @diff_19(%arg63) : (tensor<702x701x702xf32>) -> tensor<701x701x702xf32>
    %cst_1 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<701x701x702xf32>
    %26 = stablehlo.divide %24, %25 : tensor<701x701x702xf32>
    %27 = stablehlo.slice %26 [0:12, 0:701, 0:702] : (tensor<701x701x702xf32>) -> tensor<12x701x702xf32>
    %28 = stablehlo.slice %26 [689:701, 0:701, 0:702] : (tensor<701x701x702xf32>) -> tensor<12x701x702xf32>
    %29 = stablehlo.concatenate %27, %28, dim = 0 : (tensor<12x701x702xf32>, tensor<12x701x702xf32>) -> tensor<24x701x702xf32>
    %30 = stablehlo.broadcast_in_dim %arg5, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x701x702xf32>
    %31 = stablehlo.multiply %30, %arg69 : tensor<24x701x702xf32>
    %32 = stablehlo.broadcast_in_dim %arg6, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x701x702xf32>
    %33 = stablehlo.multiply %32, %29 : tensor<24x701x702xf32>
    %34 = stablehlo.add %31, %33 : tensor<24x701x702xf32>
    %35 = stablehlo.broadcast_in_dim %arg7, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x701x702xf32>
    %36 = stablehlo.multiply %29, %35 : tensor<24x701x702xf32>
    %37 = stablehlo.add %36, %34 : tensor<24x701x702xf32>
    %38 = stablehlo.slice %37 [0:12, 0:701, 0:702] : (tensor<24x701x702xf32>) -> tensor<12x701x702xf32>
    %39 = stablehlo.slice %26 [12:689, 0:701, 0:702] : (tensor<701x701x702xf32>) -> tensor<677x701x702xf32>
    %40 = stablehlo.slice %37 [12:24, 0:701, 0:702] : (tensor<24x701x702xf32>) -> tensor<12x701x702xf32>
    %41 = stablehlo.concatenate %38, %39, %40, dim = 0 : (tensor<12x701x702xf32>, tensor<677x701x702xf32>, tensor<12x701x702xf32>) -> tensor<701x701x702xf32>
    %cst_2 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<701x701x702xf32>
    %43 = stablehlo.multiply %42, %41 : tensor<701x701x702xf32>
    %44 = call @diff_32(%arg62) : (tensor<702x702x701xf32>) -> tensor<701x702x701xf32>
    %cst_3 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<701x702x701xf32>
    %46 = stablehlo.divide %44, %45 : tensor<701x702x701xf32>
    %47 = stablehlo.slice %46 [0:12, 0:702, 0:701] : (tensor<701x702x701xf32>) -> tensor<12x702x701xf32>
    %48 = stablehlo.slice %46 [689:701, 0:702, 0:701] : (tensor<701x702x701xf32>) -> tensor<12x702x701xf32>
    %49 = stablehlo.concatenate %47, %48, dim = 0 : (tensor<12x702x701xf32>, tensor<12x702x701xf32>) -> tensor<24x702x701xf32>
    %50 = stablehlo.broadcast_in_dim %arg8, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x702x701xf32>
    %51 = stablehlo.multiply %50, %arg70 : tensor<24x702x701xf32>
    %52 = stablehlo.broadcast_in_dim %arg9, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x702x701xf32>
    %53 = stablehlo.multiply %52, %49 : tensor<24x702x701xf32>
    %54 = stablehlo.add %51, %53 : tensor<24x702x701xf32>
    %55 = stablehlo.broadcast_in_dim %arg10, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x702x701xf32>
    %56 = stablehlo.multiply %49, %55 : tensor<24x702x701xf32>
    %57 = stablehlo.add %56, %54 : tensor<24x702x701xf32>
    %58 = stablehlo.slice %57 [0:12, 0:702, 0:701] : (tensor<24x702x701xf32>) -> tensor<12x702x701xf32>
    %59 = stablehlo.slice %46 [12:689, 0:702, 0:701] : (tensor<701x702x701xf32>) -> tensor<677x702x701xf32>
    %60 = stablehlo.slice %57 [12:24, 0:702, 0:701] : (tensor<24x702x701xf32>) -> tensor<12x702x701xf32>
    %61 = stablehlo.concatenate %58, %59, %60, dim = 0 : (tensor<12x702x701xf32>, tensor<677x702x701xf32>, tensor<12x702x701xf32>) -> tensor<701x702x701xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<701x702x701xf32>
    %63 = stablehlo.multiply %62, %61 : tensor<701x702x701xf32>
    %64 = call @diff_48(%arg64) : (tensor<701x702x702xf32>) -> tensor<701x702x701xf32>
    %cst_5 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<701x702x701xf32>
    %66 = stablehlo.divide %64, %65 : tensor<701x702x701xf32>
    %67 = stablehlo.slice %66 [0:701, 0:702, 0:12] : (tensor<701x702x701xf32>) -> tensor<701x702x12xf32>
    %68 = stablehlo.slice %66 [0:701, 0:702, 689:701] : (tensor<701x702x701xf32>) -> tensor<701x702x12xf32>
    %69 = stablehlo.concatenate %67, %68, dim = 2 : (tensor<701x702x12xf32>, tensor<701x702x12xf32>) -> tensor<701x702x24xf32>
    %70 = stablehlo.broadcast_in_dim %arg11, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<701x702x24xf32>
    %71 = stablehlo.multiply %70, %arg71 : tensor<701x702x24xf32>
    %72 = stablehlo.broadcast_in_dim %arg12, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<701x702x24xf32>
    %73 = stablehlo.multiply %72, %69 : tensor<701x702x24xf32>
    %74 = stablehlo.add %71, %73 : tensor<701x702x24xf32>
    %75 = stablehlo.broadcast_in_dim %arg13, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<701x702x24xf32>
    %76 = stablehlo.multiply %69, %75 : tensor<701x702x24xf32>
    %77 = stablehlo.add %76, %74 : tensor<701x702x24xf32>
    %78 = stablehlo.slice %77 [0:701, 0:702, 0:12] : (tensor<701x702x24xf32>) -> tensor<701x702x12xf32>
    %79 = stablehlo.slice %66 [0:701, 0:702, 12:689] : (tensor<701x702x701xf32>) -> tensor<701x702x677xf32>
    %80 = stablehlo.slice %77 [0:701, 0:702, 12:24] : (tensor<701x702x24xf32>) -> tensor<701x702x12xf32>
    %81 = stablehlo.concatenate %78, %79, %80, dim = 2 : (tensor<701x702x12xf32>, tensor<701x702x677xf32>, tensor<701x702x12xf32>) -> tensor<701x702x701xf32>
    %cst_6 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %82 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<701x702x701xf32>
    %83 = stablehlo.multiply %82, %81 : tensor<701x702x701xf32>
    %84 = call @diff_61(%arg63) : (tensor<702x701x702xf32>) -> tensor<702x701x701xf32>
    %cst_7 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %85 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<702x701x701xf32>
    %86 = stablehlo.divide %84, %85 : tensor<702x701x701xf32>
    %87 = stablehlo.slice %86 [0:702, 0:701, 0:12] : (tensor<702x701x701xf32>) -> tensor<702x701x12xf32>
    %88 = stablehlo.slice %86 [0:702, 0:701, 689:701] : (tensor<702x701x701xf32>) -> tensor<702x701x12xf32>
    %89 = stablehlo.concatenate %87, %88, dim = 2 : (tensor<702x701x12xf32>, tensor<702x701x12xf32>) -> tensor<702x701x24xf32>
    %90 = stablehlo.broadcast_in_dim %arg14, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<702x701x24xf32>
    %91 = stablehlo.multiply %90, %arg72 : tensor<702x701x24xf32>
    %92 = stablehlo.broadcast_in_dim %arg15, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<702x701x24xf32>
    %93 = stablehlo.multiply %92, %89 : tensor<702x701x24xf32>
    %94 = stablehlo.add %91, %93 : tensor<702x701x24xf32>
    %95 = stablehlo.broadcast_in_dim %arg16, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<702x701x24xf32>
    %96 = stablehlo.multiply %89, %95 : tensor<702x701x24xf32>
    %97 = stablehlo.add %96, %94 : tensor<702x701x24xf32>
    %98 = stablehlo.slice %97 [0:702, 0:701, 0:12] : (tensor<702x701x24xf32>) -> tensor<702x701x12xf32>
    %99 = stablehlo.slice %86 [0:702, 0:701, 12:689] : (tensor<702x701x701xf32>) -> tensor<702x701x677xf32>
    %100 = stablehlo.slice %97 [0:702, 0:701, 12:24] : (tensor<702x701x24xf32>) -> tensor<702x701x12xf32>
    %101 = stablehlo.concatenate %98, %99, %100, dim = 2 : (tensor<702x701x12xf32>, tensor<702x701x677xf32>, tensor<702x701x12xf32>) -> tensor<702x701x701xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %102 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<702x701x701xf32>
    %103 = stablehlo.multiply %102, %101 : tensor<702x701x701xf32>
    %104 = call @diff_77(%arg62) : (tensor<702x702x701xf32>) -> tensor<702x701x701xf32>
    %cst_9 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %105 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<702x701x701xf32>
    %106 = stablehlo.divide %104, %105 : tensor<702x701x701xf32>
    %107 = stablehlo.slice %106 [0:702, 0:12, 0:701] : (tensor<702x701x701xf32>) -> tensor<702x12x701xf32>
    %108 = stablehlo.slice %106 [0:702, 689:701, 0:701] : (tensor<702x701x701xf32>) -> tensor<702x12x701xf32>
    %109 = stablehlo.concatenate %107, %108, dim = 1 : (tensor<702x12x701xf32>, tensor<702x12x701xf32>) -> tensor<702x24x701xf32>
    %110 = stablehlo.broadcast_in_dim %arg17, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<702x24x701xf32>
    %111 = stablehlo.multiply %110, %arg73 : tensor<702x24x701xf32>
    %112 = stablehlo.broadcast_in_dim %arg18, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<702x24x701xf32>
    %113 = stablehlo.multiply %112, %109 : tensor<702x24x701xf32>
    %114 = stablehlo.add %111, %113 : tensor<702x24x701xf32>
    %115 = stablehlo.broadcast_in_dim %arg19, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<702x24x701xf32>
    %116 = stablehlo.multiply %109, %115 : tensor<702x24x701xf32>
    %117 = stablehlo.add %116, %114 : tensor<702x24x701xf32>
    %118 = stablehlo.slice %117 [0:702, 0:12, 0:701] : (tensor<702x24x701xf32>) -> tensor<702x12x701xf32>
    %119 = stablehlo.slice %106 [0:702, 12:689, 0:701] : (tensor<702x701x701xf32>) -> tensor<702x677x701xf32>
    %120 = stablehlo.slice %117 [0:702, 12:24, 0:701] : (tensor<702x24x701xf32>) -> tensor<702x12x701xf32>
    %121 = stablehlo.concatenate %118, %119, %120, dim = 1 : (tensor<702x12x701xf32>, tensor<702x677x701xf32>, tensor<702x12x701xf32>) -> tensor<702x701x701xf32>
    %cst_10 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %122 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<702x701x701xf32>
    %123 = stablehlo.multiply %122, %121 : tensor<702x701x701xf32>
    %124 = stablehlo.add %23, %43 : tensor<701x701x702xf32>
    %125 = stablehlo.add %63, %83 : tensor<701x702x701xf32>
    %126 = stablehlo.add %103, %123 : tensor<702x701x701xf32>
    %127 = stablehlo.broadcast_in_dim %arg20, dims = [] : (tensor<f32>) -> tensor<701x701x702xf32>
    %128 = stablehlo.multiply %127, %arg65 : tensor<701x701x702xf32>
    %129 = stablehlo.add %124, %128 : tensor<701x701x702xf32>
    %130 = stablehlo.broadcast_in_dim %arg21, dims = [] : (tensor<f32>) -> tensor<701x701x702xf32>
    %131 = stablehlo.multiply %130, %129 : tensor<701x701x702xf32>
    %132 = stablehlo.subtract %arg65, %131 : tensor<701x701x702xf32>
    %133 = stablehlo.broadcast_in_dim %arg22, dims = [] : (tensor<f32>) -> tensor<701x702x701xf32>
    %134 = stablehlo.multiply %133, %arg66 : tensor<701x702x701xf32>
    %135 = stablehlo.add %125, %134 : tensor<701x702x701xf32>
    %136 = stablehlo.broadcast_in_dim %arg23, dims = [] : (tensor<f32>) -> tensor<701x702x701xf32>
    %137 = stablehlo.multiply %136, %135 : tensor<701x702x701xf32>
    %138 = stablehlo.subtract %arg66, %137 : tensor<701x702x701xf32>
    %139 = stablehlo.broadcast_in_dim %arg24, dims = [] : (tensor<f32>) -> tensor<702x701x701xf32>
    %140 = stablehlo.multiply %139, %arg67 : tensor<702x701x701xf32>
    %141 = stablehlo.add %126, %140 : tensor<702x701x701xf32>
    %142 = stablehlo.broadcast_in_dim %arg25, dims = [] : (tensor<f32>) -> tensor<702x701x701xf32>
    %143 = stablehlo.multiply %142, %141 : tensor<702x701x701xf32>
    %144 = stablehlo.subtract %arg67, %143 : tensor<702x701x701xf32>
    %145 = stablehlo.slice %144 [0:702, 701:701, 0:701] : (tensor<702x701x701xf32>) -> tensor<702x0x701xf32>
    %146 = call @_take(%144, %arg26) : (tensor<702x701x701xf32>, tensor<1xi32>) -> tensor<702x1x701xf32>
    %147 = call @_take(%144, %arg27) : (tensor<702x701x701xf32>, tensor<1xi32>) -> tensor<702x1x701xf32>
    %148 = stablehlo.concatenate %146, %144, %147, %145, dim = 1 : (tensor<702x1x701xf32>, tensor<702x701x701xf32>, tensor<702x1x701xf32>, tensor<702x0x701xf32>) -> tensor<702x703x701xf32>
    %149 = stablehlo.slice %138 [701:701, 0:702, 0:701] : (tensor<701x702x701xf32>) -> tensor<0x702x701xf32>
    %150 = call @_take_103(%138, %arg28) : (tensor<701x702x701xf32>, tensor<1xi32>) -> tensor<1x702x701xf32>
    %151 = call @_take_103(%138, %arg29) : (tensor<701x702x701xf32>, tensor<1xi32>) -> tensor<1x702x701xf32>
    %152 = stablehlo.concatenate %150, %138, %151, %149, dim = 0 : (tensor<1x702x701xf32>, tensor<701x702x701xf32>, tensor<1x702x701xf32>, tensor<0x702x701xf32>) -> tensor<703x702x701xf32>
    %153 = stablehlo.slice %132 [701:701, 0:701, 0:702] : (tensor<701x701x702xf32>) -> tensor<0x701x702xf32>
    %154 = call @_take_111(%132, %arg30) : (tensor<701x701x702xf32>, tensor<1xi32>) -> tensor<1x701x702xf32>
    %155 = call @_take_111(%132, %arg31) : (tensor<701x701x702xf32>, tensor<1xi32>) -> tensor<1x701x702xf32>
    %156 = stablehlo.concatenate %154, %132, %155, %153, dim = 0 : (tensor<1x701x702xf32>, tensor<701x701x702xf32>, tensor<1x701x702xf32>, tensor<0x701x702xf32>) -> tensor<703x701x702xf32>
    %157 = stablehlo.slice %144 [0:702, 0:701, 701:701] : (tensor<702x701x701xf32>) -> tensor<702x701x0xf32>
    %158 = call @_take_119(%144, %arg32) : (tensor<702x701x701xf32>, tensor<1xi32>) -> tensor<702x701x1xf32>
    %159 = call @_take_119(%144, %arg33) : (tensor<702x701x701xf32>, tensor<1xi32>) -> tensor<702x701x1xf32>
    %160 = stablehlo.concatenate %158, %144, %159, %157, dim = 2 : (tensor<702x701x1xf32>, tensor<702x701x701xf32>, tensor<702x701x1xf32>, tensor<702x701x0xf32>) -> tensor<702x701x703xf32>
    %161 = stablehlo.slice %138 [0:701, 0:702, 701:701] : (tensor<701x702x701xf32>) -> tensor<701x702x0xf32>
    %162 = call @_take_127(%138, %arg34) : (tensor<701x702x701xf32>, tensor<1xi32>) -> tensor<701x702x1xf32>
    %163 = call @_take_127(%138, %arg35) : (tensor<701x702x701xf32>, tensor<1xi32>) -> tensor<701x702x1xf32>
    %164 = stablehlo.concatenate %162, %138, %163, %161, dim = 2 : (tensor<701x702x1xf32>, tensor<701x702x701xf32>, tensor<701x702x1xf32>, tensor<701x702x0xf32>) -> tensor<701x702x703xf32>
    %165 = stablehlo.slice %132 [0:701, 701:701, 0:702] : (tensor<701x701x702xf32>) -> tensor<701x0x702xf32>
    %166 = call @_take_135(%132, %arg36) : (tensor<701x701x702xf32>, tensor<1xi32>) -> tensor<701x1x702xf32>
    %167 = call @_take_135(%132, %arg37) : (tensor<701x701x702xf32>, tensor<1xi32>) -> tensor<701x1x702xf32>
    %168 = stablehlo.concatenate %166, %132, %167, %165, dim = 1 : (tensor<701x1x702xf32>, tensor<701x701x702xf32>, tensor<701x1x702xf32>, tensor<701x0x702xf32>) -> tensor<701x703x702xf32>
    %169 = stablehlo.transpose %148, dims = [1, 0, 2] : (tensor<702x703x701xf32>) -> tensor<703x702x701xf32>
    %170 = stablehlo.slice %169 [1:703, 0:702, 0:701] : (tensor<703x702x701xf32>) -> tensor<702x702x701xf32>
    %171 = stablehlo.slice %169 [0:702, 0:702, 0:701] : (tensor<703x702x701xf32>) -> tensor<702x702x701xf32>
    %172 = stablehlo.subtract %170, %171 : tensor<702x702x701xf32>
    %cst_11 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %173 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<702x702x701xf32>
    %174 = stablehlo.divide %172, %173 : tensor<702x702x701xf32>
    %175 = stablehlo.transpose %174, dims = [1, 0, 2] : (tensor<702x702x701xf32>) -> tensor<702x702x701xf32>
    %176 = stablehlo.slice %175 [0:702, 0:12, 0:701] : (tensor<702x702x701xf32>) -> tensor<702x12x701xf32>
    %177 = stablehlo.slice %175 [0:702, 690:702, 0:701] : (tensor<702x702x701xf32>) -> tensor<702x12x701xf32>
    %178 = stablehlo.concatenate %176, %177, dim = 1 : (tensor<702x12x701xf32>, tensor<702x12x701xf32>) -> tensor<702x24x701xf32>
    %179 = stablehlo.broadcast_in_dim %arg38, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<702x24x701xf32>
    %180 = stablehlo.multiply %179, %arg74 : tensor<702x24x701xf32>
    %181 = stablehlo.broadcast_in_dim %arg39, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<702x24x701xf32>
    %182 = stablehlo.multiply %181, %178 : tensor<702x24x701xf32>
    %183 = stablehlo.add %180, %182 : tensor<702x24x701xf32>
    %184 = stablehlo.broadcast_in_dim %arg40, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<702x24x701xf32>
    %185 = stablehlo.multiply %178, %184 : tensor<702x24x701xf32>
    %186 = stablehlo.add %185, %183 : tensor<702x24x701xf32>
    %187 = stablehlo.slice %186 [0:702, 0:12, 0:701] : (tensor<702x24x701xf32>) -> tensor<702x12x701xf32>
    %188 = stablehlo.slice %175 [0:702, 12:690, 0:701] : (tensor<702x702x701xf32>) -> tensor<702x678x701xf32>
    %189 = stablehlo.slice %186 [0:702, 12:24, 0:701] : (tensor<702x24x701xf32>) -> tensor<702x12x701xf32>
    %190 = stablehlo.concatenate %187, %188, %189, dim = 1 : (tensor<702x12x701xf32>, tensor<702x678x701xf32>, tensor<702x12x701xf32>) -> tensor<702x702x701xf32>
    %cst_12 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %191 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<702x702x701xf32>
    %192 = stablehlo.multiply %191, %190 : tensor<702x702x701xf32>
    %193 = stablehlo.slice %152 [1:703, 0:702, 0:701] : (tensor<703x702x701xf32>) -> tensor<702x702x701xf32>
    %194 = stablehlo.slice %152 [0:702, 0:702, 0:701] : (tensor<703x702x701xf32>) -> tensor<702x702x701xf32>
    %195 = stablehlo.subtract %193, %194 : tensor<702x702x701xf32>
    %cst_13 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %196 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<702x702x701xf32>
    %197 = stablehlo.divide %195, %196 : tensor<702x702x701xf32>
    %198 = stablehlo.slice %197 [0:12, 0:702, 0:701] : (tensor<702x702x701xf32>) -> tensor<12x702x701xf32>
    %199 = stablehlo.slice %197 [690:702, 0:702, 0:701] : (tensor<702x702x701xf32>) -> tensor<12x702x701xf32>
    %200 = stablehlo.concatenate %198, %199, dim = 0 : (tensor<12x702x701xf32>, tensor<12x702x701xf32>) -> tensor<24x702x701xf32>
    %201 = stablehlo.broadcast_in_dim %arg41, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x702x701xf32>
    %202 = stablehlo.multiply %201, %arg75 : tensor<24x702x701xf32>
    %203 = stablehlo.broadcast_in_dim %arg42, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x702x701xf32>
    %204 = stablehlo.multiply %203, %200 : tensor<24x702x701xf32>
    %205 = stablehlo.add %202, %204 : tensor<24x702x701xf32>
    %206 = stablehlo.broadcast_in_dim %arg43, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x702x701xf32>
    %207 = stablehlo.multiply %200, %206 : tensor<24x702x701xf32>
    %208 = stablehlo.add %207, %205 : tensor<24x702x701xf32>
    %209 = stablehlo.slice %208 [0:12, 0:702, 0:701] : (tensor<24x702x701xf32>) -> tensor<12x702x701xf32>
    %210 = stablehlo.slice %197 [12:690, 0:702, 0:701] : (tensor<702x702x701xf32>) -> tensor<678x702x701xf32>
    %211 = stablehlo.slice %208 [12:24, 0:702, 0:701] : (tensor<24x702x701xf32>) -> tensor<12x702x701xf32>
    %212 = stablehlo.concatenate %209, %210, %211, dim = 0 : (tensor<12x702x701xf32>, tensor<678x702x701xf32>, tensor<12x702x701xf32>) -> tensor<702x702x701xf32>
    %cst_14 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %213 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<702x702x701xf32>
    %214 = stablehlo.multiply %213, %212 : tensor<702x702x701xf32>
    %215 = stablehlo.slice %156 [1:703, 0:701, 0:702] : (tensor<703x701x702xf32>) -> tensor<702x701x702xf32>
    %216 = stablehlo.slice %156 [0:702, 0:701, 0:702] : (tensor<703x701x702xf32>) -> tensor<702x701x702xf32>
    %217 = stablehlo.subtract %215, %216 : tensor<702x701x702xf32>
    %cst_15 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %218 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<702x701x702xf32>
    %219 = stablehlo.divide %217, %218 : tensor<702x701x702xf32>
    %220 = stablehlo.slice %219 [0:12, 0:701, 0:702] : (tensor<702x701x702xf32>) -> tensor<12x701x702xf32>
    %221 = stablehlo.slice %219 [690:702, 0:701, 0:702] : (tensor<702x701x702xf32>) -> tensor<12x701x702xf32>
    %222 = stablehlo.concatenate %220, %221, dim = 0 : (tensor<12x701x702xf32>, tensor<12x701x702xf32>) -> tensor<24x701x702xf32>
    %223 = stablehlo.broadcast_in_dim %arg44, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x701x702xf32>
    %224 = stablehlo.multiply %223, %arg76 : tensor<24x701x702xf32>
    %225 = stablehlo.broadcast_in_dim %arg45, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x701x702xf32>
    %226 = stablehlo.multiply %225, %222 : tensor<24x701x702xf32>
    %227 = stablehlo.add %224, %226 : tensor<24x701x702xf32>
    %228 = stablehlo.broadcast_in_dim %arg46, dims = [0, 1, 2] : (tensor<24x1x1xf32>) -> tensor<24x701x702xf32>
    %229 = stablehlo.multiply %222, %228 : tensor<24x701x702xf32>
    %230 = stablehlo.add %229, %227 : tensor<24x701x702xf32>
    %231 = stablehlo.slice %230 [0:12, 0:701, 0:702] : (tensor<24x701x702xf32>) -> tensor<12x701x702xf32>
    %232 = stablehlo.slice %219 [12:690, 0:701, 0:702] : (tensor<702x701x702xf32>) -> tensor<678x701x702xf32>
    %233 = stablehlo.slice %230 [12:24, 0:701, 0:702] : (tensor<24x701x702xf32>) -> tensor<12x701x702xf32>
    %234 = stablehlo.concatenate %231, %232, %233, dim = 0 : (tensor<12x701x702xf32>, tensor<678x701x702xf32>, tensor<12x701x702xf32>) -> tensor<702x701x702xf32>
    %cst_16 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %235 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<702x701x702xf32>
    %236 = stablehlo.multiply %235, %234 : tensor<702x701x702xf32>
    %237 = stablehlo.transpose %160, dims = [2, 0, 1] : (tensor<702x701x703xf32>) -> tensor<703x702x701xf32>
    %238 = stablehlo.slice %237 [1:703, 0:702, 0:701] : (tensor<703x702x701xf32>) -> tensor<702x702x701xf32>
    %239 = stablehlo.slice %237 [0:702, 0:702, 0:701] : (tensor<703x702x701xf32>) -> tensor<702x702x701xf32>
    %240 = stablehlo.subtract %238, %239 : tensor<702x702x701xf32>
    %cst_17 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %241 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<702x702x701xf32>
    %242 = stablehlo.divide %240, %241 : tensor<702x702x701xf32>
    %243 = stablehlo.transpose %242, dims = [1, 2, 0] : (tensor<702x702x701xf32>) -> tensor<702x701x702xf32>
    %244 = stablehlo.slice %243 [0:702, 0:701, 0:12] : (tensor<702x701x702xf32>) -> tensor<702x701x12xf32>
    %245 = stablehlo.slice %243 [0:702, 0:701, 690:702] : (tensor<702x701x702xf32>) -> tensor<702x701x12xf32>
    %246 = stablehlo.concatenate %244, %245, dim = 2 : (tensor<702x701x12xf32>, tensor<702x701x12xf32>) -> tensor<702x701x24xf32>
    %247 = stablehlo.broadcast_in_dim %arg47, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<702x701x24xf32>
    %248 = stablehlo.multiply %247, %arg77 : tensor<702x701x24xf32>
    %249 = stablehlo.broadcast_in_dim %arg48, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<702x701x24xf32>
    %250 = stablehlo.multiply %249, %246 : tensor<702x701x24xf32>
    %251 = stablehlo.add %248, %250 : tensor<702x701x24xf32>
    %252 = stablehlo.broadcast_in_dim %arg49, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<702x701x24xf32>
    %253 = stablehlo.multiply %246, %252 : tensor<702x701x24xf32>
    %254 = stablehlo.add %253, %251 : tensor<702x701x24xf32>
    %255 = stablehlo.slice %254 [0:702, 0:701, 0:12] : (tensor<702x701x24xf32>) -> tensor<702x701x12xf32>
    %256 = stablehlo.slice %243 [0:702, 0:701, 12:690] : (tensor<702x701x702xf32>) -> tensor<702x701x678xf32>
    %257 = stablehlo.slice %254 [0:702, 0:701, 12:24] : (tensor<702x701x24xf32>) -> tensor<702x701x12xf32>
    %258 = stablehlo.concatenate %255, %256, %257, dim = 2 : (tensor<702x701x12xf32>, tensor<702x701x678xf32>, tensor<702x701x12xf32>) -> tensor<702x701x702xf32>
    %cst_18 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %259 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<702x701x702xf32>
    %260 = stablehlo.multiply %259, %258 : tensor<702x701x702xf32>
    %261 = stablehlo.transpose %164, dims = [2, 0, 1] : (tensor<701x702x703xf32>) -> tensor<703x701x702xf32>
    %262 = stablehlo.slice %261 [1:703, 0:701, 0:702] : (tensor<703x701x702xf32>) -> tensor<702x701x702xf32>
    %263 = stablehlo.slice %261 [0:702, 0:701, 0:702] : (tensor<703x701x702xf32>) -> tensor<702x701x702xf32>
    %264 = stablehlo.subtract %262, %263 : tensor<702x701x702xf32>
    %cst_19 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %265 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<702x701x702xf32>
    %266 = stablehlo.divide %264, %265 : tensor<702x701x702xf32>
    %267 = stablehlo.transpose %266, dims = [1, 2, 0] : (tensor<702x701x702xf32>) -> tensor<701x702x702xf32>
    %268 = stablehlo.slice %267 [0:701, 0:702, 0:12] : (tensor<701x702x702xf32>) -> tensor<701x702x12xf32>
    %269 = stablehlo.slice %267 [0:701, 0:702, 690:702] : (tensor<701x702x702xf32>) -> tensor<701x702x12xf32>
    %270 = stablehlo.concatenate %268, %269, dim = 2 : (tensor<701x702x12xf32>, tensor<701x702x12xf32>) -> tensor<701x702x24xf32>
    %271 = stablehlo.broadcast_in_dim %arg50, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<701x702x24xf32>
    %272 = stablehlo.multiply %271, %arg78 : tensor<701x702x24xf32>
    %273 = stablehlo.broadcast_in_dim %arg51, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<701x702x24xf32>
    %274 = stablehlo.multiply %273, %270 : tensor<701x702x24xf32>
    %275 = stablehlo.add %272, %274 : tensor<701x702x24xf32>
    %276 = stablehlo.broadcast_in_dim %arg52, dims = [0, 1, 2] : (tensor<1x1x24xf32>) -> tensor<701x702x24xf32>
    %277 = stablehlo.multiply %270, %276 : tensor<701x702x24xf32>
    %278 = stablehlo.add %277, %275 : tensor<701x702x24xf32>
    %279 = stablehlo.slice %278 [0:701, 0:702, 0:12] : (tensor<701x702x24xf32>) -> tensor<701x702x12xf32>
    %280 = stablehlo.slice %267 [0:701, 0:702, 12:690] : (tensor<701x702x702xf32>) -> tensor<701x702x678xf32>
    %281 = stablehlo.slice %278 [0:701, 0:702, 12:24] : (tensor<701x702x24xf32>) -> tensor<701x702x12xf32>
    %282 = stablehlo.concatenate %279, %280, %281, dim = 2 : (tensor<701x702x12xf32>, tensor<701x702x678xf32>, tensor<701x702x12xf32>) -> tensor<701x702x702xf32>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %283 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<701x702x702xf32>
    %284 = stablehlo.multiply %283, %282 : tensor<701x702x702xf32>
    %285 = stablehlo.transpose %168, dims = [1, 0, 2] : (tensor<701x703x702xf32>) -> tensor<703x701x702xf32>
    %286 = stablehlo.slice %285 [1:703, 0:701, 0:702] : (tensor<703x701x702xf32>) -> tensor<702x701x702xf32>
    %287 = stablehlo.slice %285 [0:702, 0:701, 0:702] : (tensor<703x701x702xf32>) -> tensor<702x701x702xf32>
    %288 = stablehlo.subtract %286, %287 : tensor<702x701x702xf32>
    %cst_21 = stablehlo.constant dense<7.99999995E-8> : tensor<f32>
    %289 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<702x701x702xf32>
    %290 = stablehlo.divide %288, %289 : tensor<702x701x702xf32>
    %291 = stablehlo.transpose %290, dims = [1, 0, 2] : (tensor<702x701x702xf32>) -> tensor<701x702x702xf32>
    %292 = stablehlo.slice %291 [0:701, 0:12, 0:702] : (tensor<701x702x702xf32>) -> tensor<701x12x702xf32>
    %293 = stablehlo.slice %291 [0:701, 690:702, 0:702] : (tensor<701x702x702xf32>) -> tensor<701x12x702xf32>
    %294 = stablehlo.concatenate %292, %293, dim = 1 : (tensor<701x12x702xf32>, tensor<701x12x702xf32>) -> tensor<701x24x702xf32>
    %295 = stablehlo.broadcast_in_dim %arg53, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<701x24x702xf32>
    %296 = stablehlo.multiply %295, %arg79 : tensor<701x24x702xf32>
    %297 = stablehlo.broadcast_in_dim %arg54, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<701x24x702xf32>
    %298 = stablehlo.multiply %297, %294 : tensor<701x24x702xf32>
    %299 = stablehlo.add %296, %298 : tensor<701x24x702xf32>
    %300 = stablehlo.broadcast_in_dim %arg55, dims = [0, 1, 2] : (tensor<1x24x1xf32>) -> tensor<701x24x702xf32>
    %301 = stablehlo.multiply %294, %300 : tensor<701x24x702xf32>
    %302 = stablehlo.add %301, %299 : tensor<701x24x702xf32>
    %303 = stablehlo.slice %302 [0:701, 0:12, 0:702] : (tensor<701x24x702xf32>) -> tensor<701x12x702xf32>
    %304 = stablehlo.slice %291 [0:701, 12:690, 0:702] : (tensor<701x702x702xf32>) -> tensor<701x678x702xf32>
    %305 = stablehlo.slice %302 [0:701, 12:24, 0:702] : (tensor<701x24x702xf32>) -> tensor<701x12x702xf32>
    %306 = stablehlo.concatenate %303, %304, %305, dim = 1 : (tensor<701x12x702xf32>, tensor<701x678x702xf32>, tensor<701x12x702xf32>) -> tensor<701x702x702xf32>
    %cst_22 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %307 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<701x702x702xf32>
    %308 = stablehlo.multiply %307, %306 : tensor<701x702x702xf32>
    %309 = stablehlo.add %192, %214 : tensor<702x702x701xf32>
    %310 = stablehlo.add %236, %260 : tensor<702x701x702xf32>
    %311 = stablehlo.add %284, %308 : tensor<701x702x702xf32>
    %312 = stablehlo.broadcast_in_dim %arg56, dims = [] : (tensor<f32>) -> tensor<702x702x701xf32>
    %313 = stablehlo.multiply %312, %arg62 : tensor<702x702x701xf32>
    %314 = stablehlo.subtract %309, %313 : tensor<702x702x701xf32>
    %315 = stablehlo.multiply %arg57, %314 : tensor<702x702x701xf32>
    %316 = stablehlo.add %arg62, %315 : tensor<702x702x701xf32>
    %317 = stablehlo.broadcast_in_dim %arg58, dims = [] : (tensor<f32>) -> tensor<702x701x702xf32>
    %318 = stablehlo.multiply %317, %arg63 : tensor<702x701x702xf32>
    %319 = stablehlo.subtract %310, %318 : tensor<702x701x702xf32>
    %320 = stablehlo.multiply %arg59, %319 : tensor<702x701x702xf32>
    %321 = stablehlo.add %arg63, %320 : tensor<702x701x702xf32>
    %322 = stablehlo.broadcast_in_dim %arg60, dims = [] : (tensor<f32>) -> tensor<701x702x702xf32>
    %323 = stablehlo.multiply %322, %arg64 : tensor<701x702x702xf32>
    %324 = stablehlo.subtract %311, %323 : tensor<701x702x702xf32>
    %325 = stablehlo.multiply %arg61, %324 : tensor<701x702x702xf32>
    %326 = stablehlo.add %arg64, %325 : tensor<701x702x702xf32>
    %327 = stablehlo.custom_call @LayoutConstraint(%316) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<702x702x701xf32>) -> tensor<702x702x701xf32>
    %328 = stablehlo.custom_call @LayoutConstraint(%321) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<702x701x702xf32>) -> tensor<702x701x702xf32>
    %329 = stablehlo.custom_call @LayoutConstraint(%326) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<701x702x702xf32>) -> tensor<701x702x702xf32>
    %330 = stablehlo.custom_call @LayoutConstraint(%132) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<701x701x702xf32>) -> tensor<701x701x702xf32>
    %331 = stablehlo.custom_call @LayoutConstraint(%138) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<701x702x701xf32>) -> tensor<701x702x701xf32>
    %332 = stablehlo.custom_call @LayoutConstraint(%144) {backend_config = "", operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<702x701x701xf32>) -> tensor<702x701x701xf32>
    %c_23 = stablehlo.constant dense<1> : tensor<i32>
    %333 = stablehlo.add %arg81, %c_23 : tensor<i32>
    return %327, %328, %329, %330, %331, %332, %14, %34, %54, %74, %94, %114, %183, %205, %227, %251, %275, %299, %3, %333 : tensor<702x702x701xf32>, tensor<702x701x702xf32>, tensor<701x702x702xf32>, tensor<701x701x702xf32>, tensor<701x702x701xf32>, tensor<702x701x701xf32>, tensor<701x24x702xf32>, tensor<24x701x702xf32>, tensor<24x702x701xf32>, tensor<701x702x24xf32>, tensor<702x701x24xf32>, tensor<702x24x701xf32>, tensor<702x24x701xf32>, tensor<24x702x701xf32>, tensor<24x701x702xf32>, tensor<702x701x24xf32>, tensor<701x702x24xf32>, tensor<701x24x702xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @diff(%arg0: tensor<701x702x702xf32>) -> tensor<701x701x702xf32> {
    %0 = stablehlo.slice %arg0 [0:701, 1:702, 0:702] : (tensor<701x702x702xf32>) -> tensor<701x701x702xf32>
    %1 = stablehlo.slice %arg0 [0:701, 0:701, 0:702] : (tensor<701x702x702xf32>) -> tensor<701x701x702xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<701x701x702xf32>
    return %2 : tensor<701x701x702xf32>
  }
  func.func private @diff_19(%arg0: tensor<702x701x702xf32>) -> tensor<701x701x702xf32> {
    %0 = stablehlo.slice %arg0 [1:702, 0:701, 0:702] : (tensor<702x701x702xf32>) -> tensor<701x701x702xf32>
    %1 = stablehlo.slice %arg0 [0:701, 0:701, 0:702] : (tensor<702x701x702xf32>) -> tensor<701x701x702xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<701x701x702xf32>
    return %2 : tensor<701x701x702xf32>
  }
  func.func private @diff_32(%arg0: tensor<702x702x701xf32>) -> tensor<701x702x701xf32> {
    %0 = stablehlo.slice %arg0 [1:702, 0:702, 0:701] : (tensor<702x702x701xf32>) -> tensor<701x702x701xf32>
    %1 = stablehlo.slice %arg0 [0:701, 0:702, 0:701] : (tensor<702x702x701xf32>) -> tensor<701x702x701xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<701x702x701xf32>
    return %2 : tensor<701x702x701xf32>
  }
  func.func private @diff_48(%arg0: tensor<701x702x702xf32>) -> tensor<701x702x701xf32> {
    %0 = stablehlo.slice %arg0 [0:701, 0:702, 1:702] : (tensor<701x702x702xf32>) -> tensor<701x702x701xf32>
    %1 = stablehlo.slice %arg0 [0:701, 0:702, 0:701] : (tensor<701x702x702xf32>) -> tensor<701x702x701xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<701x702x701xf32>
    return %2 : tensor<701x702x701xf32>
  }
  func.func private @diff_61(%arg0: tensor<702x701x702xf32>) -> tensor<702x701x701xf32> {
    %0 = stablehlo.slice %arg0 [0:702, 0:701, 1:702] : (tensor<702x701x702xf32>) -> tensor<702x701x701xf32>
    %1 = stablehlo.slice %arg0 [0:702, 0:701, 0:701] : (tensor<702x701x702xf32>) -> tensor<702x701x701xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<702x701x701xf32>
    return %2 : tensor<702x701x701xf32>
  }
  func.func private @diff_77(%arg0: tensor<702x702x701xf32>) -> tensor<702x701x701xf32> {
    %0 = stablehlo.slice %arg0 [0:702, 1:702, 0:701] : (tensor<702x702x701xf32>) -> tensor<702x701x701xf32>
    %1 = stablehlo.slice %arg0 [0:702, 0:701, 0:701] : (tensor<702x702x701xf32>) -> tensor<702x701x701xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<702x701x701xf32>
    return %2 : tensor<702x701x701xf32>
  }
  func.func private @_take(%arg0: tensor<702x701x701xf32>, %arg1: tensor<1xi32>) -> tensor<702x1x701xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<701> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<700> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 702, 1, 701>}> : (tensor<702x701x701xf32>, tensor<1x1xi32>) -> tensor<702x1x701xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<1xi1>) -> tensor<702x1x701xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<702x1x701xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<702x1x701xi1>, tensor<702x1x701xf32>
    return %15 : tensor<702x1x701xf32>
  }
  func.func private @_where(%arg0: tensor<1xi1>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>) -> tensor<1xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1xi1>, tensor<1xi32>
    return %0 : tensor<1xi32>
  }
  func.func private @_take_103(%arg0: tensor<701x702x701xf32>, %arg1: tensor<1xi32>) -> tensor<1x702x701xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<701> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<700> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 702, 701>}> : (tensor<701x702x701xf32>, tensor<1x1xi32>) -> tensor<1x702x701xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<1xi1>) -> tensor<1x702x701xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x702x701xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<1x702x701xi1>, tensor<1x702x701xf32>
    return %15 : tensor<1x702x701xf32>
  }
  func.func private @_take_111(%arg0: tensor<701x701x702xf32>, %arg1: tensor<1xi32>) -> tensor<1x701x702xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<701> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<700> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 701, 702>}> : (tensor<701x701x702xf32>, tensor<1x1xi32>) -> tensor<1x701x702xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<1xi1>) -> tensor<1x701x702xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x701x702xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<1x701x702xi1>, tensor<1x701x702xf32>
    return %15 : tensor<1x701x702xf32>
  }
  func.func private @_take_119(%arg0: tensor<702x701x701xf32>, %arg1: tensor<1xi32>) -> tensor<702x701x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<701> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<700> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 702, 701, 1>}> : (tensor<702x701x701xf32>, tensor<1x1xi32>) -> tensor<702x701x1xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<1xi1>) -> tensor<702x701x1xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<702x701x1xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<702x701x1xi1>, tensor<702x701x1xf32>
    return %15 : tensor<702x701x1xf32>
  }
  func.func private @_take_127(%arg0: tensor<701x702x701xf32>, %arg1: tensor<1xi32>) -> tensor<701x702x1xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<701> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<700> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 701, 702, 1>}> : (tensor<701x702x701xf32>, tensor<1x1xi32>) -> tensor<701x702x1xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [2] : (tensor<1xi1>) -> tensor<701x702x1xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<701x702x1xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<701x702x1xi1>, tensor<701x702x1xf32>
    return %15 : tensor<701x702x1xf32>
  }
  func.func private @_take_135(%arg0: tensor<701x701x702xf32>, %arg1: tensor<1xi32>) -> tensor<701x1x702xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_0 = stablehlo.constant dense<701> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %c_1 = stablehlo.constant dense<700> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [1] : (tensor<1xi32>) -> tensor<1x1xi32>
    %9 = stablehlo.compare  LE, %5, %8,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %10 = stablehlo.and %7, %9 : tensor<1x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %11 = stablehlo.reduce(%10 init: %c_3) applies stablehlo.and across dimensions = [1] : (tensor<1x1xi1>, tensor<i1>) -> tensor<1xi1>
    %12 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 701, 1, 702>}> : (tensor<701x701x702xf32>, tensor<1x1xi32>) -> tensor<701x1x702xf32>
    %13 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<1xi1>) -> tensor<701x1x702xi1>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<701x1x702xf32>
    %15 = stablehlo.select %13, %12, %14 : tensor<701x1x702xi1>, tensor<701x1x702xf32>
    return %15 : tensor<701x1x702xf32>
  }
}
