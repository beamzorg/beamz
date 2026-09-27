module @jit_run_scan attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["fdtd"=8]>
  func.func public @main(%arg0: tensor<10008x1251x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg1: tensor<10008x1250x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg2: tensor<10008x1251x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg3: tensor<10008x1250x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg4: tensor<10008x1251x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg5: tensor<10008x1250x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg6: tensor<10008x24x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg7: tensor<24x1250x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg8: tensor<24x1251x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg9: tensor<10008x1251x24xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg10: tensor<10008x1250x24xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg11: tensor<10008x24x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg12: tensor<10008x24x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg13: tensor<24x1251x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg14: tensor<24x1250x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg15: tensor<10008x1250x24xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg16: tensor<10008x1251x24xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg17: tensor<10008x24x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg18: tensor<2x1xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg19: tensor<2x1xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg20: tensor<2xi32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg21: tensor<2x101xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg22: tensor<2x101xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg23: tensor<2x101xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg24: tensor<2x101xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg25: tensor<242400xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg26: tensor<242400xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg27: tensor<202xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg28: tensor<f32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, []>}, %arg29: tensor<i32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, []>}, %arg30: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg31: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg32: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg33: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg34: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg35: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg36: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg37: tensor<10008x1251x1200xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg38: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg39: tensor<10008x1250x1201xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg40: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg41: tensor<10008x1251x1201xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}) -> (tensor<10008x1251x1200xf32> {jax.result_info = "result.ex"}, tensor<10008x1250x1201xf32> {jax.result_info = "result.ey"}, tensor<10008x1251x1201xf32> {jax.result_info = "result.ez"}, tensor<10008x1250x1201xf32> {jax.result_info = "result.hx"}, tensor<10008x1251x1200xf32> {jax.result_info = "result.hy"}, tensor<10008x1250x1200xf32> {jax.result_info = "result.hz"}, tensor<10008x24x1201xf32> {jax.result_info = "result.cpml_psi_h_terms[0]"}, tensor<24x1250x1201xf32> {jax.result_info = "result.cpml_psi_h_terms[1]"}, tensor<24x1251x1200xf32> {jax.result_info = "result.cpml_psi_h_terms[2]"}, tensor<10008x1251x24xf32> {jax.result_info = "result.cpml_psi_h_terms[3]"}, tensor<10008x1250x24xf32> {jax.result_info = "result.cpml_psi_h_terms[4]"}, tensor<10008x24x1200xf32> {jax.result_info = "result.cpml_psi_h_terms[5]"}, tensor<10008x24x1200xf32> {jax.result_info = "result.cpml_psi_e_terms[0]"}, tensor<24x1251x1200xf32> {jax.result_info = "result.cpml_psi_e_terms[1]"}, tensor<24x1250x1201xf32> {jax.result_info = "result.cpml_psi_e_terms[2]"}, tensor<10008x1250x24xf32> {jax.result_info = "result.cpml_psi_e_terms[3]"}, tensor<10008x1251x24xf32> {jax.result_info = "result.cpml_psi_e_terms[4]"}, tensor<10008x24x1201xf32> {jax.result_info = "result.cpml_psi_e_terms[5]"}, tensor<2x1xf32> {jax.result_info = "result.powers"}, tensor<2x1xf32> {jax.result_info = "result.timestamps"}, tensor<2xi32> {jax.result_info = "result.counts"}, tensor<2x101xf32> {jax.result_info = "result.freq_flux_re"}, tensor<2x101xf32> {jax.result_info = "result.freq_flux_im"}, tensor<2x101xf32> {jax.result_info = "result.freq_phase_re"}, tensor<2x101xf32> {jax.result_info = "result.freq_phase_im"}, tensor<242400xf32> {jax.result_info = "result.dft_vec_re"}, tensor<242400xf32> {jax.result_info = "result.dft_vec_im"}, tensor<202xf32> {jax.result_info = "result.dft_weight_sum"}, tensor<f32> {jax.result_info = "result.t"}, tensor<i32> {jax.result_info = "result.current_step"}) {
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
    %cst_20 = stablehlo.constant dense_resource<__elided__> : tensor<2x14x27x1xf32>
    %cst_21 = stablehlo.constant dense_resource<__elided__> : tensor<2x1024xf32>
    %cst_22 = stablehlo.constant dense_resource<__elided__> : tensor<2x15x26x1xf32>
    %cst_23 = stablehlo.constant dense_resource<__elided__> : tensor<2x1024xf32>
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
    %cst_44 = stablehlo.constant dense_resource<__elided__> : tensor<2x15x28x1xf32>
    %cst_45 = stablehlo.constant dense_resource<__elided__> : tensor<2x1024xf32>
    %cst_46 = stablehlo.constant dense_resource<__elided__> : tensor<2x16x27x1xf32>
    %cst_47 = stablehlo.constant dense_resource<__elided__> : tensor<2x1024xf32>
    %c_48 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_49 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_50 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_51 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_52 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_53 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_54 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_56 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_57 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_58 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_59 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_60 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_61 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_62 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_63 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_64 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_65 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_66 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_67 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_68 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_69 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_70 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_71 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_72 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %c_73 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %c_74 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %cst_75 = stablehlo.constant dense<1.250000e-01> : tensor<200x8xf32>
    %c_76 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %c_77 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %c_78 = stablehlo.constant dense<780> : tensor<200x2xi32>
    %cst_79 = stablehlo.constant dense<5.000000e-01> : tensor<200x2xf32>
    %c_80 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %c_81 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %c_82 = stablehlo.constant dense<780> : tensor<200x2xi32>
    %cst_83 = stablehlo.constant dense<5.000000e-01> : tensor<200x2xf32>
    %c_84 = stablehlo.constant dense_resource<__elided__> : tensor<200x1xi32>
    %c_85 = stablehlo.constant dense_resource<__elided__> : tensor<200x1xi32>
    %c_86 = stablehlo.constant dense<780> : tensor<200x1xi32>
    %cst_87 = stablehlo.constant dense<1.000000e+00> : tensor<200x1xf32>
    %c_88 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %c_89 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %c_90 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_91 = stablehlo.constant dense<2.500000e-01> : tensor<200x4xf32>
    %c_92 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %c_93 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %c_94 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_95 = stablehlo.constant dense<2.500000e-01> : tensor<200x4xf32>
    %cst_96 = stablehlo.constant dense_resource<__elided__> : tensor<101xf32>
    %cst_97 = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
    %c_98 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_99 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_100 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_102 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_103 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_104 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_105 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_106 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_107 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_108 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_110 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_111 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_112 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_113 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_114 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_115 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_116 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_118 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_119 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %c_120 = stablehlo.constant dense<0> : tensor<200x1xi32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<200x1xf32>
    %c_122 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %c_123 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %c_124 = stablehlo.constant dense_resource<__elided__> : tensor<200x8xi32>
    %cst_125 = stablehlo.constant dense<1.250000e-01> : tensor<200x8xf32>
    %c_126 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %c_127 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %c_128 = stablehlo.constant dense<900> : tensor<200x2xi32>
    %cst_129 = stablehlo.constant dense<5.000000e-01> : tensor<200x2xf32>
    %c_130 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %c_131 = stablehlo.constant dense_resource<__elided__> : tensor<200x2xi32>
    %c_132 = stablehlo.constant dense<900> : tensor<200x2xi32>
    %cst_133 = stablehlo.constant dense<5.000000e-01> : tensor<200x2xf32>
    %c_134 = stablehlo.constant dense_resource<__elided__> : tensor<200x1xi32>
    %c_135 = stablehlo.constant dense_resource<__elided__> : tensor<200x1xi32>
    %c_136 = stablehlo.constant dense<900> : tensor<200x1xi32>
    %cst_137 = stablehlo.constant dense<1.000000e+00> : tensor<200x1xf32>
    %c_138 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %c_139 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %c_140 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_141 = stablehlo.constant dense<2.500000e-01> : tensor<200x4xf32>
    %c_142 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %c_143 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %c_144 = stablehlo.constant dense_resource<__elided__> : tensor<200x4xi32>
    %cst_145 = stablehlo.constant dense<2.500000e-01> : tensor<200x4xf32>
    %cst_146 = stablehlo.constant dense_resource<__elided__> : tensor<101xf32>
    %cst_147 = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
    %0 = sdy.manual_computation(%arg25) in_shardings=[<@mesh, [{}]>] out_shardings=[<@mesh, [{"fdtd"}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<242400xf32>) {
      %c_149 = stablehlo.constant dense<1> : tensor<ui32>
      %c_150 = stablehlo.constant dense<8> : tensor<ui32>
      %14 = stablehlo.partition_id : tensor<ui32>
      %15 = stablehlo.divide %14, %c_149 : tensor<ui32>
      %16 = stablehlo.remainder %15, %c_150 : tensor<ui32>
      %17 = stablehlo.convert %16 : (tensor<ui32>) -> tensor<i32>
      %c_151 = stablehlo.constant dense<0> : tensor<i32>
      %18 = stablehlo.compare  EQ, %17, %c_151,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_152 = stablehlo.constant dense<0> : tensor<i32>
      %19 = func.call @_where(%18, %arg42, %c_152) : (tensor<i1>, tensor<242400xf32>, tensor<i32>) -> tensor<242400xf32>
      %20 = stablehlo.broadcast_in_dim %19, dims = [1] : (tensor<242400xf32>) -> tensor<1x242400xf32>
      sdy.return %20 : tensor<1x242400xf32>
    } : (tensor<242400xf32>) -> tensor<8x242400xf32>
    %1 = sdy.manual_computation(%arg26) in_shardings=[<@mesh, [{}]>] out_shardings=[<@mesh, [{"fdtd"}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<242400xf32>) {
      %c_149 = stablehlo.constant dense<1> : tensor<ui32>
      %c_150 = stablehlo.constant dense<8> : tensor<ui32>
      %14 = stablehlo.partition_id : tensor<ui32>
      %15 = stablehlo.divide %14, %c_149 : tensor<ui32>
      %16 = stablehlo.remainder %15, %c_150 : tensor<ui32>
      %17 = stablehlo.convert %16 : (tensor<ui32>) -> tensor<i32>
      %c_151 = stablehlo.constant dense<0> : tensor<i32>
      %18 = stablehlo.compare  EQ, %17, %c_151,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_152 = stablehlo.constant dense<0> : tensor<i32>
      %19 = func.call @_where(%18, %arg42, %c_152) : (tensor<i1>, tensor<242400xf32>, tensor<i32>) -> tensor<242400xf32>
      %20 = stablehlo.broadcast_in_dim %19, dims = [1] : (tensor<242400xf32>) -> tensor<1x242400xf32>
      sdy.return %20 : tensor<1x242400xf32>
    } : (tensor<242400xf32>) -> tensor<8x242400xf32>
    %2 = sdy.manual_computation(%arg7) in_shardings=[<@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<24x1250x1201xf32>) {
      %c_149 = stablehlo.constant dense<1> : tensor<ui32>
      %c_150 = stablehlo.constant dense<8> : tensor<ui32>
      %14 = stablehlo.partition_id : tensor<ui32>
      %15 = stablehlo.divide %14, %c_149 : tensor<ui32>
      %16 = stablehlo.remainder %15, %c_150 : tensor<ui32>
      %17 = stablehlo.convert %16 : (tensor<ui32>) -> tensor<i32>
      %c_151 = stablehlo.constant dense<1251> : tensor<i32>
      %18 = stablehlo.multiply %17, %c_151 : tensor<i32>
      %19 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_152 = stablehlo.constant dense<12> : tensor<i32>
      %20 = stablehlo.broadcast_in_dim %c_152, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %21 = stablehlo.compare  LT, %19, %20,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_153 = stablehlo.constant dense<12> : tensor<i32>
      %22 = stablehlo.broadcast_in_dim %c_153, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %23 = stablehlo.subtract %19, %22 : tensor<24xi32>
      %c_154 = stablehlo.constant dense<10000> : tensor<i32>
      %24 = stablehlo.broadcast_in_dim %c_154, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %25 = stablehlo.add %23, %24 : tensor<24xi32>
      %c_155 = stablehlo.constant dense<12> : tensor<i32>
      %26 = stablehlo.broadcast_in_dim %c_155, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %27 = stablehlo.subtract %25, %26 : tensor<24xi32>
      %28 = func.call @_where_4(%21, %19, %27) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_156 = stablehlo.constant dense<0> : tensor<i32>
      %29 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %30 = stablehlo.compare  GE, %28, %29,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_157 = stablehlo.constant dense<10000> : tensor<i32>
      %31 = stablehlo.broadcast_in_dim %c_157, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %32 = stablehlo.compare  LT, %28, %31,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %33 = stablehlo.and %30, %32 : tensor<24xi1>
      %34 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %35 = stablehlo.compare  GE, %28, %34,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_158 = stablehlo.constant dense<1251> : tensor<i32>
      %36 = stablehlo.add %18, %c_158 : tensor<i32>
      %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %38 = stablehlo.compare  LT, %28, %37,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %39 = stablehlo.and %35, %38 : tensor<24xi1>
      %40 = stablehlo.and %33, %39 : tensor<24xi1>
      %41 = stablehlo.reshape %40 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_159 = stablehlo.constant dense<true> : tensor<i1>
      %42 = stablehlo.broadcast_in_dim %c_159, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %43 = stablehlo.and %42, %41 : tensor<24x1x1xi1>
      %44 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_160 = stablehlo.constant dense<0> : tensor<i32>
      %45 = stablehlo.broadcast_in_dim %c_160, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %46 = stablehlo.compare  GE, %44, %45,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_161 = stablehlo.constant dense<1250> : tensor<i32>
      %47 = stablehlo.broadcast_in_dim %c_161, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %48 = stablehlo.compare  LT, %44, %47,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %49 = stablehlo.and %46, %48 : tensor<1250xi1>
      %50 = stablehlo.reshape %49 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %51 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1250x1xi1>
      %52 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<24x1250x1xi1>
      %53 = stablehlo.and %51, %52 : tensor<24x1250x1xi1>
      %54 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_162 = stablehlo.constant dense<0> : tensor<i32>
      %55 = stablehlo.broadcast_in_dim %c_162, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %56 = stablehlo.compare  GE, %54, %55,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_163 = stablehlo.constant dense<1201> : tensor<i32>
      %57 = stablehlo.broadcast_in_dim %c_163, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %58 = stablehlo.compare  LT, %54, %57,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %59 = stablehlo.and %56, %58 : tensor<1201xi1>
      %60 = stablehlo.reshape %59 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %61 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2] : (tensor<24x1250x1xi1>) -> tensor<24x1250x1201xi1>
      %62 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<24x1250x1201xi1>
      %63 = stablehlo.and %61, %62 : tensor<24x1250x1201xi1>
      %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %64 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f32>) -> tensor<24x1250x1201xf32>
      %65 = func.call @_where_29(%63, %arg42, %64) : (tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>, tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %66 = stablehlo.broadcast_in_dim %65, dims = [1, 2, 3] : (tensor<24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      sdy.return %66 : tensor<1x24x1250x1201xf32>
    } : (tensor<24x1250x1201xf32>) -> tensor<8x24x1250x1201xf32>
    %3 = sdy.manual_computation(%arg8) in_shardings=[<@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<24x1251x1200xf32>) {
      %c_149 = stablehlo.constant dense<1> : tensor<ui32>
      %c_150 = stablehlo.constant dense<8> : tensor<ui32>
      %14 = stablehlo.partition_id : tensor<ui32>
      %15 = stablehlo.divide %14, %c_149 : tensor<ui32>
      %16 = stablehlo.remainder %15, %c_150 : tensor<ui32>
      %17 = stablehlo.convert %16 : (tensor<ui32>) -> tensor<i32>
      %c_151 = stablehlo.constant dense<1251> : tensor<i32>
      %18 = stablehlo.multiply %17, %c_151 : tensor<i32>
      %19 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_152 = stablehlo.constant dense<12> : tensor<i32>
      %20 = stablehlo.broadcast_in_dim %c_152, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %21 = stablehlo.compare  LT, %19, %20,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_153 = stablehlo.constant dense<12> : tensor<i32>
      %22 = stablehlo.broadcast_in_dim %c_153, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %23 = stablehlo.subtract %19, %22 : tensor<24xi32>
      %c_154 = stablehlo.constant dense<10000> : tensor<i32>
      %24 = stablehlo.broadcast_in_dim %c_154, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %25 = stablehlo.add %23, %24 : tensor<24xi32>
      %c_155 = stablehlo.constant dense<12> : tensor<i32>
      %26 = stablehlo.broadcast_in_dim %c_155, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %27 = stablehlo.subtract %25, %26 : tensor<24xi32>
      %28 = func.call @_where_4(%21, %19, %27) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_156 = stablehlo.constant dense<0> : tensor<i32>
      %29 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %30 = stablehlo.compare  GE, %28, %29,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_157 = stablehlo.constant dense<10000> : tensor<i32>
      %31 = stablehlo.broadcast_in_dim %c_157, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %32 = stablehlo.compare  LT, %28, %31,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %33 = stablehlo.and %30, %32 : tensor<24xi1>
      %34 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %35 = stablehlo.compare  GE, %28, %34,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_158 = stablehlo.constant dense<1251> : tensor<i32>
      %36 = stablehlo.add %18, %c_158 : tensor<i32>
      %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %38 = stablehlo.compare  LT, %28, %37,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %39 = stablehlo.and %35, %38 : tensor<24xi1>
      %40 = stablehlo.and %33, %39 : tensor<24xi1>
      %41 = stablehlo.reshape %40 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_159 = stablehlo.constant dense<true> : tensor<i1>
      %42 = stablehlo.broadcast_in_dim %c_159, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %43 = stablehlo.and %42, %41 : tensor<24x1x1xi1>
      %44 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_160 = stablehlo.constant dense<0> : tensor<i32>
      %45 = stablehlo.broadcast_in_dim %c_160, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %46 = stablehlo.compare  GE, %44, %45,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_161 = stablehlo.constant dense<1251> : tensor<i32>
      %47 = stablehlo.broadcast_in_dim %c_161, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %48 = stablehlo.compare  LT, %44, %47,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %49 = stablehlo.and %46, %48 : tensor<1251xi1>
      %50 = stablehlo.reshape %49 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %51 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1251x1xi1>
      %52 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<24x1251x1xi1>
      %53 = stablehlo.and %51, %52 : tensor<24x1251x1xi1>
      %54 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_162 = stablehlo.constant dense<0> : tensor<i32>
      %55 = stablehlo.broadcast_in_dim %c_162, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %56 = stablehlo.compare  GE, %54, %55,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_163 = stablehlo.constant dense<1200> : tensor<i32>
      %57 = stablehlo.broadcast_in_dim %c_163, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %58 = stablehlo.compare  LT, %54, %57,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %59 = stablehlo.and %56, %58 : tensor<1200xi1>
      %60 = stablehlo.reshape %59 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %61 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2] : (tensor<24x1251x1xi1>) -> tensor<24x1251x1200xi1>
      %62 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<24x1251x1200xi1>
      %63 = stablehlo.and %61, %62 : tensor<24x1251x1200xi1>
      %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %64 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f32>) -> tensor<24x1251x1200xf32>
      %65 = func.call @_where_49(%63, %arg42, %64) : (tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>, tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %66 = stablehlo.broadcast_in_dim %65, dims = [1, 2, 3] : (tensor<24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      sdy.return %66 : tensor<1x24x1251x1200xf32>
    } : (tensor<24x1251x1200xf32>) -> tensor<8x24x1251x1200xf32>
    %4 = sdy.manual_computation(%arg13) in_shardings=[<@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<24x1251x1200xf32>) {
      %c_149 = stablehlo.constant dense<1> : tensor<ui32>
      %c_150 = stablehlo.constant dense<8> : tensor<ui32>
      %14 = stablehlo.partition_id : tensor<ui32>
      %15 = stablehlo.divide %14, %c_149 : tensor<ui32>
      %16 = stablehlo.remainder %15, %c_150 : tensor<ui32>
      %17 = stablehlo.convert %16 : (tensor<ui32>) -> tensor<i32>
      %c_151 = stablehlo.constant dense<1251> : tensor<i32>
      %18 = stablehlo.multiply %17, %c_151 : tensor<i32>
      %19 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_152 = stablehlo.constant dense<12> : tensor<i32>
      %20 = stablehlo.broadcast_in_dim %c_152, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %21 = stablehlo.compare  LT, %19, %20,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_153 = stablehlo.constant dense<12> : tensor<i32>
      %22 = stablehlo.broadcast_in_dim %c_153, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %23 = stablehlo.subtract %19, %22 : tensor<24xi32>
      %c_154 = stablehlo.constant dense<10001> : tensor<i32>
      %24 = stablehlo.broadcast_in_dim %c_154, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %25 = stablehlo.add %23, %24 : tensor<24xi32>
      %c_155 = stablehlo.constant dense<12> : tensor<i32>
      %26 = stablehlo.broadcast_in_dim %c_155, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %27 = stablehlo.subtract %25, %26 : tensor<24xi32>
      %28 = func.call @_where_4(%21, %19, %27) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_156 = stablehlo.constant dense<0> : tensor<i32>
      %29 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %30 = stablehlo.compare  GE, %28, %29,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_157 = stablehlo.constant dense<10001> : tensor<i32>
      %31 = stablehlo.broadcast_in_dim %c_157, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %32 = stablehlo.compare  LT, %28, %31,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %33 = stablehlo.and %30, %32 : tensor<24xi1>
      %34 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %35 = stablehlo.compare  GE, %28, %34,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_158 = stablehlo.constant dense<1251> : tensor<i32>
      %36 = stablehlo.add %18, %c_158 : tensor<i32>
      %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %38 = stablehlo.compare  LT, %28, %37,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %39 = stablehlo.and %35, %38 : tensor<24xi1>
      %40 = stablehlo.and %33, %39 : tensor<24xi1>
      %41 = stablehlo.reshape %40 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_159 = stablehlo.constant dense<true> : tensor<i1>
      %42 = stablehlo.broadcast_in_dim %c_159, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %43 = stablehlo.and %42, %41 : tensor<24x1x1xi1>
      %44 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_160 = stablehlo.constant dense<0> : tensor<i32>
      %45 = stablehlo.broadcast_in_dim %c_160, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %46 = stablehlo.compare  GE, %44, %45,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_161 = stablehlo.constant dense<1251> : tensor<i32>
      %47 = stablehlo.broadcast_in_dim %c_161, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %48 = stablehlo.compare  LT, %44, %47,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %49 = stablehlo.and %46, %48 : tensor<1251xi1>
      %50 = stablehlo.reshape %49 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %51 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1251x1xi1>
      %52 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<24x1251x1xi1>
      %53 = stablehlo.and %51, %52 : tensor<24x1251x1xi1>
      %54 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_162 = stablehlo.constant dense<0> : tensor<i32>
      %55 = stablehlo.broadcast_in_dim %c_162, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %56 = stablehlo.compare  GE, %54, %55,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_163 = stablehlo.constant dense<1200> : tensor<i32>
      %57 = stablehlo.broadcast_in_dim %c_163, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %58 = stablehlo.compare  LT, %54, %57,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %59 = stablehlo.and %56, %58 : tensor<1200xi1>
      %60 = stablehlo.reshape %59 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %61 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2] : (tensor<24x1251x1xi1>) -> tensor<24x1251x1200xi1>
      %62 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<24x1251x1200xi1>
      %63 = stablehlo.and %61, %62 : tensor<24x1251x1200xi1>
      %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %64 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f32>) -> tensor<24x1251x1200xf32>
      %65 = func.call @_where_49(%63, %arg42, %64) : (tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>, tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %66 = stablehlo.broadcast_in_dim %65, dims = [1, 2, 3] : (tensor<24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      sdy.return %66 : tensor<1x24x1251x1200xf32>
    } : (tensor<24x1251x1200xf32>) -> tensor<8x24x1251x1200xf32>
    %5 = sdy.manual_computation(%arg14) in_shardings=[<@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<24x1250x1201xf32>) {
      %c_149 = stablehlo.constant dense<1> : tensor<ui32>
      %c_150 = stablehlo.constant dense<8> : tensor<ui32>
      %14 = stablehlo.partition_id : tensor<ui32>
      %15 = stablehlo.divide %14, %c_149 : tensor<ui32>
      %16 = stablehlo.remainder %15, %c_150 : tensor<ui32>
      %17 = stablehlo.convert %16 : (tensor<ui32>) -> tensor<i32>
      %c_151 = stablehlo.constant dense<1251> : tensor<i32>
      %18 = stablehlo.multiply %17, %c_151 : tensor<i32>
      %19 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_152 = stablehlo.constant dense<12> : tensor<i32>
      %20 = stablehlo.broadcast_in_dim %c_152, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %21 = stablehlo.compare  LT, %19, %20,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_153 = stablehlo.constant dense<12> : tensor<i32>
      %22 = stablehlo.broadcast_in_dim %c_153, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %23 = stablehlo.subtract %19, %22 : tensor<24xi32>
      %c_154 = stablehlo.constant dense<10001> : tensor<i32>
      %24 = stablehlo.broadcast_in_dim %c_154, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %25 = stablehlo.add %23, %24 : tensor<24xi32>
      %c_155 = stablehlo.constant dense<12> : tensor<i32>
      %26 = stablehlo.broadcast_in_dim %c_155, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %27 = stablehlo.subtract %25, %26 : tensor<24xi32>
      %28 = func.call @_where_4(%21, %19, %27) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_156 = stablehlo.constant dense<0> : tensor<i32>
      %29 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %30 = stablehlo.compare  GE, %28, %29,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_157 = stablehlo.constant dense<10001> : tensor<i32>
      %31 = stablehlo.broadcast_in_dim %c_157, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %32 = stablehlo.compare  LT, %28, %31,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %33 = stablehlo.and %30, %32 : tensor<24xi1>
      %34 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %35 = stablehlo.compare  GE, %28, %34,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_158 = stablehlo.constant dense<1251> : tensor<i32>
      %36 = stablehlo.add %18, %c_158 : tensor<i32>
      %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %38 = stablehlo.compare  LT, %28, %37,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %39 = stablehlo.and %35, %38 : tensor<24xi1>
      %40 = stablehlo.and %33, %39 : tensor<24xi1>
      %41 = stablehlo.reshape %40 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_159 = stablehlo.constant dense<true> : tensor<i1>
      %42 = stablehlo.broadcast_in_dim %c_159, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %43 = stablehlo.and %42, %41 : tensor<24x1x1xi1>
      %44 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_160 = stablehlo.constant dense<0> : tensor<i32>
      %45 = stablehlo.broadcast_in_dim %c_160, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %46 = stablehlo.compare  GE, %44, %45,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_161 = stablehlo.constant dense<1250> : tensor<i32>
      %47 = stablehlo.broadcast_in_dim %c_161, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %48 = stablehlo.compare  LT, %44, %47,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %49 = stablehlo.and %46, %48 : tensor<1250xi1>
      %50 = stablehlo.reshape %49 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %51 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1250x1xi1>
      %52 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<24x1250x1xi1>
      %53 = stablehlo.and %51, %52 : tensor<24x1250x1xi1>
      %54 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_162 = stablehlo.constant dense<0> : tensor<i32>
      %55 = stablehlo.broadcast_in_dim %c_162, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %56 = stablehlo.compare  GE, %54, %55,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_163 = stablehlo.constant dense<1201> : tensor<i32>
      %57 = stablehlo.broadcast_in_dim %c_163, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %58 = stablehlo.compare  LT, %54, %57,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %59 = stablehlo.and %56, %58 : tensor<1201xi1>
      %60 = stablehlo.reshape %59 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %61 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2] : (tensor<24x1250x1xi1>) -> tensor<24x1250x1201xi1>
      %62 = stablehlo.broadcast_in_dim %60, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<24x1250x1201xi1>
      %63 = stablehlo.and %61, %62 : tensor<24x1250x1201xi1>
      %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %64 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f32>) -> tensor<24x1250x1201xf32>
      %65 = func.call @_where_29(%63, %arg42, %64) : (tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>, tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %66 = stablehlo.broadcast_in_dim %65, dims = [1, 2, 3] : (tensor<24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      sdy.return %66 : tensor<1x24x1250x1201xf32>
    } : (tensor<24x1250x1201xf32>) -> tensor<8x24x1250x1201xf32>
    %6 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %c_148 = stablehlo.constant dense<0> : tensor<i32>
    %7:191 = stablehlo.while(%iterArg = %6, %iterArg_149 = %cst, %iterArg_150 = %arg28, %iterArg_151 = %c, %iterArg_152 = %arg30, %iterArg_153 = %arg32, %iterArg_154 = %arg34, %iterArg_155 = %arg31, %iterArg_156 = %arg33, %iterArg_157 = %arg35, %iterArg_158 = %cst_0, %iterArg_159 = %cst_1, %iterArg_160 = %cst_2, %iterArg_161 = %cst_3, %iterArg_162 = %cst_4, %iterArg_163 = %cst_5, %iterArg_164 = %cst_6, %iterArg_165 = %cst_7, %iterArg_166 = %cst_8, %iterArg_167 = %cst_9, %iterArg_168 = %cst_10, %iterArg_169 = %cst_11, %iterArg_170 = %cst_12, %iterArg_171 = %cst_13, %iterArg_172 = %cst_14, %iterArg_173 = %cst_15, %iterArg_174 = %cst_16, %iterArg_175 = %cst_17, %iterArg_176 = %cst_18, %iterArg_177 = %c_19, %iterArg_178 = %cst_20, %iterArg_179 = %cst_21, %iterArg_180 = %cst_22, %iterArg_181 = %cst_23, %iterArg_182 = %c_24, %iterArg_183 = %arg36, %iterArg_184 = %arg38, %iterArg_185 = %arg40, %iterArg_186 = %arg37, %iterArg_187 = %arg39, %iterArg_188 = %arg41, %iterArg_189 = %cst_25, %iterArg_190 = %cst_26, %iterArg_191 = %cst_27, %iterArg_192 = %cst_28, %iterArg_193 = %cst_29, %iterArg_194 = %cst_30, %iterArg_195 = %cst_31, %iterArg_196 = %cst_32, %iterArg_197 = %cst_33, %iterArg_198 = %cst_34, %iterArg_199 = %cst_35, %iterArg_200 = %cst_36, %iterArg_201 = %cst_37, %iterArg_202 = %cst_38, %iterArg_203 = %cst_39, %iterArg_204 = %cst_40, %iterArg_205 = %cst_41, %iterArg_206 = %cst_42, %iterArg_207 = %c_43, %iterArg_208 = %cst_44, %iterArg_209 = %cst_45, %iterArg_210 = %cst_46, %iterArg_211 = %cst_47, %iterArg_212 = %c_48, %iterArg_213 = %c_49, %iterArg_214 = %c_50, %iterArg_215 = %cst_51, %iterArg_216 = %c_52, %iterArg_217 = %c_53, %iterArg_218 = %c_54, %iterArg_219 = %cst_55, %iterArg_220 = %c_56, %iterArg_221 = %c_57, %iterArg_222 = %c_58, %iterArg_223 = %cst_59, %iterArg_224 = %c_60, %iterArg_225 = %c_61, %iterArg_226 = %c_62, %iterArg_227 = %cst_63, %iterArg_228 = %c_64, %iterArg_229 = %c_65, %iterArg_230 = %c_66, %iterArg_231 = %cst_67, %iterArg_232 = %c_68, %iterArg_233 = %c_69, %iterArg_234 = %c_70, %iterArg_235 = %cst_71, %iterArg_236 = %c_72, %iterArg_237 = %c_73, %iterArg_238 = %c_74, %iterArg_239 = %cst_75, %iterArg_240 = %c_76, %iterArg_241 = %c_77, %iterArg_242 = %c_78, %iterArg_243 = %cst_79, %iterArg_244 = %c_80, %iterArg_245 = %c_81, %iterArg_246 = %c_82, %iterArg_247 = %cst_83, %iterArg_248 = %c_84, %iterArg_249 = %c_85, %iterArg_250 = %c_86, %iterArg_251 = %cst_87, %iterArg_252 = %c_88, %iterArg_253 = %c_89, %iterArg_254 = %c_90, %iterArg_255 = %cst_91, %iterArg_256 = %c_92, %iterArg_257 = %c_93, %iterArg_258 = %c_94, %iterArg_259 = %cst_95, %iterArg_260 = %cst_96, %iterArg_261 = %cst_97, %iterArg_262 = %c_98, %iterArg_263 = %c_99, %iterArg_264 = %c_100, %iterArg_265 = %cst_101, %iterArg_266 = %c_102, %iterArg_267 = %c_103, %iterArg_268 = %c_104, %iterArg_269 = %cst_105, %iterArg_270 = %c_106, %iterArg_271 = %c_107, %iterArg_272 = %c_108, %iterArg_273 = %cst_109, %iterArg_274 = %c_110, %iterArg_275 = %c_111, %iterArg_276 = %c_112, %iterArg_277 = %cst_113, %iterArg_278 = %c_114, %iterArg_279 = %c_115, %iterArg_280 = %c_116, %iterArg_281 = %cst_117, %iterArg_282 = %c_118, %iterArg_283 = %c_119, %iterArg_284 = %c_120, %iterArg_285 = %cst_121, %iterArg_286 = %c_122, %iterArg_287 = %c_123, %iterArg_288 = %c_124, %iterArg_289 = %cst_125, %iterArg_290 = %c_126, %iterArg_291 = %c_127, %iterArg_292 = %c_128, %iterArg_293 = %cst_129, %iterArg_294 = %c_130, %iterArg_295 = %c_131, %iterArg_296 = %c_132, %iterArg_297 = %cst_133, %iterArg_298 = %c_134, %iterArg_299 = %c_135, %iterArg_300 = %c_136, %iterArg_301 = %cst_137, %iterArg_302 = %c_138, %iterArg_303 = %c_139, %iterArg_304 = %c_140, %iterArg_305 = %cst_141, %iterArg_306 = %c_142, %iterArg_307 = %c_143, %iterArg_308 = %c_144, %iterArg_309 = %cst_145, %iterArg_310 = %cst_146, %iterArg_311 = %cst_147, %iterArg_312 = %c_148, %iterArg_313 = %arg0, %iterArg_314 = %arg1, %iterArg_315 = %arg2, %iterArg_316 = %arg3, %iterArg_317 = %arg4, %iterArg_318 = %arg5, %iterArg_319 = %arg6, %iterArg_320 = %2, %iterArg_321 = %3, %iterArg_322 = %arg9, %iterArg_323 = %arg10, %iterArg_324 = %arg11, %iterArg_325 = %arg12, %iterArg_326 = %4, %iterArg_327 = %5, %iterArg_328 = %arg15, %iterArg_329 = %arg16, %iterArg_330 = %arg17, %iterArg_331 = %arg18, %iterArg_332 = %arg19, %iterArg_333 = %arg20, %iterArg_334 = %0, %iterArg_335 = %1, %iterArg_336 = %arg27, %iterArg_337 = %arg28, %iterArg_338 = %arg29) : tensor<1024xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x14x27x1xf32>, tensor<2x1024xf32>, tensor<2x15x26x1xf32>, tensor<2x1024xf32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x15x28x1xf32>, tensor<2x1024xf32>, tensor<2x16x27x1xf32>, tensor<2x1024xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<101xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<101xf32>, tensor<6xf32>, tensor<i32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>, tensor<f32>, tensor<i32>
    cond {
      %c_339 = stablehlo.constant dense<1024> : tensor<i32>
      %14 = stablehlo.compare  LT, %iterArg_312, %c_339,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %14 : tensor<i1>
    } do {
      %14 = stablehlo.dynamic_slice %iterArg, %iterArg_312, sizes = [1] : (tensor<1024xi32>, tensor<i32>) -> tensor<1xi32>
      %15 = stablehlo.reshape %14 : (tensor<1xi32>) -> tensor<i32>
      %16:26 = func.call @closed_call(%iterArg_149, %iterArg_150, %iterArg_151, %iterArg_152, %iterArg_153, %iterArg_154, %iterArg_155, %iterArg_156, %iterArg_157, %iterArg_158, %iterArg_159, %iterArg_160, %iterArg_161, %iterArg_162, %iterArg_163, %iterArg_164, %iterArg_165, %iterArg_166, %iterArg_167, %iterArg_168, %iterArg_169, %iterArg_170, %iterArg_171, %iterArg_172, %iterArg_173, %iterArg_174, %iterArg_175, %iterArg_176, %iterArg_177, %iterArg_178, %iterArg_179, %iterArg_180, %iterArg_181, %iterArg_182, %iterArg_183, %iterArg_184, %iterArg_185, %iterArg_186, %iterArg_187, %iterArg_188, %iterArg_189, %iterArg_190, %iterArg_191, %iterArg_192, %iterArg_193, %iterArg_194, %iterArg_195, %iterArg_196, %iterArg_197, %iterArg_198, %iterArg_199, %iterArg_200, %iterArg_201, %iterArg_202, %iterArg_203, %iterArg_204, %iterArg_205, %iterArg_206, %iterArg_207, %iterArg_208, %iterArg_209, %iterArg_210, %iterArg_211, %iterArg_212, %iterArg_213, %iterArg_214, %iterArg_215, %iterArg_216, %iterArg_217, %iterArg_218, %iterArg_219, %iterArg_220, %iterArg_221, %iterArg_222, %iterArg_223, %iterArg_224, %iterArg_225, %iterArg_226, %iterArg_227, %iterArg_228, %iterArg_229, %iterArg_230, %iterArg_231, %iterArg_232, %iterArg_233, %iterArg_234, %iterArg_235, %iterArg_236, %iterArg_237, %iterArg_238, %iterArg_239, %iterArg_240, %iterArg_241, %iterArg_242, %iterArg_243, %iterArg_244, %iterArg_245, %iterArg_246, %iterArg_247, %iterArg_248, %iterArg_249, %iterArg_250, %iterArg_251, %iterArg_252, %iterArg_253, %iterArg_254, %iterArg_255, %iterArg_256, %iterArg_257, %iterArg_258, %iterArg_259, %iterArg_260, %iterArg_261, %iterArg_262, %iterArg_263, %iterArg_264, %iterArg_265, %iterArg_266, %iterArg_267, %iterArg_268, %iterArg_269, %iterArg_270, %iterArg_271, %iterArg_272, %iterArg_273, %iterArg_274, %iterArg_275, %iterArg_276, %iterArg_277, %iterArg_278, %iterArg_279, %iterArg_280, %iterArg_281, %iterArg_282, %iterArg_283, %iterArg_284, %iterArg_285, %iterArg_286, %iterArg_287, %iterArg_288, %iterArg_289, %iterArg_290, %iterArg_291, %iterArg_292, %iterArg_293, %iterArg_294, %iterArg_295, %iterArg_296, %iterArg_297, %iterArg_298, %iterArg_299, %iterArg_300, %iterArg_301, %iterArg_302, %iterArg_303, %iterArg_304, %iterArg_305, %iterArg_306, %iterArg_307, %iterArg_308, %iterArg_309, %iterArg_310, %iterArg_311, %iterArg_313, %iterArg_314, %iterArg_315, %iterArg_316, %iterArg_317, %iterArg_318, %iterArg_319, %iterArg_320, %iterArg_321, %iterArg_322, %iterArg_323, %iterArg_324, %iterArg_325, %iterArg_326, %iterArg_327, %iterArg_328, %iterArg_329, %iterArg_330, %iterArg_331, %iterArg_332, %iterArg_333, %iterArg_334, %iterArg_335, %iterArg_336, %iterArg_337, %iterArg_338, %15) : (tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x14x27x1xf32>, tensor<2x1024xf32>, tensor<2x15x26x1xf32>, tensor<2x1024xf32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x15x28x1xf32>, tensor<2x1024xf32>, tensor<2x16x27x1xf32>, tensor<2x1024xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<101xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<101xf32>, tensor<6xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>, tensor<f32>, tensor<i32>, tensor<i32>) -> (tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>, tensor<f32>, tensor<i32>)
      %c_339 = stablehlo.constant dense<1> : tensor<i32>
      %17 = stablehlo.add %iterArg_312, %c_339 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_149, %iterArg_150, %iterArg_151, %iterArg_152, %iterArg_153, %iterArg_154, %iterArg_155, %iterArg_156, %iterArg_157, %iterArg_158, %iterArg_159, %iterArg_160, %iterArg_161, %iterArg_162, %iterArg_163, %iterArg_164, %iterArg_165, %iterArg_166, %iterArg_167, %iterArg_168, %iterArg_169, %iterArg_170, %iterArg_171, %iterArg_172, %iterArg_173, %iterArg_174, %iterArg_175, %iterArg_176, %iterArg_177, %iterArg_178, %iterArg_179, %iterArg_180, %iterArg_181, %iterArg_182, %iterArg_183, %iterArg_184, %iterArg_185, %iterArg_186, %iterArg_187, %iterArg_188, %iterArg_189, %iterArg_190, %iterArg_191, %iterArg_192, %iterArg_193, %iterArg_194, %iterArg_195, %iterArg_196, %iterArg_197, %iterArg_198, %iterArg_199, %iterArg_200, %iterArg_201, %iterArg_202, %iterArg_203, %iterArg_204, %iterArg_205, %iterArg_206, %iterArg_207, %iterArg_208, %iterArg_209, %iterArg_210, %iterArg_211, %iterArg_212, %iterArg_213, %iterArg_214, %iterArg_215, %iterArg_216, %iterArg_217, %iterArg_218, %iterArg_219, %iterArg_220, %iterArg_221, %iterArg_222, %iterArg_223, %iterArg_224, %iterArg_225, %iterArg_226, %iterArg_227, %iterArg_228, %iterArg_229, %iterArg_230, %iterArg_231, %iterArg_232, %iterArg_233, %iterArg_234, %iterArg_235, %iterArg_236, %iterArg_237, %iterArg_238, %iterArg_239, %iterArg_240, %iterArg_241, %iterArg_242, %iterArg_243, %iterArg_244, %iterArg_245, %iterArg_246, %iterArg_247, %iterArg_248, %iterArg_249, %iterArg_250, %iterArg_251, %iterArg_252, %iterArg_253, %iterArg_254, %iterArg_255, %iterArg_256, %iterArg_257, %iterArg_258, %iterArg_259, %iterArg_260, %iterArg_261, %iterArg_262, %iterArg_263, %iterArg_264, %iterArg_265, %iterArg_266, %iterArg_267, %iterArg_268, %iterArg_269, %iterArg_270, %iterArg_271, %iterArg_272, %iterArg_273, %iterArg_274, %iterArg_275, %iterArg_276, %iterArg_277, %iterArg_278, %iterArg_279, %iterArg_280, %iterArg_281, %iterArg_282, %iterArg_283, %iterArg_284, %iterArg_285, %iterArg_286, %iterArg_287, %iterArg_288, %iterArg_289, %iterArg_290, %iterArg_291, %iterArg_292, %iterArg_293, %iterArg_294, %iterArg_295, %iterArg_296, %iterArg_297, %iterArg_298, %iterArg_299, %iterArg_300, %iterArg_301, %iterArg_302, %iterArg_303, %iterArg_304, %iterArg_305, %iterArg_306, %iterArg_307, %iterArg_308, %iterArg_309, %iterArg_310, %iterArg_311, %17, %16#0, %16#1, %16#2, %16#3, %16#4, %16#5, %16#6, %16#7, %16#8, %16#9, %16#10, %16#11, %16#12, %16#13, %16#14, %16#15, %16#16, %16#17, %16#18, %16#19, %16#20, %16#21, %16#22, %16#23, %16#24, %16#25 : tensor<1024xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x14x27x1xf32>, tensor<2x1024xf32>, tensor<2x15x26x1xf32>, tensor<2x1024xf32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<2x15x28x1xf32>, tensor<2x1024xf32>, tensor<2x16x27x1xf32>, tensor<2x1024xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<101xf32>, tensor<6xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<101xf32>, tensor<6xf32>, tensor<i32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>, tensor<f32>, tensor<i32>
    }
    %8 = sdy.manual_computation(%7#172) in_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] out_shardings=[<@mesh, [{}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<1x24x1250x1201xf32>) {
      %14 = stablehlo.slice %arg42 [0:1, 0:24, 0:1250, 0:1201] : (tensor<1x24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      %15 = stablehlo.reshape %14 : (tensor<1x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %16 = "stablehlo.all_reduce"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
        %17 = stablehlo.add %arg43, %arg44 : tensor<f32>
        stablehlo.return %17 : tensor<f32>
      }) : (tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      sdy.return %16 : tensor<24x1250x1201xf32>
    } : (tensor<8x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
    %9 = sdy.manual_computation(%7#173) in_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] out_shardings=[<@mesh, [{}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<1x24x1251x1200xf32>) {
      %14 = stablehlo.slice %arg42 [0:1, 0:24, 0:1251, 0:1200] : (tensor<1x24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      %15 = stablehlo.reshape %14 : (tensor<1x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %16 = "stablehlo.all_reduce"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
        %17 = stablehlo.add %arg43, %arg44 : tensor<f32>
        stablehlo.return %17 : tensor<f32>
      }) : (tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      sdy.return %16 : tensor<24x1251x1200xf32>
    } : (tensor<8x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
    %10 = sdy.manual_computation(%7#178) in_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] out_shardings=[<@mesh, [{}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<1x24x1251x1200xf32>) {
      %14 = stablehlo.slice %arg42 [0:1, 0:24, 0:1251, 0:1200] : (tensor<1x24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      %15 = stablehlo.reshape %14 : (tensor<1x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %16 = "stablehlo.all_reduce"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
        %17 = stablehlo.add %arg43, %arg44 : tensor<f32>
        stablehlo.return %17 : tensor<f32>
      }) : (tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      sdy.return %16 : tensor<24x1251x1200xf32>
    } : (tensor<8x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
    %11 = sdy.manual_computation(%7#179) in_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] out_shardings=[<@mesh, [{}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<1x24x1250x1201xf32>) {
      %14 = stablehlo.slice %arg42 [0:1, 0:24, 0:1250, 0:1201] : (tensor<1x24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      %15 = stablehlo.reshape %14 : (tensor<1x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %16 = "stablehlo.all_reduce"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
        %17 = stablehlo.add %arg43, %arg44 : tensor<f32>
        stablehlo.return %17 : tensor<f32>
      }) : (tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      sdy.return %16 : tensor<24x1250x1201xf32>
    } : (tensor<8x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
    %12 = sdy.manual_computation(%7#186) in_shardings=[<@mesh, [{"fdtd"}, {}]>] out_shardings=[<@mesh, [{}]>] manual_axes={"fdtd"} (%arg42: tensor<1x242400xf32>) {
      %14 = stablehlo.slice %arg42 [0:1, 0:242400] : (tensor<1x242400xf32>) -> tensor<1x242400xf32>
      %15 = stablehlo.reshape %14 : (tensor<1x242400xf32>) -> tensor<242400xf32>
      %16 = "stablehlo.all_reduce"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
        %17 = stablehlo.add %arg43, %arg44 : tensor<f32>
        stablehlo.return %17 : tensor<f32>
      }) : (tensor<242400xf32>) -> tensor<242400xf32>
      sdy.return %16 : tensor<242400xf32>
    } : (tensor<8x242400xf32>) -> tensor<242400xf32>
    %13 = sdy.manual_computation(%7#187) in_shardings=[<@mesh, [{"fdtd"}, {}]>] out_shardings=[<@mesh, [{}]>] manual_axes={"fdtd"} (%arg42: tensor<1x242400xf32>) {
      %14 = stablehlo.slice %arg42 [0:1, 0:242400] : (tensor<1x242400xf32>) -> tensor<1x242400xf32>
      %15 = stablehlo.reshape %14 : (tensor<1x242400xf32>) -> tensor<242400xf32>
      %16 = "stablehlo.all_reduce"(%15) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
        %17 = stablehlo.add %arg43, %arg44 : tensor<f32>
        stablehlo.return %17 : tensor<f32>
      }) : (tensor<242400xf32>) -> tensor<242400xf32>
      sdy.return %16 : tensor<242400xf32>
    } : (tensor<8x242400xf32>) -> tensor<242400xf32>
    return %7#165, %7#166, %7#167, %7#168, %7#169, %7#170, %7#171, %8, %9, %7#174, %7#175, %7#176, %7#177, %10, %11, %7#180, %7#181, %7#182, %7#183, %7#184, %7#185, %arg21, %arg22, %arg23, %arg24, %12, %13, %7#188, %7#189, %7#190 : tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<24x1250x1201xf32>, tensor<24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<24x1251x1200xf32>, tensor<24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<2x101xf32>, tensor<2x101xf32>, tensor<2x101xf32>, tensor<2x101xf32>, tensor<242400xf32>, tensor<242400xf32>, tensor<202xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<242400xf32>, %arg2: tensor<i32>) -> tensor<242400xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<242400xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<i1>, tensor<242400xf32>
    return %2 : tensor<242400xf32>
  }
  func.func private @_where_4(%arg0: tensor<24xi1>, %arg1: tensor<24xi32>, %arg2: tensor<24xi32>) -> tensor<24xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24xi1>, tensor<24xi32>
    return %0 : tensor<24xi32>
  }
  func.func private @_where_29(%arg0: tensor<24x1250x1201xi1>, %arg1: tensor<24x1250x1201xf32>, %arg2: tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>
    return %0 : tensor<24x1250x1201xf32>
  }
  func.func private @_where_49(%arg0: tensor<24x1251x1200xi1>, %arg1: tensor<24x1251x1200xf32>, %arg2: tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>
    return %0 : tensor<24x1251x1200xf32>
  }
  func.func private @closed_call(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<6x3xi32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>, %arg9: tensor<0xf32>, %arg10: tensor<1x24x1xf32>, %arg11: tensor<1x24x1xf32>, %arg12: tensor<1x24x1xf32>, %arg13: tensor<24x1x1xf32>, %arg14: tensor<24x1x1xf32>, %arg15: tensor<24x1x1xf32>, %arg16: tensor<24x1x1xf32>, %arg17: tensor<24x1x1xf32>, %arg18: tensor<24x1x1xf32>, %arg19: tensor<1x1x24xf32>, %arg20: tensor<1x1x24xf32>, %arg21: tensor<1x1x24xf32>, %arg22: tensor<1x1x24xf32>, %arg23: tensor<1x1x24xf32>, %arg24: tensor<1x1x24xf32>, %arg25: tensor<1x24x1xf32>, %arg26: tensor<1x24x1xf32>, %arg27: tensor<1x24x1xf32>, %arg28: tensor<6x5xi32>, %arg29: tensor<2x14x27x1xf32>, %arg30: tensor<2x1024xf32>, %arg31: tensor<2x15x26x1xf32>, %arg32: tensor<2x1024xf32>, %arg33: tensor<6x3xi32>, %arg34: tensor<f32>, %arg35: tensor<f32>, %arg36: tensor<f32>, %arg37: tensor<10008x1251x1200xf32>, %arg38: tensor<10008x1250x1201xf32>, %arg39: tensor<10008x1251x1201xf32>, %arg40: tensor<1x24x1xf32>, %arg41: tensor<1x24x1xf32>, %arg42: tensor<1x24x1xf32>, %arg43: tensor<24x1x1xf32>, %arg44: tensor<24x1x1xf32>, %arg45: tensor<24x1x1xf32>, %arg46: tensor<24x1x1xf32>, %arg47: tensor<24x1x1xf32>, %arg48: tensor<24x1x1xf32>, %arg49: tensor<1x1x24xf32>, %arg50: tensor<1x1x24xf32>, %arg51: tensor<1x1x24xf32>, %arg52: tensor<1x1x24xf32>, %arg53: tensor<1x1x24xf32>, %arg54: tensor<1x1x24xf32>, %arg55: tensor<1x24x1xf32>, %arg56: tensor<1x24x1xf32>, %arg57: tensor<1x24x1xf32>, %arg58: tensor<6x5xi32>, %arg59: tensor<2x15x28x1xf32>, %arg60: tensor<2x1024xf32>, %arg61: tensor<2x16x27x1xf32>, %arg62: tensor<2x1024xf32>, %arg63: tensor<200x1xi32>, %arg64: tensor<200x1xi32>, %arg65: tensor<200x1xi32>, %arg66: tensor<200x1xf32>, %arg67: tensor<200x1xi32>, %arg68: tensor<200x1xi32>, %arg69: tensor<200x1xi32>, %arg70: tensor<200x1xf32>, %arg71: tensor<200x1xi32>, %arg72: tensor<200x1xi32>, %arg73: tensor<200x1xi32>, %arg74: tensor<200x1xf32>, %arg75: tensor<200x1xi32>, %arg76: tensor<200x1xi32>, %arg77: tensor<200x1xi32>, %arg78: tensor<200x1xf32>, %arg79: tensor<200x1xi32>, %arg80: tensor<200x1xi32>, %arg81: tensor<200x1xi32>, %arg82: tensor<200x1xf32>, %arg83: tensor<200x1xi32>, %arg84: tensor<200x1xi32>, %arg85: tensor<200x1xi32>, %arg86: tensor<200x1xf32>, %arg87: tensor<200x8xi32>, %arg88: tensor<200x8xi32>, %arg89: tensor<200x8xi32>, %arg90: tensor<200x8xf32>, %arg91: tensor<200x2xi32>, %arg92: tensor<200x2xi32>, %arg93: tensor<200x2xi32>, %arg94: tensor<200x2xf32>, %arg95: tensor<200x2xi32>, %arg96: tensor<200x2xi32>, %arg97: tensor<200x2xi32>, %arg98: tensor<200x2xf32>, %arg99: tensor<200x1xi32>, %arg100: tensor<200x1xi32>, %arg101: tensor<200x1xi32>, %arg102: tensor<200x1xf32>, %arg103: tensor<200x4xi32>, %arg104: tensor<200x4xi32>, %arg105: tensor<200x4xi32>, %arg106: tensor<200x4xf32>, %arg107: tensor<200x4xi32>, %arg108: tensor<200x4xi32>, %arg109: tensor<200x4xi32>, %arg110: tensor<200x4xf32>, %arg111: tensor<101xf32>, %arg112: tensor<6xf32>, %arg113: tensor<200x1xi32>, %arg114: tensor<200x1xi32>, %arg115: tensor<200x1xi32>, %arg116: tensor<200x1xf32>, %arg117: tensor<200x1xi32>, %arg118: tensor<200x1xi32>, %arg119: tensor<200x1xi32>, %arg120: tensor<200x1xf32>, %arg121: tensor<200x1xi32>, %arg122: tensor<200x1xi32>, %arg123: tensor<200x1xi32>, %arg124: tensor<200x1xf32>, %arg125: tensor<200x1xi32>, %arg126: tensor<200x1xi32>, %arg127: tensor<200x1xi32>, %arg128: tensor<200x1xf32>, %arg129: tensor<200x1xi32>, %arg130: tensor<200x1xi32>, %arg131: tensor<200x1xi32>, %arg132: tensor<200x1xf32>, %arg133: tensor<200x1xi32>, %arg134: tensor<200x1xi32>, %arg135: tensor<200x1xi32>, %arg136: tensor<200x1xf32>, %arg137: tensor<200x8xi32>, %arg138: tensor<200x8xi32>, %arg139: tensor<200x8xi32>, %arg140: tensor<200x8xf32>, %arg141: tensor<200x2xi32>, %arg142: tensor<200x2xi32>, %arg143: tensor<200x2xi32>, %arg144: tensor<200x2xf32>, %arg145: tensor<200x2xi32>, %arg146: tensor<200x2xi32>, %arg147: tensor<200x2xi32>, %arg148: tensor<200x2xf32>, %arg149: tensor<200x1xi32>, %arg150: tensor<200x1xi32>, %arg151: tensor<200x1xi32>, %arg152: tensor<200x1xf32>, %arg153: tensor<200x4xi32>, %arg154: tensor<200x4xi32>, %arg155: tensor<200x4xi32>, %arg156: tensor<200x4xf32>, %arg157: tensor<200x4xi32>, %arg158: tensor<200x4xi32>, %arg159: tensor<200x4xi32>, %arg160: tensor<200x4xf32>, %arg161: tensor<101xf32>, %arg162: tensor<6xf32>, %arg163: tensor<10008x1251x1200xf32>, %arg164: tensor<10008x1250x1201xf32>, %arg165: tensor<10008x1251x1201xf32>, %arg166: tensor<10008x1250x1201xf32>, %arg167: tensor<10008x1251x1200xf32>, %arg168: tensor<10008x1250x1200xf32>, %arg169: tensor<10008x24x1201xf32>, %arg170: tensor<8x24x1250x1201xf32>, %arg171: tensor<8x24x1251x1200xf32>, %arg172: tensor<10008x1251x24xf32>, %arg173: tensor<10008x1250x24xf32>, %arg174: tensor<10008x24x1200xf32>, %arg175: tensor<10008x24x1200xf32>, %arg176: tensor<8x24x1251x1200xf32>, %arg177: tensor<8x24x1250x1201xf32>, %arg178: tensor<10008x1250x24xf32>, %arg179: tensor<10008x1251x24xf32>, %arg180: tensor<10008x24x1201xf32>, %arg181: tensor<2x1xf32>, %arg182: tensor<2x1xf32>, %arg183: tensor<2xi32>, %arg184: tensor<8x242400xf32>, %arg185: tensor<8x242400xf32>, %arg186: tensor<202xf32>, %arg187: tensor<f32>, %arg188: tensor<i32>, %arg189: tensor<i32>) -> (tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>, tensor<f32>, tensor<i32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg189, %c : tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<f32>
    %3 = stablehlo.add %arg1, %2 : tensor<f32>
    %4:9 = sdy.manual_computation(%arg2, %arg28, %arg166, %arg167, %arg168, %arg163, %arg164, %arg165, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg169, %arg170, %arg171, %arg172, %arg173, %arg174, %arg9, %arg9, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, []>, <@mesh, []>, <@mesh, []>, <@mesh, []>, <@mesh, []>, <@mesh, []>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>] manual_axes={"fdtd"} (%arg190: tensor<6x3xi32>, %arg191: tensor<6x5xi32>, %arg192: tensor<1251x1250x1201xf32>, %arg193: tensor<1251x1251x1200xf32>, %arg194: tensor<1251x1250x1200xf32>, %arg195: tensor<1251x1251x1200xf32>, %arg196: tensor<1251x1250x1201xf32>, %arg197: tensor<1251x1251x1201xf32>, %arg198: tensor<f32>, %arg199: tensor<f32>, %arg200: tensor<f32>, %arg201: tensor<f32>, %arg202: tensor<f32>, %arg203: tensor<f32>, %arg204: tensor<1251x24x1201xf32>, %arg205: tensor<1x24x1250x1201xf32>, %arg206: tensor<1x24x1251x1200xf32>, %arg207: tensor<1251x1251x24xf32>, %arg208: tensor<1251x1250x24xf32>, %arg209: tensor<1251x24x1200xf32>, %arg210: tensor<0xf32>, %arg211: tensor<0xf32>, %arg212: tensor<0xf32>, %arg213: tensor<1x24x1xf32>, %arg214: tensor<1x24x1xf32>, %arg215: tensor<1x24x1xf32>, %arg216: tensor<24x1x1xf32>, %arg217: tensor<24x1x1xf32>, %arg218: tensor<24x1x1xf32>, %arg219: tensor<24x1x1xf32>, %arg220: tensor<24x1x1xf32>, %arg221: tensor<24x1x1xf32>, %arg222: tensor<1x1x24xf32>, %arg223: tensor<1x1x24xf32>, %arg224: tensor<1x1x24xf32>, %arg225: tensor<1x1x24xf32>, %arg226: tensor<1x1x24xf32>, %arg227: tensor<1x1x24xf32>, %arg228: tensor<1x24x1xf32>, %arg229: tensor<1x24x1xf32>, %arg230: tensor<1x24x1xf32>) {
      %c_60 = stablehlo.constant dense<1> : tensor<ui32>
      %c_61 = stablehlo.constant dense<8> : tensor<ui32>
      %129 = stablehlo.partition_id : tensor<ui32>
      %130 = stablehlo.divide %129, %c_60 : tensor<ui32>
      %131 = stablehlo.remainder %130, %c_61 : tensor<ui32>
      %132 = stablehlo.convert %131 : (tensor<ui32>) -> tensor<i32>
      %c_62 = stablehlo.constant dense<1251> : tensor<i32>
      %133 = stablehlo.multiply %132, %c_62 : tensor<i32>
      %c_63 = stablehlo.constant dense<0> : tensor<i32>
      %134 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %135 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %c_64 = stablehlo.constant dense<0> : tensor<i32>
      %136 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %137 = stablehlo.concatenate %134, %135, %136, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
      %138 = stablehlo.broadcast_in_dim %137, dims = [1] : (tensor<3xi32>) -> tensor<1x3xi32>
      %139 = stablehlo.concatenate %138, %arg190, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
      %140 = stablehlo.slice %arg205 [0:1, 0:24, 0:1250, 0:1201] : (tensor<1x24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      %141 = stablehlo.reshape %140 : (tensor<1x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %142 = stablehlo.slice %arg206 [0:1, 0:24, 0:1251, 0:1200] : (tensor<1x24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      %143 = stablehlo.reshape %142 : (tensor<1x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %144 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %145 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %146 = stablehlo.add %144, %145 : tensor<1251xi32>
      %c_65 = stablehlo.constant dense<0> : tensor<i32>
      %147 = stablehlo.broadcast_in_dim %c_65, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %148 = stablehlo.compare  GE, %146, %147,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_66 = stablehlo.constant dense<10000> : tensor<i32>
      %149 = stablehlo.broadcast_in_dim %c_66, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %150 = stablehlo.compare  LT, %146, %149,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %151 = stablehlo.and %148, %150 : tensor<1251xi1>
      %152 = stablehlo.reshape %151 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_67 = stablehlo.constant dense<true> : tensor<i1>
      %153 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %154 = stablehlo.and %153, %152 : tensor<1251x1x1xi1>
      %155 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_68 = stablehlo.constant dense<12> : tensor<i32>
      %156 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %157 = stablehlo.compare  LT, %155, %156,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_69 = stablehlo.constant dense<12> : tensor<i32>
      %158 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %159 = stablehlo.subtract %155, %158 : tensor<24xi32>
      %c_70 = stablehlo.constant dense<1250> : tensor<i32>
      %160 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %161 = stablehlo.add %159, %160 : tensor<24xi32>
      %c_71 = stablehlo.constant dense<12> : tensor<i32>
      %162 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %163 = stablehlo.subtract %161, %162 : tensor<24xi32>
      %164 = func.call @_where_4(%157, %155, %163) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_72 = stablehlo.constant dense<0> : tensor<i32>
      %165 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %166 = stablehlo.compare  GE, %164, %165,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_73 = stablehlo.constant dense<1250> : tensor<i32>
      %167 = stablehlo.broadcast_in_dim %c_73, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %168 = stablehlo.compare  LT, %164, %167,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %169 = stablehlo.and %166, %168 : tensor<24xi1>
      %170 = stablehlo.reshape %169 : (tensor<24xi1>) -> tensor<1x24x1xi1>
      %171 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x24x1xi1>
      %172 = stablehlo.broadcast_in_dim %170, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<1251x24x1xi1>
      %173 = stablehlo.and %171, %172 : tensor<1251x24x1xi1>
      %174 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_74 = stablehlo.constant dense<0> : tensor<i32>
      %175 = stablehlo.broadcast_in_dim %c_74, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %176 = stablehlo.compare  GE, %174, %175,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_75 = stablehlo.constant dense<1201> : tensor<i32>
      %177 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %178 = stablehlo.compare  LT, %174, %177,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %179 = stablehlo.and %176, %178 : tensor<1201xi1>
      %180 = stablehlo.reshape %179 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %181 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2] : (tensor<1251x24x1xi1>) -> tensor<1251x24x1201xi1>
      %182 = stablehlo.broadcast_in_dim %180, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<1251x24x1201xi1>
      %183 = stablehlo.and %181, %182 : tensor<1251x24x1201xi1>
      %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %184 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f32>) -> tensor<1251x24x1201xf32>
      %185 = func.call @_where_84(%183, %arg204, %184) : (tensor<1251x24x1201xi1>, tensor<1251x24x1201xf32>, tensor<1251x24x1201xf32>) -> tensor<1251x24x1201xf32>
      %186 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_77 = stablehlo.constant dense<12> : tensor<i32>
      %187 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %188 = stablehlo.compare  LT, %186, %187,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_78 = stablehlo.constant dense<12> : tensor<i32>
      %189 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %190 = stablehlo.subtract %186, %189 : tensor<24xi32>
      %c_79 = stablehlo.constant dense<10000> : tensor<i32>
      %191 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %192 = stablehlo.add %190, %191 : tensor<24xi32>
      %c_80 = stablehlo.constant dense<12> : tensor<i32>
      %193 = stablehlo.broadcast_in_dim %c_80, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %194 = stablehlo.subtract %192, %193 : tensor<24xi32>
      %195 = func.call @_where_4(%188, %186, %194) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_81 = stablehlo.constant dense<0> : tensor<i32>
      %196 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %197 = stablehlo.compare  GE, %195, %196,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_82 = stablehlo.constant dense<10000> : tensor<i32>
      %198 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %199 = stablehlo.compare  LT, %195, %198,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %200 = stablehlo.and %197, %199 : tensor<24xi1>
      %201 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %202 = stablehlo.compare  GE, %195, %201,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_83 = stablehlo.constant dense<1251> : tensor<i32>
      %203 = stablehlo.add %133, %c_83 : tensor<i32>
      %204 = stablehlo.broadcast_in_dim %203, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %205 = stablehlo.compare  LT, %195, %204,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %206 = stablehlo.and %202, %205 : tensor<24xi1>
      %207 = stablehlo.and %200, %206 : tensor<24xi1>
      %208 = stablehlo.reshape %207 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_84 = stablehlo.constant dense<true> : tensor<i1>
      %209 = stablehlo.broadcast_in_dim %c_84, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %210 = stablehlo.and %209, %208 : tensor<24x1x1xi1>
      %211 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_85 = stablehlo.constant dense<0> : tensor<i32>
      %212 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %213 = stablehlo.compare  GE, %211, %212,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_86 = stablehlo.constant dense<1250> : tensor<i32>
      %214 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %215 = stablehlo.compare  LT, %211, %214,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %216 = stablehlo.and %213, %215 : tensor<1250xi1>
      %217 = stablehlo.reshape %216 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %218 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1250x1xi1>
      %219 = stablehlo.broadcast_in_dim %217, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<24x1250x1xi1>
      %220 = stablehlo.and %218, %219 : tensor<24x1250x1xi1>
      %221 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_87 = stablehlo.constant dense<0> : tensor<i32>
      %222 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %223 = stablehlo.compare  GE, %221, %222,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_88 = stablehlo.constant dense<1201> : tensor<i32>
      %224 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %225 = stablehlo.compare  LT, %221, %224,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %226 = stablehlo.and %223, %225 : tensor<1201xi1>
      %227 = stablehlo.reshape %226 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %228 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2] : (tensor<24x1250x1xi1>) -> tensor<24x1250x1201xi1>
      %229 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<24x1250x1201xi1>
      %230 = stablehlo.and %228, %229 : tensor<24x1250x1201xi1>
      %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %231 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<f32>) -> tensor<24x1250x1201xf32>
      %232 = func.call @_where_86(%230, %141, %231) : (tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>, tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %233 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_90 = stablehlo.constant dense<12> : tensor<i32>
      %234 = stablehlo.broadcast_in_dim %c_90, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %235 = stablehlo.compare  LT, %233, %234,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_91 = stablehlo.constant dense<12> : tensor<i32>
      %236 = stablehlo.broadcast_in_dim %c_91, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %237 = stablehlo.subtract %233, %236 : tensor<24xi32>
      %c_92 = stablehlo.constant dense<10000> : tensor<i32>
      %238 = stablehlo.broadcast_in_dim %c_92, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %239 = stablehlo.add %237, %238 : tensor<24xi32>
      %c_93 = stablehlo.constant dense<12> : tensor<i32>
      %240 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %241 = stablehlo.subtract %239, %240 : tensor<24xi32>
      %242 = func.call @_where_4(%235, %233, %241) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_94 = stablehlo.constant dense<0> : tensor<i32>
      %243 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %244 = stablehlo.compare  GE, %242, %243,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_95 = stablehlo.constant dense<10000> : tensor<i32>
      %245 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %246 = stablehlo.compare  LT, %242, %245,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %247 = stablehlo.and %244, %246 : tensor<24xi1>
      %248 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %249 = stablehlo.compare  GE, %242, %248,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_96 = stablehlo.constant dense<1251> : tensor<i32>
      %250 = stablehlo.add %133, %c_96 : tensor<i32>
      %251 = stablehlo.broadcast_in_dim %250, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %252 = stablehlo.compare  LT, %242, %251,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %253 = stablehlo.and %249, %252 : tensor<24xi1>
      %254 = stablehlo.and %247, %253 : tensor<24xi1>
      %255 = stablehlo.reshape %254 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_97 = stablehlo.constant dense<true> : tensor<i1>
      %256 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %257 = stablehlo.and %256, %255 : tensor<24x1x1xi1>
      %258 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_98 = stablehlo.constant dense<0> : tensor<i32>
      %259 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %260 = stablehlo.compare  GE, %258, %259,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_99 = stablehlo.constant dense<1251> : tensor<i32>
      %261 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %262 = stablehlo.compare  LT, %258, %261,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %263 = stablehlo.and %260, %262 : tensor<1251xi1>
      %264 = stablehlo.reshape %263 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %265 = stablehlo.broadcast_in_dim %257, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1251x1xi1>
      %266 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<24x1251x1xi1>
      %267 = stablehlo.and %265, %266 : tensor<24x1251x1xi1>
      %268 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_100 = stablehlo.constant dense<0> : tensor<i32>
      %269 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %270 = stablehlo.compare  GE, %268, %269,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_101 = stablehlo.constant dense<1200> : tensor<i32>
      %271 = stablehlo.broadcast_in_dim %c_101, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %272 = stablehlo.compare  LT, %268, %271,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %273 = stablehlo.and %270, %272 : tensor<1200xi1>
      %274 = stablehlo.reshape %273 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %275 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2] : (tensor<24x1251x1xi1>) -> tensor<24x1251x1200xi1>
      %276 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<24x1251x1200xi1>
      %277 = stablehlo.and %275, %276 : tensor<24x1251x1200xi1>
      %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %278 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f32>) -> tensor<24x1251x1200xf32>
      %279 = func.call @_where_87(%277, %143, %278) : (tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>, tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %280 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %281 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %282 = stablehlo.add %280, %281 : tensor<1251xi32>
      %c_103 = stablehlo.constant dense<0> : tensor<i32>
      %283 = stablehlo.broadcast_in_dim %c_103, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %284 = stablehlo.compare  GE, %282, %283,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_104 = stablehlo.constant dense<10000> : tensor<i32>
      %285 = stablehlo.broadcast_in_dim %c_104, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %286 = stablehlo.compare  LT, %282, %285,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %287 = stablehlo.and %284, %286 : tensor<1251xi1>
      %288 = stablehlo.reshape %287 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_105 = stablehlo.constant dense<true> : tensor<i1>
      %289 = stablehlo.broadcast_in_dim %c_105, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %290 = stablehlo.and %289, %288 : tensor<1251x1x1xi1>
      %291 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_106 = stablehlo.constant dense<0> : tensor<i32>
      %292 = stablehlo.broadcast_in_dim %c_106, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %293 = stablehlo.compare  GE, %291, %292,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_107 = stablehlo.constant dense<1251> : tensor<i32>
      %294 = stablehlo.broadcast_in_dim %c_107, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %295 = stablehlo.compare  LT, %291, %294,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %296 = stablehlo.and %293, %295 : tensor<1251xi1>
      %297 = stablehlo.reshape %296 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %298 = stablehlo.broadcast_in_dim %290, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x1251x1xi1>
      %299 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<1251x1251x1xi1>
      %300 = stablehlo.and %298, %299 : tensor<1251x1251x1xi1>
      %301 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_108 = stablehlo.constant dense<12> : tensor<i32>
      %302 = stablehlo.broadcast_in_dim %c_108, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %303 = stablehlo.compare  LT, %301, %302,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_109 = stablehlo.constant dense<12> : tensor<i32>
      %304 = stablehlo.broadcast_in_dim %c_109, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %305 = stablehlo.subtract %301, %304 : tensor<24xi32>
      %c_110 = stablehlo.constant dense<1200> : tensor<i32>
      %306 = stablehlo.broadcast_in_dim %c_110, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %307 = stablehlo.add %305, %306 : tensor<24xi32>
      %c_111 = stablehlo.constant dense<12> : tensor<i32>
      %308 = stablehlo.broadcast_in_dim %c_111, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %309 = stablehlo.subtract %307, %308 : tensor<24xi32>
      %310 = func.call @_where_4(%303, %301, %309) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_112 = stablehlo.constant dense<0> : tensor<i32>
      %311 = stablehlo.broadcast_in_dim %c_112, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %312 = stablehlo.compare  GE, %310, %311,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_113 = stablehlo.constant dense<1200> : tensor<i32>
      %313 = stablehlo.broadcast_in_dim %c_113, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %314 = stablehlo.compare  LT, %310, %313,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %315 = stablehlo.and %312, %314 : tensor<24xi1>
      %316 = stablehlo.reshape %315 : (tensor<24xi1>) -> tensor<1x1x24xi1>
      %317 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2] : (tensor<1251x1251x1xi1>) -> tensor<1251x1251x24xi1>
      %318 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<1251x1251x24xi1>
      %319 = stablehlo.and %317, %318 : tensor<1251x1251x24xi1>
      %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %320 = stablehlo.broadcast_in_dim %cst_114, dims = [] : (tensor<f32>) -> tensor<1251x1251x24xf32>
      %321 = func.call @_where_94(%319, %arg207, %320) : (tensor<1251x1251x24xi1>, tensor<1251x1251x24xf32>, tensor<1251x1251x24xf32>) -> tensor<1251x1251x24xf32>
      %322 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %323 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %324 = stablehlo.add %322, %323 : tensor<1251xi32>
      %c_115 = stablehlo.constant dense<0> : tensor<i32>
      %325 = stablehlo.broadcast_in_dim %c_115, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %326 = stablehlo.compare  GE, %324, %325,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_116 = stablehlo.constant dense<10001> : tensor<i32>
      %327 = stablehlo.broadcast_in_dim %c_116, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %328 = stablehlo.compare  LT, %324, %327,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %329 = stablehlo.and %326, %328 : tensor<1251xi1>
      %330 = stablehlo.reshape %329 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_117 = stablehlo.constant dense<true> : tensor<i1>
      %331 = stablehlo.broadcast_in_dim %c_117, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %332 = stablehlo.and %331, %330 : tensor<1251x1x1xi1>
      %333 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_118 = stablehlo.constant dense<0> : tensor<i32>
      %334 = stablehlo.broadcast_in_dim %c_118, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %335 = stablehlo.compare  GE, %333, %334,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_119 = stablehlo.constant dense<1250> : tensor<i32>
      %336 = stablehlo.broadcast_in_dim %c_119, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %337 = stablehlo.compare  LT, %333, %336,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %338 = stablehlo.and %335, %337 : tensor<1250xi1>
      %339 = stablehlo.reshape %338 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %340 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x1250x1xi1>
      %341 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<1251x1250x1xi1>
      %342 = stablehlo.and %340, %341 : tensor<1251x1250x1xi1>
      %343 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_120 = stablehlo.constant dense<12> : tensor<i32>
      %344 = stablehlo.broadcast_in_dim %c_120, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %345 = stablehlo.compare  LT, %343, %344,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_121 = stablehlo.constant dense<12> : tensor<i32>
      %346 = stablehlo.broadcast_in_dim %c_121, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %347 = stablehlo.subtract %343, %346 : tensor<24xi32>
      %c_122 = stablehlo.constant dense<1200> : tensor<i32>
      %348 = stablehlo.broadcast_in_dim %c_122, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %349 = stablehlo.add %347, %348 : tensor<24xi32>
      %c_123 = stablehlo.constant dense<12> : tensor<i32>
      %350 = stablehlo.broadcast_in_dim %c_123, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %351 = stablehlo.subtract %349, %350 : tensor<24xi32>
      %352 = func.call @_where_4(%345, %343, %351) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_124 = stablehlo.constant dense<0> : tensor<i32>
      %353 = stablehlo.broadcast_in_dim %c_124, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %354 = stablehlo.compare  GE, %352, %353,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_125 = stablehlo.constant dense<1200> : tensor<i32>
      %355 = stablehlo.broadcast_in_dim %c_125, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %356 = stablehlo.compare  LT, %352, %355,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %357 = stablehlo.and %354, %356 : tensor<24xi1>
      %358 = stablehlo.reshape %357 : (tensor<24xi1>) -> tensor<1x1x24xi1>
      %359 = stablehlo.broadcast_in_dim %342, dims = [0, 1, 2] : (tensor<1251x1250x1xi1>) -> tensor<1251x1250x24xi1>
      %360 = stablehlo.broadcast_in_dim %358, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<1251x1250x24xi1>
      %361 = stablehlo.and %359, %360 : tensor<1251x1250x24xi1>
      %cst_126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_126, dims = [] : (tensor<f32>) -> tensor<1251x1250x24xf32>
      %363 = func.call @_where_100(%361, %arg208, %362) : (tensor<1251x1250x24xi1>, tensor<1251x1250x24xf32>, tensor<1251x1250x24xf32>) -> tensor<1251x1250x24xf32>
      %364 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %365 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %366 = stablehlo.add %364, %365 : tensor<1251xi32>
      %c_127 = stablehlo.constant dense<0> : tensor<i32>
      %367 = stablehlo.broadcast_in_dim %c_127, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %368 = stablehlo.compare  GE, %366, %367,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_128 = stablehlo.constant dense<10001> : tensor<i32>
      %369 = stablehlo.broadcast_in_dim %c_128, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %370 = stablehlo.compare  LT, %366, %369,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %371 = stablehlo.and %368, %370 : tensor<1251xi1>
      %372 = stablehlo.reshape %371 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_129 = stablehlo.constant dense<true> : tensor<i1>
      %373 = stablehlo.broadcast_in_dim %c_129, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %374 = stablehlo.and %373, %372 : tensor<1251x1x1xi1>
      %375 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_130 = stablehlo.constant dense<12> : tensor<i32>
      %376 = stablehlo.broadcast_in_dim %c_130, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %377 = stablehlo.compare  LT, %375, %376,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_131 = stablehlo.constant dense<12> : tensor<i32>
      %378 = stablehlo.broadcast_in_dim %c_131, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %379 = stablehlo.subtract %375, %378 : tensor<24xi32>
      %c_132 = stablehlo.constant dense<1250> : tensor<i32>
      %380 = stablehlo.broadcast_in_dim %c_132, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %381 = stablehlo.add %379, %380 : tensor<24xi32>
      %c_133 = stablehlo.constant dense<12> : tensor<i32>
      %382 = stablehlo.broadcast_in_dim %c_133, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %383 = stablehlo.subtract %381, %382 : tensor<24xi32>
      %384 = func.call @_where_4(%377, %375, %383) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_134 = stablehlo.constant dense<0> : tensor<i32>
      %385 = stablehlo.broadcast_in_dim %c_134, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %386 = stablehlo.compare  GE, %384, %385,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_135 = stablehlo.constant dense<1250> : tensor<i32>
      %387 = stablehlo.broadcast_in_dim %c_135, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %388 = stablehlo.compare  LT, %384, %387,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %389 = stablehlo.and %386, %388 : tensor<24xi1>
      %390 = stablehlo.reshape %389 : (tensor<24xi1>) -> tensor<1x24x1xi1>
      %391 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x24x1xi1>
      %392 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<1251x24x1xi1>
      %393 = stablehlo.and %391, %392 : tensor<1251x24x1xi1>
      %394 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_136 = stablehlo.constant dense<0> : tensor<i32>
      %395 = stablehlo.broadcast_in_dim %c_136, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %396 = stablehlo.compare  GE, %394, %395,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_137 = stablehlo.constant dense<1200> : tensor<i32>
      %397 = stablehlo.broadcast_in_dim %c_137, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %398 = stablehlo.compare  LT, %394, %397,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %399 = stablehlo.and %396, %398 : tensor<1200xi1>
      %400 = stablehlo.reshape %399 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %401 = stablehlo.broadcast_in_dim %393, dims = [0, 1, 2] : (tensor<1251x24x1xi1>) -> tensor<1251x24x1200xi1>
      %402 = stablehlo.broadcast_in_dim %400, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<1251x24x1200xi1>
      %403 = stablehlo.and %401, %402 : tensor<1251x24x1200xi1>
      %cst_138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_138, dims = [] : (tensor<f32>) -> tensor<1251x24x1200xf32>
      %405 = func.call @_where_105(%403, %arg209, %404) : (tensor<1251x24x1200xi1>, tensor<1251x24x1200xf32>, tensor<1251x24x1200xf32>) -> tensor<1251x24x1200xf32>
      %406 = stablehlo.slice %arg195 [0:1, 0:1251, 0:1200] : (tensor<1251x1251x1200xf32>) -> tensor<1x1251x1200xf32>
      %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f32>) -> tensor<1x1251x1200xf32>
      %408 = "stablehlo.collective_permute"(%406) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense_resource<__elided__> : tensor<7x2xi64>}> : (tensor<1x1251x1200xf32>) -> tensor<1x1251x1200xf32>
      %409 = stablehlo.slice %arg196 [0:1, 0:1250, 0:1201] : (tensor<1251x1250x1201xf32>) -> tensor<1x1250x1201xf32>
      %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<f32>) -> tensor<1x1250x1201xf32>
      %411 = "stablehlo.collective_permute"(%409) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense_resource<__elided__> : tensor<7x2xi64>}> : (tensor<1x1250x1201xf32>) -> tensor<1x1250x1201xf32>
      %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<1x1251x1201xf32>
      %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_142, dims = [] : (tensor<f32>) -> tensor<1x1251x1201xf32>
      %414:9 = stablehlo.custom_call @beamz_cuda_sharded(%arg192, %arg193, %arg194, %arg195, %arg196, %arg197, %arg198, %arg199, %arg200, %arg201, %arg202, %arg203, %arg191, %arg213, %arg214, %arg215, %arg216, %arg217, %arg218, %arg219, %arg220, %arg221, %arg222, %arg223, %arg224, %arg225, %arg226, %arg227, %arg228, %arg229, %arg230, %185, %232, %279, %321, %363, %405, %arg210, %arg211, %arg212, %139, %407, %408, %410, %411, %412, %413) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 0 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<1251x1250x1201xf32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1200xf32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1201xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1251x24x1201xf32>, tensor<24x1250x1201xf32>, tensor<24x1251x1200xf32>, tensor<1251x1251x24xf32>, tensor<1251x1250x24xf32>, tensor<1251x24x1200xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<1x1251x1200xf32>, tensor<1x1251x1200xf32>, tensor<1x1250x1201xf32>, tensor<1x1250x1201xf32>, tensor<1x1251x1201xf32>, tensor<1x1251x1201xf32>) -> (tensor<1251x1250x1201xf32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1200xf32>, tensor<1251x24x1201xf32>, tensor<24x1250x1201xf32>, tensor<24x1251x1200xf32>, tensor<1251x1251x24xf32>, tensor<1251x1250x24xf32>, tensor<1251x24x1200xf32>)
      %415 = stablehlo.broadcast_in_dim %414#4, dims = [1, 2, 3] : (tensor<24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      %416 = stablehlo.broadcast_in_dim %414#5, dims = [1, 2, 3] : (tensor<24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      sdy.return %414#0, %414#1, %414#2, %414#3, %415, %416, %414#6, %414#7, %414#8 : tensor<1251x1250x1201xf32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1200xf32>, tensor<1251x24x1201xf32>, tensor<1x24x1250x1201xf32>, tensor<1x24x1251x1200xf32>, tensor<1251x1251x24xf32>, tensor<1251x1250x24xf32>, tensor<1251x24x1200xf32>
    } : (tensor<6x3xi32>, tensor<6x5xi32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>) -> (tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>)
    %5 = sdy.manual_computation(%4#1, %arg188, %arg29, %arg30) in_shardings=[<@mesh, [{"fdtd"}, {}, {}]>, <@mesh, []>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}]>] manual_axes={"fdtd"} (%arg190: tensor<1251x1251x1200xf32>, %arg191: tensor<i32>, %arg192: tensor<2x14x27x1xf32>, %arg193: tensor<2x1024xf32>) {
      %c_60 = stablehlo.constant dense<0> : tensor<i32>
      %c_61 = stablehlo.constant dense<1023> : tensor<i32>
      %129 = func.call @clip(%arg191, %c_60, %c_61) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %c_62 = stablehlo.constant dense<1> : tensor<ui32>
      %c_63 = stablehlo.constant dense<8> : tensor<ui32>
      %130 = stablehlo.partition_id : tensor<ui32>
      %131 = stablehlo.divide %130, %c_62 : tensor<ui32>
      %132 = stablehlo.remainder %131, %c_63 : tensor<ui32>
      %133 = stablehlo.convert %132 : (tensor<ui32>) -> tensor<i32>
      %c_64 = stablehlo.constant dense<1251> : tensor<i32>
      %134 = stablehlo.multiply %133, %c_64 : tensor<i32>
      %c_65 = stablehlo.constant dense<4993> : tensor<i32>
      %135 = stablehlo.subtract %c_65, %134 : tensor<i32>
      %c_66 = stablehlo.constant dense<0> : tensor<i32>
      %c_67 = stablehlo.constant dense<1237> : tensor<i32>
      %136 = func.call @clip_132(%135, %c_66, %c_67) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %137 = stablehlo.add %134, %136 : tensor<i32>
      %138 = stablehlo.iota dim = 0 : tensor<14xi32>
      %139 = stablehlo.broadcast_in_dim %137, dims = [] : (tensor<i32>) -> tensor<14xi32>
      %140 = stablehlo.add %139, %138 : tensor<14xi32>
      %c_68 = stablehlo.constant dense<4993> : tensor<i32>
      %141 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<14xi32>
      %142 = stablehlo.subtract %140, %141 : tensor<14xi32>
      %c_69 = stablehlo.constant dense<0> : tensor<i32>
      %143 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<14xi32>
      %144 = stablehlo.compare  GE, %142, %143,  SIGNED : (tensor<14xi32>, tensor<14xi32>) -> tensor<14xi1>
      %c_70 = stablehlo.constant dense<14> : tensor<i32>
      %145 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<14xi32>
      %146 = stablehlo.compare  LT, %142, %145,  SIGNED : (tensor<14xi32>, tensor<14xi32>) -> tensor<14xi1>
      %147 = stablehlo.and %144, %146 : tensor<14xi1>
      %148 = stablehlo.slice %arg192 [0:1, 0:14, 0:27, 0:1] : (tensor<2x14x27x1xf32>) -> tensor<1x14x27x1xf32>
      %149 = stablehlo.reshape %148 : (tensor<1x14x27x1xf32>) -> tensor<14x27x1xf32>
      %150 = func.call @_take(%149, %142) : (tensor<14x27x1xf32>, tensor<14xi32>) -> tensor<14x27x1xf32>
      %151 = stablehlo.reshape %147 : (tensor<14xi1>) -> tensor<14x1x1xi1>
      %c_71 = stablehlo.constant dense<0> : tensor<i32>
      %152 = stablehlo.compare  LT, %129, %c_71,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_72 = stablehlo.constant dense<1024> : tensor<i32>
      %153 = stablehlo.add %129, %c_72 : tensor<i32>
      %154 = stablehlo.select %152, %153, %129 : tensor<i1>, tensor<i32>
      %c_73 = stablehlo.constant dense<0> : tensor<i32>
      %155 = stablehlo.dynamic_slice %arg193, %c_73, %154, sizes = [1, 1] : (tensor<2x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
      %156 = stablehlo.reshape %155 : (tensor<1x1xf32>) -> tensor<f32>
      %157 = stablehlo.broadcast_in_dim %156, dims = [] : (tensor<f32>) -> tensor<14x27x1xf32>
      %158 = stablehlo.multiply %150, %157 : tensor<14x27x1xf32>
      %c_74 = stablehlo.constant dense<0> : tensor<i32>
      %159 = func.call @_where_153(%151, %158, %c_74) : (tensor<14x1x1xi1>, tensor<14x27x1xf32>, tensor<i32>) -> tensor<14x27x1xf32>
      %160 = stablehlo.iota dim = 0 : tensor<1xi32>
      %161 = stablehlo.reshape %160 : (tensor<1xi32>) -> tensor<1x1x1xi32>
      %c_75 = stablehlo.constant dense<358> : tensor<i32>
      %162 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %163 = stablehlo.add %162, %161 : tensor<1x1x1xi32>
      %c_76 = stablehlo.constant dense<1> : tensor<i32>
      %164 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %165 = stablehlo.multiply %163, %164 : tensor<1x1x1xi32>
      %c_77 = stablehlo.constant dense<0> : tensor<i32>
      %166 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %167 = stablehlo.add %166, %165 : tensor<1x1x1xi32>
      %168 = stablehlo.iota dim = 0 : tensor<27xi32>
      %169 = stablehlo.reshape %168 : (tensor<27xi32>) -> tensor<1x27x1xi32>
      %c_78 = stablehlo.constant dense<612> : tensor<i32>
      %170 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<1x27x1xi32>
      %171 = stablehlo.add %170, %169 : tensor<1x27x1xi32>
      %c_79 = stablehlo.constant dense<1200> : tensor<i32>
      %172 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<1x27x1xi32>
      %173 = stablehlo.multiply %171, %172 : tensor<1x27x1xi32>
      %174 = stablehlo.broadcast_in_dim %167, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x27x1xi32>
      %175 = stablehlo.add %174, %173 : tensor<1x27x1xi32>
      %176 = stablehlo.iota dim = 0 : tensor<14xi32>
      %177 = stablehlo.reshape %176 : (tensor<14xi32>) -> tensor<14x1x1xi32>
      %178 = stablehlo.broadcast_in_dim %136, dims = [] : (tensor<i32>) -> tensor<14x1x1xi32>
      %179 = stablehlo.add %178, %177 : tensor<14x1x1xi32>
      %c_80 = stablehlo.constant dense<1501200> : tensor<i32>
      %180 = stablehlo.broadcast_in_dim %c_80, dims = [] : (tensor<i32>) -> tensor<14x1x1xi32>
      %181 = stablehlo.multiply %179, %180 : tensor<14x1x1xi32>
      %182 = stablehlo.broadcast_in_dim %175, dims = [0, 1, 2] : (tensor<1x27x1xi32>) -> tensor<14x27x1xi32>
      %183 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2] : (tensor<14x1x1xi32>) -> tensor<14x27x1xi32>
      %184 = stablehlo.add %182, %183 : tensor<14x27x1xi32>
      %185 = stablehlo.reshape %arg190 : (tensor<1251x1251x1200xf32>) -> tensor<1878001200xf32>
      %186 = stablehlo.reshape %184 : (tensor<14x27x1xi32>) -> tensor<378xi32>
      %187 = stablehlo.reshape %159 : (tensor<14x27x1xf32>) -> tensor<378xf32>
      %c_81 = stablehlo.constant dense<0> : tensor<i32>
      %188 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<378xi32>
      %189 = stablehlo.compare  LT, %186, %188,  SIGNED : (tensor<378xi32>, tensor<378xi32>) -> tensor<378xi1>
      %c_82 = stablehlo.constant dense<1878001200> : tensor<i32>
      %190 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<378xi32>
      %191 = stablehlo.add %186, %190 : tensor<378xi32>
      %192 = stablehlo.select %189, %191, %186 : tensor<378xi1>, tensor<378xi32>
      %193 = stablehlo.broadcast_in_dim %192, dims = [0] : (tensor<378xi32>) -> tensor<378x1xi32>
      %194 = "stablehlo.scatter"(%185, %193, %187) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
      ^bb0(%arg194: tensor<f32>, %arg195: tensor<f32>):
        %257 = stablehlo.add %arg194, %arg195 : tensor<f32>
        stablehlo.return %257 : tensor<f32>
      }) : (tensor<1878001200xf32>, tensor<378x1xi32>, tensor<378xf32>) -> tensor<1878001200xf32>
      %195 = stablehlo.reshape %194 : (tensor<1878001200xf32>) -> tensor<1251x1251x1200xf32>
      %c_83 = stablehlo.constant dense<4993> : tensor<i32>
      %196 = stablehlo.subtract %c_83, %134 : tensor<i32>
      %c_84 = stablehlo.constant dense<0> : tensor<i32>
      %c_85 = stablehlo.constant dense<1237> : tensor<i32>
      %197 = func.call @clip_132(%196, %c_84, %c_85) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %198 = stablehlo.add %134, %197 : tensor<i32>
      %199 = stablehlo.iota dim = 0 : tensor<14xi32>
      %200 = stablehlo.broadcast_in_dim %198, dims = [] : (tensor<i32>) -> tensor<14xi32>
      %201 = stablehlo.add %200, %199 : tensor<14xi32>
      %c_86 = stablehlo.constant dense<4993> : tensor<i32>
      %202 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<14xi32>
      %203 = stablehlo.subtract %201, %202 : tensor<14xi32>
      %c_87 = stablehlo.constant dense<0> : tensor<i32>
      %204 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<14xi32>
      %205 = stablehlo.compare  GE, %203, %204,  SIGNED : (tensor<14xi32>, tensor<14xi32>) -> tensor<14xi1>
      %c_88 = stablehlo.constant dense<14> : tensor<i32>
      %206 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<14xi32>
      %207 = stablehlo.compare  LT, %203, %206,  SIGNED : (tensor<14xi32>, tensor<14xi32>) -> tensor<14xi1>
      %208 = stablehlo.and %205, %207 : tensor<14xi1>
      %209 = stablehlo.slice %arg192 [1:2, 0:14, 0:27, 0:1] : (tensor<2x14x27x1xf32>) -> tensor<1x14x27x1xf32>
      %210 = stablehlo.reshape %209 : (tensor<1x14x27x1xf32>) -> tensor<14x27x1xf32>
      %211 = func.call @_take(%210, %203) : (tensor<14x27x1xf32>, tensor<14xi32>) -> tensor<14x27x1xf32>
      %212 = stablehlo.reshape %208 : (tensor<14xi1>) -> tensor<14x1x1xi1>
      %c_89 = stablehlo.constant dense<0> : tensor<i32>
      %213 = stablehlo.compare  LT, %129, %c_89,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_90 = stablehlo.constant dense<1024> : tensor<i32>
      %214 = stablehlo.add %129, %c_90 : tensor<i32>
      %215 = stablehlo.select %213, %214, %129 : tensor<i1>, tensor<i32>
      %c_91 = stablehlo.constant dense<1> : tensor<i32>
      %216 = stablehlo.dynamic_slice %arg193, %c_91, %215, sizes = [1, 1] : (tensor<2x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
      %217 = stablehlo.reshape %216 : (tensor<1x1xf32>) -> tensor<f32>
      %218 = stablehlo.broadcast_in_dim %217, dims = [] : (tensor<f32>) -> tensor<14x27x1xf32>
      %219 = stablehlo.multiply %211, %218 : tensor<14x27x1xf32>
      %c_92 = stablehlo.constant dense<0> : tensor<i32>
      %220 = func.call @_where_153(%212, %219, %c_92) : (tensor<14x1x1xi1>, tensor<14x27x1xf32>, tensor<i32>) -> tensor<14x27x1xf32>
      %221 = stablehlo.iota dim = 0 : tensor<1xi32>
      %222 = stablehlo.reshape %221 : (tensor<1xi32>) -> tensor<1x1x1xi32>
      %c_93 = stablehlo.constant dense<358> : tensor<i32>
      %223 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %224 = stablehlo.add %223, %222 : tensor<1x1x1xi32>
      %c_94 = stablehlo.constant dense<1> : tensor<i32>
      %225 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %226 = stablehlo.multiply %224, %225 : tensor<1x1x1xi32>
      %c_95 = stablehlo.constant dense<0> : tensor<i32>
      %227 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %228 = stablehlo.add %227, %226 : tensor<1x1x1xi32>
      %229 = stablehlo.iota dim = 0 : tensor<27xi32>
      %230 = stablehlo.reshape %229 : (tensor<27xi32>) -> tensor<1x27x1xi32>
      %c_96 = stablehlo.constant dense<612> : tensor<i32>
      %231 = stablehlo.broadcast_in_dim %c_96, dims = [] : (tensor<i32>) -> tensor<1x27x1xi32>
      %232 = stablehlo.add %231, %230 : tensor<1x27x1xi32>
      %c_97 = stablehlo.constant dense<1200> : tensor<i32>
      %233 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<1x27x1xi32>
      %234 = stablehlo.multiply %232, %233 : tensor<1x27x1xi32>
      %235 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x27x1xi32>
      %236 = stablehlo.add %235, %234 : tensor<1x27x1xi32>
      %237 = stablehlo.iota dim = 0 : tensor<14xi32>
      %238 = stablehlo.reshape %237 : (tensor<14xi32>) -> tensor<14x1x1xi32>
      %239 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<i32>) -> tensor<14x1x1xi32>
      %240 = stablehlo.add %239, %238 : tensor<14x1x1xi32>
      %c_98 = stablehlo.constant dense<1501200> : tensor<i32>
      %241 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<14x1x1xi32>
      %242 = stablehlo.multiply %240, %241 : tensor<14x1x1xi32>
      %243 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2] : (tensor<1x27x1xi32>) -> tensor<14x27x1xi32>
      %244 = stablehlo.broadcast_in_dim %242, dims = [0, 1, 2] : (tensor<14x1x1xi32>) -> tensor<14x27x1xi32>
      %245 = stablehlo.add %243, %244 : tensor<14x27x1xi32>
      %246 = stablehlo.reshape %195 : (tensor<1251x1251x1200xf32>) -> tensor<1878001200xf32>
      %247 = stablehlo.reshape %245 : (tensor<14x27x1xi32>) -> tensor<378xi32>
      %248 = stablehlo.reshape %220 : (tensor<14x27x1xf32>) -> tensor<378xf32>
      %c_99 = stablehlo.constant dense<0> : tensor<i32>
      %249 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<378xi32>
      %250 = stablehlo.compare  LT, %247, %249,  SIGNED : (tensor<378xi32>, tensor<378xi32>) -> tensor<378xi1>
      %c_100 = stablehlo.constant dense<1878001200> : tensor<i32>
      %251 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<378xi32>
      %252 = stablehlo.add %247, %251 : tensor<378xi32>
      %253 = stablehlo.select %250, %252, %247 : tensor<378xi1>, tensor<378xi32>
      %254 = stablehlo.broadcast_in_dim %253, dims = [0] : (tensor<378xi32>) -> tensor<378x1xi32>
      %255 = "stablehlo.scatter"(%246, %254, %248) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
      ^bb0(%arg194: tensor<f32>, %arg195: tensor<f32>):
        %257 = stablehlo.add %arg194, %arg195 : tensor<f32>
        stablehlo.return %257 : tensor<f32>
      }) : (tensor<1878001200xf32>, tensor<378x1xi32>, tensor<378xf32>) -> tensor<1878001200xf32>
      %256 = stablehlo.reshape %255 : (tensor<1878001200xf32>) -> tensor<1251x1251x1200xf32>
      sdy.return %256 : tensor<1251x1251x1200xf32>
    } : (tensor<10008x1251x1200xf32>, tensor<i32>, tensor<2x14x27x1xf32>, tensor<2x1024xf32>) -> tensor<10008x1251x1200xf32>
    %6 = sdy.manual_computation(%4#2, %arg188, %arg31, %arg32) in_shardings=[<@mesh, [{"fdtd"}, {}, {}]>, <@mesh, []>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}]>] manual_axes={"fdtd"} (%arg190: tensor<1251x1250x1200xf32>, %arg191: tensor<i32>, %arg192: tensor<2x15x26x1xf32>, %arg193: tensor<2x1024xf32>) {
      %c_60 = stablehlo.constant dense<0> : tensor<i32>
      %c_61 = stablehlo.constant dense<1023> : tensor<i32>
      %129 = func.call @clip(%arg191, %c_60, %c_61) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %c_62 = stablehlo.constant dense<1> : tensor<ui32>
      %c_63 = stablehlo.constant dense<8> : tensor<ui32>
      %130 = stablehlo.partition_id : tensor<ui32>
      %131 = stablehlo.divide %130, %c_62 : tensor<ui32>
      %132 = stablehlo.remainder %131, %c_63 : tensor<ui32>
      %133 = stablehlo.convert %132 : (tensor<ui32>) -> tensor<i32>
      %c_64 = stablehlo.constant dense<1251> : tensor<i32>
      %134 = stablehlo.multiply %133, %c_64 : tensor<i32>
      %c_65 = stablehlo.constant dense<4993> : tensor<i32>
      %135 = stablehlo.subtract %c_65, %134 : tensor<i32>
      %c_66 = stablehlo.constant dense<0> : tensor<i32>
      %c_67 = stablehlo.constant dense<1236> : tensor<i32>
      %136 = func.call @clip_132(%135, %c_66, %c_67) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %137 = stablehlo.add %134, %136 : tensor<i32>
      %138 = stablehlo.iota dim = 0 : tensor<15xi32>
      %139 = stablehlo.broadcast_in_dim %137, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %140 = stablehlo.add %139, %138 : tensor<15xi32>
      %c_68 = stablehlo.constant dense<4993> : tensor<i32>
      %141 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %142 = stablehlo.subtract %140, %141 : tensor<15xi32>
      %c_69 = stablehlo.constant dense<0> : tensor<i32>
      %143 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %144 = stablehlo.compare  GE, %142, %143,  SIGNED : (tensor<15xi32>, tensor<15xi32>) -> tensor<15xi1>
      %c_70 = stablehlo.constant dense<15> : tensor<i32>
      %145 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %146 = stablehlo.compare  LT, %142, %145,  SIGNED : (tensor<15xi32>, tensor<15xi32>) -> tensor<15xi1>
      %147 = stablehlo.and %144, %146 : tensor<15xi1>
      %148 = stablehlo.slice %arg192 [0:1, 0:15, 0:26, 0:1] : (tensor<2x15x26x1xf32>) -> tensor<1x15x26x1xf32>
      %149 = stablehlo.reshape %148 : (tensor<1x15x26x1xf32>) -> tensor<15x26x1xf32>
      %150 = func.call @_take_193(%149, %142) : (tensor<15x26x1xf32>, tensor<15xi32>) -> tensor<15x26x1xf32>
      %151 = stablehlo.reshape %147 : (tensor<15xi1>) -> tensor<15x1x1xi1>
      %c_71 = stablehlo.constant dense<0> : tensor<i32>
      %152 = stablehlo.compare  LT, %129, %c_71,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_72 = stablehlo.constant dense<1024> : tensor<i32>
      %153 = stablehlo.add %129, %c_72 : tensor<i32>
      %154 = stablehlo.select %152, %153, %129 : tensor<i1>, tensor<i32>
      %c_73 = stablehlo.constant dense<0> : tensor<i32>
      %155 = stablehlo.dynamic_slice %arg193, %c_73, %154, sizes = [1, 1] : (tensor<2x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
      %156 = stablehlo.reshape %155 : (tensor<1x1xf32>) -> tensor<f32>
      %157 = stablehlo.broadcast_in_dim %156, dims = [] : (tensor<f32>) -> tensor<15x26x1xf32>
      %158 = stablehlo.multiply %150, %157 : tensor<15x26x1xf32>
      %c_74 = stablehlo.constant dense<0> : tensor<i32>
      %159 = func.call @_where_199(%151, %158, %c_74) : (tensor<15x1x1xi1>, tensor<15x26x1xf32>, tensor<i32>) -> tensor<15x26x1xf32>
      %160 = stablehlo.iota dim = 0 : tensor<1xi32>
      %161 = stablehlo.reshape %160 : (tensor<1xi32>) -> tensor<1x1x1xi32>
      %c_75 = stablehlo.constant dense<358> : tensor<i32>
      %162 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %163 = stablehlo.add %162, %161 : tensor<1x1x1xi32>
      %c_76 = stablehlo.constant dense<1> : tensor<i32>
      %164 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %165 = stablehlo.multiply %163, %164 : tensor<1x1x1xi32>
      %c_77 = stablehlo.constant dense<0> : tensor<i32>
      %166 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %167 = stablehlo.add %166, %165 : tensor<1x1x1xi32>
      %168 = stablehlo.iota dim = 0 : tensor<26xi32>
      %169 = stablehlo.reshape %168 : (tensor<26xi32>) -> tensor<1x26x1xi32>
      %c_78 = stablehlo.constant dense<612> : tensor<i32>
      %170 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<1x26x1xi32>
      %171 = stablehlo.add %170, %169 : tensor<1x26x1xi32>
      %c_79 = stablehlo.constant dense<1200> : tensor<i32>
      %172 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<1x26x1xi32>
      %173 = stablehlo.multiply %171, %172 : tensor<1x26x1xi32>
      %174 = stablehlo.broadcast_in_dim %167, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x26x1xi32>
      %175 = stablehlo.add %174, %173 : tensor<1x26x1xi32>
      %176 = stablehlo.iota dim = 0 : tensor<15xi32>
      %177 = stablehlo.reshape %176 : (tensor<15xi32>) -> tensor<15x1x1xi32>
      %178 = stablehlo.broadcast_in_dim %136, dims = [] : (tensor<i32>) -> tensor<15x1x1xi32>
      %179 = stablehlo.add %178, %177 : tensor<15x1x1xi32>
      %c_80 = stablehlo.constant dense<1500000> : tensor<i32>
      %180 = stablehlo.broadcast_in_dim %c_80, dims = [] : (tensor<i32>) -> tensor<15x1x1xi32>
      %181 = stablehlo.multiply %179, %180 : tensor<15x1x1xi32>
      %182 = stablehlo.broadcast_in_dim %175, dims = [0, 1, 2] : (tensor<1x26x1xi32>) -> tensor<15x26x1xi32>
      %183 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2] : (tensor<15x1x1xi32>) -> tensor<15x26x1xi32>
      %184 = stablehlo.add %182, %183 : tensor<15x26x1xi32>
      %185 = stablehlo.reshape %arg190 : (tensor<1251x1250x1200xf32>) -> tensor<1876500000xf32>
      %186 = stablehlo.reshape %184 : (tensor<15x26x1xi32>) -> tensor<390xi32>
      %187 = stablehlo.reshape %159 : (tensor<15x26x1xf32>) -> tensor<390xf32>
      %c_81 = stablehlo.constant dense<0> : tensor<i32>
      %188 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<390xi32>
      %189 = stablehlo.compare  LT, %186, %188,  SIGNED : (tensor<390xi32>, tensor<390xi32>) -> tensor<390xi1>
      %c_82 = stablehlo.constant dense<1876500000> : tensor<i32>
      %190 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<390xi32>
      %191 = stablehlo.add %186, %190 : tensor<390xi32>
      %192 = stablehlo.select %189, %191, %186 : tensor<390xi1>, tensor<390xi32>
      %193 = stablehlo.broadcast_in_dim %192, dims = [0] : (tensor<390xi32>) -> tensor<390x1xi32>
      %194 = "stablehlo.scatter"(%185, %193, %187) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
      ^bb0(%arg194: tensor<f32>, %arg195: tensor<f32>):
        %257 = stablehlo.add %arg194, %arg195 : tensor<f32>
        stablehlo.return %257 : tensor<f32>
      }) : (tensor<1876500000xf32>, tensor<390x1xi32>, tensor<390xf32>) -> tensor<1876500000xf32>
      %195 = stablehlo.reshape %194 : (tensor<1876500000xf32>) -> tensor<1251x1250x1200xf32>
      %c_83 = stablehlo.constant dense<4993> : tensor<i32>
      %196 = stablehlo.subtract %c_83, %134 : tensor<i32>
      %c_84 = stablehlo.constant dense<0> : tensor<i32>
      %c_85 = stablehlo.constant dense<1236> : tensor<i32>
      %197 = func.call @clip_132(%196, %c_84, %c_85) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %198 = stablehlo.add %134, %197 : tensor<i32>
      %199 = stablehlo.iota dim = 0 : tensor<15xi32>
      %200 = stablehlo.broadcast_in_dim %198, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %201 = stablehlo.add %200, %199 : tensor<15xi32>
      %c_86 = stablehlo.constant dense<4993> : tensor<i32>
      %202 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %203 = stablehlo.subtract %201, %202 : tensor<15xi32>
      %c_87 = stablehlo.constant dense<0> : tensor<i32>
      %204 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %205 = stablehlo.compare  GE, %203, %204,  SIGNED : (tensor<15xi32>, tensor<15xi32>) -> tensor<15xi1>
      %c_88 = stablehlo.constant dense<15> : tensor<i32>
      %206 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %207 = stablehlo.compare  LT, %203, %206,  SIGNED : (tensor<15xi32>, tensor<15xi32>) -> tensor<15xi1>
      %208 = stablehlo.and %205, %207 : tensor<15xi1>
      %209 = stablehlo.slice %arg192 [1:2, 0:15, 0:26, 0:1] : (tensor<2x15x26x1xf32>) -> tensor<1x15x26x1xf32>
      %210 = stablehlo.reshape %209 : (tensor<1x15x26x1xf32>) -> tensor<15x26x1xf32>
      %211 = func.call @_take_193(%210, %203) : (tensor<15x26x1xf32>, tensor<15xi32>) -> tensor<15x26x1xf32>
      %212 = stablehlo.reshape %208 : (tensor<15xi1>) -> tensor<15x1x1xi1>
      %c_89 = stablehlo.constant dense<0> : tensor<i32>
      %213 = stablehlo.compare  LT, %129, %c_89,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_90 = stablehlo.constant dense<1024> : tensor<i32>
      %214 = stablehlo.add %129, %c_90 : tensor<i32>
      %215 = stablehlo.select %213, %214, %129 : tensor<i1>, tensor<i32>
      %c_91 = stablehlo.constant dense<1> : tensor<i32>
      %216 = stablehlo.dynamic_slice %arg193, %c_91, %215, sizes = [1, 1] : (tensor<2x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
      %217 = stablehlo.reshape %216 : (tensor<1x1xf32>) -> tensor<f32>
      %218 = stablehlo.broadcast_in_dim %217, dims = [] : (tensor<f32>) -> tensor<15x26x1xf32>
      %219 = stablehlo.multiply %211, %218 : tensor<15x26x1xf32>
      %c_92 = stablehlo.constant dense<0> : tensor<i32>
      %220 = func.call @_where_199(%212, %219, %c_92) : (tensor<15x1x1xi1>, tensor<15x26x1xf32>, tensor<i32>) -> tensor<15x26x1xf32>
      %221 = stablehlo.iota dim = 0 : tensor<1xi32>
      %222 = stablehlo.reshape %221 : (tensor<1xi32>) -> tensor<1x1x1xi32>
      %c_93 = stablehlo.constant dense<358> : tensor<i32>
      %223 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %224 = stablehlo.add %223, %222 : tensor<1x1x1xi32>
      %c_94 = stablehlo.constant dense<1> : tensor<i32>
      %225 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %226 = stablehlo.multiply %224, %225 : tensor<1x1x1xi32>
      %c_95 = stablehlo.constant dense<0> : tensor<i32>
      %227 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %228 = stablehlo.add %227, %226 : tensor<1x1x1xi32>
      %229 = stablehlo.iota dim = 0 : tensor<26xi32>
      %230 = stablehlo.reshape %229 : (tensor<26xi32>) -> tensor<1x26x1xi32>
      %c_96 = stablehlo.constant dense<612> : tensor<i32>
      %231 = stablehlo.broadcast_in_dim %c_96, dims = [] : (tensor<i32>) -> tensor<1x26x1xi32>
      %232 = stablehlo.add %231, %230 : tensor<1x26x1xi32>
      %c_97 = stablehlo.constant dense<1200> : tensor<i32>
      %233 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<1x26x1xi32>
      %234 = stablehlo.multiply %232, %233 : tensor<1x26x1xi32>
      %235 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x26x1xi32>
      %236 = stablehlo.add %235, %234 : tensor<1x26x1xi32>
      %237 = stablehlo.iota dim = 0 : tensor<15xi32>
      %238 = stablehlo.reshape %237 : (tensor<15xi32>) -> tensor<15x1x1xi32>
      %239 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<i32>) -> tensor<15x1x1xi32>
      %240 = stablehlo.add %239, %238 : tensor<15x1x1xi32>
      %c_98 = stablehlo.constant dense<1500000> : tensor<i32>
      %241 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<15x1x1xi32>
      %242 = stablehlo.multiply %240, %241 : tensor<15x1x1xi32>
      %243 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2] : (tensor<1x26x1xi32>) -> tensor<15x26x1xi32>
      %244 = stablehlo.broadcast_in_dim %242, dims = [0, 1, 2] : (tensor<15x1x1xi32>) -> tensor<15x26x1xi32>
      %245 = stablehlo.add %243, %244 : tensor<15x26x1xi32>
      %246 = stablehlo.reshape %195 : (tensor<1251x1250x1200xf32>) -> tensor<1876500000xf32>
      %247 = stablehlo.reshape %245 : (tensor<15x26x1xi32>) -> tensor<390xi32>
      %248 = stablehlo.reshape %220 : (tensor<15x26x1xf32>) -> tensor<390xf32>
      %c_99 = stablehlo.constant dense<0> : tensor<i32>
      %249 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<390xi32>
      %250 = stablehlo.compare  LT, %247, %249,  SIGNED : (tensor<390xi32>, tensor<390xi32>) -> tensor<390xi1>
      %c_100 = stablehlo.constant dense<1876500000> : tensor<i32>
      %251 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<390xi32>
      %252 = stablehlo.add %247, %251 : tensor<390xi32>
      %253 = stablehlo.select %250, %252, %247 : tensor<390xi1>, tensor<390xi32>
      %254 = stablehlo.broadcast_in_dim %253, dims = [0] : (tensor<390xi32>) -> tensor<390x1xi32>
      %255 = "stablehlo.scatter"(%246, %254, %248) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
      ^bb0(%arg194: tensor<f32>, %arg195: tensor<f32>):
        %257 = stablehlo.add %arg194, %arg195 : tensor<f32>
        stablehlo.return %257 : tensor<f32>
      }) : (tensor<1876500000xf32>, tensor<390x1xi32>, tensor<390xf32>) -> tensor<1876500000xf32>
      %256 = stablehlo.reshape %255 : (tensor<1876500000xf32>) -> tensor<1251x1250x1200xf32>
      sdy.return %256 : tensor<1251x1250x1200xf32>
    } : (tensor<10008x1250x1200xf32>, tensor<i32>, tensor<2x15x26x1xf32>, tensor<2x1024xf32>) -> tensor<10008x1250x1200xf32>
    %7:9 = sdy.manual_computation(%arg33, %arg58, %arg163, %arg164, %arg165, %4#0, %5, %6, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg175, %arg176, %arg177, %arg178, %arg179, %arg180, %arg9, %arg9, %arg9, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, []>, <@mesh, []>, <@mesh, []>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>] manual_axes={"fdtd"} (%arg190: tensor<6x3xi32>, %arg191: tensor<6x5xi32>, %arg192: tensor<1251x1251x1200xf32>, %arg193: tensor<1251x1250x1201xf32>, %arg194: tensor<1251x1251x1201xf32>, %arg195: tensor<1251x1250x1201xf32>, %arg196: tensor<1251x1251x1200xf32>, %arg197: tensor<1251x1250x1200xf32>, %arg198: tensor<f32>, %arg199: tensor<f32>, %arg200: tensor<f32>, %arg201: tensor<1251x1251x1200xf32>, %arg202: tensor<1251x1250x1201xf32>, %arg203: tensor<1251x1251x1201xf32>, %arg204: tensor<1251x24x1200xf32>, %arg205: tensor<1x24x1251x1200xf32>, %arg206: tensor<1x24x1250x1201xf32>, %arg207: tensor<1251x1250x24xf32>, %arg208: tensor<1251x1251x24xf32>, %arg209: tensor<1251x24x1201xf32>, %arg210: tensor<0xf32>, %arg211: tensor<0xf32>, %arg212: tensor<0xf32>, %arg213: tensor<1x24x1xf32>, %arg214: tensor<1x24x1xf32>, %arg215: tensor<1x24x1xf32>, %arg216: tensor<24x1x1xf32>, %arg217: tensor<24x1x1xf32>, %arg218: tensor<24x1x1xf32>, %arg219: tensor<24x1x1xf32>, %arg220: tensor<24x1x1xf32>, %arg221: tensor<24x1x1xf32>, %arg222: tensor<1x1x24xf32>, %arg223: tensor<1x1x24xf32>, %arg224: tensor<1x1x24xf32>, %arg225: tensor<1x1x24xf32>, %arg226: tensor<1x1x24xf32>, %arg227: tensor<1x1x24xf32>, %arg228: tensor<1x24x1xf32>, %arg229: tensor<1x24x1xf32>, %arg230: tensor<1x24x1xf32>) {
      %c_60 = stablehlo.constant dense<1> : tensor<ui32>
      %c_61 = stablehlo.constant dense<8> : tensor<ui32>
      %129 = stablehlo.partition_id : tensor<ui32>
      %130 = stablehlo.divide %129, %c_60 : tensor<ui32>
      %131 = stablehlo.remainder %130, %c_61 : tensor<ui32>
      %132 = stablehlo.convert %131 : (tensor<ui32>) -> tensor<i32>
      %c_62 = stablehlo.constant dense<1251> : tensor<i32>
      %133 = stablehlo.multiply %132, %c_62 : tensor<i32>
      %c_63 = stablehlo.constant dense<0> : tensor<i32>
      %134 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %135 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %c_64 = stablehlo.constant dense<0> : tensor<i32>
      %136 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %137 = stablehlo.concatenate %134, %135, %136, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
      %138 = stablehlo.broadcast_in_dim %137, dims = [1] : (tensor<3xi32>) -> tensor<1x3xi32>
      %139 = stablehlo.concatenate %138, %arg190, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
      %140 = stablehlo.slice %arg205 [0:1, 0:24, 0:1251, 0:1200] : (tensor<1x24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      %141 = stablehlo.reshape %140 : (tensor<1x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %142 = stablehlo.slice %arg206 [0:1, 0:24, 0:1250, 0:1201] : (tensor<1x24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      %143 = stablehlo.reshape %142 : (tensor<1x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %144 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %145 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %146 = stablehlo.add %144, %145 : tensor<1251xi32>
      %c_65 = stablehlo.constant dense<0> : tensor<i32>
      %147 = stablehlo.broadcast_in_dim %c_65, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %148 = stablehlo.compare  GE, %146, %147,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_66 = stablehlo.constant dense<10001> : tensor<i32>
      %149 = stablehlo.broadcast_in_dim %c_66, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %150 = stablehlo.compare  LT, %146, %149,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %151 = stablehlo.and %148, %150 : tensor<1251xi1>
      %152 = stablehlo.reshape %151 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_67 = stablehlo.constant dense<true> : tensor<i1>
      %153 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %154 = stablehlo.and %153, %152 : tensor<1251x1x1xi1>
      %155 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_68 = stablehlo.constant dense<12> : tensor<i32>
      %156 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %157 = stablehlo.compare  LT, %155, %156,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_69 = stablehlo.constant dense<12> : tensor<i32>
      %158 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %159 = stablehlo.subtract %155, %158 : tensor<24xi32>
      %c_70 = stablehlo.constant dense<1251> : tensor<i32>
      %160 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %161 = stablehlo.add %159, %160 : tensor<24xi32>
      %c_71 = stablehlo.constant dense<12> : tensor<i32>
      %162 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %163 = stablehlo.subtract %161, %162 : tensor<24xi32>
      %164 = func.call @_where_4(%157, %155, %163) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_72 = stablehlo.constant dense<0> : tensor<i32>
      %165 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %166 = stablehlo.compare  GE, %164, %165,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_73 = stablehlo.constant dense<1251> : tensor<i32>
      %167 = stablehlo.broadcast_in_dim %c_73, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %168 = stablehlo.compare  LT, %164, %167,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %169 = stablehlo.and %166, %168 : tensor<24xi1>
      %170 = stablehlo.reshape %169 : (tensor<24xi1>) -> tensor<1x24x1xi1>
      %171 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x24x1xi1>
      %172 = stablehlo.broadcast_in_dim %170, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<1251x24x1xi1>
      %173 = stablehlo.and %171, %172 : tensor<1251x24x1xi1>
      %174 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_74 = stablehlo.constant dense<0> : tensor<i32>
      %175 = stablehlo.broadcast_in_dim %c_74, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %176 = stablehlo.compare  GE, %174, %175,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_75 = stablehlo.constant dense<1200> : tensor<i32>
      %177 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %178 = stablehlo.compare  LT, %174, %177,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %179 = stablehlo.and %176, %178 : tensor<1200xi1>
      %180 = stablehlo.reshape %179 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %181 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2] : (tensor<1251x24x1xi1>) -> tensor<1251x24x1200xi1>
      %182 = stablehlo.broadcast_in_dim %180, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<1251x24x1200xi1>
      %183 = stablehlo.and %181, %182 : tensor<1251x24x1200xi1>
      %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %184 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f32>) -> tensor<1251x24x1200xf32>
      %185 = func.call @_where_105(%183, %arg204, %184) : (tensor<1251x24x1200xi1>, tensor<1251x24x1200xf32>, tensor<1251x24x1200xf32>) -> tensor<1251x24x1200xf32>
      %186 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_77 = stablehlo.constant dense<12> : tensor<i32>
      %187 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %188 = stablehlo.compare  LT, %186, %187,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_78 = stablehlo.constant dense<12> : tensor<i32>
      %189 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %190 = stablehlo.subtract %186, %189 : tensor<24xi32>
      %c_79 = stablehlo.constant dense<10001> : tensor<i32>
      %191 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %192 = stablehlo.add %190, %191 : tensor<24xi32>
      %c_80 = stablehlo.constant dense<12> : tensor<i32>
      %193 = stablehlo.broadcast_in_dim %c_80, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %194 = stablehlo.subtract %192, %193 : tensor<24xi32>
      %195 = func.call @_where_4(%188, %186, %194) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_81 = stablehlo.constant dense<0> : tensor<i32>
      %196 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %197 = stablehlo.compare  GE, %195, %196,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_82 = stablehlo.constant dense<10001> : tensor<i32>
      %198 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %199 = stablehlo.compare  LT, %195, %198,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %200 = stablehlo.and %197, %199 : tensor<24xi1>
      %201 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %202 = stablehlo.compare  GE, %195, %201,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_83 = stablehlo.constant dense<1251> : tensor<i32>
      %203 = stablehlo.add %133, %c_83 : tensor<i32>
      %204 = stablehlo.broadcast_in_dim %203, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %205 = stablehlo.compare  LT, %195, %204,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %206 = stablehlo.and %202, %205 : tensor<24xi1>
      %207 = stablehlo.and %200, %206 : tensor<24xi1>
      %208 = stablehlo.reshape %207 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_84 = stablehlo.constant dense<true> : tensor<i1>
      %209 = stablehlo.broadcast_in_dim %c_84, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %210 = stablehlo.and %209, %208 : tensor<24x1x1xi1>
      %211 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_85 = stablehlo.constant dense<0> : tensor<i32>
      %212 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %213 = stablehlo.compare  GE, %211, %212,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_86 = stablehlo.constant dense<1251> : tensor<i32>
      %214 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %215 = stablehlo.compare  LT, %211, %214,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %216 = stablehlo.and %213, %215 : tensor<1251xi1>
      %217 = stablehlo.reshape %216 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %218 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1251x1xi1>
      %219 = stablehlo.broadcast_in_dim %217, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<24x1251x1xi1>
      %220 = stablehlo.and %218, %219 : tensor<24x1251x1xi1>
      %221 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_87 = stablehlo.constant dense<0> : tensor<i32>
      %222 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %223 = stablehlo.compare  GE, %221, %222,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_88 = stablehlo.constant dense<1200> : tensor<i32>
      %224 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %225 = stablehlo.compare  LT, %221, %224,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %226 = stablehlo.and %223, %225 : tensor<1200xi1>
      %227 = stablehlo.reshape %226 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %228 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2] : (tensor<24x1251x1xi1>) -> tensor<24x1251x1200xi1>
      %229 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<24x1251x1200xi1>
      %230 = stablehlo.and %228, %229 : tensor<24x1251x1200xi1>
      %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %231 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<f32>) -> tensor<24x1251x1200xf32>
      %232 = func.call @_where_87(%230, %141, %231) : (tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>, tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %233 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_90 = stablehlo.constant dense<12> : tensor<i32>
      %234 = stablehlo.broadcast_in_dim %c_90, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %235 = stablehlo.compare  LT, %233, %234,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_91 = stablehlo.constant dense<12> : tensor<i32>
      %236 = stablehlo.broadcast_in_dim %c_91, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %237 = stablehlo.subtract %233, %236 : tensor<24xi32>
      %c_92 = stablehlo.constant dense<10001> : tensor<i32>
      %238 = stablehlo.broadcast_in_dim %c_92, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %239 = stablehlo.add %237, %238 : tensor<24xi32>
      %c_93 = stablehlo.constant dense<12> : tensor<i32>
      %240 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %241 = stablehlo.subtract %239, %240 : tensor<24xi32>
      %242 = func.call @_where_4(%235, %233, %241) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_94 = stablehlo.constant dense<0> : tensor<i32>
      %243 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %244 = stablehlo.compare  GE, %242, %243,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_95 = stablehlo.constant dense<10001> : tensor<i32>
      %245 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %246 = stablehlo.compare  LT, %242, %245,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %247 = stablehlo.and %244, %246 : tensor<24xi1>
      %248 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %249 = stablehlo.compare  GE, %242, %248,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_96 = stablehlo.constant dense<1251> : tensor<i32>
      %250 = stablehlo.add %133, %c_96 : tensor<i32>
      %251 = stablehlo.broadcast_in_dim %250, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %252 = stablehlo.compare  LT, %242, %251,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %253 = stablehlo.and %249, %252 : tensor<24xi1>
      %254 = stablehlo.and %247, %253 : tensor<24xi1>
      %255 = stablehlo.reshape %254 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_97 = stablehlo.constant dense<true> : tensor<i1>
      %256 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %257 = stablehlo.and %256, %255 : tensor<24x1x1xi1>
      %258 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_98 = stablehlo.constant dense<0> : tensor<i32>
      %259 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %260 = stablehlo.compare  GE, %258, %259,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_99 = stablehlo.constant dense<1250> : tensor<i32>
      %261 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %262 = stablehlo.compare  LT, %258, %261,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %263 = stablehlo.and %260, %262 : tensor<1250xi1>
      %264 = stablehlo.reshape %263 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %265 = stablehlo.broadcast_in_dim %257, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1250x1xi1>
      %266 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<24x1250x1xi1>
      %267 = stablehlo.and %265, %266 : tensor<24x1250x1xi1>
      %268 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_100 = stablehlo.constant dense<0> : tensor<i32>
      %269 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %270 = stablehlo.compare  GE, %268, %269,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_101 = stablehlo.constant dense<1201> : tensor<i32>
      %271 = stablehlo.broadcast_in_dim %c_101, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %272 = stablehlo.compare  LT, %268, %271,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %273 = stablehlo.and %270, %272 : tensor<1201xi1>
      %274 = stablehlo.reshape %273 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %275 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2] : (tensor<24x1250x1xi1>) -> tensor<24x1250x1201xi1>
      %276 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<24x1250x1201xi1>
      %277 = stablehlo.and %275, %276 : tensor<24x1250x1201xi1>
      %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %278 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f32>) -> tensor<24x1250x1201xf32>
      %279 = func.call @_where_86(%277, %143, %278) : (tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>, tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %280 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %281 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %282 = stablehlo.add %280, %281 : tensor<1251xi32>
      %c_103 = stablehlo.constant dense<0> : tensor<i32>
      %283 = stablehlo.broadcast_in_dim %c_103, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %284 = stablehlo.compare  GE, %282, %283,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_104 = stablehlo.constant dense<10001> : tensor<i32>
      %285 = stablehlo.broadcast_in_dim %c_104, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %286 = stablehlo.compare  LT, %282, %285,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %287 = stablehlo.and %284, %286 : tensor<1251xi1>
      %288 = stablehlo.reshape %287 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_105 = stablehlo.constant dense<true> : tensor<i1>
      %289 = stablehlo.broadcast_in_dim %c_105, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %290 = stablehlo.and %289, %288 : tensor<1251x1x1xi1>
      %291 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_106 = stablehlo.constant dense<0> : tensor<i32>
      %292 = stablehlo.broadcast_in_dim %c_106, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %293 = stablehlo.compare  GE, %291, %292,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_107 = stablehlo.constant dense<1250> : tensor<i32>
      %294 = stablehlo.broadcast_in_dim %c_107, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %295 = stablehlo.compare  LT, %291, %294,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %296 = stablehlo.and %293, %295 : tensor<1250xi1>
      %297 = stablehlo.reshape %296 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %298 = stablehlo.broadcast_in_dim %290, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x1250x1xi1>
      %299 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<1251x1250x1xi1>
      %300 = stablehlo.and %298, %299 : tensor<1251x1250x1xi1>
      %301 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_108 = stablehlo.constant dense<12> : tensor<i32>
      %302 = stablehlo.broadcast_in_dim %c_108, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %303 = stablehlo.compare  LT, %301, %302,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_109 = stablehlo.constant dense<12> : tensor<i32>
      %304 = stablehlo.broadcast_in_dim %c_109, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %305 = stablehlo.subtract %301, %304 : tensor<24xi32>
      %c_110 = stablehlo.constant dense<1201> : tensor<i32>
      %306 = stablehlo.broadcast_in_dim %c_110, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %307 = stablehlo.add %305, %306 : tensor<24xi32>
      %c_111 = stablehlo.constant dense<12> : tensor<i32>
      %308 = stablehlo.broadcast_in_dim %c_111, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %309 = stablehlo.subtract %307, %308 : tensor<24xi32>
      %310 = func.call @_where_4(%303, %301, %309) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_112 = stablehlo.constant dense<0> : tensor<i32>
      %311 = stablehlo.broadcast_in_dim %c_112, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %312 = stablehlo.compare  GE, %310, %311,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_113 = stablehlo.constant dense<1201> : tensor<i32>
      %313 = stablehlo.broadcast_in_dim %c_113, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %314 = stablehlo.compare  LT, %310, %313,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %315 = stablehlo.and %312, %314 : tensor<24xi1>
      %316 = stablehlo.reshape %315 : (tensor<24xi1>) -> tensor<1x1x24xi1>
      %317 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2] : (tensor<1251x1250x1xi1>) -> tensor<1251x1250x24xi1>
      %318 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<1251x1250x24xi1>
      %319 = stablehlo.and %317, %318 : tensor<1251x1250x24xi1>
      %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %320 = stablehlo.broadcast_in_dim %cst_114, dims = [] : (tensor<f32>) -> tensor<1251x1250x24xf32>
      %321 = func.call @_where_100(%319, %arg207, %320) : (tensor<1251x1250x24xi1>, tensor<1251x1250x24xf32>, tensor<1251x1250x24xf32>) -> tensor<1251x1250x24xf32>
      %322 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %323 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %324 = stablehlo.add %322, %323 : tensor<1251xi32>
      %c_115 = stablehlo.constant dense<0> : tensor<i32>
      %325 = stablehlo.broadcast_in_dim %c_115, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %326 = stablehlo.compare  GE, %324, %325,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_116 = stablehlo.constant dense<10000> : tensor<i32>
      %327 = stablehlo.broadcast_in_dim %c_116, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %328 = stablehlo.compare  LT, %324, %327,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %329 = stablehlo.and %326, %328 : tensor<1251xi1>
      %330 = stablehlo.reshape %329 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_117 = stablehlo.constant dense<true> : tensor<i1>
      %331 = stablehlo.broadcast_in_dim %c_117, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %332 = stablehlo.and %331, %330 : tensor<1251x1x1xi1>
      %333 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_118 = stablehlo.constant dense<0> : tensor<i32>
      %334 = stablehlo.broadcast_in_dim %c_118, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %335 = stablehlo.compare  GE, %333, %334,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_119 = stablehlo.constant dense<1251> : tensor<i32>
      %336 = stablehlo.broadcast_in_dim %c_119, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %337 = stablehlo.compare  LT, %333, %336,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %338 = stablehlo.and %335, %337 : tensor<1251xi1>
      %339 = stablehlo.reshape %338 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %340 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x1251x1xi1>
      %341 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<1251x1251x1xi1>
      %342 = stablehlo.and %340, %341 : tensor<1251x1251x1xi1>
      %343 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_120 = stablehlo.constant dense<12> : tensor<i32>
      %344 = stablehlo.broadcast_in_dim %c_120, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %345 = stablehlo.compare  LT, %343, %344,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_121 = stablehlo.constant dense<12> : tensor<i32>
      %346 = stablehlo.broadcast_in_dim %c_121, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %347 = stablehlo.subtract %343, %346 : tensor<24xi32>
      %c_122 = stablehlo.constant dense<1201> : tensor<i32>
      %348 = stablehlo.broadcast_in_dim %c_122, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %349 = stablehlo.add %347, %348 : tensor<24xi32>
      %c_123 = stablehlo.constant dense<12> : tensor<i32>
      %350 = stablehlo.broadcast_in_dim %c_123, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %351 = stablehlo.subtract %349, %350 : tensor<24xi32>
      %352 = func.call @_where_4(%345, %343, %351) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_124 = stablehlo.constant dense<0> : tensor<i32>
      %353 = stablehlo.broadcast_in_dim %c_124, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %354 = stablehlo.compare  GE, %352, %353,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_125 = stablehlo.constant dense<1201> : tensor<i32>
      %355 = stablehlo.broadcast_in_dim %c_125, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %356 = stablehlo.compare  LT, %352, %355,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %357 = stablehlo.and %354, %356 : tensor<24xi1>
      %358 = stablehlo.reshape %357 : (tensor<24xi1>) -> tensor<1x1x24xi1>
      %359 = stablehlo.broadcast_in_dim %342, dims = [0, 1, 2] : (tensor<1251x1251x1xi1>) -> tensor<1251x1251x24xi1>
      %360 = stablehlo.broadcast_in_dim %358, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<1251x1251x24xi1>
      %361 = stablehlo.and %359, %360 : tensor<1251x1251x24xi1>
      %cst_126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_126, dims = [] : (tensor<f32>) -> tensor<1251x1251x24xf32>
      %363 = func.call @_where_94(%361, %arg208, %362) : (tensor<1251x1251x24xi1>, tensor<1251x1251x24xf32>, tensor<1251x1251x24xf32>) -> tensor<1251x1251x24xf32>
      %364 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %365 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %366 = stablehlo.add %364, %365 : tensor<1251xi32>
      %c_127 = stablehlo.constant dense<0> : tensor<i32>
      %367 = stablehlo.broadcast_in_dim %c_127, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %368 = stablehlo.compare  GE, %366, %367,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_128 = stablehlo.constant dense<10000> : tensor<i32>
      %369 = stablehlo.broadcast_in_dim %c_128, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %370 = stablehlo.compare  LT, %366, %369,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %371 = stablehlo.and %368, %370 : tensor<1251xi1>
      %372 = stablehlo.reshape %371 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_129 = stablehlo.constant dense<true> : tensor<i1>
      %373 = stablehlo.broadcast_in_dim %c_129, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %374 = stablehlo.and %373, %372 : tensor<1251x1x1xi1>
      %375 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_130 = stablehlo.constant dense<12> : tensor<i32>
      %376 = stablehlo.broadcast_in_dim %c_130, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %377 = stablehlo.compare  LT, %375, %376,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_131 = stablehlo.constant dense<12> : tensor<i32>
      %378 = stablehlo.broadcast_in_dim %c_131, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %379 = stablehlo.subtract %375, %378 : tensor<24xi32>
      %c_132 = stablehlo.constant dense<1251> : tensor<i32>
      %380 = stablehlo.broadcast_in_dim %c_132, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %381 = stablehlo.add %379, %380 : tensor<24xi32>
      %c_133 = stablehlo.constant dense<12> : tensor<i32>
      %382 = stablehlo.broadcast_in_dim %c_133, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %383 = stablehlo.subtract %381, %382 : tensor<24xi32>
      %384 = func.call @_where_4(%377, %375, %383) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_134 = stablehlo.constant dense<0> : tensor<i32>
      %385 = stablehlo.broadcast_in_dim %c_134, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %386 = stablehlo.compare  GE, %384, %385,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_135 = stablehlo.constant dense<1251> : tensor<i32>
      %387 = stablehlo.broadcast_in_dim %c_135, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %388 = stablehlo.compare  LT, %384, %387,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %389 = stablehlo.and %386, %388 : tensor<24xi1>
      %390 = stablehlo.reshape %389 : (tensor<24xi1>) -> tensor<1x24x1xi1>
      %391 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x24x1xi1>
      %392 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<1251x24x1xi1>
      %393 = stablehlo.and %391, %392 : tensor<1251x24x1xi1>
      %394 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_136 = stablehlo.constant dense<0> : tensor<i32>
      %395 = stablehlo.broadcast_in_dim %c_136, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %396 = stablehlo.compare  GE, %394, %395,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_137 = stablehlo.constant dense<1201> : tensor<i32>
      %397 = stablehlo.broadcast_in_dim %c_137, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %398 = stablehlo.compare  LT, %394, %397,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %399 = stablehlo.and %396, %398 : tensor<1201xi1>
      %400 = stablehlo.reshape %399 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %401 = stablehlo.broadcast_in_dim %393, dims = [0, 1, 2] : (tensor<1251x24x1xi1>) -> tensor<1251x24x1201xi1>
      %402 = stablehlo.broadcast_in_dim %400, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<1251x24x1201xi1>
      %403 = stablehlo.and %401, %402 : tensor<1251x24x1201xi1>
      %cst_138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_138, dims = [] : (tensor<f32>) -> tensor<1251x24x1201xf32>
      %405 = func.call @_where_84(%403, %arg209, %404) : (tensor<1251x24x1201xi1>, tensor<1251x24x1201xf32>, tensor<1251x24x1201xf32>) -> tensor<1251x24x1201xf32>
      %406 = stablehlo.slice %arg195 [1250:1251, 0:1250, 0:1201] : (tensor<1251x1250x1201xf32>) -> tensor<1x1250x1201xf32>
      %407 = "stablehlo.collective_permute"(%406) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense_resource<__elided__> : tensor<7x2xi64>}> : (tensor<1x1250x1201xf32>) -> tensor<1x1250x1201xf32>
      %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f32>) -> tensor<1x1250x1201xf32>
      %409 = stablehlo.slice %arg196 [1250:1251, 0:1251, 0:1200] : (tensor<1251x1251x1200xf32>) -> tensor<1x1251x1200xf32>
      %410 = "stablehlo.collective_permute"(%409) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense_resource<__elided__> : tensor<7x2xi64>}> : (tensor<1x1251x1200xf32>) -> tensor<1x1251x1200xf32>
      %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %411 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<f32>) -> tensor<1x1251x1200xf32>
      %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<1x1250x1200xf32>
      %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_142, dims = [] : (tensor<f32>) -> tensor<1x1250x1200xf32>
      %414:9 = stablehlo.custom_call @beamz_cuda_sharded(%arg192, %arg193, %arg194, %arg195, %arg196, %arg197, %arg198, %arg199, %arg200, %arg201, %arg202, %arg203, %arg191, %arg213, %arg214, %arg215, %arg216, %arg217, %arg218, %arg219, %arg220, %arg221, %arg222, %arg223, %arg224, %arg225, %arg226, %arg227, %arg228, %arg229, %arg230, %185, %232, %279, %321, %363, %405, %arg210, %arg211, %arg212, %139, %407, %408, %410, %411, %412, %413) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 1 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<1251x1251x1200xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1201xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1200xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1201xf32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1251x24x1200xf32>, tensor<24x1251x1200xf32>, tensor<24x1250x1201xf32>, tensor<1251x1250x24xf32>, tensor<1251x1251x24xf32>, tensor<1251x24x1201xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<1x1250x1201xf32>, tensor<1x1250x1201xf32>, tensor<1x1251x1200xf32>, tensor<1x1251x1200xf32>, tensor<1x1250x1200xf32>, tensor<1x1250x1200xf32>) -> (tensor<1251x1251x1200xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1201xf32>, tensor<1251x24x1200xf32>, tensor<24x1251x1200xf32>, tensor<24x1250x1201xf32>, tensor<1251x1250x24xf32>, tensor<1251x1251x24xf32>, tensor<1251x24x1201xf32>)
      %415 = stablehlo.broadcast_in_dim %414#4, dims = [1, 2, 3] : (tensor<24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      %416 = stablehlo.broadcast_in_dim %414#5, dims = [1, 2, 3] : (tensor<24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      sdy.return %414#0, %414#1, %414#2, %414#3, %415, %416, %414#6, %414#7, %414#8 : tensor<1251x1251x1200xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1201xf32>, tensor<1251x24x1200xf32>, tensor<1x24x1251x1200xf32>, tensor<1x24x1250x1201xf32>, tensor<1251x1250x24xf32>, tensor<1251x1251x24xf32>, tensor<1251x24x1201xf32>
    } : (tensor<6x3xi32>, tensor<6x5xi32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>) -> (tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>)
    %8 = sdy.manual_computation(%7#1, %arg188, %arg59, %arg60) in_shardings=[<@mesh, [{"fdtd"}, {}, {}]>, <@mesh, []>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}]>] manual_axes={"fdtd"} (%arg190: tensor<1251x1250x1201xf32>, %arg191: tensor<i32>, %arg192: tensor<2x15x28x1xf32>, %arg193: tensor<2x1024xf32>) {
      %c_60 = stablehlo.constant dense<0> : tensor<i32>
      %c_61 = stablehlo.constant dense<1023> : tensor<i32>
      %129 = func.call @clip(%arg191, %c_60, %c_61) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %c_62 = stablehlo.constant dense<1> : tensor<ui32>
      %c_63 = stablehlo.constant dense<8> : tensor<ui32>
      %130 = stablehlo.partition_id : tensor<ui32>
      %131 = stablehlo.divide %130, %c_62 : tensor<ui32>
      %132 = stablehlo.remainder %131, %c_63 : tensor<ui32>
      %133 = stablehlo.convert %132 : (tensor<ui32>) -> tensor<i32>
      %c_64 = stablehlo.constant dense<1251> : tensor<i32>
      %134 = stablehlo.multiply %133, %c_64 : tensor<i32>
      %c_65 = stablehlo.constant dense<4993> : tensor<i32>
      %135 = stablehlo.subtract %c_65, %134 : tensor<i32>
      %c_66 = stablehlo.constant dense<0> : tensor<i32>
      %c_67 = stablehlo.constant dense<1236> : tensor<i32>
      %136 = func.call @clip_132(%135, %c_66, %c_67) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %137 = stablehlo.add %134, %136 : tensor<i32>
      %138 = stablehlo.iota dim = 0 : tensor<15xi32>
      %139 = stablehlo.broadcast_in_dim %137, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %140 = stablehlo.add %139, %138 : tensor<15xi32>
      %c_68 = stablehlo.constant dense<4993> : tensor<i32>
      %141 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %142 = stablehlo.subtract %140, %141 : tensor<15xi32>
      %c_69 = stablehlo.constant dense<0> : tensor<i32>
      %143 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %144 = stablehlo.compare  GE, %142, %143,  SIGNED : (tensor<15xi32>, tensor<15xi32>) -> tensor<15xi1>
      %c_70 = stablehlo.constant dense<15> : tensor<i32>
      %145 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %146 = stablehlo.compare  LT, %142, %145,  SIGNED : (tensor<15xi32>, tensor<15xi32>) -> tensor<15xi1>
      %147 = stablehlo.and %144, %146 : tensor<15xi1>
      %148 = stablehlo.slice %arg192 [0:1, 0:15, 0:28, 0:1] : (tensor<2x15x28x1xf32>) -> tensor<1x15x28x1xf32>
      %149 = stablehlo.reshape %148 : (tensor<1x15x28x1xf32>) -> tensor<15x28x1xf32>
      %150 = func.call @_take_236(%149, %142) : (tensor<15x28x1xf32>, tensor<15xi32>) -> tensor<15x28x1xf32>
      %151 = stablehlo.reshape %147 : (tensor<15xi1>) -> tensor<15x1x1xi1>
      %c_71 = stablehlo.constant dense<0> : tensor<i32>
      %152 = stablehlo.compare  LT, %129, %c_71,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_72 = stablehlo.constant dense<1024> : tensor<i32>
      %153 = stablehlo.add %129, %c_72 : tensor<i32>
      %154 = stablehlo.select %152, %153, %129 : tensor<i1>, tensor<i32>
      %c_73 = stablehlo.constant dense<0> : tensor<i32>
      %155 = stablehlo.dynamic_slice %arg193, %c_73, %154, sizes = [1, 1] : (tensor<2x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
      %156 = stablehlo.reshape %155 : (tensor<1x1xf32>) -> tensor<f32>
      %157 = stablehlo.broadcast_in_dim %156, dims = [] : (tensor<f32>) -> tensor<15x28x1xf32>
      %158 = stablehlo.multiply %150, %157 : tensor<15x28x1xf32>
      %c_74 = stablehlo.constant dense<0> : tensor<i32>
      %159 = func.call @_where_240(%151, %158, %c_74) : (tensor<15x1x1xi1>, tensor<15x28x1xf32>, tensor<i32>) -> tensor<15x28x1xf32>
      %160 = stablehlo.iota dim = 0 : tensor<1xi32>
      %161 = stablehlo.reshape %160 : (tensor<1xi32>) -> tensor<1x1x1xi32>
      %c_75 = stablehlo.constant dense<358> : tensor<i32>
      %162 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %163 = stablehlo.add %162, %161 : tensor<1x1x1xi32>
      %c_76 = stablehlo.constant dense<1> : tensor<i32>
      %164 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %165 = stablehlo.multiply %163, %164 : tensor<1x1x1xi32>
      %c_77 = stablehlo.constant dense<0> : tensor<i32>
      %166 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %167 = stablehlo.add %166, %165 : tensor<1x1x1xi32>
      %168 = stablehlo.iota dim = 0 : tensor<28xi32>
      %169 = stablehlo.reshape %168 : (tensor<28xi32>) -> tensor<1x28x1xi32>
      %c_78 = stablehlo.constant dense<611> : tensor<i32>
      %170 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<1x28x1xi32>
      %171 = stablehlo.add %170, %169 : tensor<1x28x1xi32>
      %c_79 = stablehlo.constant dense<1201> : tensor<i32>
      %172 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<1x28x1xi32>
      %173 = stablehlo.multiply %171, %172 : tensor<1x28x1xi32>
      %174 = stablehlo.broadcast_in_dim %167, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x28x1xi32>
      %175 = stablehlo.add %174, %173 : tensor<1x28x1xi32>
      %176 = stablehlo.iota dim = 0 : tensor<15xi32>
      %177 = stablehlo.reshape %176 : (tensor<15xi32>) -> tensor<15x1x1xi32>
      %178 = stablehlo.broadcast_in_dim %136, dims = [] : (tensor<i32>) -> tensor<15x1x1xi32>
      %179 = stablehlo.add %178, %177 : tensor<15x1x1xi32>
      %c_80 = stablehlo.constant dense<1501250> : tensor<i32>
      %180 = stablehlo.broadcast_in_dim %c_80, dims = [] : (tensor<i32>) -> tensor<15x1x1xi32>
      %181 = stablehlo.multiply %179, %180 : tensor<15x1x1xi32>
      %182 = stablehlo.broadcast_in_dim %175, dims = [0, 1, 2] : (tensor<1x28x1xi32>) -> tensor<15x28x1xi32>
      %183 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2] : (tensor<15x1x1xi32>) -> tensor<15x28x1xi32>
      %184 = stablehlo.add %182, %183 : tensor<15x28x1xi32>
      %185 = stablehlo.reshape %arg190 : (tensor<1251x1250x1201xf32>) -> tensor<1878063750xf32>
      %186 = stablehlo.reshape %184 : (tensor<15x28x1xi32>) -> tensor<420xi32>
      %187 = stablehlo.reshape %159 : (tensor<15x28x1xf32>) -> tensor<420xf32>
      %c_81 = stablehlo.constant dense<0> : tensor<i32>
      %188 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<420xi32>
      %189 = stablehlo.compare  LT, %186, %188,  SIGNED : (tensor<420xi32>, tensor<420xi32>) -> tensor<420xi1>
      %c_82 = stablehlo.constant dense<1878063750> : tensor<i32>
      %190 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<420xi32>
      %191 = stablehlo.add %186, %190 : tensor<420xi32>
      %192 = stablehlo.select %189, %191, %186 : tensor<420xi1>, tensor<420xi32>
      %193 = stablehlo.broadcast_in_dim %192, dims = [0] : (tensor<420xi32>) -> tensor<420x1xi32>
      %194 = "stablehlo.scatter"(%185, %193, %187) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
      ^bb0(%arg194: tensor<f32>, %arg195: tensor<f32>):
        %257 = stablehlo.add %arg194, %arg195 : tensor<f32>
        stablehlo.return %257 : tensor<f32>
      }) : (tensor<1878063750xf32>, tensor<420x1xi32>, tensor<420xf32>) -> tensor<1878063750xf32>
      %195 = stablehlo.reshape %194 : (tensor<1878063750xf32>) -> tensor<1251x1250x1201xf32>
      %c_83 = stablehlo.constant dense<4993> : tensor<i32>
      %196 = stablehlo.subtract %c_83, %134 : tensor<i32>
      %c_84 = stablehlo.constant dense<0> : tensor<i32>
      %c_85 = stablehlo.constant dense<1236> : tensor<i32>
      %197 = func.call @clip_132(%196, %c_84, %c_85) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %198 = stablehlo.add %134, %197 : tensor<i32>
      %199 = stablehlo.iota dim = 0 : tensor<15xi32>
      %200 = stablehlo.broadcast_in_dim %198, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %201 = stablehlo.add %200, %199 : tensor<15xi32>
      %c_86 = stablehlo.constant dense<4993> : tensor<i32>
      %202 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %203 = stablehlo.subtract %201, %202 : tensor<15xi32>
      %c_87 = stablehlo.constant dense<0> : tensor<i32>
      %204 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %205 = stablehlo.compare  GE, %203, %204,  SIGNED : (tensor<15xi32>, tensor<15xi32>) -> tensor<15xi1>
      %c_88 = stablehlo.constant dense<15> : tensor<i32>
      %206 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<15xi32>
      %207 = stablehlo.compare  LT, %203, %206,  SIGNED : (tensor<15xi32>, tensor<15xi32>) -> tensor<15xi1>
      %208 = stablehlo.and %205, %207 : tensor<15xi1>
      %209 = stablehlo.slice %arg192 [1:2, 0:15, 0:28, 0:1] : (tensor<2x15x28x1xf32>) -> tensor<1x15x28x1xf32>
      %210 = stablehlo.reshape %209 : (tensor<1x15x28x1xf32>) -> tensor<15x28x1xf32>
      %211 = func.call @_take_236(%210, %203) : (tensor<15x28x1xf32>, tensor<15xi32>) -> tensor<15x28x1xf32>
      %212 = stablehlo.reshape %208 : (tensor<15xi1>) -> tensor<15x1x1xi1>
      %c_89 = stablehlo.constant dense<0> : tensor<i32>
      %213 = stablehlo.compare  LT, %129, %c_89,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_90 = stablehlo.constant dense<1024> : tensor<i32>
      %214 = stablehlo.add %129, %c_90 : tensor<i32>
      %215 = stablehlo.select %213, %214, %129 : tensor<i1>, tensor<i32>
      %c_91 = stablehlo.constant dense<1> : tensor<i32>
      %216 = stablehlo.dynamic_slice %arg193, %c_91, %215, sizes = [1, 1] : (tensor<2x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
      %217 = stablehlo.reshape %216 : (tensor<1x1xf32>) -> tensor<f32>
      %218 = stablehlo.broadcast_in_dim %217, dims = [] : (tensor<f32>) -> tensor<15x28x1xf32>
      %219 = stablehlo.multiply %211, %218 : tensor<15x28x1xf32>
      %c_92 = stablehlo.constant dense<0> : tensor<i32>
      %220 = func.call @_where_240(%212, %219, %c_92) : (tensor<15x1x1xi1>, tensor<15x28x1xf32>, tensor<i32>) -> tensor<15x28x1xf32>
      %221 = stablehlo.iota dim = 0 : tensor<1xi32>
      %222 = stablehlo.reshape %221 : (tensor<1xi32>) -> tensor<1x1x1xi32>
      %c_93 = stablehlo.constant dense<358> : tensor<i32>
      %223 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %224 = stablehlo.add %223, %222 : tensor<1x1x1xi32>
      %c_94 = stablehlo.constant dense<1> : tensor<i32>
      %225 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %226 = stablehlo.multiply %224, %225 : tensor<1x1x1xi32>
      %c_95 = stablehlo.constant dense<0> : tensor<i32>
      %227 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %228 = stablehlo.add %227, %226 : tensor<1x1x1xi32>
      %229 = stablehlo.iota dim = 0 : tensor<28xi32>
      %230 = stablehlo.reshape %229 : (tensor<28xi32>) -> tensor<1x28x1xi32>
      %c_96 = stablehlo.constant dense<611> : tensor<i32>
      %231 = stablehlo.broadcast_in_dim %c_96, dims = [] : (tensor<i32>) -> tensor<1x28x1xi32>
      %232 = stablehlo.add %231, %230 : tensor<1x28x1xi32>
      %c_97 = stablehlo.constant dense<1201> : tensor<i32>
      %233 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<1x28x1xi32>
      %234 = stablehlo.multiply %232, %233 : tensor<1x28x1xi32>
      %235 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x28x1xi32>
      %236 = stablehlo.add %235, %234 : tensor<1x28x1xi32>
      %237 = stablehlo.iota dim = 0 : tensor<15xi32>
      %238 = stablehlo.reshape %237 : (tensor<15xi32>) -> tensor<15x1x1xi32>
      %239 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<i32>) -> tensor<15x1x1xi32>
      %240 = stablehlo.add %239, %238 : tensor<15x1x1xi32>
      %c_98 = stablehlo.constant dense<1501250> : tensor<i32>
      %241 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<15x1x1xi32>
      %242 = stablehlo.multiply %240, %241 : tensor<15x1x1xi32>
      %243 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2] : (tensor<1x28x1xi32>) -> tensor<15x28x1xi32>
      %244 = stablehlo.broadcast_in_dim %242, dims = [0, 1, 2] : (tensor<15x1x1xi32>) -> tensor<15x28x1xi32>
      %245 = stablehlo.add %243, %244 : tensor<15x28x1xi32>
      %246 = stablehlo.reshape %195 : (tensor<1251x1250x1201xf32>) -> tensor<1878063750xf32>
      %247 = stablehlo.reshape %245 : (tensor<15x28x1xi32>) -> tensor<420xi32>
      %248 = stablehlo.reshape %220 : (tensor<15x28x1xf32>) -> tensor<420xf32>
      %c_99 = stablehlo.constant dense<0> : tensor<i32>
      %249 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<420xi32>
      %250 = stablehlo.compare  LT, %247, %249,  SIGNED : (tensor<420xi32>, tensor<420xi32>) -> tensor<420xi1>
      %c_100 = stablehlo.constant dense<1878063750> : tensor<i32>
      %251 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<420xi32>
      %252 = stablehlo.add %247, %251 : tensor<420xi32>
      %253 = stablehlo.select %250, %252, %247 : tensor<420xi1>, tensor<420xi32>
      %254 = stablehlo.broadcast_in_dim %253, dims = [0] : (tensor<420xi32>) -> tensor<420x1xi32>
      %255 = "stablehlo.scatter"(%246, %254, %248) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
      ^bb0(%arg194: tensor<f32>, %arg195: tensor<f32>):
        %257 = stablehlo.add %arg194, %arg195 : tensor<f32>
        stablehlo.return %257 : tensor<f32>
      }) : (tensor<1878063750xf32>, tensor<420x1xi32>, tensor<420xf32>) -> tensor<1878063750xf32>
      %256 = stablehlo.reshape %255 : (tensor<1878063750xf32>) -> tensor<1251x1250x1201xf32>
      sdy.return %256 : tensor<1251x1250x1201xf32>
    } : (tensor<10008x1250x1201xf32>, tensor<i32>, tensor<2x15x28x1xf32>, tensor<2x1024xf32>) -> tensor<10008x1250x1201xf32>
    %9 = sdy.manual_computation(%7#2, %arg188, %arg61, %arg62) in_shardings=[<@mesh, [{"fdtd"}, {}, {}]>, <@mesh, []>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}]>] manual_axes={"fdtd"} (%arg190: tensor<1251x1251x1201xf32>, %arg191: tensor<i32>, %arg192: tensor<2x16x27x1xf32>, %arg193: tensor<2x1024xf32>) {
      %c_60 = stablehlo.constant dense<0> : tensor<i32>
      %c_61 = stablehlo.constant dense<1023> : tensor<i32>
      %129 = func.call @clip(%arg191, %c_60, %c_61) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %c_62 = stablehlo.constant dense<1> : tensor<ui32>
      %c_63 = stablehlo.constant dense<8> : tensor<ui32>
      %130 = stablehlo.partition_id : tensor<ui32>
      %131 = stablehlo.divide %130, %c_62 : tensor<ui32>
      %132 = stablehlo.remainder %131, %c_63 : tensor<ui32>
      %133 = stablehlo.convert %132 : (tensor<ui32>) -> tensor<i32>
      %c_64 = stablehlo.constant dense<1251> : tensor<i32>
      %134 = stablehlo.multiply %133, %c_64 : tensor<i32>
      %c_65 = stablehlo.constant dense<4992> : tensor<i32>
      %135 = stablehlo.subtract %c_65, %134 : tensor<i32>
      %c_66 = stablehlo.constant dense<0> : tensor<i32>
      %c_67 = stablehlo.constant dense<1235> : tensor<i32>
      %136 = func.call @clip_132(%135, %c_66, %c_67) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %137 = stablehlo.add %134, %136 : tensor<i32>
      %138 = stablehlo.iota dim = 0 : tensor<16xi32>
      %139 = stablehlo.broadcast_in_dim %137, dims = [] : (tensor<i32>) -> tensor<16xi32>
      %140 = stablehlo.add %139, %138 : tensor<16xi32>
      %c_68 = stablehlo.constant dense<4992> : tensor<i32>
      %141 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<16xi32>
      %142 = stablehlo.subtract %140, %141 : tensor<16xi32>
      %c_69 = stablehlo.constant dense<0> : tensor<i32>
      %143 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<16xi32>
      %144 = stablehlo.compare  GE, %142, %143,  SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
      %c_70 = stablehlo.constant dense<16> : tensor<i32>
      %145 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<16xi32>
      %146 = stablehlo.compare  LT, %142, %145,  SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
      %147 = stablehlo.and %144, %146 : tensor<16xi1>
      %148 = stablehlo.slice %arg192 [0:1, 0:16, 0:27, 0:1] : (tensor<2x16x27x1xf32>) -> tensor<1x16x27x1xf32>
      %149 = stablehlo.reshape %148 : (tensor<1x16x27x1xf32>) -> tensor<16x27x1xf32>
      %150 = func.call @_take_271(%149, %142) : (tensor<16x27x1xf32>, tensor<16xi32>) -> tensor<16x27x1xf32>
      %151 = stablehlo.reshape %147 : (tensor<16xi1>) -> tensor<16x1x1xi1>
      %c_71 = stablehlo.constant dense<0> : tensor<i32>
      %152 = stablehlo.compare  LT, %129, %c_71,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_72 = stablehlo.constant dense<1024> : tensor<i32>
      %153 = stablehlo.add %129, %c_72 : tensor<i32>
      %154 = stablehlo.select %152, %153, %129 : tensor<i1>, tensor<i32>
      %c_73 = stablehlo.constant dense<0> : tensor<i32>
      %155 = stablehlo.dynamic_slice %arg193, %c_73, %154, sizes = [1, 1] : (tensor<2x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
      %156 = stablehlo.reshape %155 : (tensor<1x1xf32>) -> tensor<f32>
      %157 = stablehlo.broadcast_in_dim %156, dims = [] : (tensor<f32>) -> tensor<16x27x1xf32>
      %158 = stablehlo.multiply %150, %157 : tensor<16x27x1xf32>
      %c_74 = stablehlo.constant dense<0> : tensor<i32>
      %159 = func.call @_where_277(%151, %158, %c_74) : (tensor<16x1x1xi1>, tensor<16x27x1xf32>, tensor<i32>) -> tensor<16x27x1xf32>
      %160 = stablehlo.iota dim = 0 : tensor<1xi32>
      %161 = stablehlo.reshape %160 : (tensor<1xi32>) -> tensor<1x1x1xi32>
      %c_75 = stablehlo.constant dense<358> : tensor<i32>
      %162 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %163 = stablehlo.add %162, %161 : tensor<1x1x1xi32>
      %c_76 = stablehlo.constant dense<1> : tensor<i32>
      %164 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %165 = stablehlo.multiply %163, %164 : tensor<1x1x1xi32>
      %c_77 = stablehlo.constant dense<0> : tensor<i32>
      %166 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %167 = stablehlo.add %166, %165 : tensor<1x1x1xi32>
      %168 = stablehlo.iota dim = 0 : tensor<27xi32>
      %169 = stablehlo.reshape %168 : (tensor<27xi32>) -> tensor<1x27x1xi32>
      %c_78 = stablehlo.constant dense<612> : tensor<i32>
      %170 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<1x27x1xi32>
      %171 = stablehlo.add %170, %169 : tensor<1x27x1xi32>
      %c_79 = stablehlo.constant dense<1201> : tensor<i32>
      %172 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<1x27x1xi32>
      %173 = stablehlo.multiply %171, %172 : tensor<1x27x1xi32>
      %174 = stablehlo.broadcast_in_dim %167, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x27x1xi32>
      %175 = stablehlo.add %174, %173 : tensor<1x27x1xi32>
      %176 = stablehlo.iota dim = 0 : tensor<16xi32>
      %177 = stablehlo.reshape %176 : (tensor<16xi32>) -> tensor<16x1x1xi32>
      %178 = stablehlo.broadcast_in_dim %136, dims = [] : (tensor<i32>) -> tensor<16x1x1xi32>
      %179 = stablehlo.add %178, %177 : tensor<16x1x1xi32>
      %c_80 = stablehlo.constant dense<1502451> : tensor<i32>
      %180 = stablehlo.broadcast_in_dim %c_80, dims = [] : (tensor<i32>) -> tensor<16x1x1xi32>
      %181 = stablehlo.multiply %179, %180 : tensor<16x1x1xi32>
      %182 = stablehlo.broadcast_in_dim %175, dims = [0, 1, 2] : (tensor<1x27x1xi32>) -> tensor<16x27x1xi32>
      %183 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2] : (tensor<16x1x1xi32>) -> tensor<16x27x1xi32>
      %184 = stablehlo.add %182, %183 : tensor<16x27x1xi32>
      %185 = stablehlo.reshape %arg190 : (tensor<1251x1251x1201xf32>) -> tensor<1879566201xf32>
      %186 = stablehlo.reshape %184 : (tensor<16x27x1xi32>) -> tensor<432xi32>
      %187 = stablehlo.reshape %159 : (tensor<16x27x1xf32>) -> tensor<432xf32>
      %c_81 = stablehlo.constant dense<0> : tensor<i32>
      %188 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<432xi32>
      %189 = stablehlo.compare  LT, %186, %188,  SIGNED : (tensor<432xi32>, tensor<432xi32>) -> tensor<432xi1>
      %c_82 = stablehlo.constant dense<1879566201> : tensor<i32>
      %190 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<432xi32>
      %191 = stablehlo.add %186, %190 : tensor<432xi32>
      %192 = stablehlo.select %189, %191, %186 : tensor<432xi1>, tensor<432xi32>
      %193 = stablehlo.broadcast_in_dim %192, dims = [0] : (tensor<432xi32>) -> tensor<432x1xi32>
      %194 = "stablehlo.scatter"(%185, %193, %187) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
      ^bb0(%arg194: tensor<f32>, %arg195: tensor<f32>):
        %257 = stablehlo.add %arg194, %arg195 : tensor<f32>
        stablehlo.return %257 : tensor<f32>
      }) : (tensor<1879566201xf32>, tensor<432x1xi32>, tensor<432xf32>) -> tensor<1879566201xf32>
      %195 = stablehlo.reshape %194 : (tensor<1879566201xf32>) -> tensor<1251x1251x1201xf32>
      %c_83 = stablehlo.constant dense<4992> : tensor<i32>
      %196 = stablehlo.subtract %c_83, %134 : tensor<i32>
      %c_84 = stablehlo.constant dense<0> : tensor<i32>
      %c_85 = stablehlo.constant dense<1235> : tensor<i32>
      %197 = func.call @clip_132(%196, %c_84, %c_85) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %198 = stablehlo.add %134, %197 : tensor<i32>
      %199 = stablehlo.iota dim = 0 : tensor<16xi32>
      %200 = stablehlo.broadcast_in_dim %198, dims = [] : (tensor<i32>) -> tensor<16xi32>
      %201 = stablehlo.add %200, %199 : tensor<16xi32>
      %c_86 = stablehlo.constant dense<4992> : tensor<i32>
      %202 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<16xi32>
      %203 = stablehlo.subtract %201, %202 : tensor<16xi32>
      %c_87 = stablehlo.constant dense<0> : tensor<i32>
      %204 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<16xi32>
      %205 = stablehlo.compare  GE, %203, %204,  SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
      %c_88 = stablehlo.constant dense<16> : tensor<i32>
      %206 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<16xi32>
      %207 = stablehlo.compare  LT, %203, %206,  SIGNED : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi1>
      %208 = stablehlo.and %205, %207 : tensor<16xi1>
      %209 = stablehlo.slice %arg192 [1:2, 0:16, 0:27, 0:1] : (tensor<2x16x27x1xf32>) -> tensor<1x16x27x1xf32>
      %210 = stablehlo.reshape %209 : (tensor<1x16x27x1xf32>) -> tensor<16x27x1xf32>
      %211 = func.call @_take_271(%210, %203) : (tensor<16x27x1xf32>, tensor<16xi32>) -> tensor<16x27x1xf32>
      %212 = stablehlo.reshape %208 : (tensor<16xi1>) -> tensor<16x1x1xi1>
      %c_89 = stablehlo.constant dense<0> : tensor<i32>
      %213 = stablehlo.compare  LT, %129, %c_89,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_90 = stablehlo.constant dense<1024> : tensor<i32>
      %214 = stablehlo.add %129, %c_90 : tensor<i32>
      %215 = stablehlo.select %213, %214, %129 : tensor<i1>, tensor<i32>
      %c_91 = stablehlo.constant dense<1> : tensor<i32>
      %216 = stablehlo.dynamic_slice %arg193, %c_91, %215, sizes = [1, 1] : (tensor<2x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
      %217 = stablehlo.reshape %216 : (tensor<1x1xf32>) -> tensor<f32>
      %218 = stablehlo.broadcast_in_dim %217, dims = [] : (tensor<f32>) -> tensor<16x27x1xf32>
      %219 = stablehlo.multiply %211, %218 : tensor<16x27x1xf32>
      %c_92 = stablehlo.constant dense<0> : tensor<i32>
      %220 = func.call @_where_277(%212, %219, %c_92) : (tensor<16x1x1xi1>, tensor<16x27x1xf32>, tensor<i32>) -> tensor<16x27x1xf32>
      %221 = stablehlo.iota dim = 0 : tensor<1xi32>
      %222 = stablehlo.reshape %221 : (tensor<1xi32>) -> tensor<1x1x1xi32>
      %c_93 = stablehlo.constant dense<358> : tensor<i32>
      %223 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %224 = stablehlo.add %223, %222 : tensor<1x1x1xi32>
      %c_94 = stablehlo.constant dense<1> : tensor<i32>
      %225 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %226 = stablehlo.multiply %224, %225 : tensor<1x1x1xi32>
      %c_95 = stablehlo.constant dense<0> : tensor<i32>
      %227 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<1x1x1xi32>
      %228 = stablehlo.add %227, %226 : tensor<1x1x1xi32>
      %229 = stablehlo.iota dim = 0 : tensor<27xi32>
      %230 = stablehlo.reshape %229 : (tensor<27xi32>) -> tensor<1x27x1xi32>
      %c_96 = stablehlo.constant dense<612> : tensor<i32>
      %231 = stablehlo.broadcast_in_dim %c_96, dims = [] : (tensor<i32>) -> tensor<1x27x1xi32>
      %232 = stablehlo.add %231, %230 : tensor<1x27x1xi32>
      %c_97 = stablehlo.constant dense<1201> : tensor<i32>
      %233 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<1x27x1xi32>
      %234 = stablehlo.multiply %232, %233 : tensor<1x27x1xi32>
      %235 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x27x1xi32>
      %236 = stablehlo.add %235, %234 : tensor<1x27x1xi32>
      %237 = stablehlo.iota dim = 0 : tensor<16xi32>
      %238 = stablehlo.reshape %237 : (tensor<16xi32>) -> tensor<16x1x1xi32>
      %239 = stablehlo.broadcast_in_dim %197, dims = [] : (tensor<i32>) -> tensor<16x1x1xi32>
      %240 = stablehlo.add %239, %238 : tensor<16x1x1xi32>
      %c_98 = stablehlo.constant dense<1502451> : tensor<i32>
      %241 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<16x1x1xi32>
      %242 = stablehlo.multiply %240, %241 : tensor<16x1x1xi32>
      %243 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2] : (tensor<1x27x1xi32>) -> tensor<16x27x1xi32>
      %244 = stablehlo.broadcast_in_dim %242, dims = [0, 1, 2] : (tensor<16x1x1xi32>) -> tensor<16x27x1xi32>
      %245 = stablehlo.add %243, %244 : tensor<16x27x1xi32>
      %246 = stablehlo.reshape %195 : (tensor<1251x1251x1201xf32>) -> tensor<1879566201xf32>
      %247 = stablehlo.reshape %245 : (tensor<16x27x1xi32>) -> tensor<432xi32>
      %248 = stablehlo.reshape %220 : (tensor<16x27x1xf32>) -> tensor<432xf32>
      %c_99 = stablehlo.constant dense<0> : tensor<i32>
      %249 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<432xi32>
      %250 = stablehlo.compare  LT, %247, %249,  SIGNED : (tensor<432xi32>, tensor<432xi32>) -> tensor<432xi1>
      %c_100 = stablehlo.constant dense<1879566201> : tensor<i32>
      %251 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<432xi32>
      %252 = stablehlo.add %247, %251 : tensor<432xi32>
      %253 = stablehlo.select %250, %252, %247 : tensor<432xi1>, tensor<432xi32>
      %254 = stablehlo.broadcast_in_dim %253, dims = [0] : (tensor<432xi32>) -> tensor<432x1xi32>
      %255 = "stablehlo.scatter"(%246, %254, %248) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
      ^bb0(%arg194: tensor<f32>, %arg195: tensor<f32>):
        %257 = stablehlo.add %arg194, %arg195 : tensor<f32>
        stablehlo.return %257 : tensor<f32>
      }) : (tensor<1879566201xf32>, tensor<432x1xi32>, tensor<432xf32>) -> tensor<1879566201xf32>
      %256 = stablehlo.reshape %255 : (tensor<1879566201xf32>) -> tensor<1251x1251x1201xf32>
      sdy.return %256 : tensor<1251x1251x1201xf32>
    } : (tensor<10008x1251x1201xf32>, tensor<i32>, tensor<2x16x27x1xf32>, tensor<2x1024xf32>) -> tensor<10008x1251x1201xf32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.add %arg188, %c_0 : tensor<i32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.maximum %c_1, %c_2 : tensor<i32>
    %12 = call @remainder(%10, %11) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %13 = stablehlo.compare  EQ, %12, %c_3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %14 = stablehlo.slice %arg183 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %15 = stablehlo.reshape %14 : (tensor<1xi32>) -> tensor<i32>
    %c_4 = stablehlo.constant dense<1> : tensor<i32>
    %16 = stablehlo.compare  LT, %15, %c_4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %17 = stablehlo.and %13, %16 : tensor<i1>
    %c_5 = stablehlo.constant dense<false> : tensor<i1>
    %18 = stablehlo.and %17, %c_5 : tensor<i1>
    %c_6 = stablehlo.constant dense<false> : tensor<i1>
    %19 = stablehlo.or %18, %c_6 : tensor<i1>
    %20 = stablehlo.convert %19 : (tensor<i1>) -> tensor<i32>
    %21 = "stablehlo.case"(%20) ({
      %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      stablehlo.return %cst_60 : tensor<f32>
    }, {
      %c_60 = stablehlo.constant dense<0> : tensor<i32>
      %129 = stablehlo.broadcast_in_dim %c_60, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %130 = stablehlo.compare  LT, %arg63, %129,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_61 = stablehlo.constant dense<10008> : tensor<i32>
      %131 = stablehlo.broadcast_in_dim %c_61, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %132 = stablehlo.add %arg63, %131 : tensor<200x1xi32>
      %133 = stablehlo.select %130, %132, %arg63 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_62 = stablehlo.constant dense<0> : tensor<i32>
      %134 = stablehlo.broadcast_in_dim %c_62, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %135 = stablehlo.compare  LT, %arg64, %134,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_63 = stablehlo.constant dense<1251> : tensor<i32>
      %136 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %137 = stablehlo.add %arg64, %136 : tensor<200x1xi32>
      %138 = stablehlo.select %135, %137, %arg64 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_64 = stablehlo.constant dense<0> : tensor<i32>
      %139 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %140 = stablehlo.compare  LT, %arg65, %139,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_65 = stablehlo.constant dense<1200> : tensor<i32>
      %141 = stablehlo.broadcast_in_dim %c_65, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %142 = stablehlo.add %arg65, %141 : tensor<200x1xi32>
      %143 = stablehlo.select %140, %142, %arg65 : tensor<200x1xi1>, tensor<200x1xi32>
      %144 = stablehlo.broadcast_in_dim %133, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %145 = stablehlo.broadcast_in_dim %138, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %146 = stablehlo.broadcast_in_dim %143, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %147 = stablehlo.concatenate %144, %145, %146, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %148 = "stablehlo.gather"(%7#0, %147) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1251x1200xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %149 = stablehlo.multiply %148, %arg66 : tensor<200x1xf32>
      %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %150 = stablehlo.reduce(%149 init: %cst_66) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_67 = stablehlo.constant dense<0> : tensor<i32>
      %151 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %152 = stablehlo.compare  LT, %arg67, %151,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_68 = stablehlo.constant dense<10008> : tensor<i32>
      %153 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %154 = stablehlo.add %arg67, %153 : tensor<200x1xi32>
      %155 = stablehlo.select %152, %154, %arg67 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_69 = stablehlo.constant dense<0> : tensor<i32>
      %156 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %157 = stablehlo.compare  LT, %arg68, %156,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_70 = stablehlo.constant dense<1250> : tensor<i32>
      %158 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %159 = stablehlo.add %arg68, %158 : tensor<200x1xi32>
      %160 = stablehlo.select %157, %159, %arg68 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_71 = stablehlo.constant dense<0> : tensor<i32>
      %161 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %162 = stablehlo.compare  LT, %arg69, %161,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_72 = stablehlo.constant dense<1201> : tensor<i32>
      %163 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %164 = stablehlo.add %arg69, %163 : tensor<200x1xi32>
      %165 = stablehlo.select %162, %164, %arg69 : tensor<200x1xi1>, tensor<200x1xi32>
      %166 = stablehlo.broadcast_in_dim %155, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %167 = stablehlo.broadcast_in_dim %160, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %168 = stablehlo.broadcast_in_dim %165, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %169 = stablehlo.concatenate %166, %167, %168, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %170 = "stablehlo.gather"(%8, %169) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1250x1201xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %171 = stablehlo.multiply %170, %arg70 : tensor<200x1xf32>
      %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %172 = stablehlo.reduce(%171 init: %cst_73) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_74 = stablehlo.constant dense<0> : tensor<i32>
      %173 = stablehlo.broadcast_in_dim %c_74, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %174 = stablehlo.compare  LT, %arg71, %173,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_75 = stablehlo.constant dense<10008> : tensor<i32>
      %175 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %176 = stablehlo.add %arg71, %175 : tensor<200x1xi32>
      %177 = stablehlo.select %174, %176, %arg71 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_76 = stablehlo.constant dense<0> : tensor<i32>
      %178 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %179 = stablehlo.compare  LT, %arg72, %178,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_77 = stablehlo.constant dense<1251> : tensor<i32>
      %180 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %181 = stablehlo.add %arg72, %180 : tensor<200x1xi32>
      %182 = stablehlo.select %179, %181, %arg72 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_78 = stablehlo.constant dense<0> : tensor<i32>
      %183 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %184 = stablehlo.compare  LT, %arg73, %183,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_79 = stablehlo.constant dense<1201> : tensor<i32>
      %185 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %186 = stablehlo.add %arg73, %185 : tensor<200x1xi32>
      %187 = stablehlo.select %184, %186, %arg73 : tensor<200x1xi1>, tensor<200x1xi32>
      %188 = stablehlo.broadcast_in_dim %177, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %189 = stablehlo.broadcast_in_dim %182, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %190 = stablehlo.broadcast_in_dim %187, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %191 = stablehlo.concatenate %188, %189, %190, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %192 = "stablehlo.gather"(%9, %191) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1251x1201xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %193 = stablehlo.multiply %192, %arg74 : tensor<200x1xf32>
      %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %194 = stablehlo.reduce(%193 init: %cst_80) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_81 = stablehlo.constant dense<0> : tensor<i32>
      %195 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %196 = stablehlo.compare  LT, %arg75, %195,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_82 = stablehlo.constant dense<10008> : tensor<i32>
      %197 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %198 = stablehlo.add %arg75, %197 : tensor<200x1xi32>
      %199 = stablehlo.select %196, %198, %arg75 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_83 = stablehlo.constant dense<0> : tensor<i32>
      %200 = stablehlo.broadcast_in_dim %c_83, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %201 = stablehlo.compare  LT, %arg76, %200,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_84 = stablehlo.constant dense<1250> : tensor<i32>
      %202 = stablehlo.broadcast_in_dim %c_84, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %203 = stablehlo.add %arg76, %202 : tensor<200x1xi32>
      %204 = stablehlo.select %201, %203, %arg76 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_85 = stablehlo.constant dense<0> : tensor<i32>
      %205 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %206 = stablehlo.compare  LT, %arg77, %205,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_86 = stablehlo.constant dense<1201> : tensor<i32>
      %207 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %208 = stablehlo.add %arg77, %207 : tensor<200x1xi32>
      %209 = stablehlo.select %206, %208, %arg77 : tensor<200x1xi1>, tensor<200x1xi32>
      %210 = stablehlo.broadcast_in_dim %199, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %211 = stablehlo.broadcast_in_dim %204, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %212 = stablehlo.broadcast_in_dim %209, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %213 = stablehlo.concatenate %210, %211, %212, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %214 = "stablehlo.gather"(%4#0, %213) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1250x1201xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %215 = stablehlo.multiply %214, %arg78 : tensor<200x1xf32>
      %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %216 = stablehlo.reduce(%215 init: %cst_87) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_88 = stablehlo.constant dense<0> : tensor<i32>
      %217 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %218 = stablehlo.compare  LT, %arg79, %217,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_89 = stablehlo.constant dense<10008> : tensor<i32>
      %219 = stablehlo.broadcast_in_dim %c_89, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %220 = stablehlo.add %arg79, %219 : tensor<200x1xi32>
      %221 = stablehlo.select %218, %220, %arg79 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_90 = stablehlo.constant dense<0> : tensor<i32>
      %222 = stablehlo.broadcast_in_dim %c_90, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %223 = stablehlo.compare  LT, %arg80, %222,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_91 = stablehlo.constant dense<1251> : tensor<i32>
      %224 = stablehlo.broadcast_in_dim %c_91, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %225 = stablehlo.add %arg80, %224 : tensor<200x1xi32>
      %226 = stablehlo.select %223, %225, %arg80 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_92 = stablehlo.constant dense<0> : tensor<i32>
      %227 = stablehlo.broadcast_in_dim %c_92, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %228 = stablehlo.compare  LT, %arg81, %227,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_93 = stablehlo.constant dense<1200> : tensor<i32>
      %229 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %230 = stablehlo.add %arg81, %229 : tensor<200x1xi32>
      %231 = stablehlo.select %228, %230, %arg81 : tensor<200x1xi1>, tensor<200x1xi32>
      %232 = stablehlo.broadcast_in_dim %221, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %233 = stablehlo.broadcast_in_dim %226, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %234 = stablehlo.broadcast_in_dim %231, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %235 = stablehlo.concatenate %232, %233, %234, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %236 = "stablehlo.gather"(%5, %235) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1251x1200xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %237 = stablehlo.multiply %236, %arg82 : tensor<200x1xf32>
      %cst_94 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %238 = stablehlo.reduce(%237 init: %cst_94) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_95 = stablehlo.constant dense<0> : tensor<i32>
      %239 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %240 = stablehlo.compare  LT, %arg83, %239,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_96 = stablehlo.constant dense<10008> : tensor<i32>
      %241 = stablehlo.broadcast_in_dim %c_96, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %242 = stablehlo.add %arg83, %241 : tensor<200x1xi32>
      %243 = stablehlo.select %240, %242, %arg83 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_97 = stablehlo.constant dense<0> : tensor<i32>
      %244 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %245 = stablehlo.compare  LT, %arg84, %244,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_98 = stablehlo.constant dense<1250> : tensor<i32>
      %246 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %247 = stablehlo.add %arg84, %246 : tensor<200x1xi32>
      %248 = stablehlo.select %245, %247, %arg84 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_99 = stablehlo.constant dense<0> : tensor<i32>
      %249 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %250 = stablehlo.compare  LT, %arg85, %249,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_100 = stablehlo.constant dense<1200> : tensor<i32>
      %251 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %252 = stablehlo.add %arg85, %251 : tensor<200x1xi32>
      %253 = stablehlo.select %250, %252, %arg85 : tensor<200x1xi1>, tensor<200x1xi32>
      %254 = stablehlo.broadcast_in_dim %243, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %255 = stablehlo.broadcast_in_dim %248, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %256 = stablehlo.broadcast_in_dim %253, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %257 = stablehlo.concatenate %254, %255, %256, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %258 = "stablehlo.gather"(%6, %257) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1250x1200xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %259 = stablehlo.multiply %258, %arg86 : tensor<200x1xf32>
      %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %260 = stablehlo.reduce(%259 init: %cst_101) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %261 = stablehlo.broadcast_in_dim %150, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %262 = stablehlo.broadcast_in_dim %172, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %263 = stablehlo.broadcast_in_dim %194, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %264 = stablehlo.broadcast_in_dim %216, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %265 = stablehlo.broadcast_in_dim %238, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %266 = stablehlo.broadcast_in_dim %260, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %267 = stablehlo.concatenate %261, %262, %263, %264, %265, %266, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %268 = stablehlo.slice %267 [1:2, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %269 = stablehlo.reshape %268 : (tensor<1x200xf32>) -> tensor<200xf32>
      %270 = stablehlo.slice %267 [2:3, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %271 = stablehlo.reshape %270 : (tensor<1x200xf32>) -> tensor<200xf32>
      %272 = stablehlo.slice %267 [4:5, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %273 = stablehlo.reshape %272 : (tensor<1x200xf32>) -> tensor<200xf32>
      %274 = stablehlo.slice %267 [5:6, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %275 = stablehlo.reshape %274 : (tensor<1x200xf32>) -> tensor<200xf32>
      %276 = stablehlo.multiply %269, %275 : tensor<200xf32>
      %277 = stablehlo.multiply %271, %273 : tensor<200xf32>
      %278 = stablehlo.subtract %276, %277 : tensor<200xf32>
      %cst_102 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %280 = stablehlo.multiply %278, %279 : tensor<200xf32>
      %cst_103 = stablehlo.constant dense<7.485380e-15> : tensor<f32>
      %281 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %282 = stablehlo.multiply %280, %281 : tensor<200xf32>
      %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%282 init: %cst_104) applies stablehlo.add across dimensions = [0] : (tensor<200xf32>, tensor<f32>) -> tensor<f32>
      stablehlo.return %283 : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = call @_where_332(%18, %21, %cst) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %23 = stablehlo.slice %arg183 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %24 = stablehlo.reshape %23 : (tensor<1xi32>) -> tensor<i32>
    %c_7 = stablehlo.constant dense<0> : tensor<i32>
    %25 = stablehlo.minimum %24, %c_7 : tensor<i32>
    %c_8 = stablehlo.constant dense<0> : tensor<i32>
    %26 = stablehlo.compare  LT, %25, %c_8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_9 = stablehlo.constant dense<1> : tensor<i32>
    %27 = stablehlo.add %25, %c_9 : tensor<i32>
    %28 = stablehlo.select %26, %27, %25 : tensor<i1>, tensor<i32>
    %c_10 = stablehlo.constant dense<0> : tensor<i32>
    %29 = stablehlo.dynamic_slice %arg181, %c_10, %28, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %30 = stablehlo.reshape %29 : (tensor<1x1xf32>) -> tensor<f32>
    %c_11 = stablehlo.constant dense<0> : tensor<i32>
    %31 = stablehlo.compare  LT, %25, %c_11,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_12 = stablehlo.constant dense<1> : tensor<i32>
    %32 = stablehlo.add %25, %c_12 : tensor<i32>
    %33 = stablehlo.select %31, %32, %25 : tensor<i1>, tensor<i32>
    %c_13 = stablehlo.constant dense<0> : tensor<i32>
    %34 = stablehlo.dynamic_slice %arg182, %c_13, %33, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %35 = stablehlo.reshape %34 : (tensor<1x1xf32>) -> tensor<f32>
    %36 = call @_where_337(%18, %22, %30) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_14 = stablehlo.constant dense<0> : tensor<i32>
    %37 = stablehlo.compare  LT, %25, %c_14,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_15 = stablehlo.constant dense<1> : tensor<i32>
    %38 = stablehlo.add %25, %c_15 : tensor<i32>
    %39 = stablehlo.select %37, %38, %25 : tensor<i1>, tensor<i32>
    %c_16 = stablehlo.constant dense<0> : tensor<i32>
    %40 = stablehlo.broadcast_in_dim %c_16, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %41 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %42 = stablehlo.concatenate %40, %41, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %43 = "stablehlo.scatter"(%arg181, %42, %36) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg190: tensor<f32>, %arg191: tensor<f32>):
      stablehlo.return %arg191 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %44 = call @_where_337(%18, %3, %35) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_17 = stablehlo.constant dense<0> : tensor<i32>
    %45 = stablehlo.compare  LT, %25, %c_17,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_18 = stablehlo.constant dense<1> : tensor<i32>
    %46 = stablehlo.add %25, %c_18 : tensor<i32>
    %47 = stablehlo.select %45, %46, %25 : tensor<i1>, tensor<i32>
    %c_19 = stablehlo.constant dense<0> : tensor<i32>
    %48 = stablehlo.broadcast_in_dim %c_19, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %49 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %50 = stablehlo.concatenate %48, %49, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %51 = "stablehlo.scatter"(%arg182, %50, %44) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg190: tensor<f32>, %arg191: tensor<f32>):
      stablehlo.return %arg191 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %52 = stablehlo.slice %arg183 [0:1] : (tensor<2xi32>) -> tensor<1xi32>
    %53 = stablehlo.reshape %52 : (tensor<1xi32>) -> tensor<i32>
    %c_20 = stablehlo.constant dense<1> : tensor<i32>
    %c_21 = stablehlo.constant dense<0> : tensor<i32>
    %54 = call @_where_342(%18, %c_20, %c_21) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %55 = stablehlo.convert %54 : tensor<i32>
    %56 = stablehlo.add %53, %55 : tensor<i32>
    %c_22 = stablehlo.constant dense<0> : tensor<i32>
    %57 = stablehlo.broadcast_in_dim %c_22, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %58 = "stablehlo.scatter"(%arg183, %57, %56) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg190: tensor<i32>, %arg191: tensor<i32>):
      stablehlo.return %arg191 : tensor<i32>
    }) : (tensor<2xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %59 = stablehlo.compare  GE, %3, %cst_23,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %c_24 = stablehlo.constant dense<true> : tensor<i1>
    %60 = stablehlo.and %c_24, %59 : tensor<i1>
    %cst_25 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %61 = stablehlo.compare  LE, %3, %cst_25,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %62 = stablehlo.and %60, %61 : tensor<i1>
    %c_26 = stablehlo.constant dense<1> : tensor<i32>
    %c_27 = stablehlo.constant dense<1> : tensor<i32>
    %63 = stablehlo.maximum %c_26, %c_27 : tensor<i32>
    %64 = call @remainder(%arg188, %63) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_28 = stablehlo.constant dense<0> : tensor<i32>
    %65 = stablehlo.compare  EQ, %64, %c_28,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %66 = stablehlo.and %62, %65 : tensor<i1>
    %67 = stablehlo.convert %66 : (tensor<i1>) -> tensor<i32>
    %68:3 = "stablehlo.case"(%67) ({
      stablehlo.return %arg184, %arg185, %arg186 : tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>
    }, {
      %129:3 = sdy.manual_computation(%arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %7#0, %8, %9, %4#0, %5, %6, %arg184, %arg185, %arg186, %3, %arg0) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}]>, <@mesh, [{"fdtd"}, {}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"fdtd"}, {}]>, <@mesh, [{"fdtd"}, {}]>, <@mesh, [{}]>] manual_axes={"fdtd"} (%arg190: tensor<200x8xi32>, %arg191: tensor<200x8xi32>, %arg192: tensor<200x8xi32>, %arg193: tensor<200x8xf32>, %arg194: tensor<200x2xi32>, %arg195: tensor<200x2xi32>, %arg196: tensor<200x2xi32>, %arg197: tensor<200x2xf32>, %arg198: tensor<200x2xi32>, %arg199: tensor<200x2xi32>, %arg200: tensor<200x2xi32>, %arg201: tensor<200x2xf32>, %arg202: tensor<200x1xi32>, %arg203: tensor<200x1xi32>, %arg204: tensor<200x1xi32>, %arg205: tensor<200x1xf32>, %arg206: tensor<200x4xi32>, %arg207: tensor<200x4xi32>, %arg208: tensor<200x4xi32>, %arg209: tensor<200x4xf32>, %arg210: tensor<200x4xi32>, %arg211: tensor<200x4xi32>, %arg212: tensor<200x4xi32>, %arg213: tensor<200x4xf32>, %arg214: tensor<101xf32>, %arg215: tensor<6xf32>, %arg216: tensor<1251x1251x1200xf32>, %arg217: tensor<1251x1250x1201xf32>, %arg218: tensor<1251x1251x1201xf32>, %arg219: tensor<1251x1250x1201xf32>, %arg220: tensor<1251x1251x1200xf32>, %arg221: tensor<1251x1250x1200xf32>, %arg222: tensor<1x242400xf32>, %arg223: tensor<1x242400xf32>, %arg224: tensor<202xf32>, %arg225: tensor<f32>, %arg226: tensor<f32>) {
        %c_60 = stablehlo.constant dense<1> : tensor<ui32>
        %c_61 = stablehlo.constant dense<8> : tensor<ui32>
        %130 = stablehlo.partition_id : tensor<ui32>
        %131 = stablehlo.divide %130, %c_60 : tensor<ui32>
        %132 = stablehlo.remainder %131, %c_61 : tensor<ui32>
        %133 = stablehlo.convert %132 : (tensor<ui32>) -> tensor<i32>
        %c_62 = stablehlo.constant dense<1251> : tensor<i32>
        %134 = stablehlo.multiply %133, %c_62 : tensor<i32>
        %135 = stablehlo.broadcast_in_dim %134, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %136 = stablehlo.subtract %arg190, %135 : tensor<200x8xi32>
        %c_63 = stablehlo.constant dense<0> : tensor<i32>
        %137 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %138 = stablehlo.compare  GE, %136, %137,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
        %c_64 = stablehlo.constant dense<1251> : tensor<i32>
        %139 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %140 = stablehlo.compare  LT, %136, %139,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
        %141 = stablehlo.and %138, %140 : tensor<200x8xi1>
        %c_65 = stablehlo.constant dense<0> : tensor<i32>
        %c_66 = stablehlo.constant dense<1250> : tensor<i32>
        %142 = func.call @clip_356(%136, %c_65, %c_66) : (tensor<200x8xi32>, tensor<i32>, tensor<i32>) -> tensor<200x8xi32>
        %c_67 = stablehlo.constant dense<0> : tensor<i32>
        %143 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %144 = stablehlo.compare  LT, %142, %143,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
        %c_68 = stablehlo.constant dense<1251> : tensor<i32>
        %145 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %146 = stablehlo.add %142, %145 : tensor<200x8xi32>
        %147 = stablehlo.select %144, %146, %142 : tensor<200x8xi1>, tensor<200x8xi32>
        %c_69 = stablehlo.constant dense<0> : tensor<i32>
        %148 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %149 = stablehlo.compare  LT, %arg191, %148,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
        %c_70 = stablehlo.constant dense<1251> : tensor<i32>
        %150 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %151 = stablehlo.add %arg191, %150 : tensor<200x8xi32>
        %152 = stablehlo.select %149, %151, %arg191 : tensor<200x8xi1>, tensor<200x8xi32>
        %c_71 = stablehlo.constant dense<0> : tensor<i32>
        %153 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %154 = stablehlo.compare  LT, %arg192, %153,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
        %c_72 = stablehlo.constant dense<1200> : tensor<i32>
        %155 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %156 = stablehlo.add %arg192, %155 : tensor<200x8xi32>
        %157 = stablehlo.select %154, %156, %arg192 : tensor<200x8xi1>, tensor<200x8xi32>
        %158 = stablehlo.broadcast_in_dim %147, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
        %159 = stablehlo.broadcast_in_dim %152, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
        %160 = stablehlo.broadcast_in_dim %157, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
        %161 = stablehlo.concatenate %158, %159, %160, dim = 2 : (tensor<200x8x1xi32>, tensor<200x8x1xi32>, tensor<200x8x1xi32>) -> tensor<200x8x3xi32>
        %162 = "stablehlo.gather"(%arg216, %161) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1251x1200xf32>, tensor<200x8x3xi32>) -> tensor<200x8xf32>
        %c_73 = stablehlo.constant dense<0> : tensor<i32>
        %163 = func.call @_where_369(%141, %162, %c_73) : (tensor<200x8xi1>, tensor<200x8xf32>, tensor<i32>) -> tensor<200x8xf32>
        %164 = stablehlo.multiply %163, %arg193 : tensor<200x8xf32>
        %cst_74 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %165 = stablehlo.reduce(%164 init: %cst_74) applies stablehlo.add across dimensions = [1] : (tensor<200x8xf32>, tensor<f32>) -> tensor<200xf32>
        %c_75 = stablehlo.constant dense<1> : tensor<ui32>
        %c_76 = stablehlo.constant dense<8> : tensor<ui32>
        %166 = stablehlo.partition_id : tensor<ui32>
        %167 = stablehlo.divide %166, %c_75 : tensor<ui32>
        %168 = stablehlo.remainder %167, %c_76 : tensor<ui32>
        %169 = stablehlo.convert %168 : (tensor<ui32>) -> tensor<i32>
        %c_77 = stablehlo.constant dense<1251> : tensor<i32>
        %170 = stablehlo.multiply %169, %c_77 : tensor<i32>
        %171 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %172 = stablehlo.subtract %arg194, %171 : tensor<200x2xi32>
        %c_78 = stablehlo.constant dense<0> : tensor<i32>
        %173 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %174 = stablehlo.compare  GE, %172, %173,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_79 = stablehlo.constant dense<1251> : tensor<i32>
        %175 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %176 = stablehlo.compare  LT, %172, %175,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %177 = stablehlo.and %174, %176 : tensor<200x2xi1>
        %c_80 = stablehlo.constant dense<0> : tensor<i32>
        %c_81 = stablehlo.constant dense<1250> : tensor<i32>
        %178 = func.call @clip_381(%172, %c_80, %c_81) : (tensor<200x2xi32>, tensor<i32>, tensor<i32>) -> tensor<200x2xi32>
        %c_82 = stablehlo.constant dense<0> : tensor<i32>
        %179 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %180 = stablehlo.compare  LT, %178, %179,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_83 = stablehlo.constant dense<1251> : tensor<i32>
        %181 = stablehlo.broadcast_in_dim %c_83, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %182 = stablehlo.add %178, %181 : tensor<200x2xi32>
        %183 = stablehlo.select %180, %182, %178 : tensor<200x2xi1>, tensor<200x2xi32>
        %c_84 = stablehlo.constant dense<0> : tensor<i32>
        %184 = stablehlo.broadcast_in_dim %c_84, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %185 = stablehlo.compare  LT, %arg195, %184,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_85 = stablehlo.constant dense<1250> : tensor<i32>
        %186 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %187 = stablehlo.add %arg195, %186 : tensor<200x2xi32>
        %188 = stablehlo.select %185, %187, %arg195 : tensor<200x2xi1>, tensor<200x2xi32>
        %c_86 = stablehlo.constant dense<0> : tensor<i32>
        %189 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %190 = stablehlo.compare  LT, %arg196, %189,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_87 = stablehlo.constant dense<1201> : tensor<i32>
        %191 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %192 = stablehlo.add %arg196, %191 : tensor<200x2xi32>
        %193 = stablehlo.select %190, %192, %arg196 : tensor<200x2xi1>, tensor<200x2xi32>
        %194 = stablehlo.broadcast_in_dim %183, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %195 = stablehlo.broadcast_in_dim %188, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %196 = stablehlo.broadcast_in_dim %193, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %197 = stablehlo.concatenate %194, %195, %196, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
        %198 = "stablehlo.gather"(%arg217, %197) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1250x1201xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
        %c_88 = stablehlo.constant dense<0> : tensor<i32>
        %199 = func.call @_where_394(%177, %198, %c_88) : (tensor<200x2xi1>, tensor<200x2xf32>, tensor<i32>) -> tensor<200x2xf32>
        %200 = stablehlo.multiply %199, %arg197 : tensor<200x2xf32>
        %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %201 = stablehlo.reduce(%200 init: %cst_89) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
        %c_90 = stablehlo.constant dense<1> : tensor<ui32>
        %c_91 = stablehlo.constant dense<8> : tensor<ui32>
        %202 = stablehlo.partition_id : tensor<ui32>
        %203 = stablehlo.divide %202, %c_90 : tensor<ui32>
        %204 = stablehlo.remainder %203, %c_91 : tensor<ui32>
        %205 = stablehlo.convert %204 : (tensor<ui32>) -> tensor<i32>
        %c_92 = stablehlo.constant dense<1251> : tensor<i32>
        %206 = stablehlo.multiply %205, %c_92 : tensor<i32>
        %207 = stablehlo.broadcast_in_dim %206, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %208 = stablehlo.subtract %arg198, %207 : tensor<200x2xi32>
        %c_93 = stablehlo.constant dense<0> : tensor<i32>
        %209 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %210 = stablehlo.compare  GE, %208, %209,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_94 = stablehlo.constant dense<1251> : tensor<i32>
        %211 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %212 = stablehlo.compare  LT, %208, %211,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %213 = stablehlo.and %210, %212 : tensor<200x2xi1>
        %c_95 = stablehlo.constant dense<0> : tensor<i32>
        %c_96 = stablehlo.constant dense<1250> : tensor<i32>
        %214 = func.call @clip_381(%208, %c_95, %c_96) : (tensor<200x2xi32>, tensor<i32>, tensor<i32>) -> tensor<200x2xi32>
        %c_97 = stablehlo.constant dense<0> : tensor<i32>
        %215 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %216 = stablehlo.compare  LT, %214, %215,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_98 = stablehlo.constant dense<1251> : tensor<i32>
        %217 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %218 = stablehlo.add %214, %217 : tensor<200x2xi32>
        %219 = stablehlo.select %216, %218, %214 : tensor<200x2xi1>, tensor<200x2xi32>
        %c_99 = stablehlo.constant dense<0> : tensor<i32>
        %220 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %221 = stablehlo.compare  LT, %arg199, %220,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_100 = stablehlo.constant dense<1251> : tensor<i32>
        %222 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %223 = stablehlo.add %arg199, %222 : tensor<200x2xi32>
        %224 = stablehlo.select %221, %223, %arg199 : tensor<200x2xi1>, tensor<200x2xi32>
        %c_101 = stablehlo.constant dense<0> : tensor<i32>
        %225 = stablehlo.broadcast_in_dim %c_101, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %226 = stablehlo.compare  LT, %arg200, %225,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_102 = stablehlo.constant dense<1201> : tensor<i32>
        %227 = stablehlo.broadcast_in_dim %c_102, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %228 = stablehlo.add %arg200, %227 : tensor<200x2xi32>
        %229 = stablehlo.select %226, %228, %arg200 : tensor<200x2xi1>, tensor<200x2xi32>
        %230 = stablehlo.broadcast_in_dim %219, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %231 = stablehlo.broadcast_in_dim %224, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %232 = stablehlo.broadcast_in_dim %229, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %233 = stablehlo.concatenate %230, %231, %232, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
        %234 = "stablehlo.gather"(%arg218, %233) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1251x1201xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
        %c_103 = stablehlo.constant dense<0> : tensor<i32>
        %235 = func.call @_where_394(%213, %234, %c_103) : (tensor<200x2xi1>, tensor<200x2xf32>, tensor<i32>) -> tensor<200x2xf32>
        %236 = stablehlo.multiply %235, %arg201 : tensor<200x2xf32>
        %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %237 = stablehlo.reduce(%236 init: %cst_104) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
        %c_105 = stablehlo.constant dense<1> : tensor<ui32>
        %c_106 = stablehlo.constant dense<8> : tensor<ui32>
        %238 = stablehlo.partition_id : tensor<ui32>
        %239 = stablehlo.divide %238, %c_105 : tensor<ui32>
        %240 = stablehlo.remainder %239, %c_106 : tensor<ui32>
        %241 = stablehlo.convert %240 : (tensor<ui32>) -> tensor<i32>
        %c_107 = stablehlo.constant dense<1251> : tensor<i32>
        %242 = stablehlo.multiply %241, %c_107 : tensor<i32>
        %243 = stablehlo.broadcast_in_dim %242, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %244 = stablehlo.subtract %arg202, %243 : tensor<200x1xi32>
        %c_108 = stablehlo.constant dense<0> : tensor<i32>
        %245 = stablehlo.broadcast_in_dim %c_108, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %246 = stablehlo.compare  GE, %244, %245,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
        %c_109 = stablehlo.constant dense<1251> : tensor<i32>
        %247 = stablehlo.broadcast_in_dim %c_109, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %248 = stablehlo.compare  LT, %244, %247,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
        %249 = stablehlo.and %246, %248 : tensor<200x1xi1>
        %c_110 = stablehlo.constant dense<0> : tensor<i32>
        %c_111 = stablehlo.constant dense<1250> : tensor<i32>
        %250 = func.call @clip_407(%244, %c_110, %c_111) : (tensor<200x1xi32>, tensor<i32>, tensor<i32>) -> tensor<200x1xi32>
        %c_112 = stablehlo.constant dense<0> : tensor<i32>
        %251 = stablehlo.broadcast_in_dim %c_112, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %252 = stablehlo.compare  LT, %250, %251,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
        %c_113 = stablehlo.constant dense<1251> : tensor<i32>
        %253 = stablehlo.broadcast_in_dim %c_113, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %254 = stablehlo.add %250, %253 : tensor<200x1xi32>
        %255 = stablehlo.select %252, %254, %250 : tensor<200x1xi1>, tensor<200x1xi32>
        %c_114 = stablehlo.constant dense<0> : tensor<i32>
        %256 = stablehlo.broadcast_in_dim %c_114, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %257 = stablehlo.compare  LT, %arg203, %256,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
        %c_115 = stablehlo.constant dense<1250> : tensor<i32>
        %258 = stablehlo.broadcast_in_dim %c_115, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %259 = stablehlo.add %arg203, %258 : tensor<200x1xi32>
        %260 = stablehlo.select %257, %259, %arg203 : tensor<200x1xi1>, tensor<200x1xi32>
        %c_116 = stablehlo.constant dense<0> : tensor<i32>
        %261 = stablehlo.broadcast_in_dim %c_116, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %262 = stablehlo.compare  LT, %arg204, %261,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
        %c_117 = stablehlo.constant dense<1201> : tensor<i32>
        %263 = stablehlo.broadcast_in_dim %c_117, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %264 = stablehlo.add %arg204, %263 : tensor<200x1xi32>
        %265 = stablehlo.select %262, %264, %arg204 : tensor<200x1xi1>, tensor<200x1xi32>
        %266 = stablehlo.broadcast_in_dim %255, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
        %267 = stablehlo.broadcast_in_dim %260, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
        %268 = stablehlo.broadcast_in_dim %265, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
        %269 = stablehlo.concatenate %266, %267, %268, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
        %270 = "stablehlo.gather"(%arg219, %269) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1250x1201xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
        %c_118 = stablehlo.constant dense<0> : tensor<i32>
        %271 = func.call @_where_420(%249, %270, %c_118) : (tensor<200x1xi1>, tensor<200x1xf32>, tensor<i32>) -> tensor<200x1xf32>
        %272 = stablehlo.multiply %271, %arg205 : tensor<200x1xf32>
        %cst_119 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %273 = stablehlo.reduce(%272 init: %cst_119) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
        %c_120 = stablehlo.constant dense<1> : tensor<ui32>
        %c_121 = stablehlo.constant dense<8> : tensor<ui32>
        %274 = stablehlo.partition_id : tensor<ui32>
        %275 = stablehlo.divide %274, %c_120 : tensor<ui32>
        %276 = stablehlo.remainder %275, %c_121 : tensor<ui32>
        %277 = stablehlo.convert %276 : (tensor<ui32>) -> tensor<i32>
        %c_122 = stablehlo.constant dense<1251> : tensor<i32>
        %278 = stablehlo.multiply %277, %c_122 : tensor<i32>
        %279 = stablehlo.broadcast_in_dim %278, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %280 = stablehlo.subtract %arg206, %279 : tensor<200x4xi32>
        %c_123 = stablehlo.constant dense<0> : tensor<i32>
        %281 = stablehlo.broadcast_in_dim %c_123, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %282 = stablehlo.compare  GE, %280, %281,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_124 = stablehlo.constant dense<1251> : tensor<i32>
        %283 = stablehlo.broadcast_in_dim %c_124, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %284 = stablehlo.compare  LT, %280, %283,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %285 = stablehlo.and %282, %284 : tensor<200x4xi1>
        %c_125 = stablehlo.constant dense<0> : tensor<i32>
        %c_126 = stablehlo.constant dense<1250> : tensor<i32>
        %286 = func.call @clip_432(%280, %c_125, %c_126) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
        %c_127 = stablehlo.constant dense<0> : tensor<i32>
        %287 = stablehlo.broadcast_in_dim %c_127, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %288 = stablehlo.compare  LT, %286, %287,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_128 = stablehlo.constant dense<1251> : tensor<i32>
        %289 = stablehlo.broadcast_in_dim %c_128, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %290 = stablehlo.add %286, %289 : tensor<200x4xi32>
        %291 = stablehlo.select %288, %290, %286 : tensor<200x4xi1>, tensor<200x4xi32>
        %c_129 = stablehlo.constant dense<0> : tensor<i32>
        %292 = stablehlo.broadcast_in_dim %c_129, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %293 = stablehlo.compare  LT, %arg207, %292,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_130 = stablehlo.constant dense<1251> : tensor<i32>
        %294 = stablehlo.broadcast_in_dim %c_130, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %295 = stablehlo.add %arg207, %294 : tensor<200x4xi32>
        %296 = stablehlo.select %293, %295, %arg207 : tensor<200x4xi1>, tensor<200x4xi32>
        %c_131 = stablehlo.constant dense<0> : tensor<i32>
        %297 = stablehlo.broadcast_in_dim %c_131, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %298 = stablehlo.compare  LT, %arg208, %297,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_132 = stablehlo.constant dense<1200> : tensor<i32>
        %299 = stablehlo.broadcast_in_dim %c_132, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %300 = stablehlo.add %arg208, %299 : tensor<200x4xi32>
        %301 = stablehlo.select %298, %300, %arg208 : tensor<200x4xi1>, tensor<200x4xi32>
        %302 = stablehlo.broadcast_in_dim %291, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %303 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %304 = stablehlo.broadcast_in_dim %301, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %305 = stablehlo.concatenate %302, %303, %304, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
        %306 = "stablehlo.gather"(%arg220, %305) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1251x1200xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
        %c_133 = stablehlo.constant dense<0> : tensor<i32>
        %307 = func.call @_where_445(%285, %306, %c_133) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
        %308 = stablehlo.multiply %307, %arg209 : tensor<200x4xf32>
        %cst_134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %309 = stablehlo.reduce(%308 init: %cst_134) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
        %c_135 = stablehlo.constant dense<1> : tensor<ui32>
        %c_136 = stablehlo.constant dense<8> : tensor<ui32>
        %310 = stablehlo.partition_id : tensor<ui32>
        %311 = stablehlo.divide %310, %c_135 : tensor<ui32>
        %312 = stablehlo.remainder %311, %c_136 : tensor<ui32>
        %313 = stablehlo.convert %312 : (tensor<ui32>) -> tensor<i32>
        %c_137 = stablehlo.constant dense<1251> : tensor<i32>
        %314 = stablehlo.multiply %313, %c_137 : tensor<i32>
        %315 = stablehlo.broadcast_in_dim %314, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %316 = stablehlo.subtract %arg210, %315 : tensor<200x4xi32>
        %c_138 = stablehlo.constant dense<0> : tensor<i32>
        %317 = stablehlo.broadcast_in_dim %c_138, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %318 = stablehlo.compare  GE, %316, %317,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_139 = stablehlo.constant dense<1251> : tensor<i32>
        %319 = stablehlo.broadcast_in_dim %c_139, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %320 = stablehlo.compare  LT, %316, %319,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %321 = stablehlo.and %318, %320 : tensor<200x4xi1>
        %c_140 = stablehlo.constant dense<0> : tensor<i32>
        %c_141 = stablehlo.constant dense<1250> : tensor<i32>
        %322 = func.call @clip_432(%316, %c_140, %c_141) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
        %c_142 = stablehlo.constant dense<0> : tensor<i32>
        %323 = stablehlo.broadcast_in_dim %c_142, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %324 = stablehlo.compare  LT, %322, %323,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_143 = stablehlo.constant dense<1251> : tensor<i32>
        %325 = stablehlo.broadcast_in_dim %c_143, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %326 = stablehlo.add %322, %325 : tensor<200x4xi32>
        %327 = stablehlo.select %324, %326, %322 : tensor<200x4xi1>, tensor<200x4xi32>
        %c_144 = stablehlo.constant dense<0> : tensor<i32>
        %328 = stablehlo.broadcast_in_dim %c_144, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %329 = stablehlo.compare  LT, %arg211, %328,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_145 = stablehlo.constant dense<1250> : tensor<i32>
        %330 = stablehlo.broadcast_in_dim %c_145, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %331 = stablehlo.add %arg211, %330 : tensor<200x4xi32>
        %332 = stablehlo.select %329, %331, %arg211 : tensor<200x4xi1>, tensor<200x4xi32>
        %c_146 = stablehlo.constant dense<0> : tensor<i32>
        %333 = stablehlo.broadcast_in_dim %c_146, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %334 = stablehlo.compare  LT, %arg212, %333,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_147 = stablehlo.constant dense<1200> : tensor<i32>
        %335 = stablehlo.broadcast_in_dim %c_147, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %336 = stablehlo.add %arg212, %335 : tensor<200x4xi32>
        %337 = stablehlo.select %334, %336, %arg212 : tensor<200x4xi1>, tensor<200x4xi32>
        %338 = stablehlo.broadcast_in_dim %327, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %339 = stablehlo.broadcast_in_dim %332, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %340 = stablehlo.broadcast_in_dim %337, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %341 = stablehlo.concatenate %338, %339, %340, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
        %342 = "stablehlo.gather"(%arg221, %341) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1250x1200xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
        %c_148 = stablehlo.constant dense<0> : tensor<i32>
        %343 = func.call @_where_445(%321, %342, %c_148) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
        %344 = stablehlo.multiply %343, %arg213 : tensor<200x4xf32>
        %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %345 = stablehlo.reduce(%344 init: %cst_149) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
        %346 = stablehlo.slice %arg222 [0:1, 0:242400] : (tensor<1x242400xf32>) -> tensor<1x242400xf32>
        %347 = stablehlo.reshape %346 : (tensor<1x242400xf32>) -> tensor<242400xf32>
        %348 = stablehlo.slice %arg223 [0:1, 0:242400] : (tensor<1x242400xf32>) -> tensor<1x242400xf32>
        %349 = stablehlo.reshape %348 : (tensor<1x242400xf32>) -> tensor<242400xf32>
        %350 = stablehlo.broadcast_in_dim %165, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %351 = stablehlo.broadcast_in_dim %201, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %352 = stablehlo.broadcast_in_dim %237, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %353 = stablehlo.broadcast_in_dim %273, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %354 = stablehlo.broadcast_in_dim %309, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %355 = stablehlo.broadcast_in_dim %345, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %356 = stablehlo.concatenate %350, %351, %352, %353, %354, %355, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
        %cst_150 = stablehlo.constant dense<6.28318548> : tensor<f32>
        %357 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f32>) -> tensor<101xf32>
        %358 = stablehlo.multiply %357, %arg214 : tensor<101xf32>
        %359 = stablehlo.optimization_barrier %358 : tensor<101xf32>
        %360 = stablehlo.optimization_barrier %arg225 : tensor<f32>
        %361 = stablehlo.broadcast_in_dim %360, dims = [] : (tensor<f32>) -> tensor<101xf32>
        %362 = stablehlo.multiply %359, %361 : tensor<101xf32>
        %363 = stablehlo.cosine %362 : tensor<101xf32>
        %364 = stablehlo.sine %362 : tensor<101xf32>
        %cst_151 = stablehlo.constant dense<0x7F800000> : tensor<f32>
        %cst_152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %365 = stablehlo.subtract %cst_151, %cst_152 : tensor<f32>
        %cst_153 = stablehlo.constant dense<1.000000e-30> : tensor<f32>
        %366 = stablehlo.maximum %365, %cst_153 : tensor<f32>
        %cst_154 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %367 = stablehlo.subtract %arg225, %cst_154 : tensor<f32>
        %368 = stablehlo.divide %367, %366 : tensor<f32>
        %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %cst_156 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %369 = func.call @clip_462(%368, %cst_155, %cst_156) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
        %cst_157 = stablehlo.constant dense<6.28318548> : tensor<f32>
        %370 = stablehlo.multiply %cst_157, %369 : tensor<f32>
        %371 = stablehlo.cosine %370 : tensor<f32>
        %cst_158 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %372 = stablehlo.subtract %cst_158, %371 : tensor<f32>
        %cst_159 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
        %373 = stablehlo.multiply %cst_159, %372 : tensor<f32>
        %cst_160 = stablehlo.constant dense<0x7F800000> : tensor<f32>
        %374 = stablehlo.is_finite %cst_160 : (tensor<f32>) -> tensor<i1>
        %c_161 = stablehlo.constant dense<false> : tensor<i1>
        %375 = stablehlo.and %c_161, %374 : tensor<i1>
        %cst_162 = stablehlo.constant dense<0x7F800000> : tensor<f32>
        %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %376 = stablehlo.compare  GT, %cst_162, %cst_163,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %377 = stablehlo.and %375, %376 : tensor<i1>
        %cst_164 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %378 = func.call @_where_468(%377, %373, %cst_164) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
        %cst_165 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %379 = stablehlo.maximum %378, %cst_165 : tensor<f32>
        %cst_166 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %380 = stablehlo.multiply %arg226, %cst_166 : tensor<f32>
        %cst_167 = stablehlo.constant dense<0x4D8EF3C2> : tensor<f32>
        %381 = stablehlo.multiply %380, %cst_167 : tensor<f32>
        %cst_168 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
        %382 = stablehlo.divide %381, %cst_168 : tensor<f32>
        %cst_169 = stablehlo.constant dense<6.28318548> : tensor<f32>
        %383 = stablehlo.sqrt %cst_169 : tensor<f32>
        %384 = stablehlo.divide %382, %383 : tensor<f32>
        %385 = stablehlo.multiply %379, %384 : tensor<f32>
        %c_170 = stablehlo.constant dense<0> : tensor<i32>
        %c_171 = stablehlo.constant dense<1> : tensor<i32>
        %386 = stablehlo.compare  EQ, %c_170, %c_171,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %387 = func.call @_where_471(%386, %385, %379) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
        %388 = stablehlo.broadcast_in_dim %arg215, dims = [0] : (tensor<6xf32>) -> tensor<6x1x1xf32>
        %389 = stablehlo.broadcast_in_dim %387, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
        %390 = stablehlo.multiply %389, %388 : tensor<6x1x1xf32>
        %391 = stablehlo.dot_general %356, %363, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<101xf32>) -> tensor<6x200x101xf32>
        %392 = stablehlo.transpose %391, dims = [0, 2, 1] : (tensor<6x200x101xf32>) -> tensor<6x101x200xf32>
        %393 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x101x200xf32>
        %394 = stablehlo.multiply %393, %392 : tensor<6x101x200xf32>
        %395 = stablehlo.broadcast_in_dim %387, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
        %396 = stablehlo.multiply %395, %388 : tensor<6x1x1xf32>
        %397 = stablehlo.dot_general %356, %364, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<101xf32>) -> tensor<6x200x101xf32>
        %398 = stablehlo.transpose %397, dims = [0, 2, 1] : (tensor<6x200x101xf32>) -> tensor<6x101x200xf32>
        %399 = stablehlo.broadcast_in_dim %396, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x101x200xf32>
        %400 = stablehlo.multiply %399, %398 : tensor<6x101x200xf32>
        %401 = stablehlo.reshape %394 : (tensor<6x101x200xf32>) -> tensor<121200xf32>
        %c_172 = stablehlo.constant dense<0> : tensor<i32>
        %402 = stablehlo.broadcast_in_dim %c_172, dims = [] : (tensor<i32>) -> tensor<1xi32>
        %403 = "stablehlo.scatter"(%347, %402, %401) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
        ^bb0(%arg227: tensor<f32>, %arg228: tensor<f32>):
          %412 = stablehlo.add %arg227, %arg228 : tensor<f32>
          stablehlo.return %412 : tensor<f32>
        }) : (tensor<242400xf32>, tensor<1xi32>, tensor<121200xf32>) -> tensor<242400xf32>
        %404 = stablehlo.reshape %400 : (tensor<6x101x200xf32>) -> tensor<121200xf32>
        %c_173 = stablehlo.constant dense<0> : tensor<i32>
        %405 = stablehlo.broadcast_in_dim %c_173, dims = [] : (tensor<i32>) -> tensor<1xi32>
        %406 = "stablehlo.scatter"(%349, %405, %404) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
        ^bb0(%arg227: tensor<f32>, %arg228: tensor<f32>):
          %412 = stablehlo.add %arg227, %arg228 : tensor<f32>
          stablehlo.return %412 : tensor<f32>
        }) : (tensor<242400xf32>, tensor<1xi32>, tensor<121200xf32>) -> tensor<242400xf32>
        %407 = stablehlo.broadcast_in_dim %378, dims = [] : (tensor<f32>) -> tensor<101xf32>
        %c_174 = stablehlo.constant dense<0> : tensor<i32>
        %408 = stablehlo.broadcast_in_dim %c_174, dims = [] : (tensor<i32>) -> tensor<1xi32>
        %409 = "stablehlo.scatter"(%arg224, %408, %407) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
        ^bb0(%arg227: tensor<f32>, %arg228: tensor<f32>):
          %412 = stablehlo.add %arg227, %arg228 : tensor<f32>
          stablehlo.return %412 : tensor<f32>
        }) : (tensor<202xf32>, tensor<1xi32>, tensor<101xf32>) -> tensor<202xf32>
        %410 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<242400xf32>) -> tensor<1x242400xf32>
        %411 = stablehlo.broadcast_in_dim %406, dims = [1] : (tensor<242400xf32>) -> tensor<1x242400xf32>
        sdy.return %410, %411, %409 : tensor<1x242400xf32>, tensor<1x242400xf32>, tensor<202xf32>
      } : (tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<101xf32>, tensor<6xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>, tensor<f32>, tensor<f32>) -> (tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>)
      stablehlo.return %129#0, %129#1, %129#2 : tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>
    }) : (tensor<i32>) -> (tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>)
    %c_29 = stablehlo.constant dense<1> : tensor<i32>
    %69 = stablehlo.add %arg188, %c_29 : tensor<i32>
    %c_30 = stablehlo.constant dense<1> : tensor<i32>
    %c_31 = stablehlo.constant dense<1> : tensor<i32>
    %70 = stablehlo.maximum %c_30, %c_31 : tensor<i32>
    %71 = call @remainder(%69, %70) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_32 = stablehlo.constant dense<0> : tensor<i32>
    %72 = stablehlo.compare  EQ, %71, %c_32,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %73 = stablehlo.slice %58 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %74 = stablehlo.reshape %73 : (tensor<1xi32>) -> tensor<i32>
    %c_33 = stablehlo.constant dense<1> : tensor<i32>
    %75 = stablehlo.compare  LT, %74, %c_33,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %76 = stablehlo.and %72, %75 : tensor<i1>
    %c_34 = stablehlo.constant dense<false> : tensor<i1>
    %77 = stablehlo.and %76, %c_34 : tensor<i1>
    %c_35 = stablehlo.constant dense<false> : tensor<i1>
    %78 = stablehlo.or %77, %c_35 : tensor<i1>
    %79 = stablehlo.convert %78 : (tensor<i1>) -> tensor<i32>
    %80 = "stablehlo.case"(%79) ({
      %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      stablehlo.return %cst_60 : tensor<f32>
    }, {
      %c_60 = stablehlo.constant dense<0> : tensor<i32>
      %129 = stablehlo.broadcast_in_dim %c_60, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %130 = stablehlo.compare  LT, %arg113, %129,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_61 = stablehlo.constant dense<10008> : tensor<i32>
      %131 = stablehlo.broadcast_in_dim %c_61, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %132 = stablehlo.add %arg113, %131 : tensor<200x1xi32>
      %133 = stablehlo.select %130, %132, %arg113 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_62 = stablehlo.constant dense<0> : tensor<i32>
      %134 = stablehlo.broadcast_in_dim %c_62, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %135 = stablehlo.compare  LT, %arg114, %134,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_63 = stablehlo.constant dense<1251> : tensor<i32>
      %136 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %137 = stablehlo.add %arg114, %136 : tensor<200x1xi32>
      %138 = stablehlo.select %135, %137, %arg114 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_64 = stablehlo.constant dense<0> : tensor<i32>
      %139 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %140 = stablehlo.compare  LT, %arg115, %139,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_65 = stablehlo.constant dense<1200> : tensor<i32>
      %141 = stablehlo.broadcast_in_dim %c_65, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %142 = stablehlo.add %arg115, %141 : tensor<200x1xi32>
      %143 = stablehlo.select %140, %142, %arg115 : tensor<200x1xi1>, tensor<200x1xi32>
      %144 = stablehlo.broadcast_in_dim %133, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %145 = stablehlo.broadcast_in_dim %138, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %146 = stablehlo.broadcast_in_dim %143, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %147 = stablehlo.concatenate %144, %145, %146, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %148 = "stablehlo.gather"(%7#0, %147) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1251x1200xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %149 = stablehlo.multiply %148, %arg116 : tensor<200x1xf32>
      %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %150 = stablehlo.reduce(%149 init: %cst_66) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_67 = stablehlo.constant dense<0> : tensor<i32>
      %151 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %152 = stablehlo.compare  LT, %arg117, %151,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_68 = stablehlo.constant dense<10008> : tensor<i32>
      %153 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %154 = stablehlo.add %arg117, %153 : tensor<200x1xi32>
      %155 = stablehlo.select %152, %154, %arg117 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_69 = stablehlo.constant dense<0> : tensor<i32>
      %156 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %157 = stablehlo.compare  LT, %arg118, %156,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_70 = stablehlo.constant dense<1250> : tensor<i32>
      %158 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %159 = stablehlo.add %arg118, %158 : tensor<200x1xi32>
      %160 = stablehlo.select %157, %159, %arg118 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_71 = stablehlo.constant dense<0> : tensor<i32>
      %161 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %162 = stablehlo.compare  LT, %arg119, %161,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_72 = stablehlo.constant dense<1201> : tensor<i32>
      %163 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %164 = stablehlo.add %arg119, %163 : tensor<200x1xi32>
      %165 = stablehlo.select %162, %164, %arg119 : tensor<200x1xi1>, tensor<200x1xi32>
      %166 = stablehlo.broadcast_in_dim %155, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %167 = stablehlo.broadcast_in_dim %160, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %168 = stablehlo.broadcast_in_dim %165, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %169 = stablehlo.concatenate %166, %167, %168, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %170 = "stablehlo.gather"(%8, %169) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1250x1201xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %171 = stablehlo.multiply %170, %arg120 : tensor<200x1xf32>
      %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %172 = stablehlo.reduce(%171 init: %cst_73) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_74 = stablehlo.constant dense<0> : tensor<i32>
      %173 = stablehlo.broadcast_in_dim %c_74, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %174 = stablehlo.compare  LT, %arg121, %173,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_75 = stablehlo.constant dense<10008> : tensor<i32>
      %175 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %176 = stablehlo.add %arg121, %175 : tensor<200x1xi32>
      %177 = stablehlo.select %174, %176, %arg121 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_76 = stablehlo.constant dense<0> : tensor<i32>
      %178 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %179 = stablehlo.compare  LT, %arg122, %178,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_77 = stablehlo.constant dense<1251> : tensor<i32>
      %180 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %181 = stablehlo.add %arg122, %180 : tensor<200x1xi32>
      %182 = stablehlo.select %179, %181, %arg122 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_78 = stablehlo.constant dense<0> : tensor<i32>
      %183 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %184 = stablehlo.compare  LT, %arg123, %183,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_79 = stablehlo.constant dense<1201> : tensor<i32>
      %185 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %186 = stablehlo.add %arg123, %185 : tensor<200x1xi32>
      %187 = stablehlo.select %184, %186, %arg123 : tensor<200x1xi1>, tensor<200x1xi32>
      %188 = stablehlo.broadcast_in_dim %177, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %189 = stablehlo.broadcast_in_dim %182, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %190 = stablehlo.broadcast_in_dim %187, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %191 = stablehlo.concatenate %188, %189, %190, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %192 = "stablehlo.gather"(%9, %191) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1251x1201xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %193 = stablehlo.multiply %192, %arg124 : tensor<200x1xf32>
      %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %194 = stablehlo.reduce(%193 init: %cst_80) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_81 = stablehlo.constant dense<0> : tensor<i32>
      %195 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %196 = stablehlo.compare  LT, %arg125, %195,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_82 = stablehlo.constant dense<10008> : tensor<i32>
      %197 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %198 = stablehlo.add %arg125, %197 : tensor<200x1xi32>
      %199 = stablehlo.select %196, %198, %arg125 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_83 = stablehlo.constant dense<0> : tensor<i32>
      %200 = stablehlo.broadcast_in_dim %c_83, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %201 = stablehlo.compare  LT, %arg126, %200,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_84 = stablehlo.constant dense<1250> : tensor<i32>
      %202 = stablehlo.broadcast_in_dim %c_84, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %203 = stablehlo.add %arg126, %202 : tensor<200x1xi32>
      %204 = stablehlo.select %201, %203, %arg126 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_85 = stablehlo.constant dense<0> : tensor<i32>
      %205 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %206 = stablehlo.compare  LT, %arg127, %205,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_86 = stablehlo.constant dense<1201> : tensor<i32>
      %207 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %208 = stablehlo.add %arg127, %207 : tensor<200x1xi32>
      %209 = stablehlo.select %206, %208, %arg127 : tensor<200x1xi1>, tensor<200x1xi32>
      %210 = stablehlo.broadcast_in_dim %199, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %211 = stablehlo.broadcast_in_dim %204, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %212 = stablehlo.broadcast_in_dim %209, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %213 = stablehlo.concatenate %210, %211, %212, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %214 = "stablehlo.gather"(%4#0, %213) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1250x1201xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %215 = stablehlo.multiply %214, %arg128 : tensor<200x1xf32>
      %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %216 = stablehlo.reduce(%215 init: %cst_87) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_88 = stablehlo.constant dense<0> : tensor<i32>
      %217 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %218 = stablehlo.compare  LT, %arg129, %217,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_89 = stablehlo.constant dense<10008> : tensor<i32>
      %219 = stablehlo.broadcast_in_dim %c_89, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %220 = stablehlo.add %arg129, %219 : tensor<200x1xi32>
      %221 = stablehlo.select %218, %220, %arg129 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_90 = stablehlo.constant dense<0> : tensor<i32>
      %222 = stablehlo.broadcast_in_dim %c_90, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %223 = stablehlo.compare  LT, %arg130, %222,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_91 = stablehlo.constant dense<1251> : tensor<i32>
      %224 = stablehlo.broadcast_in_dim %c_91, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %225 = stablehlo.add %arg130, %224 : tensor<200x1xi32>
      %226 = stablehlo.select %223, %225, %arg130 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_92 = stablehlo.constant dense<0> : tensor<i32>
      %227 = stablehlo.broadcast_in_dim %c_92, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %228 = stablehlo.compare  LT, %arg131, %227,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_93 = stablehlo.constant dense<1200> : tensor<i32>
      %229 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %230 = stablehlo.add %arg131, %229 : tensor<200x1xi32>
      %231 = stablehlo.select %228, %230, %arg131 : tensor<200x1xi1>, tensor<200x1xi32>
      %232 = stablehlo.broadcast_in_dim %221, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %233 = stablehlo.broadcast_in_dim %226, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %234 = stablehlo.broadcast_in_dim %231, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %235 = stablehlo.concatenate %232, %233, %234, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %236 = "stablehlo.gather"(%5, %235) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1251x1200xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %237 = stablehlo.multiply %236, %arg132 : tensor<200x1xf32>
      %cst_94 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %238 = stablehlo.reduce(%237 init: %cst_94) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %c_95 = stablehlo.constant dense<0> : tensor<i32>
      %239 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %240 = stablehlo.compare  LT, %arg133, %239,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_96 = stablehlo.constant dense<10008> : tensor<i32>
      %241 = stablehlo.broadcast_in_dim %c_96, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %242 = stablehlo.add %arg133, %241 : tensor<200x1xi32>
      %243 = stablehlo.select %240, %242, %arg133 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_97 = stablehlo.constant dense<0> : tensor<i32>
      %244 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %245 = stablehlo.compare  LT, %arg134, %244,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_98 = stablehlo.constant dense<1250> : tensor<i32>
      %246 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %247 = stablehlo.add %arg134, %246 : tensor<200x1xi32>
      %248 = stablehlo.select %245, %247, %arg134 : tensor<200x1xi1>, tensor<200x1xi32>
      %c_99 = stablehlo.constant dense<0> : tensor<i32>
      %249 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %250 = stablehlo.compare  LT, %arg135, %249,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
      %c_100 = stablehlo.constant dense<1200> : tensor<i32>
      %251 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
      %252 = stablehlo.add %arg135, %251 : tensor<200x1xi32>
      %253 = stablehlo.select %250, %252, %arg135 : tensor<200x1xi1>, tensor<200x1xi32>
      %254 = stablehlo.broadcast_in_dim %243, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %255 = stablehlo.broadcast_in_dim %248, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %256 = stablehlo.broadcast_in_dim %253, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
      %257 = stablehlo.concatenate %254, %255, %256, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
      %258 = "stablehlo.gather"(%6, %257) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<10008x1250x1200xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
      %259 = stablehlo.multiply %258, %arg136 : tensor<200x1xf32>
      %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %260 = stablehlo.reduce(%259 init: %cst_101) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
      %261 = stablehlo.broadcast_in_dim %150, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %262 = stablehlo.broadcast_in_dim %172, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %263 = stablehlo.broadcast_in_dim %194, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %264 = stablehlo.broadcast_in_dim %216, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %265 = stablehlo.broadcast_in_dim %238, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %266 = stablehlo.broadcast_in_dim %260, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
      %267 = stablehlo.concatenate %261, %262, %263, %264, %265, %266, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
      %268 = stablehlo.slice %267 [1:2, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %269 = stablehlo.reshape %268 : (tensor<1x200xf32>) -> tensor<200xf32>
      %270 = stablehlo.slice %267 [2:3, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %271 = stablehlo.reshape %270 : (tensor<1x200xf32>) -> tensor<200xf32>
      %272 = stablehlo.slice %267 [4:5, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %273 = stablehlo.reshape %272 : (tensor<1x200xf32>) -> tensor<200xf32>
      %274 = stablehlo.slice %267 [5:6, 0:200] : (tensor<6x200xf32>) -> tensor<1x200xf32>
      %275 = stablehlo.reshape %274 : (tensor<1x200xf32>) -> tensor<200xf32>
      %276 = stablehlo.multiply %269, %275 : tensor<200xf32>
      %277 = stablehlo.multiply %271, %273 : tensor<200xf32>
      %278 = stablehlo.subtract %276, %277 : tensor<200xf32>
      %cst_102 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %280 = stablehlo.multiply %278, %279 : tensor<200xf32>
      %cst_103 = stablehlo.constant dense<7.485380e-15> : tensor<f32>
      %281 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<200xf32>
      %282 = stablehlo.multiply %280, %281 : tensor<200xf32>
      %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %283 = stablehlo.reduce(%282 init: %cst_104) applies stablehlo.add across dimensions = [0] : (tensor<200xf32>, tensor<f32>) -> tensor<f32>
      stablehlo.return %283 : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %81 = call @_where_332(%77, %80, %cst_36) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %82 = stablehlo.slice %58 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %83 = stablehlo.reshape %82 : (tensor<1xi32>) -> tensor<i32>
    %c_37 = stablehlo.constant dense<0> : tensor<i32>
    %84 = stablehlo.minimum %83, %c_37 : tensor<i32>
    %c_38 = stablehlo.constant dense<0> : tensor<i32>
    %85 = stablehlo.compare  LT, %84, %c_38,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_39 = stablehlo.constant dense<1> : tensor<i32>
    %86 = stablehlo.add %84, %c_39 : tensor<i32>
    %87 = stablehlo.select %85, %86, %84 : tensor<i1>, tensor<i32>
    %c_40 = stablehlo.constant dense<1> : tensor<i32>
    %88 = stablehlo.dynamic_slice %43, %c_40, %87, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %89 = stablehlo.reshape %88 : (tensor<1x1xf32>) -> tensor<f32>
    %c_41 = stablehlo.constant dense<0> : tensor<i32>
    %90 = stablehlo.compare  LT, %84, %c_41,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_42 = stablehlo.constant dense<1> : tensor<i32>
    %91 = stablehlo.add %84, %c_42 : tensor<i32>
    %92 = stablehlo.select %90, %91, %84 : tensor<i1>, tensor<i32>
    %c_43 = stablehlo.constant dense<1> : tensor<i32>
    %93 = stablehlo.dynamic_slice %51, %c_43, %92, sizes = [1, 1] : (tensor<2x1xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
    %94 = stablehlo.reshape %93 : (tensor<1x1xf32>) -> tensor<f32>
    %95 = call @_where_337(%77, %81, %89) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_44 = stablehlo.constant dense<0> : tensor<i32>
    %96 = stablehlo.compare  LT, %84, %c_44,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_45 = stablehlo.constant dense<1> : tensor<i32>
    %97 = stablehlo.add %84, %c_45 : tensor<i32>
    %98 = stablehlo.select %96, %97, %84 : tensor<i1>, tensor<i32>
    %c_46 = stablehlo.constant dense<1> : tensor<i32>
    %99 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %100 = stablehlo.broadcast_in_dim %98, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %101 = stablehlo.concatenate %99, %100, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %102 = "stablehlo.scatter"(%43, %101, %95) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg190: tensor<f32>, %arg191: tensor<f32>):
      stablehlo.return %arg191 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %103 = call @_where_337(%77, %3, %94) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c_47 = stablehlo.constant dense<0> : tensor<i32>
    %104 = stablehlo.compare  LT, %84, %c_47,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_48 = stablehlo.constant dense<1> : tensor<i32>
    %105 = stablehlo.add %84, %c_48 : tensor<i32>
    %106 = stablehlo.select %104, %105, %84 : tensor<i1>, tensor<i32>
    %c_49 = stablehlo.constant dense<1> : tensor<i32>
    %107 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %108 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %109 = stablehlo.concatenate %107, %108, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %110 = "stablehlo.scatter"(%51, %109, %103) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true}> ({
    ^bb0(%arg190: tensor<f32>, %arg191: tensor<f32>):
      stablehlo.return %arg191 : tensor<f32>
    }) : (tensor<2x1xf32>, tensor<2xi32>, tensor<f32>) -> tensor<2x1xf32>
    %111 = stablehlo.slice %58 [1:2] : (tensor<2xi32>) -> tensor<1xi32>
    %112 = stablehlo.reshape %111 : (tensor<1xi32>) -> tensor<i32>
    %c_50 = stablehlo.constant dense<1> : tensor<i32>
    %c_51 = stablehlo.constant dense<0> : tensor<i32>
    %113 = call @_where_342(%77, %c_50, %c_51) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %114 = stablehlo.convert %113 : tensor<i32>
    %115 = stablehlo.add %112, %114 : tensor<i32>
    %c_52 = stablehlo.constant dense<1> : tensor<i32>
    %116 = stablehlo.broadcast_in_dim %c_52, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %117 = "stablehlo.scatter"(%58, %116, %115) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg190: tensor<i32>, %arg191: tensor<i32>):
      stablehlo.return %arg191 : tensor<i32>
    }) : (tensor<2xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
    %cst_53 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %118 = stablehlo.compare  GE, %3, %cst_53,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %c_54 = stablehlo.constant dense<true> : tensor<i1>
    %119 = stablehlo.and %c_54, %118 : tensor<i1>
    %cst_55 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %120 = stablehlo.compare  LE, %3, %cst_55,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %121 = stablehlo.and %119, %120 : tensor<i1>
    %c_56 = stablehlo.constant dense<1> : tensor<i32>
    %c_57 = stablehlo.constant dense<1> : tensor<i32>
    %122 = stablehlo.maximum %c_56, %c_57 : tensor<i32>
    %123 = call @remainder(%arg188, %122) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %c_58 = stablehlo.constant dense<0> : tensor<i32>
    %124 = stablehlo.compare  EQ, %123, %c_58,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %125 = stablehlo.and %121, %124 : tensor<i1>
    %126 = stablehlo.convert %125 : (tensor<i1>) -> tensor<i32>
    %127:3 = "stablehlo.case"(%126) ({
      stablehlo.return %68#0, %68#1, %68#2 : tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>
    }, {
      %129:3 = sdy.manual_computation(%arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %arg162, %7#0, %8, %9, %4#0, %5, %6, %68#0, %68#1, %68#2, %3, %arg0) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}]>, <@mesh, [{"fdtd"}, {}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"fdtd"}, {}]>, <@mesh, [{"fdtd"}, {}]>, <@mesh, [{}]>] manual_axes={"fdtd"} (%arg190: tensor<200x8xi32>, %arg191: tensor<200x8xi32>, %arg192: tensor<200x8xi32>, %arg193: tensor<200x8xf32>, %arg194: tensor<200x2xi32>, %arg195: tensor<200x2xi32>, %arg196: tensor<200x2xi32>, %arg197: tensor<200x2xf32>, %arg198: tensor<200x2xi32>, %arg199: tensor<200x2xi32>, %arg200: tensor<200x2xi32>, %arg201: tensor<200x2xf32>, %arg202: tensor<200x1xi32>, %arg203: tensor<200x1xi32>, %arg204: tensor<200x1xi32>, %arg205: tensor<200x1xf32>, %arg206: tensor<200x4xi32>, %arg207: tensor<200x4xi32>, %arg208: tensor<200x4xi32>, %arg209: tensor<200x4xf32>, %arg210: tensor<200x4xi32>, %arg211: tensor<200x4xi32>, %arg212: tensor<200x4xi32>, %arg213: tensor<200x4xf32>, %arg214: tensor<101xf32>, %arg215: tensor<6xf32>, %arg216: tensor<1251x1251x1200xf32>, %arg217: tensor<1251x1250x1201xf32>, %arg218: tensor<1251x1251x1201xf32>, %arg219: tensor<1251x1250x1201xf32>, %arg220: tensor<1251x1251x1200xf32>, %arg221: tensor<1251x1250x1200xf32>, %arg222: tensor<1x242400xf32>, %arg223: tensor<1x242400xf32>, %arg224: tensor<202xf32>, %arg225: tensor<f32>, %arg226: tensor<f32>) {
        %c_60 = stablehlo.constant dense<1> : tensor<ui32>
        %c_61 = stablehlo.constant dense<8> : tensor<ui32>
        %130 = stablehlo.partition_id : tensor<ui32>
        %131 = stablehlo.divide %130, %c_60 : tensor<ui32>
        %132 = stablehlo.remainder %131, %c_61 : tensor<ui32>
        %133 = stablehlo.convert %132 : (tensor<ui32>) -> tensor<i32>
        %c_62 = stablehlo.constant dense<1251> : tensor<i32>
        %134 = stablehlo.multiply %133, %c_62 : tensor<i32>
        %135 = stablehlo.broadcast_in_dim %134, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %136 = stablehlo.subtract %arg190, %135 : tensor<200x8xi32>
        %c_63 = stablehlo.constant dense<0> : tensor<i32>
        %137 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %138 = stablehlo.compare  GE, %136, %137,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
        %c_64 = stablehlo.constant dense<1251> : tensor<i32>
        %139 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %140 = stablehlo.compare  LT, %136, %139,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
        %141 = stablehlo.and %138, %140 : tensor<200x8xi1>
        %c_65 = stablehlo.constant dense<0> : tensor<i32>
        %c_66 = stablehlo.constant dense<1250> : tensor<i32>
        %142 = func.call @clip_356(%136, %c_65, %c_66) : (tensor<200x8xi32>, tensor<i32>, tensor<i32>) -> tensor<200x8xi32>
        %c_67 = stablehlo.constant dense<0> : tensor<i32>
        %143 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %144 = stablehlo.compare  LT, %142, %143,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
        %c_68 = stablehlo.constant dense<1251> : tensor<i32>
        %145 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %146 = stablehlo.add %142, %145 : tensor<200x8xi32>
        %147 = stablehlo.select %144, %146, %142 : tensor<200x8xi1>, tensor<200x8xi32>
        %c_69 = stablehlo.constant dense<0> : tensor<i32>
        %148 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %149 = stablehlo.compare  LT, %arg191, %148,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
        %c_70 = stablehlo.constant dense<1251> : tensor<i32>
        %150 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %151 = stablehlo.add %arg191, %150 : tensor<200x8xi32>
        %152 = stablehlo.select %149, %151, %arg191 : tensor<200x8xi1>, tensor<200x8xi32>
        %c_71 = stablehlo.constant dense<0> : tensor<i32>
        %153 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %154 = stablehlo.compare  LT, %arg192, %153,  SIGNED : (tensor<200x8xi32>, tensor<200x8xi32>) -> tensor<200x8xi1>
        %c_72 = stablehlo.constant dense<1200> : tensor<i32>
        %155 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
        %156 = stablehlo.add %arg192, %155 : tensor<200x8xi32>
        %157 = stablehlo.select %154, %156, %arg192 : tensor<200x8xi1>, tensor<200x8xi32>
        %158 = stablehlo.broadcast_in_dim %147, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
        %159 = stablehlo.broadcast_in_dim %152, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
        %160 = stablehlo.broadcast_in_dim %157, dims = [0, 1] : (tensor<200x8xi32>) -> tensor<200x8x1xi32>
        %161 = stablehlo.concatenate %158, %159, %160, dim = 2 : (tensor<200x8x1xi32>, tensor<200x8x1xi32>, tensor<200x8x1xi32>) -> tensor<200x8x3xi32>
        %162 = "stablehlo.gather"(%arg216, %161) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1251x1200xf32>, tensor<200x8x3xi32>) -> tensor<200x8xf32>
        %c_73 = stablehlo.constant dense<0> : tensor<i32>
        %163 = func.call @_where_369(%141, %162, %c_73) : (tensor<200x8xi1>, tensor<200x8xf32>, tensor<i32>) -> tensor<200x8xf32>
        %164 = stablehlo.multiply %163, %arg193 : tensor<200x8xf32>
        %cst_74 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %165 = stablehlo.reduce(%164 init: %cst_74) applies stablehlo.add across dimensions = [1] : (tensor<200x8xf32>, tensor<f32>) -> tensor<200xf32>
        %c_75 = stablehlo.constant dense<1> : tensor<ui32>
        %c_76 = stablehlo.constant dense<8> : tensor<ui32>
        %166 = stablehlo.partition_id : tensor<ui32>
        %167 = stablehlo.divide %166, %c_75 : tensor<ui32>
        %168 = stablehlo.remainder %167, %c_76 : tensor<ui32>
        %169 = stablehlo.convert %168 : (tensor<ui32>) -> tensor<i32>
        %c_77 = stablehlo.constant dense<1251> : tensor<i32>
        %170 = stablehlo.multiply %169, %c_77 : tensor<i32>
        %171 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %172 = stablehlo.subtract %arg194, %171 : tensor<200x2xi32>
        %c_78 = stablehlo.constant dense<0> : tensor<i32>
        %173 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %174 = stablehlo.compare  GE, %172, %173,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_79 = stablehlo.constant dense<1251> : tensor<i32>
        %175 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %176 = stablehlo.compare  LT, %172, %175,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %177 = stablehlo.and %174, %176 : tensor<200x2xi1>
        %c_80 = stablehlo.constant dense<0> : tensor<i32>
        %c_81 = stablehlo.constant dense<1250> : tensor<i32>
        %178 = func.call @clip_381(%172, %c_80, %c_81) : (tensor<200x2xi32>, tensor<i32>, tensor<i32>) -> tensor<200x2xi32>
        %c_82 = stablehlo.constant dense<0> : tensor<i32>
        %179 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %180 = stablehlo.compare  LT, %178, %179,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_83 = stablehlo.constant dense<1251> : tensor<i32>
        %181 = stablehlo.broadcast_in_dim %c_83, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %182 = stablehlo.add %178, %181 : tensor<200x2xi32>
        %183 = stablehlo.select %180, %182, %178 : tensor<200x2xi1>, tensor<200x2xi32>
        %c_84 = stablehlo.constant dense<0> : tensor<i32>
        %184 = stablehlo.broadcast_in_dim %c_84, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %185 = stablehlo.compare  LT, %arg195, %184,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_85 = stablehlo.constant dense<1250> : tensor<i32>
        %186 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %187 = stablehlo.add %arg195, %186 : tensor<200x2xi32>
        %188 = stablehlo.select %185, %187, %arg195 : tensor<200x2xi1>, tensor<200x2xi32>
        %c_86 = stablehlo.constant dense<0> : tensor<i32>
        %189 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %190 = stablehlo.compare  LT, %arg196, %189,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_87 = stablehlo.constant dense<1201> : tensor<i32>
        %191 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %192 = stablehlo.add %arg196, %191 : tensor<200x2xi32>
        %193 = stablehlo.select %190, %192, %arg196 : tensor<200x2xi1>, tensor<200x2xi32>
        %194 = stablehlo.broadcast_in_dim %183, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %195 = stablehlo.broadcast_in_dim %188, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %196 = stablehlo.broadcast_in_dim %193, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %197 = stablehlo.concatenate %194, %195, %196, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
        %198 = "stablehlo.gather"(%arg217, %197) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1250x1201xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
        %c_88 = stablehlo.constant dense<0> : tensor<i32>
        %199 = func.call @_where_394(%177, %198, %c_88) : (tensor<200x2xi1>, tensor<200x2xf32>, tensor<i32>) -> tensor<200x2xf32>
        %200 = stablehlo.multiply %199, %arg197 : tensor<200x2xf32>
        %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %201 = stablehlo.reduce(%200 init: %cst_89) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
        %c_90 = stablehlo.constant dense<1> : tensor<ui32>
        %c_91 = stablehlo.constant dense<8> : tensor<ui32>
        %202 = stablehlo.partition_id : tensor<ui32>
        %203 = stablehlo.divide %202, %c_90 : tensor<ui32>
        %204 = stablehlo.remainder %203, %c_91 : tensor<ui32>
        %205 = stablehlo.convert %204 : (tensor<ui32>) -> tensor<i32>
        %c_92 = stablehlo.constant dense<1251> : tensor<i32>
        %206 = stablehlo.multiply %205, %c_92 : tensor<i32>
        %207 = stablehlo.broadcast_in_dim %206, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %208 = stablehlo.subtract %arg198, %207 : tensor<200x2xi32>
        %c_93 = stablehlo.constant dense<0> : tensor<i32>
        %209 = stablehlo.broadcast_in_dim %c_93, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %210 = stablehlo.compare  GE, %208, %209,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_94 = stablehlo.constant dense<1251> : tensor<i32>
        %211 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %212 = stablehlo.compare  LT, %208, %211,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %213 = stablehlo.and %210, %212 : tensor<200x2xi1>
        %c_95 = stablehlo.constant dense<0> : tensor<i32>
        %c_96 = stablehlo.constant dense<1250> : tensor<i32>
        %214 = func.call @clip_381(%208, %c_95, %c_96) : (tensor<200x2xi32>, tensor<i32>, tensor<i32>) -> tensor<200x2xi32>
        %c_97 = stablehlo.constant dense<0> : tensor<i32>
        %215 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %216 = stablehlo.compare  LT, %214, %215,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_98 = stablehlo.constant dense<1251> : tensor<i32>
        %217 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %218 = stablehlo.add %214, %217 : tensor<200x2xi32>
        %219 = stablehlo.select %216, %218, %214 : tensor<200x2xi1>, tensor<200x2xi32>
        %c_99 = stablehlo.constant dense<0> : tensor<i32>
        %220 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %221 = stablehlo.compare  LT, %arg199, %220,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_100 = stablehlo.constant dense<1251> : tensor<i32>
        %222 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %223 = stablehlo.add %arg199, %222 : tensor<200x2xi32>
        %224 = stablehlo.select %221, %223, %arg199 : tensor<200x2xi1>, tensor<200x2xi32>
        %c_101 = stablehlo.constant dense<0> : tensor<i32>
        %225 = stablehlo.broadcast_in_dim %c_101, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %226 = stablehlo.compare  LT, %arg200, %225,  SIGNED : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<200x2xi1>
        %c_102 = stablehlo.constant dense<1201> : tensor<i32>
        %227 = stablehlo.broadcast_in_dim %c_102, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
        %228 = stablehlo.add %arg200, %227 : tensor<200x2xi32>
        %229 = stablehlo.select %226, %228, %arg200 : tensor<200x2xi1>, tensor<200x2xi32>
        %230 = stablehlo.broadcast_in_dim %219, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %231 = stablehlo.broadcast_in_dim %224, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %232 = stablehlo.broadcast_in_dim %229, dims = [0, 1] : (tensor<200x2xi32>) -> tensor<200x2x1xi32>
        %233 = stablehlo.concatenate %230, %231, %232, dim = 2 : (tensor<200x2x1xi32>, tensor<200x2x1xi32>, tensor<200x2x1xi32>) -> tensor<200x2x3xi32>
        %234 = "stablehlo.gather"(%arg218, %233) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1251x1201xf32>, tensor<200x2x3xi32>) -> tensor<200x2xf32>
        %c_103 = stablehlo.constant dense<0> : tensor<i32>
        %235 = func.call @_where_394(%213, %234, %c_103) : (tensor<200x2xi1>, tensor<200x2xf32>, tensor<i32>) -> tensor<200x2xf32>
        %236 = stablehlo.multiply %235, %arg201 : tensor<200x2xf32>
        %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %237 = stablehlo.reduce(%236 init: %cst_104) applies stablehlo.add across dimensions = [1] : (tensor<200x2xf32>, tensor<f32>) -> tensor<200xf32>
        %c_105 = stablehlo.constant dense<1> : tensor<ui32>
        %c_106 = stablehlo.constant dense<8> : tensor<ui32>
        %238 = stablehlo.partition_id : tensor<ui32>
        %239 = stablehlo.divide %238, %c_105 : tensor<ui32>
        %240 = stablehlo.remainder %239, %c_106 : tensor<ui32>
        %241 = stablehlo.convert %240 : (tensor<ui32>) -> tensor<i32>
        %c_107 = stablehlo.constant dense<1251> : tensor<i32>
        %242 = stablehlo.multiply %241, %c_107 : tensor<i32>
        %243 = stablehlo.broadcast_in_dim %242, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %244 = stablehlo.subtract %arg202, %243 : tensor<200x1xi32>
        %c_108 = stablehlo.constant dense<0> : tensor<i32>
        %245 = stablehlo.broadcast_in_dim %c_108, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %246 = stablehlo.compare  GE, %244, %245,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
        %c_109 = stablehlo.constant dense<1251> : tensor<i32>
        %247 = stablehlo.broadcast_in_dim %c_109, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %248 = stablehlo.compare  LT, %244, %247,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
        %249 = stablehlo.and %246, %248 : tensor<200x1xi1>
        %c_110 = stablehlo.constant dense<0> : tensor<i32>
        %c_111 = stablehlo.constant dense<1250> : tensor<i32>
        %250 = func.call @clip_407(%244, %c_110, %c_111) : (tensor<200x1xi32>, tensor<i32>, tensor<i32>) -> tensor<200x1xi32>
        %c_112 = stablehlo.constant dense<0> : tensor<i32>
        %251 = stablehlo.broadcast_in_dim %c_112, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %252 = stablehlo.compare  LT, %250, %251,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
        %c_113 = stablehlo.constant dense<1251> : tensor<i32>
        %253 = stablehlo.broadcast_in_dim %c_113, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %254 = stablehlo.add %250, %253 : tensor<200x1xi32>
        %255 = stablehlo.select %252, %254, %250 : tensor<200x1xi1>, tensor<200x1xi32>
        %c_114 = stablehlo.constant dense<0> : tensor<i32>
        %256 = stablehlo.broadcast_in_dim %c_114, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %257 = stablehlo.compare  LT, %arg203, %256,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
        %c_115 = stablehlo.constant dense<1250> : tensor<i32>
        %258 = stablehlo.broadcast_in_dim %c_115, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %259 = stablehlo.add %arg203, %258 : tensor<200x1xi32>
        %260 = stablehlo.select %257, %259, %arg203 : tensor<200x1xi1>, tensor<200x1xi32>
        %c_116 = stablehlo.constant dense<0> : tensor<i32>
        %261 = stablehlo.broadcast_in_dim %c_116, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %262 = stablehlo.compare  LT, %arg204, %261,  SIGNED : (tensor<200x1xi32>, tensor<200x1xi32>) -> tensor<200x1xi1>
        %c_117 = stablehlo.constant dense<1201> : tensor<i32>
        %263 = stablehlo.broadcast_in_dim %c_117, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
        %264 = stablehlo.add %arg204, %263 : tensor<200x1xi32>
        %265 = stablehlo.select %262, %264, %arg204 : tensor<200x1xi1>, tensor<200x1xi32>
        %266 = stablehlo.broadcast_in_dim %255, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
        %267 = stablehlo.broadcast_in_dim %260, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
        %268 = stablehlo.broadcast_in_dim %265, dims = [0, 1] : (tensor<200x1xi32>) -> tensor<200x1x1xi32>
        %269 = stablehlo.concatenate %266, %267, %268, dim = 2 : (tensor<200x1x1xi32>, tensor<200x1x1xi32>, tensor<200x1x1xi32>) -> tensor<200x1x3xi32>
        %270 = "stablehlo.gather"(%arg219, %269) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1250x1201xf32>, tensor<200x1x3xi32>) -> tensor<200x1xf32>
        %c_118 = stablehlo.constant dense<0> : tensor<i32>
        %271 = func.call @_where_420(%249, %270, %c_118) : (tensor<200x1xi1>, tensor<200x1xf32>, tensor<i32>) -> tensor<200x1xf32>
        %272 = stablehlo.multiply %271, %arg205 : tensor<200x1xf32>
        %cst_119 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %273 = stablehlo.reduce(%272 init: %cst_119) applies stablehlo.add across dimensions = [1] : (tensor<200x1xf32>, tensor<f32>) -> tensor<200xf32>
        %c_120 = stablehlo.constant dense<1> : tensor<ui32>
        %c_121 = stablehlo.constant dense<8> : tensor<ui32>
        %274 = stablehlo.partition_id : tensor<ui32>
        %275 = stablehlo.divide %274, %c_120 : tensor<ui32>
        %276 = stablehlo.remainder %275, %c_121 : tensor<ui32>
        %277 = stablehlo.convert %276 : (tensor<ui32>) -> tensor<i32>
        %c_122 = stablehlo.constant dense<1251> : tensor<i32>
        %278 = stablehlo.multiply %277, %c_122 : tensor<i32>
        %279 = stablehlo.broadcast_in_dim %278, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %280 = stablehlo.subtract %arg206, %279 : tensor<200x4xi32>
        %c_123 = stablehlo.constant dense<0> : tensor<i32>
        %281 = stablehlo.broadcast_in_dim %c_123, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %282 = stablehlo.compare  GE, %280, %281,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_124 = stablehlo.constant dense<1251> : tensor<i32>
        %283 = stablehlo.broadcast_in_dim %c_124, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %284 = stablehlo.compare  LT, %280, %283,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %285 = stablehlo.and %282, %284 : tensor<200x4xi1>
        %c_125 = stablehlo.constant dense<0> : tensor<i32>
        %c_126 = stablehlo.constant dense<1250> : tensor<i32>
        %286 = func.call @clip_432(%280, %c_125, %c_126) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
        %c_127 = stablehlo.constant dense<0> : tensor<i32>
        %287 = stablehlo.broadcast_in_dim %c_127, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %288 = stablehlo.compare  LT, %286, %287,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_128 = stablehlo.constant dense<1251> : tensor<i32>
        %289 = stablehlo.broadcast_in_dim %c_128, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %290 = stablehlo.add %286, %289 : tensor<200x4xi32>
        %291 = stablehlo.select %288, %290, %286 : tensor<200x4xi1>, tensor<200x4xi32>
        %c_129 = stablehlo.constant dense<0> : tensor<i32>
        %292 = stablehlo.broadcast_in_dim %c_129, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %293 = stablehlo.compare  LT, %arg207, %292,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_130 = stablehlo.constant dense<1251> : tensor<i32>
        %294 = stablehlo.broadcast_in_dim %c_130, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %295 = stablehlo.add %arg207, %294 : tensor<200x4xi32>
        %296 = stablehlo.select %293, %295, %arg207 : tensor<200x4xi1>, tensor<200x4xi32>
        %c_131 = stablehlo.constant dense<0> : tensor<i32>
        %297 = stablehlo.broadcast_in_dim %c_131, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %298 = stablehlo.compare  LT, %arg208, %297,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_132 = stablehlo.constant dense<1200> : tensor<i32>
        %299 = stablehlo.broadcast_in_dim %c_132, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %300 = stablehlo.add %arg208, %299 : tensor<200x4xi32>
        %301 = stablehlo.select %298, %300, %arg208 : tensor<200x4xi1>, tensor<200x4xi32>
        %302 = stablehlo.broadcast_in_dim %291, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %303 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %304 = stablehlo.broadcast_in_dim %301, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %305 = stablehlo.concatenate %302, %303, %304, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
        %306 = "stablehlo.gather"(%arg220, %305) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1251x1200xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
        %c_133 = stablehlo.constant dense<0> : tensor<i32>
        %307 = func.call @_where_445(%285, %306, %c_133) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
        %308 = stablehlo.multiply %307, %arg209 : tensor<200x4xf32>
        %cst_134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %309 = stablehlo.reduce(%308 init: %cst_134) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
        %c_135 = stablehlo.constant dense<1> : tensor<ui32>
        %c_136 = stablehlo.constant dense<8> : tensor<ui32>
        %310 = stablehlo.partition_id : tensor<ui32>
        %311 = stablehlo.divide %310, %c_135 : tensor<ui32>
        %312 = stablehlo.remainder %311, %c_136 : tensor<ui32>
        %313 = stablehlo.convert %312 : (tensor<ui32>) -> tensor<i32>
        %c_137 = stablehlo.constant dense<1251> : tensor<i32>
        %314 = stablehlo.multiply %313, %c_137 : tensor<i32>
        %315 = stablehlo.broadcast_in_dim %314, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %316 = stablehlo.subtract %arg210, %315 : tensor<200x4xi32>
        %c_138 = stablehlo.constant dense<0> : tensor<i32>
        %317 = stablehlo.broadcast_in_dim %c_138, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %318 = stablehlo.compare  GE, %316, %317,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_139 = stablehlo.constant dense<1251> : tensor<i32>
        %319 = stablehlo.broadcast_in_dim %c_139, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %320 = stablehlo.compare  LT, %316, %319,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %321 = stablehlo.and %318, %320 : tensor<200x4xi1>
        %c_140 = stablehlo.constant dense<0> : tensor<i32>
        %c_141 = stablehlo.constant dense<1250> : tensor<i32>
        %322 = func.call @clip_432(%316, %c_140, %c_141) : (tensor<200x4xi32>, tensor<i32>, tensor<i32>) -> tensor<200x4xi32>
        %c_142 = stablehlo.constant dense<0> : tensor<i32>
        %323 = stablehlo.broadcast_in_dim %c_142, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %324 = stablehlo.compare  LT, %322, %323,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_143 = stablehlo.constant dense<1251> : tensor<i32>
        %325 = stablehlo.broadcast_in_dim %c_143, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %326 = stablehlo.add %322, %325 : tensor<200x4xi32>
        %327 = stablehlo.select %324, %326, %322 : tensor<200x4xi1>, tensor<200x4xi32>
        %c_144 = stablehlo.constant dense<0> : tensor<i32>
        %328 = stablehlo.broadcast_in_dim %c_144, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %329 = stablehlo.compare  LT, %arg211, %328,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_145 = stablehlo.constant dense<1250> : tensor<i32>
        %330 = stablehlo.broadcast_in_dim %c_145, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %331 = stablehlo.add %arg211, %330 : tensor<200x4xi32>
        %332 = stablehlo.select %329, %331, %arg211 : tensor<200x4xi1>, tensor<200x4xi32>
        %c_146 = stablehlo.constant dense<0> : tensor<i32>
        %333 = stablehlo.broadcast_in_dim %c_146, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %334 = stablehlo.compare  LT, %arg212, %333,  SIGNED : (tensor<200x4xi32>, tensor<200x4xi32>) -> tensor<200x4xi1>
        %c_147 = stablehlo.constant dense<1200> : tensor<i32>
        %335 = stablehlo.broadcast_in_dim %c_147, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
        %336 = stablehlo.add %arg212, %335 : tensor<200x4xi32>
        %337 = stablehlo.select %334, %336, %arg212 : tensor<200x4xi1>, tensor<200x4xi32>
        %338 = stablehlo.broadcast_in_dim %327, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %339 = stablehlo.broadcast_in_dim %332, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %340 = stablehlo.broadcast_in_dim %337, dims = [0, 1] : (tensor<200x4xi32>) -> tensor<200x4x1xi32>
        %341 = stablehlo.concatenate %338, %339, %340, dim = 2 : (tensor<200x4x1xi32>, tensor<200x4x1xi32>, tensor<200x4x1xi32>) -> tensor<200x4x3xi32>
        %342 = "stablehlo.gather"(%arg221, %341) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1251x1250x1200xf32>, tensor<200x4x3xi32>) -> tensor<200x4xf32>
        %c_148 = stablehlo.constant dense<0> : tensor<i32>
        %343 = func.call @_where_445(%321, %342, %c_148) : (tensor<200x4xi1>, tensor<200x4xf32>, tensor<i32>) -> tensor<200x4xf32>
        %344 = stablehlo.multiply %343, %arg213 : tensor<200x4xf32>
        %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %345 = stablehlo.reduce(%344 init: %cst_149) applies stablehlo.add across dimensions = [1] : (tensor<200x4xf32>, tensor<f32>) -> tensor<200xf32>
        %346 = stablehlo.slice %arg222 [0:1, 0:242400] : (tensor<1x242400xf32>) -> tensor<1x242400xf32>
        %347 = stablehlo.reshape %346 : (tensor<1x242400xf32>) -> tensor<242400xf32>
        %348 = stablehlo.slice %arg223 [0:1, 0:242400] : (tensor<1x242400xf32>) -> tensor<1x242400xf32>
        %349 = stablehlo.reshape %348 : (tensor<1x242400xf32>) -> tensor<242400xf32>
        %350 = stablehlo.broadcast_in_dim %165, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %351 = stablehlo.broadcast_in_dim %201, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %352 = stablehlo.broadcast_in_dim %237, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %353 = stablehlo.broadcast_in_dim %273, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %354 = stablehlo.broadcast_in_dim %309, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %355 = stablehlo.broadcast_in_dim %345, dims = [1] : (tensor<200xf32>) -> tensor<1x200xf32>
        %356 = stablehlo.concatenate %350, %351, %352, %353, %354, %355, dim = 0 : (tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>, tensor<1x200xf32>) -> tensor<6x200xf32>
        %cst_150 = stablehlo.constant dense<6.28318548> : tensor<f32>
        %357 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f32>) -> tensor<101xf32>
        %358 = stablehlo.multiply %357, %arg214 : tensor<101xf32>
        %359 = stablehlo.optimization_barrier %358 : tensor<101xf32>
        %360 = stablehlo.optimization_barrier %arg225 : tensor<f32>
        %361 = stablehlo.broadcast_in_dim %360, dims = [] : (tensor<f32>) -> tensor<101xf32>
        %362 = stablehlo.multiply %359, %361 : tensor<101xf32>
        %363 = stablehlo.cosine %362 : tensor<101xf32>
        %364 = stablehlo.sine %362 : tensor<101xf32>
        %cst_151 = stablehlo.constant dense<0x7F800000> : tensor<f32>
        %cst_152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %365 = stablehlo.subtract %cst_151, %cst_152 : tensor<f32>
        %cst_153 = stablehlo.constant dense<1.000000e-30> : tensor<f32>
        %366 = stablehlo.maximum %365, %cst_153 : tensor<f32>
        %cst_154 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %367 = stablehlo.subtract %arg225, %cst_154 : tensor<f32>
        %368 = stablehlo.divide %367, %366 : tensor<f32>
        %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %cst_156 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %369 = func.call @clip_462(%368, %cst_155, %cst_156) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
        %cst_157 = stablehlo.constant dense<6.28318548> : tensor<f32>
        %370 = stablehlo.multiply %cst_157, %369 : tensor<f32>
        %371 = stablehlo.cosine %370 : tensor<f32>
        %cst_158 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %372 = stablehlo.subtract %cst_158, %371 : tensor<f32>
        %cst_159 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
        %373 = stablehlo.multiply %cst_159, %372 : tensor<f32>
        %cst_160 = stablehlo.constant dense<0x7F800000> : tensor<f32>
        %374 = stablehlo.is_finite %cst_160 : (tensor<f32>) -> tensor<i1>
        %c_161 = stablehlo.constant dense<false> : tensor<i1>
        %375 = stablehlo.and %c_161, %374 : tensor<i1>
        %cst_162 = stablehlo.constant dense<0x7F800000> : tensor<f32>
        %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %376 = stablehlo.compare  GT, %cst_162, %cst_163,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        %377 = stablehlo.and %375, %376 : tensor<i1>
        %cst_164 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %378 = func.call @_where_468(%377, %373, %cst_164) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
        %cst_165 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %379 = stablehlo.maximum %378, %cst_165 : tensor<f32>
        %cst_166 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %380 = stablehlo.multiply %arg226, %cst_166 : tensor<f32>
        %cst_167 = stablehlo.constant dense<0x4D8EF3C2> : tensor<f32>
        %381 = stablehlo.multiply %380, %cst_167 : tensor<f32>
        %cst_168 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
        %382 = stablehlo.divide %381, %cst_168 : tensor<f32>
        %cst_169 = stablehlo.constant dense<6.28318548> : tensor<f32>
        %383 = stablehlo.sqrt %cst_169 : tensor<f32>
        %384 = stablehlo.divide %382, %383 : tensor<f32>
        %385 = stablehlo.multiply %379, %384 : tensor<f32>
        %c_170 = stablehlo.constant dense<0> : tensor<i32>
        %c_171 = stablehlo.constant dense<1> : tensor<i32>
        %386 = stablehlo.compare  EQ, %c_170, %c_171,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %387 = func.call @_where_471(%386, %385, %379) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
        %388 = stablehlo.broadcast_in_dim %arg215, dims = [0] : (tensor<6xf32>) -> tensor<6x1x1xf32>
        %389 = stablehlo.broadcast_in_dim %387, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
        %390 = stablehlo.multiply %389, %388 : tensor<6x1x1xf32>
        %391 = stablehlo.dot_general %356, %363, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<101xf32>) -> tensor<6x200x101xf32>
        %392 = stablehlo.transpose %391, dims = [0, 2, 1] : (tensor<6x200x101xf32>) -> tensor<6x101x200xf32>
        %393 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x101x200xf32>
        %394 = stablehlo.multiply %393, %392 : tensor<6x101x200xf32>
        %395 = stablehlo.broadcast_in_dim %387, dims = [] : (tensor<f32>) -> tensor<6x1x1xf32>
        %396 = stablehlo.multiply %395, %388 : tensor<6x1x1xf32>
        %397 = stablehlo.dot_general %356, %364, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<6x200xf32>, tensor<101xf32>) -> tensor<6x200x101xf32>
        %398 = stablehlo.transpose %397, dims = [0, 2, 1] : (tensor<6x200x101xf32>) -> tensor<6x101x200xf32>
        %399 = stablehlo.broadcast_in_dim %396, dims = [0, 1, 2] : (tensor<6x1x1xf32>) -> tensor<6x101x200xf32>
        %400 = stablehlo.multiply %399, %398 : tensor<6x101x200xf32>
        %401 = stablehlo.reshape %394 : (tensor<6x101x200xf32>) -> tensor<121200xf32>
        %c_172 = stablehlo.constant dense<121200> : tensor<i32>
        %402 = stablehlo.broadcast_in_dim %c_172, dims = [] : (tensor<i32>) -> tensor<1xi32>
        %403 = "stablehlo.scatter"(%347, %402, %401) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
        ^bb0(%arg227: tensor<f32>, %arg228: tensor<f32>):
          %412 = stablehlo.add %arg227, %arg228 : tensor<f32>
          stablehlo.return %412 : tensor<f32>
        }) : (tensor<242400xf32>, tensor<1xi32>, tensor<121200xf32>) -> tensor<242400xf32>
        %404 = stablehlo.reshape %400 : (tensor<6x101x200xf32>) -> tensor<121200xf32>
        %c_173 = stablehlo.constant dense<121200> : tensor<i32>
        %405 = stablehlo.broadcast_in_dim %c_173, dims = [] : (tensor<i32>) -> tensor<1xi32>
        %406 = "stablehlo.scatter"(%349, %405, %404) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
        ^bb0(%arg227: tensor<f32>, %arg228: tensor<f32>):
          %412 = stablehlo.add %arg227, %arg228 : tensor<f32>
          stablehlo.return %412 : tensor<f32>
        }) : (tensor<242400xf32>, tensor<1xi32>, tensor<121200xf32>) -> tensor<242400xf32>
        %407 = stablehlo.broadcast_in_dim %378, dims = [] : (tensor<f32>) -> tensor<101xf32>
        %c_174 = stablehlo.constant dense<101> : tensor<i32>
        %408 = stablehlo.broadcast_in_dim %c_174, dims = [] : (tensor<i32>) -> tensor<1xi32>
        %409 = "stablehlo.scatter"(%arg224, %408, %407) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
        ^bb0(%arg227: tensor<f32>, %arg228: tensor<f32>):
          %412 = stablehlo.add %arg227, %arg228 : tensor<f32>
          stablehlo.return %412 : tensor<f32>
        }) : (tensor<202xf32>, tensor<1xi32>, tensor<101xf32>) -> tensor<202xf32>
        %410 = stablehlo.broadcast_in_dim %403, dims = [1] : (tensor<242400xf32>) -> tensor<1x242400xf32>
        %411 = stablehlo.broadcast_in_dim %406, dims = [1] : (tensor<242400xf32>) -> tensor<1x242400xf32>
        sdy.return %410, %411, %409 : tensor<1x242400xf32>, tensor<1x242400xf32>, tensor<202xf32>
      } : (tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xi32>, tensor<200x8xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xi32>, tensor<200x2xf32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xi32>, tensor<200x1xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xi32>, tensor<200x4xf32>, tensor<101xf32>, tensor<6xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>, tensor<f32>, tensor<f32>) -> (tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>)
      stablehlo.return %129#0, %129#1, %129#2 : tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>
    }) : (tensor<i32>) -> (tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>)
    %c_59 = stablehlo.constant dense<1> : tensor<i32>
    %128 = stablehlo.add %arg188, %c_59 : tensor<i32>
    return %7#0, %8, %9, %4#0, %5, %6, %4#3, %4#4, %4#5, %4#6, %4#7, %4#8, %7#3, %7#4, %7#5, %7#6, %7#7, %7#8, %102, %110, %117, %127#0, %127#1, %127#2, %3, %128 : tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2xi32>, tensor<8x242400xf32>, tensor<8x242400xf32>, tensor<202xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @_where_84(%arg0: tensor<1251x24x1201xi1>, %arg1: tensor<1251x24x1201xf32>, %arg2: tensor<1251x24x1201xf32>) -> tensor<1251x24x1201xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1251x24x1201xi1>, tensor<1251x24x1201xf32>
    return %0 : tensor<1251x24x1201xf32>
  }
  func.func private @_where_86(%arg0: tensor<24x1250x1201xi1>, %arg1: tensor<24x1250x1201xf32>, %arg2: tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>
    return %0 : tensor<24x1250x1201xf32>
  }
  func.func private @_where_87(%arg0: tensor<24x1251x1200xi1>, %arg1: tensor<24x1251x1200xf32>, %arg2: tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>
    return %0 : tensor<24x1251x1200xf32>
  }
  func.func private @_where_94(%arg0: tensor<1251x1251x24xi1>, %arg1: tensor<1251x1251x24xf32>, %arg2: tensor<1251x1251x24xf32>) -> tensor<1251x1251x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1251x1251x24xi1>, tensor<1251x1251x24xf32>
    return %0 : tensor<1251x1251x24xf32>
  }
  func.func private @_where_100(%arg0: tensor<1251x1250x24xi1>, %arg1: tensor<1251x1250x24xf32>, %arg2: tensor<1251x1250x24xf32>) -> tensor<1251x1250x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1251x1250x24xi1>, tensor<1251x1250x24xf32>
    return %0 : tensor<1251x1250x24xf32>
  }
  func.func private @_where_105(%arg0: tensor<1251x24x1200xi1>, %arg1: tensor<1251x24x1200xf32>, %arg2: tensor<1251x24x1200xf32>) -> tensor<1251x24x1200xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1251x24x1200xi1>, tensor<1251x24x1200xf32>
    return %0 : tensor<1251x24x1200xf32>
  }
  func.func private @clip(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %arg2 : tensor<i32>
    %3 = stablehlo.minimum %2, %1 : tensor<i32>
    return %3 : tensor<i32>
  }
  func.func private @clip_132(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %arg2 : tensor<i32>
    %3 = stablehlo.minimum %2, %1 : tensor<i32>
    return %3 : tensor<i32>
  }
  func.func private @_take(%arg0: tensor<14x27x1xf32>, %arg1: tensor<14xi32>) -> tensor<14x27x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<14xi32>) -> tensor<14x1xi32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 27, 1>}> : (tensor<14x27x1xf32>, tensor<14x1xi32>) -> tensor<14x27x1xf32>
    return %1 : tensor<14x27x1xf32>
  }
  func.func private @_where_153(%arg0: tensor<14x1x1xi1>, %arg1: tensor<14x27x1xf32>, %arg2: tensor<i32>) -> tensor<14x27x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<14x1x1xi1>) -> tensor<14x27x1xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<14x27x1xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<14x27x1xi1>, tensor<14x27x1xf32>
    return %3 : tensor<14x27x1xf32>
  }
  func.func private @_take_193(%arg0: tensor<15x26x1xf32>, %arg1: tensor<15xi32>) -> tensor<15x26x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<15xi32>) -> tensor<15x1xi32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 26, 1>}> : (tensor<15x26x1xf32>, tensor<15x1xi32>) -> tensor<15x26x1xf32>
    return %1 : tensor<15x26x1xf32>
  }
  func.func private @_where_199(%arg0: tensor<15x1x1xi1>, %arg1: tensor<15x26x1xf32>, %arg2: tensor<i32>) -> tensor<15x26x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<15x1x1xi1>) -> tensor<15x26x1xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<15x26x1xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<15x26x1xi1>, tensor<15x26x1xf32>
    return %3 : tensor<15x26x1xf32>
  }
  func.func private @_take_236(%arg0: tensor<15x28x1xf32>, %arg1: tensor<15xi32>) -> tensor<15x28x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<15xi32>) -> tensor<15x1xi32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 28, 1>}> : (tensor<15x28x1xf32>, tensor<15x1xi32>) -> tensor<15x28x1xf32>
    return %1 : tensor<15x28x1xf32>
  }
  func.func private @_where_240(%arg0: tensor<15x1x1xi1>, %arg1: tensor<15x28x1xf32>, %arg2: tensor<i32>) -> tensor<15x28x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<15x1x1xi1>) -> tensor<15x28x1xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<15x28x1xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<15x28x1xi1>, tensor<15x28x1xf32>
    return %3 : tensor<15x28x1xf32>
  }
  func.func private @_take_271(%arg0: tensor<16x27x1xf32>, %arg1: tensor<16xi32>) -> tensor<16x27x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 27, 1>}> : (tensor<16x27x1xf32>, tensor<16x1xi32>) -> tensor<16x27x1xf32>
    return %1 : tensor<16x27x1xf32>
  }
  func.func private @_where_277(%arg0: tensor<16x1x1xi1>, %arg1: tensor<16x27x1xf32>, %arg2: tensor<i32>) -> tensor<16x27x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<16x1x1xi1>) -> tensor<16x27x1xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<16x27x1xf32>
    %3 = stablehlo.select %1, %arg1, %2 : tensor<16x27x1xi1>, tensor<16x27x1xf32>
    return %3 : tensor<16x27x1xf32>
  }
  func.func private @remainder(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = call @_where_299(%0, %c_0, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
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
  func.func private @_where_299(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @_where_332(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_337(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_342(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @clip_356(%arg0: tensor<200x8xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x8xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x8xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x8xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x8xi32>
    return %5 : tensor<200x8xi32>
  }
  func.func private @_where_369(%arg0: tensor<200x8xi1>, %arg1: tensor<200x8xf32>, %arg2: tensor<i32>) -> tensor<200x8xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x8xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x8xi1>, tensor<200x8xf32>
    return %2 : tensor<200x8xf32>
  }
  func.func private @clip_381(%arg0: tensor<200x2xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x2xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x2xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x2xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x2xi32>
    return %5 : tensor<200x2xi32>
  }
  func.func private @_where_394(%arg0: tensor<200x2xi1>, %arg1: tensor<200x2xf32>, %arg2: tensor<i32>) -> tensor<200x2xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x2xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x2xi1>, tensor<200x2xf32>
    return %2 : tensor<200x2xf32>
  }
  func.func private @clip_407(%arg0: tensor<200x1xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x1xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x1xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x1xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x1xi32>
    return %5 : tensor<200x1xi32>
  }
  func.func private @_where_420(%arg0: tensor<200x1xi1>, %arg1: tensor<200x1xf32>, %arg2: tensor<i32>) -> tensor<200x1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x1xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x1xi1>, tensor<200x1xf32>
    return %2 : tensor<200x1xf32>
  }
  func.func private @clip_432(%arg0: tensor<200x4xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<200x4xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<200x4xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<200x4xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<200x4xi32>
    return %5 : tensor<200x4xi32>
  }
  func.func private @_where_445(%arg0: tensor<200x4xi1>, %arg1: tensor<200x4xf32>, %arg2: tensor<i32>) -> tensor<200x4xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<200x4xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<200x4xi1>, tensor<200x4xf32>
    return %2 : tensor<200x4xf32>
  }
  func.func private @clip_462(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg1 : tensor<f32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<f32>
    %2 = stablehlo.convert %arg2 : tensor<f32>
    %3 = stablehlo.minimum %2, %1 : tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @_where_468(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @_where_471(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
}
