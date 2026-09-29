module @jit_run_scan attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["fdtd"=8]>
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<10008x1251x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg1: tensor<10008x1250x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg2: tensor<10008x1251x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg3: tensor<10008x1250x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg4: tensor<10008x1251x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg5: tensor<10008x1250x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg6: tensor<10008x24x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg7: tensor<24x1250x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg8: tensor<24x1251x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg9: tensor<10008x1251x24xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg10: tensor<10008x1250x24xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg11: tensor<10008x24x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg12: tensor<10008x24x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg13: tensor<24x1251x1200xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg14: tensor<24x1250x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}, %arg15: tensor<10008x1250x24xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg16: tensor<10008x1251x24xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg17: tensor<10008x24x1201xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg18: tensor<0x0xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg19: tensor<0x0xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg20: tensor<0xi32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg21: tensor<0x0xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg22: tensor<0x0xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg23: tensor<0x0xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg24: tensor<0x0xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg25: tensor<0xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg26: tensor<0xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg27: tensor<0xf32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg28: tensor<f32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, []>}, %arg29: tensor<i32> {jax.buffer_donor = true, sdy.sharding = #sdy.sharding<@mesh, []>}, %arg30: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg31: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg32: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg33: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg34: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg35: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, %arg36: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg37: tensor<10008x1251x1200xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg38: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg39: tensor<10008x1250x1201xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}, %arg40: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg41: tensor<10008x1251x1201xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"fdtd"}, {}, {}]>}) -> (tensor<10008x1251x1200xf32> {jax.result_info = "result.ex"}, tensor<10008x1250x1201xf32> {jax.result_info = "result.ey"}, tensor<10008x1251x1201xf32> {jax.result_info = "result.ez"}, tensor<10008x1250x1201xf32> {jax.result_info = "result.hx"}, tensor<10008x1251x1200xf32> {jax.result_info = "result.hy"}, tensor<10008x1250x1200xf32> {jax.result_info = "result.hz"}, tensor<10008x24x1201xf32> {jax.result_info = "result.cpml_psi_h_terms[0]"}, tensor<24x1250x1201xf32> {jax.result_info = "result.cpml_psi_h_terms[1]"}, tensor<24x1251x1200xf32> {jax.result_info = "result.cpml_psi_h_terms[2]"}, tensor<10008x1251x24xf32> {jax.result_info = "result.cpml_psi_h_terms[3]"}, tensor<10008x1250x24xf32> {jax.result_info = "result.cpml_psi_h_terms[4]"}, tensor<10008x24x1200xf32> {jax.result_info = "result.cpml_psi_h_terms[5]"}, tensor<10008x24x1200xf32> {jax.result_info = "result.cpml_psi_e_terms[0]"}, tensor<24x1251x1200xf32> {jax.result_info = "result.cpml_psi_e_terms[1]"}, tensor<24x1250x1201xf32> {jax.result_info = "result.cpml_psi_e_terms[2]"}, tensor<10008x1250x24xf32> {jax.result_info = "result.cpml_psi_e_terms[3]"}, tensor<10008x1251x24xf32> {jax.result_info = "result.cpml_psi_e_terms[4]"}, tensor<10008x24x1201xf32> {jax.result_info = "result.cpml_psi_e_terms[5]"}, tensor<0x0xf32> {jax.result_info = "result.powers"}, tensor<0x0xf32> {jax.result_info = "result.timestamps"}, tensor<0xi32> {jax.result_info = "result.counts"}, tensor<0x0xf32> {jax.result_info = "result.freq_flux_re"}, tensor<0x0xf32> {jax.result_info = "result.freq_flux_im"}, tensor<0x0xf32> {jax.result_info = "result.freq_phase_re"}, tensor<0x0xf32> {jax.result_info = "result.freq_phase_im"}, tensor<0xf32> {jax.result_info = "result.dft_vec_re"}, tensor<0xf32> {jax.result_info = "result.dft_vec_im"}, tensor<0xf32> {jax.result_info = "result.dft_weight_sum"}, tensor<f32> {jax.result_info = "result.t"}, tensor<i32> {jax.result_info = "result.current_step"}) {
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
    %0 = sdy.manual_computation(%arg7) in_shardings=[<@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<24x1250x1201xf32>) {
      %c_41 = stablehlo.constant dense<1> : tensor<ui32>
      %c_42 = stablehlo.constant dense<8> : tensor<ui32>
      %10 = stablehlo.partition_id : tensor<ui32>
      %11 = stablehlo.divide %10, %c_41 : tensor<ui32>
      %12 = stablehlo.remainder %11, %c_42 : tensor<ui32>
      %13 = stablehlo.convert %12 : (tensor<ui32>) -> tensor<i32>
      %c_43 = stablehlo.constant dense<1251> : tensor<i32>
      %14 = stablehlo.multiply %13, %c_43 : tensor<i32>
      %15 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_44 = stablehlo.constant dense<12> : tensor<i32>
      %16 = stablehlo.broadcast_in_dim %c_44, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %17 = stablehlo.compare  LT, %15, %16,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_45 = stablehlo.constant dense<12> : tensor<i32>
      %18 = stablehlo.broadcast_in_dim %c_45, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %19 = stablehlo.subtract %15, %18 : tensor<24xi32>
      %c_46 = stablehlo.constant dense<10000> : tensor<i32>
      %20 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %21 = stablehlo.add %19, %20 : tensor<24xi32>
      %c_47 = stablehlo.constant dense<12> : tensor<i32>
      %22 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %23 = stablehlo.subtract %21, %22 : tensor<24xi32>
      %24 = func.call @_where(%17, %15, %23) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_48 = stablehlo.constant dense<0> : tensor<i32>
      %25 = stablehlo.broadcast_in_dim %c_48, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %26 = stablehlo.compare  GE, %24, %25,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_49 = stablehlo.constant dense<10000> : tensor<i32>
      %27 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %28 = stablehlo.compare  LT, %24, %27,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %29 = stablehlo.and %26, %28 : tensor<24xi1>
      %30 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %31 = stablehlo.compare  GE, %24, %30,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_50 = stablehlo.constant dense<1251> : tensor<i32>
      %32 = stablehlo.add %14, %c_50 : tensor<i32>
      %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %34 = stablehlo.compare  LT, %24, %33,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %35 = stablehlo.and %31, %34 : tensor<24xi1>
      %36 = stablehlo.and %29, %35 : tensor<24xi1>
      %37 = stablehlo.reshape %36 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_51 = stablehlo.constant dense<true> : tensor<i1>
      %38 = stablehlo.broadcast_in_dim %c_51, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %39 = stablehlo.and %38, %37 : tensor<24x1x1xi1>
      %40 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_52 = stablehlo.constant dense<0> : tensor<i32>
      %41 = stablehlo.broadcast_in_dim %c_52, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %42 = stablehlo.compare  GE, %40, %41,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_53 = stablehlo.constant dense<1250> : tensor<i32>
      %43 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %44 = stablehlo.compare  LT, %40, %43,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %45 = stablehlo.and %42, %44 : tensor<1250xi1>
      %46 = stablehlo.reshape %45 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %47 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1250x1xi1>
      %48 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<24x1250x1xi1>
      %49 = stablehlo.and %47, %48 : tensor<24x1250x1xi1>
      %50 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_54 = stablehlo.constant dense<0> : tensor<i32>
      %51 = stablehlo.broadcast_in_dim %c_54, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %52 = stablehlo.compare  GE, %50, %51,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_55 = stablehlo.constant dense<1201> : tensor<i32>
      %53 = stablehlo.broadcast_in_dim %c_55, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %54 = stablehlo.compare  LT, %50, %53,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %55 = stablehlo.and %52, %54 : tensor<1201xi1>
      %56 = stablehlo.reshape %55 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %57 = stablehlo.broadcast_in_dim %49, dims = [0, 1, 2] : (tensor<24x1250x1xi1>) -> tensor<24x1250x1201xi1>
      %58 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<24x1250x1201xi1>
      %59 = stablehlo.and %57, %58 : tensor<24x1250x1201xi1>
      %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %60 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f32>) -> tensor<24x1250x1201xf32>
      %61 = func.call @_where_22(%59, %arg42, %60) : (tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>, tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %62 = stablehlo.broadcast_in_dim %61, dims = [1, 2, 3] : (tensor<24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      sdy.return %62 : tensor<1x24x1250x1201xf32>
    } : (tensor<24x1250x1201xf32>) -> tensor<8x24x1250x1201xf32>
    %1 = sdy.manual_computation(%arg8) in_shardings=[<@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<24x1251x1200xf32>) {
      %c_41 = stablehlo.constant dense<1> : tensor<ui32>
      %c_42 = stablehlo.constant dense<8> : tensor<ui32>
      %10 = stablehlo.partition_id : tensor<ui32>
      %11 = stablehlo.divide %10, %c_41 : tensor<ui32>
      %12 = stablehlo.remainder %11, %c_42 : tensor<ui32>
      %13 = stablehlo.convert %12 : (tensor<ui32>) -> tensor<i32>
      %c_43 = stablehlo.constant dense<1251> : tensor<i32>
      %14 = stablehlo.multiply %13, %c_43 : tensor<i32>
      %15 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_44 = stablehlo.constant dense<12> : tensor<i32>
      %16 = stablehlo.broadcast_in_dim %c_44, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %17 = stablehlo.compare  LT, %15, %16,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_45 = stablehlo.constant dense<12> : tensor<i32>
      %18 = stablehlo.broadcast_in_dim %c_45, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %19 = stablehlo.subtract %15, %18 : tensor<24xi32>
      %c_46 = stablehlo.constant dense<10000> : tensor<i32>
      %20 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %21 = stablehlo.add %19, %20 : tensor<24xi32>
      %c_47 = stablehlo.constant dense<12> : tensor<i32>
      %22 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %23 = stablehlo.subtract %21, %22 : tensor<24xi32>
      %24 = func.call @_where(%17, %15, %23) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_48 = stablehlo.constant dense<0> : tensor<i32>
      %25 = stablehlo.broadcast_in_dim %c_48, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %26 = stablehlo.compare  GE, %24, %25,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_49 = stablehlo.constant dense<10000> : tensor<i32>
      %27 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %28 = stablehlo.compare  LT, %24, %27,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %29 = stablehlo.and %26, %28 : tensor<24xi1>
      %30 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %31 = stablehlo.compare  GE, %24, %30,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_50 = stablehlo.constant dense<1251> : tensor<i32>
      %32 = stablehlo.add %14, %c_50 : tensor<i32>
      %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %34 = stablehlo.compare  LT, %24, %33,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %35 = stablehlo.and %31, %34 : tensor<24xi1>
      %36 = stablehlo.and %29, %35 : tensor<24xi1>
      %37 = stablehlo.reshape %36 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_51 = stablehlo.constant dense<true> : tensor<i1>
      %38 = stablehlo.broadcast_in_dim %c_51, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %39 = stablehlo.and %38, %37 : tensor<24x1x1xi1>
      %40 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_52 = stablehlo.constant dense<0> : tensor<i32>
      %41 = stablehlo.broadcast_in_dim %c_52, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %42 = stablehlo.compare  GE, %40, %41,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_53 = stablehlo.constant dense<1251> : tensor<i32>
      %43 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %44 = stablehlo.compare  LT, %40, %43,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %45 = stablehlo.and %42, %44 : tensor<1251xi1>
      %46 = stablehlo.reshape %45 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %47 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1251x1xi1>
      %48 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<24x1251x1xi1>
      %49 = stablehlo.and %47, %48 : tensor<24x1251x1xi1>
      %50 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_54 = stablehlo.constant dense<0> : tensor<i32>
      %51 = stablehlo.broadcast_in_dim %c_54, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %52 = stablehlo.compare  GE, %50, %51,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_55 = stablehlo.constant dense<1200> : tensor<i32>
      %53 = stablehlo.broadcast_in_dim %c_55, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %54 = stablehlo.compare  LT, %50, %53,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %55 = stablehlo.and %52, %54 : tensor<1200xi1>
      %56 = stablehlo.reshape %55 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %57 = stablehlo.broadcast_in_dim %49, dims = [0, 1, 2] : (tensor<24x1251x1xi1>) -> tensor<24x1251x1200xi1>
      %58 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<24x1251x1200xi1>
      %59 = stablehlo.and %57, %58 : tensor<24x1251x1200xi1>
      %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %60 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f32>) -> tensor<24x1251x1200xf32>
      %61 = func.call @_where_42(%59, %arg42, %60) : (tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>, tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %62 = stablehlo.broadcast_in_dim %61, dims = [1, 2, 3] : (tensor<24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      sdy.return %62 : tensor<1x24x1251x1200xf32>
    } : (tensor<24x1251x1200xf32>) -> tensor<8x24x1251x1200xf32>
    %2 = sdy.manual_computation(%arg13) in_shardings=[<@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<24x1251x1200xf32>) {
      %c_41 = stablehlo.constant dense<1> : tensor<ui32>
      %c_42 = stablehlo.constant dense<8> : tensor<ui32>
      %10 = stablehlo.partition_id : tensor<ui32>
      %11 = stablehlo.divide %10, %c_41 : tensor<ui32>
      %12 = stablehlo.remainder %11, %c_42 : tensor<ui32>
      %13 = stablehlo.convert %12 : (tensor<ui32>) -> tensor<i32>
      %c_43 = stablehlo.constant dense<1251> : tensor<i32>
      %14 = stablehlo.multiply %13, %c_43 : tensor<i32>
      %15 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_44 = stablehlo.constant dense<12> : tensor<i32>
      %16 = stablehlo.broadcast_in_dim %c_44, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %17 = stablehlo.compare  LT, %15, %16,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_45 = stablehlo.constant dense<12> : tensor<i32>
      %18 = stablehlo.broadcast_in_dim %c_45, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %19 = stablehlo.subtract %15, %18 : tensor<24xi32>
      %c_46 = stablehlo.constant dense<10001> : tensor<i32>
      %20 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %21 = stablehlo.add %19, %20 : tensor<24xi32>
      %c_47 = stablehlo.constant dense<12> : tensor<i32>
      %22 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %23 = stablehlo.subtract %21, %22 : tensor<24xi32>
      %24 = func.call @_where(%17, %15, %23) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_48 = stablehlo.constant dense<0> : tensor<i32>
      %25 = stablehlo.broadcast_in_dim %c_48, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %26 = stablehlo.compare  GE, %24, %25,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_49 = stablehlo.constant dense<10001> : tensor<i32>
      %27 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %28 = stablehlo.compare  LT, %24, %27,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %29 = stablehlo.and %26, %28 : tensor<24xi1>
      %30 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %31 = stablehlo.compare  GE, %24, %30,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_50 = stablehlo.constant dense<1251> : tensor<i32>
      %32 = stablehlo.add %14, %c_50 : tensor<i32>
      %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %34 = stablehlo.compare  LT, %24, %33,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %35 = stablehlo.and %31, %34 : tensor<24xi1>
      %36 = stablehlo.and %29, %35 : tensor<24xi1>
      %37 = stablehlo.reshape %36 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_51 = stablehlo.constant dense<true> : tensor<i1>
      %38 = stablehlo.broadcast_in_dim %c_51, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %39 = stablehlo.and %38, %37 : tensor<24x1x1xi1>
      %40 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_52 = stablehlo.constant dense<0> : tensor<i32>
      %41 = stablehlo.broadcast_in_dim %c_52, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %42 = stablehlo.compare  GE, %40, %41,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_53 = stablehlo.constant dense<1251> : tensor<i32>
      %43 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %44 = stablehlo.compare  LT, %40, %43,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %45 = stablehlo.and %42, %44 : tensor<1251xi1>
      %46 = stablehlo.reshape %45 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %47 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1251x1xi1>
      %48 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<24x1251x1xi1>
      %49 = stablehlo.and %47, %48 : tensor<24x1251x1xi1>
      %50 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_54 = stablehlo.constant dense<0> : tensor<i32>
      %51 = stablehlo.broadcast_in_dim %c_54, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %52 = stablehlo.compare  GE, %50, %51,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_55 = stablehlo.constant dense<1200> : tensor<i32>
      %53 = stablehlo.broadcast_in_dim %c_55, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %54 = stablehlo.compare  LT, %50, %53,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %55 = stablehlo.and %52, %54 : tensor<1200xi1>
      %56 = stablehlo.reshape %55 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %57 = stablehlo.broadcast_in_dim %49, dims = [0, 1, 2] : (tensor<24x1251x1xi1>) -> tensor<24x1251x1200xi1>
      %58 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<24x1251x1200xi1>
      %59 = stablehlo.and %57, %58 : tensor<24x1251x1200xi1>
      %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %60 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f32>) -> tensor<24x1251x1200xf32>
      %61 = func.call @_where_42(%59, %arg42, %60) : (tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>, tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %62 = stablehlo.broadcast_in_dim %61, dims = [1, 2, 3] : (tensor<24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      sdy.return %62 : tensor<1x24x1251x1200xf32>
    } : (tensor<24x1251x1200xf32>) -> tensor<8x24x1251x1200xf32>
    %3 = sdy.manual_computation(%arg14) in_shardings=[<@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<24x1250x1201xf32>) {
      %c_41 = stablehlo.constant dense<1> : tensor<ui32>
      %c_42 = stablehlo.constant dense<8> : tensor<ui32>
      %10 = stablehlo.partition_id : tensor<ui32>
      %11 = stablehlo.divide %10, %c_41 : tensor<ui32>
      %12 = stablehlo.remainder %11, %c_42 : tensor<ui32>
      %13 = stablehlo.convert %12 : (tensor<ui32>) -> tensor<i32>
      %c_43 = stablehlo.constant dense<1251> : tensor<i32>
      %14 = stablehlo.multiply %13, %c_43 : tensor<i32>
      %15 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_44 = stablehlo.constant dense<12> : tensor<i32>
      %16 = stablehlo.broadcast_in_dim %c_44, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %17 = stablehlo.compare  LT, %15, %16,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_45 = stablehlo.constant dense<12> : tensor<i32>
      %18 = stablehlo.broadcast_in_dim %c_45, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %19 = stablehlo.subtract %15, %18 : tensor<24xi32>
      %c_46 = stablehlo.constant dense<10001> : tensor<i32>
      %20 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %21 = stablehlo.add %19, %20 : tensor<24xi32>
      %c_47 = stablehlo.constant dense<12> : tensor<i32>
      %22 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %23 = stablehlo.subtract %21, %22 : tensor<24xi32>
      %24 = func.call @_where(%17, %15, %23) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_48 = stablehlo.constant dense<0> : tensor<i32>
      %25 = stablehlo.broadcast_in_dim %c_48, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %26 = stablehlo.compare  GE, %24, %25,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_49 = stablehlo.constant dense<10001> : tensor<i32>
      %27 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %28 = stablehlo.compare  LT, %24, %27,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %29 = stablehlo.and %26, %28 : tensor<24xi1>
      %30 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %31 = stablehlo.compare  GE, %24, %30,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_50 = stablehlo.constant dense<1251> : tensor<i32>
      %32 = stablehlo.add %14, %c_50 : tensor<i32>
      %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %34 = stablehlo.compare  LT, %24, %33,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %35 = stablehlo.and %31, %34 : tensor<24xi1>
      %36 = stablehlo.and %29, %35 : tensor<24xi1>
      %37 = stablehlo.reshape %36 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_51 = stablehlo.constant dense<true> : tensor<i1>
      %38 = stablehlo.broadcast_in_dim %c_51, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %39 = stablehlo.and %38, %37 : tensor<24x1x1xi1>
      %40 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_52 = stablehlo.constant dense<0> : tensor<i32>
      %41 = stablehlo.broadcast_in_dim %c_52, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %42 = stablehlo.compare  GE, %40, %41,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_53 = stablehlo.constant dense<1250> : tensor<i32>
      %43 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %44 = stablehlo.compare  LT, %40, %43,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %45 = stablehlo.and %42, %44 : tensor<1250xi1>
      %46 = stablehlo.reshape %45 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %47 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1250x1xi1>
      %48 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<24x1250x1xi1>
      %49 = stablehlo.and %47, %48 : tensor<24x1250x1xi1>
      %50 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_54 = stablehlo.constant dense<0> : tensor<i32>
      %51 = stablehlo.broadcast_in_dim %c_54, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %52 = stablehlo.compare  GE, %50, %51,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_55 = stablehlo.constant dense<1201> : tensor<i32>
      %53 = stablehlo.broadcast_in_dim %c_55, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %54 = stablehlo.compare  LT, %50, %53,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %55 = stablehlo.and %52, %54 : tensor<1201xi1>
      %56 = stablehlo.reshape %55 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %57 = stablehlo.broadcast_in_dim %49, dims = [0, 1, 2] : (tensor<24x1250x1xi1>) -> tensor<24x1250x1201xi1>
      %58 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<24x1250x1201xi1>
      %59 = stablehlo.and %57, %58 : tensor<24x1250x1201xi1>
      %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %60 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f32>) -> tensor<24x1250x1201xf32>
      %61 = func.call @_where_22(%59, %arg42, %60) : (tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>, tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %62 = stablehlo.broadcast_in_dim %61, dims = [1, 2, 3] : (tensor<24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      sdy.return %62 : tensor<1x24x1250x1201xf32>
    } : (tensor<24x1250x1201xf32>) -> tensor<8x24x1250x1201xf32>
    %4 = stablehlo.iota dim = 0 : tensor<32xi32>
    %c_40 = stablehlo.constant dense<0> : tensor<i32>
    %5:77 = stablehlo.while(%iterArg = %4, %iterArg_41 = %cst, %iterArg_42 = %arg28, %iterArg_43 = %c, %iterArg_44 = %arg30, %iterArg_45 = %arg32, %iterArg_46 = %arg34, %iterArg_47 = %arg31, %iterArg_48 = %arg33, %iterArg_49 = %arg35, %iterArg_50 = %cst_0, %iterArg_51 = %cst_1, %iterArg_52 = %cst_2, %iterArg_53 = %cst_3, %iterArg_54 = %cst_4, %iterArg_55 = %cst_5, %iterArg_56 = %cst_6, %iterArg_57 = %cst_7, %iterArg_58 = %cst_8, %iterArg_59 = %cst_9, %iterArg_60 = %cst_10, %iterArg_61 = %cst_11, %iterArg_62 = %cst_12, %iterArg_63 = %cst_13, %iterArg_64 = %cst_14, %iterArg_65 = %cst_15, %iterArg_66 = %cst_16, %iterArg_67 = %cst_17, %iterArg_68 = %cst_18, %iterArg_69 = %c_19, %iterArg_70 = %c_20, %iterArg_71 = %arg36, %iterArg_72 = %arg38, %iterArg_73 = %arg40, %iterArg_74 = %arg37, %iterArg_75 = %arg39, %iterArg_76 = %arg41, %iterArg_77 = %cst_21, %iterArg_78 = %cst_22, %iterArg_79 = %cst_23, %iterArg_80 = %cst_24, %iterArg_81 = %cst_25, %iterArg_82 = %cst_26, %iterArg_83 = %cst_27, %iterArg_84 = %cst_28, %iterArg_85 = %cst_29, %iterArg_86 = %cst_30, %iterArg_87 = %cst_31, %iterArg_88 = %cst_32, %iterArg_89 = %cst_33, %iterArg_90 = %cst_34, %iterArg_91 = %cst_35, %iterArg_92 = %cst_36, %iterArg_93 = %cst_37, %iterArg_94 = %cst_38, %iterArg_95 = %c_39, %iterArg_96 = %c_40, %iterArg_97 = %arg0, %iterArg_98 = %arg1, %iterArg_99 = %arg2, %iterArg_100 = %arg3, %iterArg_101 = %arg4, %iterArg_102 = %arg5, %iterArg_103 = %arg6, %iterArg_104 = %0, %iterArg_105 = %1, %iterArg_106 = %arg9, %iterArg_107 = %arg10, %iterArg_108 = %arg11, %iterArg_109 = %arg12, %iterArg_110 = %2, %iterArg_111 = %3, %iterArg_112 = %arg15, %iterArg_113 = %arg16, %iterArg_114 = %arg17, %iterArg_115 = %arg28, %iterArg_116 = %arg29) : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<i32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<f32>, tensor<i32>
    cond {
      %c_117 = stablehlo.constant dense<32> : tensor<i32>
      %10 = stablehlo.compare  LT, %iterArg_96, %c_117,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %10 : tensor<i1>
    } do {
      %10 = stablehlo.dynamic_slice %iterArg, %iterArg_96, sizes = [1] : (tensor<32xi32>, tensor<i32>) -> tensor<1xi32>
      %11 = stablehlo.reshape %10 : (tensor<1xi32>) -> tensor<i32>
      %12:20 = func.call @closed_call(%iterArg_41, %iterArg_42, %iterArg_43, %iterArg_44, %iterArg_45, %iterArg_46, %iterArg_47, %iterArg_48, %iterArg_49, %iterArg_50, %iterArg_51, %iterArg_52, %iterArg_53, %iterArg_54, %iterArg_55, %iterArg_56, %iterArg_57, %iterArg_58, %iterArg_59, %iterArg_60, %iterArg_61, %iterArg_62, %iterArg_63, %iterArg_64, %iterArg_65, %iterArg_66, %iterArg_67, %iterArg_68, %iterArg_69, %iterArg_70, %iterArg_71, %iterArg_72, %iterArg_73, %iterArg_74, %iterArg_75, %iterArg_76, %iterArg_77, %iterArg_78, %iterArg_79, %iterArg_80, %iterArg_81, %iterArg_82, %iterArg_83, %iterArg_84, %iterArg_85, %iterArg_86, %iterArg_87, %iterArg_88, %iterArg_89, %iterArg_90, %iterArg_91, %iterArg_92, %iterArg_93, %iterArg_94, %iterArg_95, %iterArg_97, %iterArg_98, %iterArg_99, %iterArg_100, %iterArg_101, %iterArg_102, %iterArg_103, %iterArg_104, %iterArg_105, %iterArg_106, %iterArg_107, %iterArg_108, %iterArg_109, %iterArg_110, %iterArg_111, %iterArg_112, %iterArg_113, %iterArg_114, %iterArg_115, %iterArg_116, %11) : (tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<f32>, tensor<i32>, tensor<i32>) -> (tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<f32>, tensor<i32>)
      %c_117 = stablehlo.constant dense<1> : tensor<i32>
      %13 = stablehlo.add %iterArg_96, %c_117 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_41, %iterArg_42, %iterArg_43, %iterArg_44, %iterArg_45, %iterArg_46, %iterArg_47, %iterArg_48, %iterArg_49, %iterArg_50, %iterArg_51, %iterArg_52, %iterArg_53, %iterArg_54, %iterArg_55, %iterArg_56, %iterArg_57, %iterArg_58, %iterArg_59, %iterArg_60, %iterArg_61, %iterArg_62, %iterArg_63, %iterArg_64, %iterArg_65, %iterArg_66, %iterArg_67, %iterArg_68, %iterArg_69, %iterArg_70, %iterArg_71, %iterArg_72, %iterArg_73, %iterArg_74, %iterArg_75, %iterArg_76, %iterArg_77, %iterArg_78, %iterArg_79, %iterArg_80, %iterArg_81, %iterArg_82, %iterArg_83, %iterArg_84, %iterArg_85, %iterArg_86, %iterArg_87, %iterArg_88, %iterArg_89, %iterArg_90, %iterArg_91, %iterArg_92, %iterArg_93, %iterArg_94, %iterArg_95, %13, %12#0, %12#1, %12#2, %12#3, %12#4, %12#5, %12#6, %12#7, %12#8, %12#9, %12#10, %12#11, %12#12, %12#13, %12#14, %12#15, %12#16, %12#17, %12#18, %12#19 : tensor<32xi32>, tensor<f32>, tensor<f32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<6x3xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<6x5xi32>, tensor<i32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<f32>, tensor<i32>
    }
    %6 = sdy.manual_computation(%5#64) in_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] out_shardings=[<@mesh, [{}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<1x24x1250x1201xf32>) {
      %10 = stablehlo.slice %arg42 [0:1, 0:24, 0:1250, 0:1201] : (tensor<1x24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      %11 = stablehlo.reshape %10 : (tensor<1x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %12 = "stablehlo.all_reduce"(%11) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
        %13 = stablehlo.add %arg43, %arg44 : tensor<f32>
        stablehlo.return %13 : tensor<f32>
      }) : (tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      sdy.return %12 : tensor<24x1250x1201xf32>
    } : (tensor<8x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
    %7 = sdy.manual_computation(%5#65) in_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] out_shardings=[<@mesh, [{}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<1x24x1251x1200xf32>) {
      %10 = stablehlo.slice %arg42 [0:1, 0:24, 0:1251, 0:1200] : (tensor<1x24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      %11 = stablehlo.reshape %10 : (tensor<1x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %12 = "stablehlo.all_reduce"(%11) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
        %13 = stablehlo.add %arg43, %arg44 : tensor<f32>
        stablehlo.return %13 : tensor<f32>
      }) : (tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      sdy.return %12 : tensor<24x1251x1200xf32>
    } : (tensor<8x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
    %8 = sdy.manual_computation(%5#70) in_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] out_shardings=[<@mesh, [{}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<1x24x1251x1200xf32>) {
      %10 = stablehlo.slice %arg42 [0:1, 0:24, 0:1251, 0:1200] : (tensor<1x24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      %11 = stablehlo.reshape %10 : (tensor<1x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %12 = "stablehlo.all_reduce"(%11) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
        %13 = stablehlo.add %arg43, %arg44 : tensor<f32>
        stablehlo.return %13 : tensor<f32>
      }) : (tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      sdy.return %12 : tensor<24x1251x1200xf32>
    } : (tensor<8x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
    %9 = sdy.manual_computation(%5#71) in_shardings=[<@mesh, [{"fdtd"}, {}, {}, {}]>] out_shardings=[<@mesh, [{}, {}, {}]>] manual_axes={"fdtd"} (%arg42: tensor<1x24x1250x1201xf32>) {
      %10 = stablehlo.slice %arg42 [0:1, 0:24, 0:1250, 0:1201] : (tensor<1x24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      %11 = stablehlo.reshape %10 : (tensor<1x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %12 = "stablehlo.all_reduce"(%11) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
        %13 = stablehlo.add %arg43, %arg44 : tensor<f32>
        stablehlo.return %13 : tensor<f32>
      }) : (tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      sdy.return %12 : tensor<24x1250x1201xf32>
    } : (tensor<8x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
    return %5#57, %5#58, %5#59, %5#60, %5#61, %5#62, %5#63, %6, %7, %5#66, %5#67, %5#68, %5#69, %8, %9, %5#72, %5#73, %5#74, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %5#75, %5#76 : tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<24x1250x1201xf32>, tensor<24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<24x1251x1200xf32>, tensor<24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0xi32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @_where(%arg0: tensor<24xi1>, %arg1: tensor<24xi32>, %arg2: tensor<24xi32>) -> tensor<24xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24xi1>, tensor<24xi32>
    return %0 : tensor<24xi32>
  }
  func.func private @_where_22(%arg0: tensor<24x1250x1201xi1>, %arg1: tensor<24x1250x1201xf32>, %arg2: tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>
    return %0 : tensor<24x1250x1201xf32>
  }
  func.func private @_where_42(%arg0: tensor<24x1251x1200xi1>, %arg1: tensor<24x1251x1200xf32>, %arg2: tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>
    return %0 : tensor<24x1251x1200xf32>
  }
  func.func private @closed_call(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<6x3xi32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>, %arg9: tensor<0xf32>, %arg10: tensor<1x24x1xf32>, %arg11: tensor<1x24x1xf32>, %arg12: tensor<1x24x1xf32>, %arg13: tensor<24x1x1xf32>, %arg14: tensor<24x1x1xf32>, %arg15: tensor<24x1x1xf32>, %arg16: tensor<24x1x1xf32>, %arg17: tensor<24x1x1xf32>, %arg18: tensor<24x1x1xf32>, %arg19: tensor<1x1x24xf32>, %arg20: tensor<1x1x24xf32>, %arg21: tensor<1x1x24xf32>, %arg22: tensor<1x1x24xf32>, %arg23: tensor<1x1x24xf32>, %arg24: tensor<1x1x24xf32>, %arg25: tensor<1x24x1xf32>, %arg26: tensor<1x24x1xf32>, %arg27: tensor<1x24x1xf32>, %arg28: tensor<6x5xi32>, %arg29: tensor<6x3xi32>, %arg30: tensor<f32>, %arg31: tensor<f32>, %arg32: tensor<f32>, %arg33: tensor<10008x1251x1200xf32>, %arg34: tensor<10008x1250x1201xf32>, %arg35: tensor<10008x1251x1201xf32>, %arg36: tensor<1x24x1xf32>, %arg37: tensor<1x24x1xf32>, %arg38: tensor<1x24x1xf32>, %arg39: tensor<24x1x1xf32>, %arg40: tensor<24x1x1xf32>, %arg41: tensor<24x1x1xf32>, %arg42: tensor<24x1x1xf32>, %arg43: tensor<24x1x1xf32>, %arg44: tensor<24x1x1xf32>, %arg45: tensor<1x1x24xf32>, %arg46: tensor<1x1x24xf32>, %arg47: tensor<1x1x24xf32>, %arg48: tensor<1x1x24xf32>, %arg49: tensor<1x1x24xf32>, %arg50: tensor<1x1x24xf32>, %arg51: tensor<1x24x1xf32>, %arg52: tensor<1x24x1xf32>, %arg53: tensor<1x24x1xf32>, %arg54: tensor<6x5xi32>, %arg55: tensor<10008x1251x1200xf32>, %arg56: tensor<10008x1250x1201xf32>, %arg57: tensor<10008x1251x1201xf32>, %arg58: tensor<10008x1250x1201xf32>, %arg59: tensor<10008x1251x1200xf32>, %arg60: tensor<10008x1250x1200xf32>, %arg61: tensor<10008x24x1201xf32>, %arg62: tensor<8x24x1250x1201xf32>, %arg63: tensor<8x24x1251x1200xf32>, %arg64: tensor<10008x1251x24xf32>, %arg65: tensor<10008x1250x24xf32>, %arg66: tensor<10008x24x1200xf32>, %arg67: tensor<10008x24x1200xf32>, %arg68: tensor<8x24x1251x1200xf32>, %arg69: tensor<8x24x1250x1201xf32>, %arg70: tensor<10008x1250x24xf32>, %arg71: tensor<10008x1251x24xf32>, %arg72: tensor<10008x24x1201xf32>, %arg73: tensor<f32>, %arg74: tensor<i32>, %arg75: tensor<i32>) -> (tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<f32>, tensor<i32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg75, %c : tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<f32>
    %3 = stablehlo.add %arg1, %2 : tensor<f32>
    %4:9 = sdy.manual_computation(%arg2, %arg28, %arg58, %arg59, %arg60, %arg55, %arg56, %arg57, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg9, %arg9, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, []>, <@mesh, []>, <@mesh, []>, <@mesh, []>, <@mesh, []>, <@mesh, []>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>] manual_axes={"fdtd"} (%arg76: tensor<6x3xi32>, %arg77: tensor<6x5xi32>, %arg78: tensor<1251x1250x1201xf32>, %arg79: tensor<1251x1251x1200xf32>, %arg80: tensor<1251x1250x1200xf32>, %arg81: tensor<1251x1251x1200xf32>, %arg82: tensor<1251x1250x1201xf32>, %arg83: tensor<1251x1251x1201xf32>, %arg84: tensor<f32>, %arg85: tensor<f32>, %arg86: tensor<f32>, %arg87: tensor<f32>, %arg88: tensor<f32>, %arg89: tensor<f32>, %arg90: tensor<1251x24x1201xf32>, %arg91: tensor<1x24x1250x1201xf32>, %arg92: tensor<1x24x1251x1200xf32>, %arg93: tensor<1251x1251x24xf32>, %arg94: tensor<1251x1250x24xf32>, %arg95: tensor<1251x24x1200xf32>, %arg96: tensor<0xf32>, %arg97: tensor<0xf32>, %arg98: tensor<0xf32>, %arg99: tensor<1x24x1xf32>, %arg100: tensor<1x24x1xf32>, %arg101: tensor<1x24x1xf32>, %arg102: tensor<24x1x1xf32>, %arg103: tensor<24x1x1xf32>, %arg104: tensor<24x1x1xf32>, %arg105: tensor<24x1x1xf32>, %arg106: tensor<24x1x1xf32>, %arg107: tensor<24x1x1xf32>, %arg108: tensor<1x1x24xf32>, %arg109: tensor<1x1x24xf32>, %arg110: tensor<1x1x24xf32>, %arg111: tensor<1x1x24xf32>, %arg112: tensor<1x1x24xf32>, %arg113: tensor<1x1x24xf32>, %arg114: tensor<1x24x1xf32>, %arg115: tensor<1x24x1xf32>, %arg116: tensor<1x24x1xf32>) {
      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
      %c_2 = stablehlo.constant dense<8> : tensor<ui32>
      %7 = stablehlo.partition_id : tensor<ui32>
      %8 = stablehlo.divide %7, %c_1 : tensor<ui32>
      %9 = stablehlo.remainder %8, %c_2 : tensor<ui32>
      %10 = stablehlo.convert %9 : (tensor<ui32>) -> tensor<i32>
      %c_3 = stablehlo.constant dense<1251> : tensor<i32>
      %11 = stablehlo.multiply %10, %c_3 : tensor<i32>
      %c_4 = stablehlo.constant dense<0> : tensor<i32>
      %12 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %13 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %c_5 = stablehlo.constant dense<0> : tensor<i32>
      %14 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %15 = stablehlo.concatenate %12, %13, %14, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
      %16 = stablehlo.broadcast_in_dim %15, dims = [1] : (tensor<3xi32>) -> tensor<1x3xi32>
      %17 = stablehlo.concatenate %16, %arg76, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
      %18 = stablehlo.slice %arg91 [0:1, 0:24, 0:1250, 0:1201] : (tensor<1x24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      %19 = stablehlo.reshape %18 : (tensor<1x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %20 = stablehlo.slice %arg92 [0:1, 0:24, 0:1251, 0:1200] : (tensor<1x24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      %21 = stablehlo.reshape %20 : (tensor<1x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %22 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %23 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %24 = stablehlo.add %22, %23 : tensor<1251xi32>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %25 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %26 = stablehlo.compare  GE, %24, %25,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_7 = stablehlo.constant dense<10000> : tensor<i32>
      %27 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %28 = stablehlo.compare  LT, %24, %27,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %29 = stablehlo.and %26, %28 : tensor<1251xi1>
      %30 = stablehlo.reshape %29 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_8 = stablehlo.constant dense<true> : tensor<i1>
      %31 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %32 = stablehlo.and %31, %30 : tensor<1251x1x1xi1>
      %33 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_9 = stablehlo.constant dense<12> : tensor<i32>
      %34 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %35 = stablehlo.compare  LT, %33, %34,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_10 = stablehlo.constant dense<12> : tensor<i32>
      %36 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %37 = stablehlo.subtract %33, %36 : tensor<24xi32>
      %c_11 = stablehlo.constant dense<1250> : tensor<i32>
      %38 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %39 = stablehlo.add %37, %38 : tensor<24xi32>
      %c_12 = stablehlo.constant dense<12> : tensor<i32>
      %40 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %41 = stablehlo.subtract %39, %40 : tensor<24xi32>
      %42 = func.call @_where(%35, %33, %41) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_13 = stablehlo.constant dense<0> : tensor<i32>
      %43 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %44 = stablehlo.compare  GE, %42, %43,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_14 = stablehlo.constant dense<1250> : tensor<i32>
      %45 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %46 = stablehlo.compare  LT, %42, %45,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %47 = stablehlo.and %44, %46 : tensor<24xi1>
      %48 = stablehlo.reshape %47 : (tensor<24xi1>) -> tensor<1x24x1xi1>
      %49 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x24x1xi1>
      %50 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<1251x24x1xi1>
      %51 = stablehlo.and %49, %50 : tensor<1251x24x1xi1>
      %52 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_15 = stablehlo.constant dense<0> : tensor<i32>
      %53 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %54 = stablehlo.compare  GE, %52, %53,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_16 = stablehlo.constant dense<1201> : tensor<i32>
      %55 = stablehlo.broadcast_in_dim %c_16, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %56 = stablehlo.compare  LT, %52, %55,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %57 = stablehlo.and %54, %56 : tensor<1201xi1>
      %58 = stablehlo.reshape %57 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %59 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2] : (tensor<1251x24x1xi1>) -> tensor<1251x24x1201xi1>
      %60 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<1251x24x1201xi1>
      %61 = stablehlo.and %59, %60 : tensor<1251x24x1201xi1>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1251x24x1201xf32>
      %63 = func.call @_where_76(%61, %arg90, %62) : (tensor<1251x24x1201xi1>, tensor<1251x24x1201xf32>, tensor<1251x24x1201xf32>) -> tensor<1251x24x1201xf32>
      %64 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_17 = stablehlo.constant dense<12> : tensor<i32>
      %65 = stablehlo.broadcast_in_dim %c_17, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %66 = stablehlo.compare  LT, %64, %65,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_18 = stablehlo.constant dense<12> : tensor<i32>
      %67 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %68 = stablehlo.subtract %64, %67 : tensor<24xi32>
      %c_19 = stablehlo.constant dense<10000> : tensor<i32>
      %69 = stablehlo.broadcast_in_dim %c_19, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %70 = stablehlo.add %68, %69 : tensor<24xi32>
      %c_20 = stablehlo.constant dense<12> : tensor<i32>
      %71 = stablehlo.broadcast_in_dim %c_20, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %72 = stablehlo.subtract %70, %71 : tensor<24xi32>
      %73 = func.call @_where(%66, %64, %72) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_21 = stablehlo.constant dense<0> : tensor<i32>
      %74 = stablehlo.broadcast_in_dim %c_21, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %75 = stablehlo.compare  GE, %73, %74,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_22 = stablehlo.constant dense<10000> : tensor<i32>
      %76 = stablehlo.broadcast_in_dim %c_22, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %77 = stablehlo.compare  LT, %73, %76,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %78 = stablehlo.and %75, %77 : tensor<24xi1>
      %79 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %80 = stablehlo.compare  GE, %73, %79,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_23 = stablehlo.constant dense<1251> : tensor<i32>
      %81 = stablehlo.add %11, %c_23 : tensor<i32>
      %82 = stablehlo.broadcast_in_dim %81, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %83 = stablehlo.compare  LT, %73, %82,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %84 = stablehlo.and %80, %83 : tensor<24xi1>
      %85 = stablehlo.and %78, %84 : tensor<24xi1>
      %86 = stablehlo.reshape %85 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_24 = stablehlo.constant dense<true> : tensor<i1>
      %87 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %88 = stablehlo.and %87, %86 : tensor<24x1x1xi1>
      %89 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_25 = stablehlo.constant dense<0> : tensor<i32>
      %90 = stablehlo.broadcast_in_dim %c_25, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %91 = stablehlo.compare  GE, %89, %90,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_26 = stablehlo.constant dense<1250> : tensor<i32>
      %92 = stablehlo.broadcast_in_dim %c_26, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %93 = stablehlo.compare  LT, %89, %92,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %94 = stablehlo.and %91, %93 : tensor<1250xi1>
      %95 = stablehlo.reshape %94 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %96 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1250x1xi1>
      %97 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<24x1250x1xi1>
      %98 = stablehlo.and %96, %97 : tensor<24x1250x1xi1>
      %99 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_27 = stablehlo.constant dense<0> : tensor<i32>
      %100 = stablehlo.broadcast_in_dim %c_27, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %101 = stablehlo.compare  GE, %99, %100,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_28 = stablehlo.constant dense<1201> : tensor<i32>
      %102 = stablehlo.broadcast_in_dim %c_28, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %103 = stablehlo.compare  LT, %99, %102,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %104 = stablehlo.and %101, %103 : tensor<1201xi1>
      %105 = stablehlo.reshape %104 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %106 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2] : (tensor<24x1250x1xi1>) -> tensor<24x1250x1201xi1>
      %107 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<24x1250x1201xi1>
      %108 = stablehlo.and %106, %107 : tensor<24x1250x1201xi1>
      %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %109 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<24x1250x1201xf32>
      %110 = func.call @_where_78(%108, %19, %109) : (tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>, tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %111 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_30 = stablehlo.constant dense<12> : tensor<i32>
      %112 = stablehlo.broadcast_in_dim %c_30, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %113 = stablehlo.compare  LT, %111, %112,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_31 = stablehlo.constant dense<12> : tensor<i32>
      %114 = stablehlo.broadcast_in_dim %c_31, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %115 = stablehlo.subtract %111, %114 : tensor<24xi32>
      %c_32 = stablehlo.constant dense<10000> : tensor<i32>
      %116 = stablehlo.broadcast_in_dim %c_32, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %117 = stablehlo.add %115, %116 : tensor<24xi32>
      %c_33 = stablehlo.constant dense<12> : tensor<i32>
      %118 = stablehlo.broadcast_in_dim %c_33, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %119 = stablehlo.subtract %117, %118 : tensor<24xi32>
      %120 = func.call @_where(%113, %111, %119) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_34 = stablehlo.constant dense<0> : tensor<i32>
      %121 = stablehlo.broadcast_in_dim %c_34, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %122 = stablehlo.compare  GE, %120, %121,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_35 = stablehlo.constant dense<10000> : tensor<i32>
      %123 = stablehlo.broadcast_in_dim %c_35, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %124 = stablehlo.compare  LT, %120, %123,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %125 = stablehlo.and %122, %124 : tensor<24xi1>
      %126 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %127 = stablehlo.compare  GE, %120, %126,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_36 = stablehlo.constant dense<1251> : tensor<i32>
      %128 = stablehlo.add %11, %c_36 : tensor<i32>
      %129 = stablehlo.broadcast_in_dim %128, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %130 = stablehlo.compare  LT, %120, %129,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %131 = stablehlo.and %127, %130 : tensor<24xi1>
      %132 = stablehlo.and %125, %131 : tensor<24xi1>
      %133 = stablehlo.reshape %132 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_37 = stablehlo.constant dense<true> : tensor<i1>
      %134 = stablehlo.broadcast_in_dim %c_37, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %135 = stablehlo.and %134, %133 : tensor<24x1x1xi1>
      %136 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_38 = stablehlo.constant dense<0> : tensor<i32>
      %137 = stablehlo.broadcast_in_dim %c_38, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %138 = stablehlo.compare  GE, %136, %137,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_39 = stablehlo.constant dense<1251> : tensor<i32>
      %139 = stablehlo.broadcast_in_dim %c_39, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %140 = stablehlo.compare  LT, %136, %139,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %141 = stablehlo.and %138, %140 : tensor<1251xi1>
      %142 = stablehlo.reshape %141 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %143 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1251x1xi1>
      %144 = stablehlo.broadcast_in_dim %142, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<24x1251x1xi1>
      %145 = stablehlo.and %143, %144 : tensor<24x1251x1xi1>
      %146 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_40 = stablehlo.constant dense<0> : tensor<i32>
      %147 = stablehlo.broadcast_in_dim %c_40, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %148 = stablehlo.compare  GE, %146, %147,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_41 = stablehlo.constant dense<1200> : tensor<i32>
      %149 = stablehlo.broadcast_in_dim %c_41, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %150 = stablehlo.compare  LT, %146, %149,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %151 = stablehlo.and %148, %150 : tensor<1200xi1>
      %152 = stablehlo.reshape %151 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %153 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2] : (tensor<24x1251x1xi1>) -> tensor<24x1251x1200xi1>
      %154 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<24x1251x1200xi1>
      %155 = stablehlo.and %153, %154 : tensor<24x1251x1200xi1>
      %cst_42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %156 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<24x1251x1200xf32>
      %157 = func.call @_where_79(%155, %21, %156) : (tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>, tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %158 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %159 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %160 = stablehlo.add %158, %159 : tensor<1251xi32>
      %c_43 = stablehlo.constant dense<0> : tensor<i32>
      %161 = stablehlo.broadcast_in_dim %c_43, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %162 = stablehlo.compare  GE, %160, %161,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_44 = stablehlo.constant dense<10000> : tensor<i32>
      %163 = stablehlo.broadcast_in_dim %c_44, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %164 = stablehlo.compare  LT, %160, %163,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %165 = stablehlo.and %162, %164 : tensor<1251xi1>
      %166 = stablehlo.reshape %165 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_45 = stablehlo.constant dense<true> : tensor<i1>
      %167 = stablehlo.broadcast_in_dim %c_45, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %168 = stablehlo.and %167, %166 : tensor<1251x1x1xi1>
      %169 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_46 = stablehlo.constant dense<0> : tensor<i32>
      %170 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %171 = stablehlo.compare  GE, %169, %170,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_47 = stablehlo.constant dense<1251> : tensor<i32>
      %172 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %173 = stablehlo.compare  LT, %169, %172,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %174 = stablehlo.and %171, %173 : tensor<1251xi1>
      %175 = stablehlo.reshape %174 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %176 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x1251x1xi1>
      %177 = stablehlo.broadcast_in_dim %175, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<1251x1251x1xi1>
      %178 = stablehlo.and %176, %177 : tensor<1251x1251x1xi1>
      %179 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_48 = stablehlo.constant dense<12> : tensor<i32>
      %180 = stablehlo.broadcast_in_dim %c_48, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %181 = stablehlo.compare  LT, %179, %180,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_49 = stablehlo.constant dense<12> : tensor<i32>
      %182 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %183 = stablehlo.subtract %179, %182 : tensor<24xi32>
      %c_50 = stablehlo.constant dense<1200> : tensor<i32>
      %184 = stablehlo.broadcast_in_dim %c_50, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %185 = stablehlo.add %183, %184 : tensor<24xi32>
      %c_51 = stablehlo.constant dense<12> : tensor<i32>
      %186 = stablehlo.broadcast_in_dim %c_51, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %187 = stablehlo.subtract %185, %186 : tensor<24xi32>
      %188 = func.call @_where(%181, %179, %187) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_52 = stablehlo.constant dense<0> : tensor<i32>
      %189 = stablehlo.broadcast_in_dim %c_52, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %190 = stablehlo.compare  GE, %188, %189,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_53 = stablehlo.constant dense<1200> : tensor<i32>
      %191 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %192 = stablehlo.compare  LT, %188, %191,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %193 = stablehlo.and %190, %192 : tensor<24xi1>
      %194 = stablehlo.reshape %193 : (tensor<24xi1>) -> tensor<1x1x24xi1>
      %195 = stablehlo.broadcast_in_dim %178, dims = [0, 1, 2] : (tensor<1251x1251x1xi1>) -> tensor<1251x1251x24xi1>
      %196 = stablehlo.broadcast_in_dim %194, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<1251x1251x24xi1>
      %197 = stablehlo.and %195, %196 : tensor<1251x1251x24xi1>
      %cst_54 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %198 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f32>) -> tensor<1251x1251x24xf32>
      %199 = func.call @_where_86(%197, %arg93, %198) : (tensor<1251x1251x24xi1>, tensor<1251x1251x24xf32>, tensor<1251x1251x24xf32>) -> tensor<1251x1251x24xf32>
      %200 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %201 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %202 = stablehlo.add %200, %201 : tensor<1251xi32>
      %c_55 = stablehlo.constant dense<0> : tensor<i32>
      %203 = stablehlo.broadcast_in_dim %c_55, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %204 = stablehlo.compare  GE, %202, %203,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_56 = stablehlo.constant dense<10001> : tensor<i32>
      %205 = stablehlo.broadcast_in_dim %c_56, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %206 = stablehlo.compare  LT, %202, %205,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %207 = stablehlo.and %204, %206 : tensor<1251xi1>
      %208 = stablehlo.reshape %207 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_57 = stablehlo.constant dense<true> : tensor<i1>
      %209 = stablehlo.broadcast_in_dim %c_57, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %210 = stablehlo.and %209, %208 : tensor<1251x1x1xi1>
      %211 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_58 = stablehlo.constant dense<0> : tensor<i32>
      %212 = stablehlo.broadcast_in_dim %c_58, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %213 = stablehlo.compare  GE, %211, %212,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_59 = stablehlo.constant dense<1250> : tensor<i32>
      %214 = stablehlo.broadcast_in_dim %c_59, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %215 = stablehlo.compare  LT, %211, %214,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %216 = stablehlo.and %213, %215 : tensor<1250xi1>
      %217 = stablehlo.reshape %216 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %218 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x1250x1xi1>
      %219 = stablehlo.broadcast_in_dim %217, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<1251x1250x1xi1>
      %220 = stablehlo.and %218, %219 : tensor<1251x1250x1xi1>
      %221 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_60 = stablehlo.constant dense<12> : tensor<i32>
      %222 = stablehlo.broadcast_in_dim %c_60, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %223 = stablehlo.compare  LT, %221, %222,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_61 = stablehlo.constant dense<12> : tensor<i32>
      %224 = stablehlo.broadcast_in_dim %c_61, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %225 = stablehlo.subtract %221, %224 : tensor<24xi32>
      %c_62 = stablehlo.constant dense<1200> : tensor<i32>
      %226 = stablehlo.broadcast_in_dim %c_62, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %227 = stablehlo.add %225, %226 : tensor<24xi32>
      %c_63 = stablehlo.constant dense<12> : tensor<i32>
      %228 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %229 = stablehlo.subtract %227, %228 : tensor<24xi32>
      %230 = func.call @_where(%223, %221, %229) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_64 = stablehlo.constant dense<0> : tensor<i32>
      %231 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %232 = stablehlo.compare  GE, %230, %231,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_65 = stablehlo.constant dense<1200> : tensor<i32>
      %233 = stablehlo.broadcast_in_dim %c_65, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %234 = stablehlo.compare  LT, %230, %233,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %235 = stablehlo.and %232, %234 : tensor<24xi1>
      %236 = stablehlo.reshape %235 : (tensor<24xi1>) -> tensor<1x1x24xi1>
      %237 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2] : (tensor<1251x1250x1xi1>) -> tensor<1251x1250x24xi1>
      %238 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<1251x1250x24xi1>
      %239 = stablehlo.and %237, %238 : tensor<1251x1250x24xi1>
      %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %240 = stablehlo.broadcast_in_dim %cst_66, dims = [] : (tensor<f32>) -> tensor<1251x1250x24xf32>
      %241 = func.call @_where_92(%239, %arg94, %240) : (tensor<1251x1250x24xi1>, tensor<1251x1250x24xf32>, tensor<1251x1250x24xf32>) -> tensor<1251x1250x24xf32>
      %242 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %243 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %244 = stablehlo.add %242, %243 : tensor<1251xi32>
      %c_67 = stablehlo.constant dense<0> : tensor<i32>
      %245 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %246 = stablehlo.compare  GE, %244, %245,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_68 = stablehlo.constant dense<10001> : tensor<i32>
      %247 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %248 = stablehlo.compare  LT, %244, %247,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %249 = stablehlo.and %246, %248 : tensor<1251xi1>
      %250 = stablehlo.reshape %249 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_69 = stablehlo.constant dense<true> : tensor<i1>
      %251 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %252 = stablehlo.and %251, %250 : tensor<1251x1x1xi1>
      %253 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_70 = stablehlo.constant dense<12> : tensor<i32>
      %254 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %255 = stablehlo.compare  LT, %253, %254,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_71 = stablehlo.constant dense<12> : tensor<i32>
      %256 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %257 = stablehlo.subtract %253, %256 : tensor<24xi32>
      %c_72 = stablehlo.constant dense<1250> : tensor<i32>
      %258 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %259 = stablehlo.add %257, %258 : tensor<24xi32>
      %c_73 = stablehlo.constant dense<12> : tensor<i32>
      %260 = stablehlo.broadcast_in_dim %c_73, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %261 = stablehlo.subtract %259, %260 : tensor<24xi32>
      %262 = func.call @_where(%255, %253, %261) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_74 = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c_74, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %264 = stablehlo.compare  GE, %262, %263,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_75 = stablehlo.constant dense<1250> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %266 = stablehlo.compare  LT, %262, %265,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %267 = stablehlo.and %264, %266 : tensor<24xi1>
      %268 = stablehlo.reshape %267 : (tensor<24xi1>) -> tensor<1x24x1xi1>
      %269 = stablehlo.broadcast_in_dim %252, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x24x1xi1>
      %270 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<1251x24x1xi1>
      %271 = stablehlo.and %269, %270 : tensor<1251x24x1xi1>
      %272 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_76 = stablehlo.constant dense<0> : tensor<i32>
      %273 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %274 = stablehlo.compare  GE, %272, %273,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_77 = stablehlo.constant dense<1200> : tensor<i32>
      %275 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %276 = stablehlo.compare  LT, %272, %275,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %277 = stablehlo.and %274, %276 : tensor<1200xi1>
      %278 = stablehlo.reshape %277 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %279 = stablehlo.broadcast_in_dim %271, dims = [0, 1, 2] : (tensor<1251x24x1xi1>) -> tensor<1251x24x1200xi1>
      %280 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<1251x24x1200xi1>
      %281 = stablehlo.and %279, %280 : tensor<1251x24x1200xi1>
      %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %282 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f32>) -> tensor<1251x24x1200xf32>
      %283 = func.call @_where_97(%281, %arg95, %282) : (tensor<1251x24x1200xi1>, tensor<1251x24x1200xf32>, tensor<1251x24x1200xf32>) -> tensor<1251x24x1200xf32>
      %284 = stablehlo.slice %arg81 [0:1, 0:1251, 0:1200] : (tensor<1251x1251x1200xf32>) -> tensor<1x1251x1200xf32>
      %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %285 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f32>) -> tensor<1x1251x1200xf32>
      %286 = "stablehlo.collective_permute"(%284) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense_resource<__elided__> : tensor<7x2xi64>}> : (tensor<1x1251x1200xf32>) -> tensor<1x1251x1200xf32>
      %287 = stablehlo.slice %arg82 [0:1, 0:1250, 0:1201] : (tensor<1251x1250x1201xf32>) -> tensor<1x1250x1201xf32>
      %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %288 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<1x1250x1201xf32>
      %289 = "stablehlo.collective_permute"(%287) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense_resource<__elided__> : tensor<7x2xi64>}> : (tensor<1x1250x1201xf32>) -> tensor<1x1250x1201xf32>
      %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<1x1251x1201xf32>
      %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_82, dims = [] : (tensor<f32>) -> tensor<1x1251x1201xf32>
      %292:9 = stablehlo.custom_call @beamz_cuda_sharded(%arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg77, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %63, %110, %157, %199, %241, %283, %arg96, %arg97, %arg98, %17, %285, %286, %288, %289, %290, %291) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 0 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<1251x1250x1201xf32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1200xf32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1201xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1251x24x1201xf32>, tensor<24x1250x1201xf32>, tensor<24x1251x1200xf32>, tensor<1251x1251x24xf32>, tensor<1251x1250x24xf32>, tensor<1251x24x1200xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<1x1251x1200xf32>, tensor<1x1251x1200xf32>, tensor<1x1250x1201xf32>, tensor<1x1250x1201xf32>, tensor<1x1251x1201xf32>, tensor<1x1251x1201xf32>) -> (tensor<1251x1250x1201xf32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1200xf32>, tensor<1251x24x1201xf32>, tensor<24x1250x1201xf32>, tensor<24x1251x1200xf32>, tensor<1251x1251x24xf32>, tensor<1251x1250x24xf32>, tensor<1251x24x1200xf32>)
      %293 = stablehlo.broadcast_in_dim %292#4, dims = [1, 2, 3] : (tensor<24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      %294 = stablehlo.broadcast_in_dim %292#5, dims = [1, 2, 3] : (tensor<24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      sdy.return %292#0, %292#1, %292#2, %292#3, %293, %294, %292#6, %292#7, %292#8 : tensor<1251x1250x1201xf32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1200xf32>, tensor<1251x24x1201xf32>, tensor<1x24x1250x1201xf32>, tensor<1x24x1251x1200xf32>, tensor<1251x1251x24xf32>, tensor<1251x1250x24xf32>, tensor<1251x24x1200xf32>
    } : (tensor<6x3xi32>, tensor<6x5xi32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>) -> (tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>)
    %5:9 = sdy.manual_computation(%arg29, %arg54, %arg55, %arg56, %arg57, %4#0, %4#1, %4#2, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg9, %arg9, %arg9, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, []>, <@mesh, []>, <@mesh, []>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>, <@mesh, [{"fdtd"}, {}, {}]>] manual_axes={"fdtd"} (%arg76: tensor<6x3xi32>, %arg77: tensor<6x5xi32>, %arg78: tensor<1251x1251x1200xf32>, %arg79: tensor<1251x1250x1201xf32>, %arg80: tensor<1251x1251x1201xf32>, %arg81: tensor<1251x1250x1201xf32>, %arg82: tensor<1251x1251x1200xf32>, %arg83: tensor<1251x1250x1200xf32>, %arg84: tensor<f32>, %arg85: tensor<f32>, %arg86: tensor<f32>, %arg87: tensor<1251x1251x1200xf32>, %arg88: tensor<1251x1250x1201xf32>, %arg89: tensor<1251x1251x1201xf32>, %arg90: tensor<1251x24x1200xf32>, %arg91: tensor<1x24x1251x1200xf32>, %arg92: tensor<1x24x1250x1201xf32>, %arg93: tensor<1251x1250x24xf32>, %arg94: tensor<1251x1251x24xf32>, %arg95: tensor<1251x24x1201xf32>, %arg96: tensor<0xf32>, %arg97: tensor<0xf32>, %arg98: tensor<0xf32>, %arg99: tensor<1x24x1xf32>, %arg100: tensor<1x24x1xf32>, %arg101: tensor<1x24x1xf32>, %arg102: tensor<24x1x1xf32>, %arg103: tensor<24x1x1xf32>, %arg104: tensor<24x1x1xf32>, %arg105: tensor<24x1x1xf32>, %arg106: tensor<24x1x1xf32>, %arg107: tensor<24x1x1xf32>, %arg108: tensor<1x1x24xf32>, %arg109: tensor<1x1x24xf32>, %arg110: tensor<1x1x24xf32>, %arg111: tensor<1x1x24xf32>, %arg112: tensor<1x1x24xf32>, %arg113: tensor<1x1x24xf32>, %arg114: tensor<1x24x1xf32>, %arg115: tensor<1x24x1xf32>, %arg116: tensor<1x24x1xf32>) {
      %c_1 = stablehlo.constant dense<1> : tensor<ui32>
      %c_2 = stablehlo.constant dense<8> : tensor<ui32>
      %7 = stablehlo.partition_id : tensor<ui32>
      %8 = stablehlo.divide %7, %c_1 : tensor<ui32>
      %9 = stablehlo.remainder %8, %c_2 : tensor<ui32>
      %10 = stablehlo.convert %9 : (tensor<ui32>) -> tensor<i32>
      %c_3 = stablehlo.constant dense<1251> : tensor<i32>
      %11 = stablehlo.multiply %10, %c_3 : tensor<i32>
      %c_4 = stablehlo.constant dense<0> : tensor<i32>
      %12 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %13 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %c_5 = stablehlo.constant dense<0> : tensor<i32>
      %14 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<1xi32>
      %15 = stablehlo.concatenate %12, %13, %14, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
      %16 = stablehlo.broadcast_in_dim %15, dims = [1] : (tensor<3xi32>) -> tensor<1x3xi32>
      %17 = stablehlo.concatenate %16, %arg76, dim = 0 : (tensor<1x3xi32>, tensor<6x3xi32>) -> tensor<7x3xi32>
      %18 = stablehlo.slice %arg91 [0:1, 0:24, 0:1251, 0:1200] : (tensor<1x24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      %19 = stablehlo.reshape %18 : (tensor<1x24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %20 = stablehlo.slice %arg92 [0:1, 0:24, 0:1250, 0:1201] : (tensor<1x24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      %21 = stablehlo.reshape %20 : (tensor<1x24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %22 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %23 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %24 = stablehlo.add %22, %23 : tensor<1251xi32>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %25 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %26 = stablehlo.compare  GE, %24, %25,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_7 = stablehlo.constant dense<10001> : tensor<i32>
      %27 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %28 = stablehlo.compare  LT, %24, %27,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %29 = stablehlo.and %26, %28 : tensor<1251xi1>
      %30 = stablehlo.reshape %29 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_8 = stablehlo.constant dense<true> : tensor<i1>
      %31 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %32 = stablehlo.and %31, %30 : tensor<1251x1x1xi1>
      %33 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_9 = stablehlo.constant dense<12> : tensor<i32>
      %34 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %35 = stablehlo.compare  LT, %33, %34,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_10 = stablehlo.constant dense<12> : tensor<i32>
      %36 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %37 = stablehlo.subtract %33, %36 : tensor<24xi32>
      %c_11 = stablehlo.constant dense<1251> : tensor<i32>
      %38 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %39 = stablehlo.add %37, %38 : tensor<24xi32>
      %c_12 = stablehlo.constant dense<12> : tensor<i32>
      %40 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %41 = stablehlo.subtract %39, %40 : tensor<24xi32>
      %42 = func.call @_where(%35, %33, %41) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_13 = stablehlo.constant dense<0> : tensor<i32>
      %43 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %44 = stablehlo.compare  GE, %42, %43,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_14 = stablehlo.constant dense<1251> : tensor<i32>
      %45 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %46 = stablehlo.compare  LT, %42, %45,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %47 = stablehlo.and %44, %46 : tensor<24xi1>
      %48 = stablehlo.reshape %47 : (tensor<24xi1>) -> tensor<1x24x1xi1>
      %49 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x24x1xi1>
      %50 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<1251x24x1xi1>
      %51 = stablehlo.and %49, %50 : tensor<1251x24x1xi1>
      %52 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_15 = stablehlo.constant dense<0> : tensor<i32>
      %53 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %54 = stablehlo.compare  GE, %52, %53,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_16 = stablehlo.constant dense<1200> : tensor<i32>
      %55 = stablehlo.broadcast_in_dim %c_16, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %56 = stablehlo.compare  LT, %52, %55,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %57 = stablehlo.and %54, %56 : tensor<1200xi1>
      %58 = stablehlo.reshape %57 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %59 = stablehlo.broadcast_in_dim %51, dims = [0, 1, 2] : (tensor<1251x24x1xi1>) -> tensor<1251x24x1200xi1>
      %60 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<1251x24x1200xi1>
      %61 = stablehlo.and %59, %60 : tensor<1251x24x1200xi1>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1251x24x1200xf32>
      %63 = func.call @_where_97(%61, %arg90, %62) : (tensor<1251x24x1200xi1>, tensor<1251x24x1200xf32>, tensor<1251x24x1200xf32>) -> tensor<1251x24x1200xf32>
      %64 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_17 = stablehlo.constant dense<12> : tensor<i32>
      %65 = stablehlo.broadcast_in_dim %c_17, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %66 = stablehlo.compare  LT, %64, %65,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_18 = stablehlo.constant dense<12> : tensor<i32>
      %67 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %68 = stablehlo.subtract %64, %67 : tensor<24xi32>
      %c_19 = stablehlo.constant dense<10001> : tensor<i32>
      %69 = stablehlo.broadcast_in_dim %c_19, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %70 = stablehlo.add %68, %69 : tensor<24xi32>
      %c_20 = stablehlo.constant dense<12> : tensor<i32>
      %71 = stablehlo.broadcast_in_dim %c_20, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %72 = stablehlo.subtract %70, %71 : tensor<24xi32>
      %73 = func.call @_where(%66, %64, %72) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_21 = stablehlo.constant dense<0> : tensor<i32>
      %74 = stablehlo.broadcast_in_dim %c_21, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %75 = stablehlo.compare  GE, %73, %74,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_22 = stablehlo.constant dense<10001> : tensor<i32>
      %76 = stablehlo.broadcast_in_dim %c_22, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %77 = stablehlo.compare  LT, %73, %76,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %78 = stablehlo.and %75, %77 : tensor<24xi1>
      %79 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %80 = stablehlo.compare  GE, %73, %79,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_23 = stablehlo.constant dense<1251> : tensor<i32>
      %81 = stablehlo.add %11, %c_23 : tensor<i32>
      %82 = stablehlo.broadcast_in_dim %81, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %83 = stablehlo.compare  LT, %73, %82,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %84 = stablehlo.and %80, %83 : tensor<24xi1>
      %85 = stablehlo.and %78, %84 : tensor<24xi1>
      %86 = stablehlo.reshape %85 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_24 = stablehlo.constant dense<true> : tensor<i1>
      %87 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %88 = stablehlo.and %87, %86 : tensor<24x1x1xi1>
      %89 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_25 = stablehlo.constant dense<0> : tensor<i32>
      %90 = stablehlo.broadcast_in_dim %c_25, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %91 = stablehlo.compare  GE, %89, %90,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_26 = stablehlo.constant dense<1251> : tensor<i32>
      %92 = stablehlo.broadcast_in_dim %c_26, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %93 = stablehlo.compare  LT, %89, %92,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %94 = stablehlo.and %91, %93 : tensor<1251xi1>
      %95 = stablehlo.reshape %94 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %96 = stablehlo.broadcast_in_dim %88, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1251x1xi1>
      %97 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<24x1251x1xi1>
      %98 = stablehlo.and %96, %97 : tensor<24x1251x1xi1>
      %99 = stablehlo.iota dim = 0 : tensor<1200xi32>
      %c_27 = stablehlo.constant dense<0> : tensor<i32>
      %100 = stablehlo.broadcast_in_dim %c_27, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %101 = stablehlo.compare  GE, %99, %100,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %c_28 = stablehlo.constant dense<1200> : tensor<i32>
      %102 = stablehlo.broadcast_in_dim %c_28, dims = [] : (tensor<i32>) -> tensor<1200xi32>
      %103 = stablehlo.compare  LT, %99, %102,  SIGNED : (tensor<1200xi32>, tensor<1200xi32>) -> tensor<1200xi1>
      %104 = stablehlo.and %101, %103 : tensor<1200xi1>
      %105 = stablehlo.reshape %104 : (tensor<1200xi1>) -> tensor<1x1x1200xi1>
      %106 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2] : (tensor<24x1251x1xi1>) -> tensor<24x1251x1200xi1>
      %107 = stablehlo.broadcast_in_dim %105, dims = [0, 1, 2] : (tensor<1x1x1200xi1>) -> tensor<24x1251x1200xi1>
      %108 = stablehlo.and %106, %107 : tensor<24x1251x1200xi1>
      %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %109 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<24x1251x1200xf32>
      %110 = func.call @_where_79(%108, %19, %109) : (tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>, tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32>
      %111 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_30 = stablehlo.constant dense<12> : tensor<i32>
      %112 = stablehlo.broadcast_in_dim %c_30, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %113 = stablehlo.compare  LT, %111, %112,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_31 = stablehlo.constant dense<12> : tensor<i32>
      %114 = stablehlo.broadcast_in_dim %c_31, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %115 = stablehlo.subtract %111, %114 : tensor<24xi32>
      %c_32 = stablehlo.constant dense<10001> : tensor<i32>
      %116 = stablehlo.broadcast_in_dim %c_32, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %117 = stablehlo.add %115, %116 : tensor<24xi32>
      %c_33 = stablehlo.constant dense<12> : tensor<i32>
      %118 = stablehlo.broadcast_in_dim %c_33, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %119 = stablehlo.subtract %117, %118 : tensor<24xi32>
      %120 = func.call @_where(%113, %111, %119) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_34 = stablehlo.constant dense<0> : tensor<i32>
      %121 = stablehlo.broadcast_in_dim %c_34, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %122 = stablehlo.compare  GE, %120, %121,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_35 = stablehlo.constant dense<10001> : tensor<i32>
      %123 = stablehlo.broadcast_in_dim %c_35, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %124 = stablehlo.compare  LT, %120, %123,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %125 = stablehlo.and %122, %124 : tensor<24xi1>
      %126 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %127 = stablehlo.compare  GE, %120, %126,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_36 = stablehlo.constant dense<1251> : tensor<i32>
      %128 = stablehlo.add %11, %c_36 : tensor<i32>
      %129 = stablehlo.broadcast_in_dim %128, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %130 = stablehlo.compare  LT, %120, %129,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %131 = stablehlo.and %127, %130 : tensor<24xi1>
      %132 = stablehlo.and %125, %131 : tensor<24xi1>
      %133 = stablehlo.reshape %132 : (tensor<24xi1>) -> tensor<24x1x1xi1>
      %c_37 = stablehlo.constant dense<true> : tensor<i1>
      %134 = stablehlo.broadcast_in_dim %c_37, dims = [] : (tensor<i1>) -> tensor<24x1x1xi1>
      %135 = stablehlo.and %134, %133 : tensor<24x1x1xi1>
      %136 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_38 = stablehlo.constant dense<0> : tensor<i32>
      %137 = stablehlo.broadcast_in_dim %c_38, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %138 = stablehlo.compare  GE, %136, %137,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_39 = stablehlo.constant dense<1250> : tensor<i32>
      %139 = stablehlo.broadcast_in_dim %c_39, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %140 = stablehlo.compare  LT, %136, %139,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %141 = stablehlo.and %138, %140 : tensor<1250xi1>
      %142 = stablehlo.reshape %141 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %143 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2] : (tensor<24x1x1xi1>) -> tensor<24x1250x1xi1>
      %144 = stablehlo.broadcast_in_dim %142, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<24x1250x1xi1>
      %145 = stablehlo.and %143, %144 : tensor<24x1250x1xi1>
      %146 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_40 = stablehlo.constant dense<0> : tensor<i32>
      %147 = stablehlo.broadcast_in_dim %c_40, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %148 = stablehlo.compare  GE, %146, %147,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_41 = stablehlo.constant dense<1201> : tensor<i32>
      %149 = stablehlo.broadcast_in_dim %c_41, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %150 = stablehlo.compare  LT, %146, %149,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %151 = stablehlo.and %148, %150 : tensor<1201xi1>
      %152 = stablehlo.reshape %151 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %153 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2] : (tensor<24x1250x1xi1>) -> tensor<24x1250x1201xi1>
      %154 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<24x1250x1201xi1>
      %155 = stablehlo.and %153, %154 : tensor<24x1250x1201xi1>
      %cst_42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %156 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<24x1250x1201xf32>
      %157 = func.call @_where_78(%155, %21, %156) : (tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>, tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32>
      %158 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %159 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %160 = stablehlo.add %158, %159 : tensor<1251xi32>
      %c_43 = stablehlo.constant dense<0> : tensor<i32>
      %161 = stablehlo.broadcast_in_dim %c_43, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %162 = stablehlo.compare  GE, %160, %161,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_44 = stablehlo.constant dense<10001> : tensor<i32>
      %163 = stablehlo.broadcast_in_dim %c_44, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %164 = stablehlo.compare  LT, %160, %163,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %165 = stablehlo.and %162, %164 : tensor<1251xi1>
      %166 = stablehlo.reshape %165 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_45 = stablehlo.constant dense<true> : tensor<i1>
      %167 = stablehlo.broadcast_in_dim %c_45, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %168 = stablehlo.and %167, %166 : tensor<1251x1x1xi1>
      %169 = stablehlo.iota dim = 0 : tensor<1250xi32>
      %c_46 = stablehlo.constant dense<0> : tensor<i32>
      %170 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %171 = stablehlo.compare  GE, %169, %170,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %c_47 = stablehlo.constant dense<1250> : tensor<i32>
      %172 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<1250xi32>
      %173 = stablehlo.compare  LT, %169, %172,  SIGNED : (tensor<1250xi32>, tensor<1250xi32>) -> tensor<1250xi1>
      %174 = stablehlo.and %171, %173 : tensor<1250xi1>
      %175 = stablehlo.reshape %174 : (tensor<1250xi1>) -> tensor<1x1250x1xi1>
      %176 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x1250x1xi1>
      %177 = stablehlo.broadcast_in_dim %175, dims = [0, 1, 2] : (tensor<1x1250x1xi1>) -> tensor<1251x1250x1xi1>
      %178 = stablehlo.and %176, %177 : tensor<1251x1250x1xi1>
      %179 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_48 = stablehlo.constant dense<12> : tensor<i32>
      %180 = stablehlo.broadcast_in_dim %c_48, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %181 = stablehlo.compare  LT, %179, %180,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_49 = stablehlo.constant dense<12> : tensor<i32>
      %182 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %183 = stablehlo.subtract %179, %182 : tensor<24xi32>
      %c_50 = stablehlo.constant dense<1201> : tensor<i32>
      %184 = stablehlo.broadcast_in_dim %c_50, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %185 = stablehlo.add %183, %184 : tensor<24xi32>
      %c_51 = stablehlo.constant dense<12> : tensor<i32>
      %186 = stablehlo.broadcast_in_dim %c_51, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %187 = stablehlo.subtract %185, %186 : tensor<24xi32>
      %188 = func.call @_where(%181, %179, %187) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_52 = stablehlo.constant dense<0> : tensor<i32>
      %189 = stablehlo.broadcast_in_dim %c_52, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %190 = stablehlo.compare  GE, %188, %189,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_53 = stablehlo.constant dense<1201> : tensor<i32>
      %191 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %192 = stablehlo.compare  LT, %188, %191,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %193 = stablehlo.and %190, %192 : tensor<24xi1>
      %194 = stablehlo.reshape %193 : (tensor<24xi1>) -> tensor<1x1x24xi1>
      %195 = stablehlo.broadcast_in_dim %178, dims = [0, 1, 2] : (tensor<1251x1250x1xi1>) -> tensor<1251x1250x24xi1>
      %196 = stablehlo.broadcast_in_dim %194, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<1251x1250x24xi1>
      %197 = stablehlo.and %195, %196 : tensor<1251x1250x24xi1>
      %cst_54 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %198 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f32>) -> tensor<1251x1250x24xf32>
      %199 = func.call @_where_92(%197, %arg93, %198) : (tensor<1251x1250x24xi1>, tensor<1251x1250x24xf32>, tensor<1251x1250x24xf32>) -> tensor<1251x1250x24xf32>
      %200 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %201 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %202 = stablehlo.add %200, %201 : tensor<1251xi32>
      %c_55 = stablehlo.constant dense<0> : tensor<i32>
      %203 = stablehlo.broadcast_in_dim %c_55, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %204 = stablehlo.compare  GE, %202, %203,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_56 = stablehlo.constant dense<10000> : tensor<i32>
      %205 = stablehlo.broadcast_in_dim %c_56, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %206 = stablehlo.compare  LT, %202, %205,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %207 = stablehlo.and %204, %206 : tensor<1251xi1>
      %208 = stablehlo.reshape %207 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_57 = stablehlo.constant dense<true> : tensor<i1>
      %209 = stablehlo.broadcast_in_dim %c_57, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %210 = stablehlo.and %209, %208 : tensor<1251x1x1xi1>
      %211 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %c_58 = stablehlo.constant dense<0> : tensor<i32>
      %212 = stablehlo.broadcast_in_dim %c_58, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %213 = stablehlo.compare  GE, %211, %212,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_59 = stablehlo.constant dense<1251> : tensor<i32>
      %214 = stablehlo.broadcast_in_dim %c_59, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %215 = stablehlo.compare  LT, %211, %214,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %216 = stablehlo.and %213, %215 : tensor<1251xi1>
      %217 = stablehlo.reshape %216 : (tensor<1251xi1>) -> tensor<1x1251x1xi1>
      %218 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x1251x1xi1>
      %219 = stablehlo.broadcast_in_dim %217, dims = [0, 1, 2] : (tensor<1x1251x1xi1>) -> tensor<1251x1251x1xi1>
      %220 = stablehlo.and %218, %219 : tensor<1251x1251x1xi1>
      %221 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_60 = stablehlo.constant dense<12> : tensor<i32>
      %222 = stablehlo.broadcast_in_dim %c_60, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %223 = stablehlo.compare  LT, %221, %222,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_61 = stablehlo.constant dense<12> : tensor<i32>
      %224 = stablehlo.broadcast_in_dim %c_61, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %225 = stablehlo.subtract %221, %224 : tensor<24xi32>
      %c_62 = stablehlo.constant dense<1201> : tensor<i32>
      %226 = stablehlo.broadcast_in_dim %c_62, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %227 = stablehlo.add %225, %226 : tensor<24xi32>
      %c_63 = stablehlo.constant dense<12> : tensor<i32>
      %228 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %229 = stablehlo.subtract %227, %228 : tensor<24xi32>
      %230 = func.call @_where(%223, %221, %229) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_64 = stablehlo.constant dense<0> : tensor<i32>
      %231 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %232 = stablehlo.compare  GE, %230, %231,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_65 = stablehlo.constant dense<1201> : tensor<i32>
      %233 = stablehlo.broadcast_in_dim %c_65, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %234 = stablehlo.compare  LT, %230, %233,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %235 = stablehlo.and %232, %234 : tensor<24xi1>
      %236 = stablehlo.reshape %235 : (tensor<24xi1>) -> tensor<1x1x24xi1>
      %237 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2] : (tensor<1251x1251x1xi1>) -> tensor<1251x1251x24xi1>
      %238 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2] : (tensor<1x1x24xi1>) -> tensor<1251x1251x24xi1>
      %239 = stablehlo.and %237, %238 : tensor<1251x1251x24xi1>
      %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %240 = stablehlo.broadcast_in_dim %cst_66, dims = [] : (tensor<f32>) -> tensor<1251x1251x24xf32>
      %241 = func.call @_where_86(%239, %arg94, %240) : (tensor<1251x1251x24xi1>, tensor<1251x1251x24xf32>, tensor<1251x1251x24xf32>) -> tensor<1251x1251x24xf32>
      %242 = stablehlo.iota dim = 0 : tensor<1251xi32>
      %243 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %244 = stablehlo.add %242, %243 : tensor<1251xi32>
      %c_67 = stablehlo.constant dense<0> : tensor<i32>
      %245 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %246 = stablehlo.compare  GE, %244, %245,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %c_68 = stablehlo.constant dense<10000> : tensor<i32>
      %247 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i32>) -> tensor<1251xi32>
      %248 = stablehlo.compare  LT, %244, %247,  SIGNED : (tensor<1251xi32>, tensor<1251xi32>) -> tensor<1251xi1>
      %249 = stablehlo.and %246, %248 : tensor<1251xi1>
      %250 = stablehlo.reshape %249 : (tensor<1251xi1>) -> tensor<1251x1x1xi1>
      %c_69 = stablehlo.constant dense<true> : tensor<i1>
      %251 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i1>) -> tensor<1251x1x1xi1>
      %252 = stablehlo.and %251, %250 : tensor<1251x1x1xi1>
      %253 = stablehlo.iota dim = 0 : tensor<24xi32>
      %c_70 = stablehlo.constant dense<12> : tensor<i32>
      %254 = stablehlo.broadcast_in_dim %c_70, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %255 = stablehlo.compare  LT, %253, %254,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_71 = stablehlo.constant dense<12> : tensor<i32>
      %256 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %257 = stablehlo.subtract %253, %256 : tensor<24xi32>
      %c_72 = stablehlo.constant dense<1251> : tensor<i32>
      %258 = stablehlo.broadcast_in_dim %c_72, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %259 = stablehlo.add %257, %258 : tensor<24xi32>
      %c_73 = stablehlo.constant dense<12> : tensor<i32>
      %260 = stablehlo.broadcast_in_dim %c_73, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %261 = stablehlo.subtract %259, %260 : tensor<24xi32>
      %262 = func.call @_where(%255, %253, %261) : (tensor<24xi1>, tensor<24xi32>, tensor<24xi32>) -> tensor<24xi32>
      %c_74 = stablehlo.constant dense<0> : tensor<i32>
      %263 = stablehlo.broadcast_in_dim %c_74, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %264 = stablehlo.compare  GE, %262, %263,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %c_75 = stablehlo.constant dense<1251> : tensor<i32>
      %265 = stablehlo.broadcast_in_dim %c_75, dims = [] : (tensor<i32>) -> tensor<24xi32>
      %266 = stablehlo.compare  LT, %262, %265,  SIGNED : (tensor<24xi32>, tensor<24xi32>) -> tensor<24xi1>
      %267 = stablehlo.and %264, %266 : tensor<24xi1>
      %268 = stablehlo.reshape %267 : (tensor<24xi1>) -> tensor<1x24x1xi1>
      %269 = stablehlo.broadcast_in_dim %252, dims = [0, 1, 2] : (tensor<1251x1x1xi1>) -> tensor<1251x24x1xi1>
      %270 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2] : (tensor<1x24x1xi1>) -> tensor<1251x24x1xi1>
      %271 = stablehlo.and %269, %270 : tensor<1251x24x1xi1>
      %272 = stablehlo.iota dim = 0 : tensor<1201xi32>
      %c_76 = stablehlo.constant dense<0> : tensor<i32>
      %273 = stablehlo.broadcast_in_dim %c_76, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %274 = stablehlo.compare  GE, %272, %273,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %c_77 = stablehlo.constant dense<1201> : tensor<i32>
      %275 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<1201xi32>
      %276 = stablehlo.compare  LT, %272, %275,  SIGNED : (tensor<1201xi32>, tensor<1201xi32>) -> tensor<1201xi1>
      %277 = stablehlo.and %274, %276 : tensor<1201xi1>
      %278 = stablehlo.reshape %277 : (tensor<1201xi1>) -> tensor<1x1x1201xi1>
      %279 = stablehlo.broadcast_in_dim %271, dims = [0, 1, 2] : (tensor<1251x24x1xi1>) -> tensor<1251x24x1201xi1>
      %280 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 2] : (tensor<1x1x1201xi1>) -> tensor<1251x24x1201xi1>
      %281 = stablehlo.and %279, %280 : tensor<1251x24x1201xi1>
      %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %282 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f32>) -> tensor<1251x24x1201xf32>
      %283 = func.call @_where_76(%281, %arg95, %282) : (tensor<1251x24x1201xi1>, tensor<1251x24x1201xf32>, tensor<1251x24x1201xf32>) -> tensor<1251x24x1201xf32>
      %284 = stablehlo.slice %arg81 [1250:1251, 0:1250, 0:1201] : (tensor<1251x1250x1201xf32>) -> tensor<1x1250x1201xf32>
      %285 = "stablehlo.collective_permute"(%284) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense_resource<__elided__> : tensor<7x2xi64>}> : (tensor<1x1250x1201xf32>) -> tensor<1x1250x1201xf32>
      %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %286 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f32>) -> tensor<1x1250x1201xf32>
      %287 = stablehlo.slice %arg82 [1250:1251, 0:1251, 0:1200] : (tensor<1251x1251x1200xf32>) -> tensor<1x1251x1200xf32>
      %288 = "stablehlo.collective_permute"(%287) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense_resource<__elided__> : tensor<7x2xi64>}> : (tensor<1x1251x1200xf32>) -> tensor<1x1251x1200xf32>
      %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %289 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<1x1251x1200xf32>
      %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<1x1250x1200xf32>
      %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %291 = stablehlo.broadcast_in_dim %cst_82, dims = [] : (tensor<f32>) -> tensor<1x1250x1200xf32>
      %292:9 = stablehlo.custom_call @beamz_cuda_sharded(%arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg77, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %63, %110, %157, %199, %241, %283, %arg96, %arg97, %arg98, %17, %285, %286, %288, %289, %290, %291) {backend_config = "", mhlo.backend_config = {abi_version = 21 : i32, boundary_code = 3072 : i32, cuda_flags = 128 : i32, dt = 1.46363323E-16 : f32, metric_kind = 0 : i32, nterms = 6 : i32, phase = 1 : i32, resolution = 7.99999995E-8 : f32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 31, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 32, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [5], operand_index = 33, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [6], operand_index = 34, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [7], operand_index = 35, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [8], operand_index = 36, operand_tuple_indices = []>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<1251x1251x1200xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1201xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1200xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<1251x1251x1200xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1201xf32>, tensor<6x5xi32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1251x24x1200xf32>, tensor<24x1251x1200xf32>, tensor<24x1250x1201xf32>, tensor<1251x1250x24xf32>, tensor<1251x1251x24xf32>, tensor<1251x24x1201xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<7x3xi32>, tensor<1x1250x1201xf32>, tensor<1x1250x1201xf32>, tensor<1x1251x1200xf32>, tensor<1x1251x1200xf32>, tensor<1x1250x1200xf32>, tensor<1x1250x1200xf32>) -> (tensor<1251x1251x1200xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1201xf32>, tensor<1251x24x1200xf32>, tensor<24x1251x1200xf32>, tensor<24x1250x1201xf32>, tensor<1251x1250x24xf32>, tensor<1251x1251x24xf32>, tensor<1251x24x1201xf32>)
      %293 = stablehlo.broadcast_in_dim %292#4, dims = [1, 2, 3] : (tensor<24x1251x1200xf32>) -> tensor<1x24x1251x1200xf32>
      %294 = stablehlo.broadcast_in_dim %292#5, dims = [1, 2, 3] : (tensor<24x1250x1201xf32>) -> tensor<1x24x1250x1201xf32>
      sdy.return %292#0, %292#1, %292#2, %292#3, %293, %294, %292#6, %292#7, %292#8 : tensor<1251x1251x1200xf32>, tensor<1251x1250x1201xf32>, tensor<1251x1251x1201xf32>, tensor<1251x24x1200xf32>, tensor<1x24x1251x1200xf32>, tensor<1x24x1250x1201xf32>, tensor<1251x1250x24xf32>, tensor<1251x1251x24xf32>, tensor<1251x24x1201xf32>
    } : (tensor<6x3xi32>, tensor<6x5xi32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<24x1x1xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x1x24xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>, tensor<1x24x1xf32>) -> (tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>)
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %6 = stablehlo.add %arg74, %c_0 : tensor<i32>
    return %5#0, %5#1, %5#2, %4#0, %4#1, %4#2, %4#3, %4#4, %4#5, %4#6, %4#7, %4#8, %5#3, %5#4, %5#5, %5#6, %5#7, %5#8, %3, %6 : tensor<10008x1251x1200xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1201xf32>, tensor<10008x1250x1201xf32>, tensor<10008x1251x1200xf32>, tensor<10008x1250x1200xf32>, tensor<10008x24x1201xf32>, tensor<8x24x1250x1201xf32>, tensor<8x24x1251x1200xf32>, tensor<10008x1251x24xf32>, tensor<10008x1250x24xf32>, tensor<10008x24x1200xf32>, tensor<10008x24x1200xf32>, tensor<8x24x1251x1200xf32>, tensor<8x24x1250x1201xf32>, tensor<10008x1250x24xf32>, tensor<10008x1251x24xf32>, tensor<10008x24x1201xf32>, tensor<f32>, tensor<i32>
  }
  func.func private @_where_76(%arg0: tensor<1251x24x1201xi1>, %arg1: tensor<1251x24x1201xf32>, %arg2: tensor<1251x24x1201xf32>) -> tensor<1251x24x1201xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1251x24x1201xi1>, tensor<1251x24x1201xf32>
    return %0 : tensor<1251x24x1201xf32>
  }
  func.func private @_where_78(%arg0: tensor<24x1250x1201xi1>, %arg1: tensor<24x1250x1201xf32>, %arg2: tensor<24x1250x1201xf32>) -> tensor<24x1250x1201xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x1250x1201xi1>, tensor<24x1250x1201xf32>
    return %0 : tensor<24x1250x1201xf32>
  }
  func.func private @_where_79(%arg0: tensor<24x1251x1200xi1>, %arg1: tensor<24x1251x1200xf32>, %arg2: tensor<24x1251x1200xf32>) -> tensor<24x1251x1200xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<24x1251x1200xi1>, tensor<24x1251x1200xf32>
    return %0 : tensor<24x1251x1200xf32>
  }
  func.func private @_where_86(%arg0: tensor<1251x1251x24xi1>, %arg1: tensor<1251x1251x24xf32>, %arg2: tensor<1251x1251x24xf32>) -> tensor<1251x1251x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1251x1251x24xi1>, tensor<1251x1251x24xf32>
    return %0 : tensor<1251x1251x24xf32>
  }
  func.func private @_where_92(%arg0: tensor<1251x1250x24xi1>, %arg1: tensor<1251x1250x24xf32>, %arg2: tensor<1251x1250x24xf32>) -> tensor<1251x1250x24xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1251x1250x24xi1>, tensor<1251x1250x24xf32>
    return %0 : tensor<1251x1250x24xf32>
  }
  func.func private @_where_97(%arg0: tensor<1251x24x1200xi1>, %arg1: tensor<1251x24x1200xf32>, %arg2: tensor<1251x24x1200xf32>) -> tensor<1251x24x1200xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1251x24x1200xi1>, tensor<1251x24x1200xf32>
    return %0 : tensor<1251x24x1200xf32>
  }
}
