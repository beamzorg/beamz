"""End-to-end test suite through the real MCP stdio protocol for BeamZ MCP Server."""
import asyncio, json, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call(s, name, **kw):
    r = await s.call_tool(name, kw)
    if getattr(r, "is_error", getattr(r, "isError", False)):
        return "ERROR: " + "".join(getattr(c, "text", "") for c in r.content)
    sc = getattr(r, "structured_content", getattr(r, "structuredContent", None))
    if sc is not None:
        return sc.get("result", sc)
    blocks = [getattr(c, "text", "") for c in r.content]
    if len(blocks) == 1:
        return json.loads(blocks[0]) if blocks[0][:1] in "{[" else blocks[0]
    return [json.loads(b) if b[:1] in "{[" else b for b in blocks]

async def main():
    print("=" * 60)
    print("Running BeamZ MCP Server End-to-End Test Suite")
    print("=" * 60)

    params = StdioServerParameters(command=sys.executable, args=["server.py"])
    async with stdio_client(params) as (rd, wr), ClientSession(rd, wr) as s:
        await s.initialize()
        tools = [t.name for t in (await s.list_tools()).tools]
        print(f"\n[Test 1] Discovered {len(tools)} MCP Tools:")
        for t in tools:
            print(f"  - {t}")
        assert "add_circle" in tools
        assert "add_ring" in tools
        assert "add_polygon" in tools
        assert "add_gaussian_source" in tools
        assert "add_time_monitor" in tools
        assert "plot_field_snapshot" in tools

        # --- Test 2D Backward Compatibility ---
        print("\n[Test 2] 2D Scene with Circle, Ring, and Polygon:")
        sid_2d = await call(s, "create_scene", domain=dict(width_um=8, height_um=8, resolution_um=0.04))
        # Add Circle
        r_c = await call(s, "add_circle", scene_id=sid_2d, circle=dict(name="disc", x_um=2.5, y_um=2.5, radius_um=0.8, index=3.48))
        print(" ", r_c)
        # Add Ring
        r_rg = await call(s, "add_ring", scene_id=sid_2d, ring=dict(name="ring_res", x_um=5.5, y_um=5.5, inner_radius_um=0.7, outer_radius_um=1.2, index=3.48))
        print(" ", r_rg)
        # Add Polygon
        r_p = await call(s, "add_polygon", scene_id=sid_2d, polygon=dict(name="triangle", vertices=[[2.0, 5.0], [3.5, 5.0], [2.8, 6.8]], index=3.48))
        print(" ", r_p)
        
        # Add Gaussian source
        r_gs = await call(s, "add_gaussian_source", scene_id=sid_2d, source=dict(name="g_src", x_um=1.5, y_um=2.5, waist_radius_um=0.5, direction="+x"))
        print(" ", r_gs)
        
        # Add Time monitor
        r_tm = await call(s, "add_time_monitor", scene_id=sid_2d, monitor=dict(name="time_snap", components=["Ez"], interval=10))
        print(" ", r_tm)

        print("  Running simulation with wait=True...")
        res_geom = await call(s, "run_simulation", scene_id=sid_2d, wait=True, timeout_s=30)
        print(f"  Status: {res_geom['status']}, Elapsed: {res_geom.get('elapsed_s')}s")
        assert res_geom["status"] == "done"
        
        # Verify time data
        tdata = await call(s, "get_time_data", job_id=res_geom["job_id"], monitor="time_snap")
        print(f"  Time snapshots captured: {tdata['num_snapshots']} frames, shape: {tdata['snapshot_shape']}")
        assert tdata["num_snapshots"] > 0

        # Plot field snapshot
        img_snap = await s.call_tool("plot_field_snapshot", dict(job_id=res_geom["job_id"], monitor="time_snap", component="Ez", step_index=-1))
        print(f"  Field snapshot rendered: content type: {img_snap.content[0].type}")

        # --- Test 3D Mode Analysis ---
        print("\n[Test 3] 3D Waveguide with ModeMonitor:")
        sid_3d = await call(s, "create_scene", domain=dict(width_um=4, height_um=2.0, depth_um=2.0, resolution_um=0.05))
        await call(s, "add_structure", scene_id=sid_3d, structure=dict(
            name="si_wg", x_um=0, y_um=0.75, z_um=0.75, width_um=4, height_um=0.5, depth_um=0.5, index=2.5
        ))
        await call(s, "add_source", scene_id=sid_3d, source=dict(
            name="mode_src", x_um=1.0, y_um=1.0, z_um=1.0, span_um=1.5, span_z_um=1.5
        ))
        await call(s, "add_mode_monitor", scene_id=sid_3d, monitor=dict(
            name="m_out", x_um=3.0, y_um=1.0, z_um=1.0, span_um=1.5, span_z_um=1.5
        ))

        modes_3d = await call(s, "solve_modes", scene_id=sid_3d)
        print(f"  Pre-solved mode: n_eff = {modes_3d[0]['neff_real']} (guided: {modes_3d[0]['is_guided']})")
        assert modes_3d[0]["is_guided"] is True

        res_3d = await call(s, "run_simulation", scene_id=sid_3d, wait=True, timeout_s=60)
        print(f"  3D simulation status: {res_3d['status']}, Elapsed: {res_3d.get('elapsed_s')}s")
        assert res_3d["status"] == "done"

        # --- Test 4: Single-Shot simulate_device Tool ---
        print("\n[Test 4] Single-Shot simulate_device Execution:")
        res_single = await call(s, "simulate_device",
            domain=dict(width_um=8, height_um=4, resolution_um=0.04),
            structures=[dict(name="wg", x_um=0, y_um=1.75, width_um=8, height_um=0.5, index=3.48)],
            sources=[dict(name="src", x_um=1.5, y_um=2.0, span_um=2.0)],
            monitors=[
                dict(name="ref", x_um=2.0, y_um=2.0, span_um=1.8),
                dict(name="out", x_um=6.5, y_um=2.0, span_um=1.8),
            ],
            timeout_s=30,
        )
        print(f"  simulate_device success: {res_single.get('success')}, status: {res_single.get('status')}")
        print(f"  Transmission summary: {res_single.get('transmission_summary')}")
        assert res_single.get("success") is True
        assert res_single.get("status") == "done"

        # --- Test 5: Prompts Discovery ---
        print("\n[Test 5] MCP Prompts Discovery:")
        prompts = [p.name for p in (await s.list_prompts()).prompts]
        print(f"  Discovered {len(prompts)} prompts: {prompts}")
        assert "design_y_splitter" in prompts
        assert "design_straight_waveguide" in prompts
        assert "design_microring_resonator" in prompts
        assert "design_wdm" in prompts

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
