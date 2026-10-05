Compile-time probes: each file calls one `#[noinline]` kernel wrapper from
`../../temporal.fut` with a dummy context. Run from futhark/ with a symlink
or copies of yee.fut and temporal.fut next to them:

    timeout 120 futhark dev -v -e shell.fut > /dev/null   # time per pass
    futhark dev shell.fut | wc -l                         # unsimplified IR size

Times on 2026-10-04: core 10 s, shell 35 s, dft 6 s.
