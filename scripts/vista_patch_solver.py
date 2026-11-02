"""
Patch flood_solver.py on Vista for the historical-storm runs.   Run once:  python3 vista_patch_solver.py

  1. removes the two debugging workarounds (0.5 s dt floor, 20 m depth cap) - they hid a 1000x rain-unit bug
  2. scenario names instead of integer return periods:  python3 flood_solver.py ev3
        reads runs/storm_<name>.rain, writes runs/output/<name>/
  3. default run length 24 h (86400 s)
  4. prints total rain and ABORTS if peak intensity is implausible (> 250 mm/hr) - catches unit mistakes

Makes a timestamped backup first. Every required edit must match or nothing is written.
"""
import re, shutil, sys, time, os

PATH = os.path.join(os.environ.get('SCRATCH', '.'), 'bgflood', 'flood_solver.py')
if len(sys.argv) > 1:
    PATH = sys.argv[1]

src = open(PATH, encoding='utf-8').read()
report = []
if "f'storm_{rp}.rain'" in src and 'ABORT: peak rain' in src and 'final state saved' in src:
    print('Already patched - nothing to do.')
    sys.exit(0)


def edit(name, pattern, repl, required=True, flags=0):
    global src
    new, n = re.subn(pattern, repl, src, flags=flags)
    if n == 0:
        if required:
            sys.exit(f'ABORT: required edit "{name}" did not match - nothing written.')
        report.append(f'  skip  {name} (not present)')
    else:
        src = new
        report.append(f'  done  {name} ({n}x)')


# 1. workarounds
edit('remove dt floor', r'dt = max\(min\((.*)\), 0\.5\)', r'dt = min(\1)', required=False)
edit('remove depth cap',
     r'cp\.minimum\((cp\.maximum\(H_d \+ dt \* \(r_rate - dQx/dx - dQy/dy\), 0\.0\)), 20\.0\)', r'\1',
     required=False)

# 2. scenario names
edit('rain path', r"f'storm_\{rp\}yr\.rain'", r"f'storm_{rp}.rain'")
edit('output dir', r"f'\{rp\}yr'", r"f'{rp}'")
edit('argv as string', r'rp = int\(sys\.argv\[1\]\)', r'rp = sys.argv[1]')
edit('log label start', r'=== \{rp\}-yr storm ===', r'=== {rp} storm ===', required=False)
edit('log label done', r'Done \{rp\}-yr', r'Done {rp}', required=False)

# 3. run length
edit('24 h default', r'(def run_storm\(rp, total_time=)\d+', r'\g<1>86400')

# 4. unit guard + total rain print, right after the Peak rain print
guard = (
    '\n    peak_mmhr = rain_r.max()*3600*1000\n'
    '    total_mm = float(((rain_r[1:]+rain_r[:-1])*0.5*np.diff(rain_t)).sum())*1000\n'
    '    print(f"Total rain: {total_mm:.1f} mm ({total_mm/25.4:.2f} in)", flush=True)\n'
    '    if peak_mmhr > 250:\n'
    '        print(f"ABORT: peak rain {peak_mmhr:.0f} mm/hr is implausible - rain file units wrong (solver expects m/s)", flush=True)\n'
    '        sys.exit(2)\n')
if 'ABORT: peak rain' in src:
    report.append('  skip  unit guard (already present)')
else:
    edit('unit guard', r'(\n    print\(f"Peak rain: [^\n]*\n)', lambda m: m.group(1) + guard.lstrip('\n'))

# 5. always save the final state (the deployed loop increments t after its output check, so 24 h was never written)
final_save = (
    '    if not os.path.exists(os.path.join(out_dir, f\'depth_{int(t):06d}s.tif\')):\n'
    '        H_np = cp.asnumpy(H_d); H_np[~valid] = -9999.0\n'
    '        with rasterio.open(os.path.join(out_dir, f\'depth_{int(t):06d}s.tif\'), \'w\', **profile) as dst:\n'
    '            dst.write(H_np, 1)\n'
    '        print(f"  final state saved at t={t/3600:.2f}hr", flush=True)\n\n')
if 'final state saved' in src:
    report.append('  skip  final-state save (already present)')
else:
    edit('final-state save', r"(\n)(    files = sorted\(\[f for f in os\.listdir\(out_dir\))", lambda m: m.group(1) + final_save + m.group(2))

compile(src, PATH, 'exec')          # syntax check before writing
backup = PATH + '.bak_' + time.strftime('%Y%m%d_%H%M%S')
shutil.copy2(PATH, backup)
open(PATH, 'w', encoding='utf-8').write(src)
print('\n'.join(report))
print(f'\nPatched {PATH}\nBackup  {backup}')
