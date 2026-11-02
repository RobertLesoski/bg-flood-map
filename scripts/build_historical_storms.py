"""
Build six historical 24-hour rainfall blocks for the Vista flood solver from public gauge data.

Source: Bowling Green-Warren County Regional Airport ASOS (KBWG), hourly precipitation
        Iowa Environmental Mesonet archive (NOAA/FAA ASOS) - https://mesonet.agron.iastate.edu

Outputs (in OUT_DIR):
  storm_ev1.rain ... storm_ev6.rain   solver input: line 1 = N, then "seconds  rate_m_per_s"
  storms.json                         metadata for the viewer (labels, totals, hourly depths)

The Vista solver reads rain values as ALREADY m/s (rain_r.append(rr)), so this writes m/s.
Rain is applied uniformly over the domain (single gauge) - a simplification.

Usage:  python build_historical_storms.py            (stdlib only)
"""
import csv, datetime as dt, json, os, urllib.request

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'storms_out')
CACHE   = os.path.join(OUT_DIR, 'bwg_hourly.csv')
IEM_URL = ('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=BWG&data=p01i'
           '&year1=1996&month1=1&day1=1&year2=2025&month2=12&day2=31&tz=Etc/UTC'
           '&format=onlycomma&latlon=no&elev=no&missing=M&trace=0.0001&direct=no&report_type=3')

# id, label, 24-h window start (UTC), note
EVENTS = [
    ('ev1', 'Light all-day rain',        '2013-12-06 00:00', 'Light rain falling in 18 of 24 hours'),
    ('ev2', 'Steady moderate rain',      '2015-03-04 00:00', 'Steady rain in 23 of 24 hours'),
    ('ev3', 'Heavy storm',               '2022-06-06 15:00', 'Heavy rain, one of the top 20 24-hour totals since 1996'),
    ('ev4', 'April 2025 flood',          '2025-04-03 13:00', 'Day 1-2 of the April 2-6, 2025 historic Bowling Green flood (NWS Louisville)'),
    ('ev5', 'Remnants of Hurricane Harvey', '2017-09-01 02:00', 'Harvey remnants, Sept 1, 2017 - daily rainfall record at Bowling Green (NWS Louisville)'),
    ('ev6', 'May 2010 record flood',     '2010-05-01 18:00', 'May 1-2, 2010 flood; NWS puts Warren County near 10 in. over two days - gauge record has gaps'),
]
MAX_FILL_GAP_H = 3          # linearly interpolate missing runs up to this long when flanked by data
IN_TO_M = 0.0254


def load_series():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(CACHE):
        print('Downloading KBWG hourly precipitation from IEM ...')
        urllib.request.urlretrieve(IEM_URL, CACHE)
    series = {}
    with open(CACHE, newline='') as fh:
        for r in csv.DictReader(fh):
            v = r['p01i']
            if v in ('M', ''):
                continue
            t = dt.datetime.strptime(r['valid'], '%Y-%m-%d %H:%M').replace(minute=0)
            series[t] = max(series.get(t, 0.0), float(v))
    return series


def window(series, start):
    """24 hourly depths (inches) starting at `start`, with short gaps interpolated."""
    vals = [series.get(start + dt.timedelta(hours=h)) for h in range(24)]
    filled = unfilled = 0
    i = 0
    while i < 24:
        if vals[i] is not None:
            i += 1
            continue
        j = i
        while j < 24 and vals[j] is None:
            j += 1
        n = j - i
        prev = vals[i - 1] if i > 0 else series.get(start + dt.timedelta(hours=i - 1))
        nxt = vals[j] if j < 24 else series.get(start + dt.timedelta(hours=j))
        if n <= MAX_FILL_GAP_H and prev is not None and nxt is not None:
            for k in range(n):
                vals[i + k] = prev + (nxt - prev) * (k + 1) / (n + 1)
            filled += n
        else:
            for k in range(n):
                vals[i + k] = 0.0
            unfilled += n
        i = j
    return vals, filled, unfilled


def write_rain_file(path, hourly_in):
    """Hourly depth (in) -> piecewise-constant intensity in m/s, linear interp between breakpoints."""
    pts = []
    for h, d_in in enumerate(hourly_in):
        rate = d_in * IN_TO_M / 3600.0
        pts.append((h * 3600.0, rate))
        pts.append((h * 3600.0 + 3599.0, rate))
    pts.append((24 * 3600.0, 0.0))
    with open(path, 'w') as fh:
        fh.write(f'{len(pts)}\n')
        for t, r in pts:
            fh.write(f'{t:.1f} {r:.12e}\n')


def read_back_total_mm(path):
    with open(path) as fh:
        n = int(fh.readline())
        pts = [tuple(map(float, fh.readline().split())) for _ in range(n)]
    area = sum((pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2 for i in range(n - 1))
    return area * 1000.0


def main():
    series = load_series()
    meta = []
    print(f"{'id':<4} {'start (UTC)':<17} {'total in':>8} {'peak in/h':>9} {'filled':>6} {'gaps':>5}  file check")
    for eid, label, start_s, note in EVENTS:
        start = dt.datetime.strptime(start_s, '%Y-%m-%d %H:%M')
        hourly, filled, unfilled = window(series, start)
        total_in = sum(hourly)
        path = os.path.join(OUT_DIR, f'storm_{eid}.rain')
        write_rain_file(path, hourly)
        back_mm = read_back_total_mm(path)
        err = abs(back_mm - total_in * 25.4) / (total_in * 25.4)
        assert err < 0.005, f'{eid}: rain file integrates to {back_mm:.1f} mm, expected {total_in*25.4:.1f}'
        print(f'{eid:<4} {start_s:<17} {total_in:8.2f} {max(hourly):9.2f} {filled:6d} {unfilled:5d}  {back_mm:.1f} mm OK')
        cum, c = [], 0.0
        for d in hourly:
            c += d
            cum.append(round(c, 3))
        meta.append({
            'id': eid, 'label': label, 'note': note, 'start_utc': start_s,
            'total_in': round(total_in, 2), 'peak_in_per_hr': round(max(hourly), 2),
            'hourly_in': [round(d, 3) for d in hourly], 'cum_in': cum,
            'gaps_interpolated_h': filled, 'gaps_unfilled_h': unfilled,
            'source': 'KBWG ASOS hourly precipitation, Iowa Environmental Mesonet'})
    with open(os.path.join(OUT_DIR, 'storms.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print('\nWrote', OUT_DIR)


if __name__ == '__main__':
    main()
