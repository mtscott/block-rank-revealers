"""The Perdigão tower array as a column-block matrix.

Data: NCAR/EOL quality-controlled 5-minute ISFS surface-flux data, geographic coordinate,
tilt corrected (Perdigão), version 1.2, https://doi.org/10.26023/ZDMJ-D1TY-FG14: one
NetCDF file per day, ``isfs_qc_tiltcor_YYYYMMDD.nc``.

Two steps:

1. ``extract`` (needs netCDF4) pulls the channels we use from the daily files into one
   compact .npz:

       python -m blockrr.perdigao path/to/nc_dir data/perdigao_iop.npz --heights 2m 10m

2. ``load`` builds a problem from that .npz: one block per tower, holding that tower's
   10 m sonic channels (``MEANS`` = u, v, w, tc; ``TURB`` adds the second moments).
   Towers with a channel present in fewer than 90% of the records are left out, records
   in which a kept tower has a gap are dropped, and every column is standardized.

In the 5-minute product the 10 m sonic channels and their second moments exist on 47 of
the 48 towers; air temperature and humidity on 18; pressure only at 2 m on 14.  So a block
that is the same on every tower has to come from the 10 m sonic.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

MEANS = ("u", "v", "w", "tc")
TURB = ("u", "v", "w", "tc", "w'w'", "u'w'", "w'tc'", "tc'tc'")

_FIRST = re.compile(r"^(u|v|w|tc|T|RH|P|spd|dir|h2o|co2)_(\d+m)_([a-z]+\d+)$")
_SECOND = re.compile(r"^(u|v|w|tc)_(u|v|w|tc)__(\d+m)_([a-z]+\d+)$")


def extract(src: str, out: str, *, start: str = "20170501", end: str = "20170615",
            heights=("2m", "10m", "20m")) -> None:
    """Collect first moments (u v w tc T RH P spd dir h2o co2), the sonic second moments
    (ISFS names x_y__<height>_<site>, stored as x'y') and tower coordinates from the
    daily files in ``src`` into ``out`` (.npz).  Columns are "var|height|site"; time is
    epoch seconds; a variable missing on a day becomes NaN.  The default window, the
    intensive operational period 1 May - 15 June 2017, gives 13,248 records."""
    from netCDF4 import Dataset  # optional dependency

    keep_h = set(heights)
    files = sorted(f for f in glob.glob(os.path.join(src, "isfs_qc_tiltcor_*.nc"))
                   if start <= os.path.basename(f)[16:24] <= end)
    if not files:
        raise FileNotFoundError(f"no isfs_qc_tiltcor_*.nc files in {src}")
    cols, times, latlon, n_prev = {}, [], {}, 0
    for f in files:
        ds = Dataset(f)
        t = np.asarray(ds.variables["time"][:], float)
        base = ds.variables["time"].units.split("since")[1].strip()[:19]
        times.append(np.datetime64(base.replace(" ", "T")) + (t * 1e3).astype("timedelta64[ms]"))
        n, seen = t.size, set()
        for nm, var in ds.variables.items():
            m1, m2 = _FIRST.match(nm), _SECOND.match(nm)
            if m1 and m1.group(2) in keep_h:
                key = f"{m1.group(1)}|{m1.group(2)}|{m1.group(3)}"
            elif m2 and m2.group(3) in keep_h:
                key = f"{m2.group(1)}'{m2.group(2)}'|{m2.group(3)}|{m2.group(4)}"
            elif nm.startswith(("latitude_", "longitude_")):
                latlon[nm] = float(np.ma.filled(var[:], np.nan).ravel()[0])
                continue
            else:
                continue
            # keep the mask: np.asarray() would turn masked cells into the 1e37 _FillValue
            arr = np.ma.filled(var[:].astype(np.float32), np.nan).ravel()[:n]
            arr[np.abs(arr) >= 1e36] = np.nan
            cols.setdefault(key, [np.full(n_prev, np.nan, np.float32)]).append(arr)
            seen.add(key)
        for key in cols:
            if key not in seen:
                cols[key].append(np.full(n, np.nan, np.float32))
        n_prev += n
        ds.close()
    ks = sorted(cols)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, time=np.concatenate(times).astype("datetime64[s]").astype(np.int64),
                        keys=np.array(ks), data=np.stack([np.concatenate(cols[k]) for k in ks], 1),
                        ll_keys=np.array(sorted(latlon)), ll_vals=np.array([latlon[k] for k in sorted(latlon)]),
                        files=np.array([os.path.basename(f) for f in files]))
    print(len(files), "files", n_prev, "records", len(ks), "columns ->", out)


def load(npz_path: str, channels=MEANS, height: str = "10m", min_complete: float = 0.90) -> dict:
    """The tower-block matrix for ``channels``: dict(A, b, names, latlon, t0, t1, m, dropped_rows)."""
    z = np.load(npz_path)
    keys = [str(k) for k in z["keys"]]
    X = z["data"].astype(np.float64)
    X[np.abs(X) >= 1e36] = np.nan          # NetCDF _FillValue (1e37) means missing
    idx = {k: i for i, k in enumerate(keys)}
    sites = sorted({k.split("|")[2] for k in keys})
    cols, names = {}, []
    for s in sites:
        ii = [idx.get(f"{c}|{height}|{s}") for c in channels]
        if None in ii:
            continue
        ok = np.isfinite(X[:, ii]).all(1)
        if ok.mean() >= min_complete:
            cols[s] = ii
            names.append(s)
    rows = np.all([np.isfinite(X[:, cols[s]]).all(1) for s in names], axis=0)
    A = np.concatenate([X[rows][:, cols[s]] for s in names], axis=1)
    A = (A - A.mean(0)) / np.where(A.std(0) > 0, A.std(0), 1.0)
    ll = dict(zip([str(k) for k in z["ll_keys"]], z["ll_vals"]))
    latlon = [(ll.get(f"latitude_{s}"), ll.get(f"longitude_{s}")) for s in names]
    t = z["time"][rows]
    return {"A": A, "b": len(channels), "names": names, "latlon": latlon,
            "t0": int(t[0]), "t1": int(t[-1]), "m": A.shape[0],
            "dropped_rows": int((~rows).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract the Perdigão ISFS channels into one .npz")
    ap.add_argument("src", help="folder with the isfs_qc_tiltcor_YYYYMMDD.nc files")
    ap.add_argument("out", help="output .npz")
    ap.add_argument("--start", default="20170501")
    ap.add_argument("--end", default="20170615")
    ap.add_argument("--heights", nargs="+", default=["2m", "10m", "20m"])
    a = ap.parse_args(argv)
    extract(a.src, a.out, start=a.start, end=a.end, heights=a.heights)


if __name__ == "__main__":
    main()
