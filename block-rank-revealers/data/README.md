# Data

The Perdigão experiments use the NCAR/EOL *Quality Controlled 5-minute ISFS surface flux
data, geographic coordinate, tilt corrected (Perdigão), Version 1.2*,
[doi:10.26023/ZDMJ-D1TY-FG14](https://doi.org/10.26023/ZDMJ-D1TY-FG14). The data are not
redistributed here.

1. From the dataset page, order or download the daily NetCDF files for the intensive
   operational period, `isfs_qc_tiltcor_20170501.nc` to `isfs_qc_tiltcor_20170615.nc`
   (46 files), and unpack them into one folder, for example `data/nc/`.
2. Extract the channels used in the paper into one compressed file (needs `netCDF4`,
   `pip install -e ".[data]"`):

   ```bash
   python -m blockrr.perdigao data/nc data/perdigao_iop.npz --heights 2m 10m
   ```

   This reads the 46 files and writes 13,248 five-minute records of 850 channels
   (`var|height|site`) with the tower coordinates.

`blockrr.perdigao.load` then builds the three tower problems of the paper from that file.
