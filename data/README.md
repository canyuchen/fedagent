# Data

The WebShop and ALFWorld environment data are **not** bundled with this
repository — they are public but large (the full WebShop catalog alone is
~5.2 GB). Fetch them with the top-level downloader:

```bash
bash ../download_data.sh           # both environments
bash ../download_data.sh --webshop # WebShop only
```

**Shipped small variants.** Three small WebShop files are committed and back the
`webshop.use_small: true` code path (used by the default / smoke configs):

- `items_shuffle_1000.json`
- `items_ins_v2_1000.json`
- `items_human_ins.json`

The full catalog (`items_shuffle.json`, `items_ins_v2.json`) and the ALFWorld
game cache are only required for full-scale runs and are gitignored.

> TODO(author): confirm the final on-disk location each downloaded artifact must
> land in (see the paths consumed by `tools/resolve_paths.py` and the vendored
> WebShop / ALFWorld envs).
