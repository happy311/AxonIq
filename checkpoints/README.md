# checkpoints/

Drop your **3** trained model checkpoints in this folder — one per slice
orientation (axial, coronal, sagittal). Nothing else needs to change; the
loader (`api/mri_engine/checkpoints.py`) reads the orientation from *inside*
each file, not from its filename, so any 3 filenames work as long as each
checkpoint is one of the 3 orientations and none repeats.

Example:
```
checkpoints/
├── fold0_axial.pt
├── fold0_coronal.pt
└── fold0_sagittal.pt
```

## Expected file format

Each `.pt` file must be a `torch.save()`'d dict with these keys (this is the
same format the training notebooks already write):

```python
{
    "model_state":          <state_dict for the smp.Unet resnet34+SCSE architecture>,
    "orientation":           "axial" | "coronal" | "sagittal",
    "modalities":             ["FLAIR"],
    "in_channels":            7,          # (2 * STACK_HALF + 1) for FLAIR-only
    "fold":                   0,          # optional, metadata only
    "validation_fold_dice":   0.42,       # optional, metadata only
}
```

- **`modalities` must be exactly `["FLAIR"]`** — this deployment is FLAIR-only,
  same as the original server. A checkpoint trained on a different modality
  set will be rejected at startup with a clear error.
- **Exactly one checkpoint per orientation is required.** Fewer than 3, more
  than 3, or two files claiming the same orientation will fail startup with
  a message telling you exactly what's missing/duplicated.

## Where to get 3 checkpoints from a bigger fold×orientation set

If your training notebooks saved a full fold×orientation grid (e.g.
`fold0_axial.pt` … `fold4_sagittal.pt`), just pick your best fold for each
orientation (e.g. by `validation_fold_dice`) and copy only those 3 files
here — the app doesn't need the rest.

## After adding the files

Startup logs will show either:
```
[MRI Checkpoints] loaded fold0_axial.pt (orientation=axial, fold=0, val_dice=0.4123)
[MRI Checkpoints] loaded fold0_coronal.pt (orientation=coronal, fold=0, val_dice=0.3987)
[MRI Checkpoints] loaded fold0_sagittal.pt (orientation=sagittal, fold=0, val_dice=0.4056)
[MRI Engine] Ready — 3 models loaded, CPU-only.
```
or a startup error telling you exactly what's wrong. Check `GET /health` on
the running app — the `mri_engine` block reports `ready: true/false` and any
`init_error`.
