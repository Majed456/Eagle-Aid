# Dataset provenance

Earlier project conversations recorded these two Roboflow sources for the merged three-class dataset:

| Role | Workspace | Project | Version |
|---|---|---|---|
| Fire | sean-cftrp | fire-z2n21 | 1 |
| Damaged buildings and victims | sanasxd | sanas24_gazebo | 4 |

This provenance is recovered from project history. Original downloads, source YAML files, dataset license records, merge code and final split manifests were not included in the uploaded package; these identifiers have not been independently checked against current source pages.

The earlier mapping recorded Gazebo `Derrumbado` → class 0 (damaged_buildings), `Victima` → class 1 (victims), and Fire → class 2 (fire). Other labels, including the Gazebo building/safe-zone category, were excluded. The final class order is confirmed by the supplied checkpoint; the per-source mapping still needs confirmation from original merge files.

Do not infer training/test counts from the detection CSVs, or create a purported original data.yaml from memory. Historical counts and split proportions are omitted until the exact merged dataset is available.

## Reproducibility checklist

1. Recover the original dataset export YAML files and license/attribution information.
2. Recover the merge/remapping script and final data.yaml.
3. Record the train/validation/test file lists and check duplicate images across splits.
4. Recover training code and results.csv/plots; match them to the checkpoint hash.
5. Evaluate on a held-out set, recording per-class precision/recall and both mAP metrics.
