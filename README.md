# Does the brain keep track of which words are still possible?

Analysis code for a study of cohort effects in intracranial recordings of natural speech, using the public *Podcast*
ECoG dataset (OpenNeuro ds005574): nine patients, one 30-minute English podcast, 1,268 usable electrodes.

An encoding model predicts high-gamma activity from the speech. Three cohort quantities per speech sound (log cohort
size, cohort surprisal, distance from the uniqueness point) are isolated by shuffling only their values among speech
sounds matched on phoneme identity, position in the word and word length, and the real assignment is tested against
205 such shuffles of the same story.

## The analysis, in order

| Script | What it does |
|---|---|
| `align_phonemes.py` | measures the onset of every speech sound from the audio |
| `whisper_past_only_features.py` | Whisper features that use past audio only, and a 500-ms control |
| `fit_encoding_model.py` | fits the model for one feature set, one condition and every patient |
| `permutation_test.py` |  the cohort values against 205 matched shuffles |
| `permutation_distribution.py` | the statistic of every assignment, for the group and per patient |
| `simulation_false_positives.py` | why that test, and not a test across patients |
| `cohort_contribution.py` | the contribution against five shuffled sets, with the planted-signal check |
| `other_shuffles.py` | two other shuffles of the same values |
| `model_free_electrodes.py` | the contribution on electrodes chosen without the model |
| `deep_models.py` | the contribution with GPT-2 or Whisper added |
| `whisper_past_only.py` | the same with a Whisper restricted to past audio, and what the test can recover |
| `anatomy.py` | where the contribution lies |
| `effect_size.py` | its size as a share of prediction accuracy |
| `prediction_accuracy.py` | how well each model predicts |
| `shuffle_properties.py` | what the matched shuffle exchanges |
| `alignment_quality.py` | how precise the measured onsets are |
| `tables.py` | the two tables of the paper |

`common.py` holds the shared helpers. `model/` holds the encoding model itself: banded ridge regression with one
penalty per feature group (`banded_ridge.py`), lagged design matrices and event series (`encoding.py`), and the
features built from the dataset (`features.py`).

## Run

```
pip install numpy pandas scipy scikit-learn pyarrow h5py mne
export PODCAST_ROOT=/path/to/the/dataset   # the folder holding data/ds005574
./run_all.sh                               # every fit, then every number and table
python -m pytest tests                     # unit tests on synthetic data
```

## Settings

Penalty search over 60 random candidates up to 1e7 per feature group, word and sound events of 50 ms, measured
phoneme onsets, 21 lags over one second, five contiguous cross-validation blocks, fixed seeds. `run_all.sh` sets
these through the environment.

## Data

OpenNeuro ds005574 (CC0). The recordings are not in this repository.
