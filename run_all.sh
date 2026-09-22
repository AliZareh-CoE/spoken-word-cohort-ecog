#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export PENALTY_MAX_LOG10=7 PENALTY_CANDIDATES=60 EVENT_WINDOW_50MS=1 PHONEME_ONSETS=realigned
fit() { OPENBLAS_NUM_THREADS=4 python fit_encoding_model.py --variant "$1" --cond "$2" ${3:+--plant $3} --jobs "${JOBS:-9}"; }

python align_phonemes.py
python whisper_past_only_features.py
for c in real $(seq -f "matched%g" 1 205); do fit spectrogram "$c"; done
for c in $(seq -f "position_length%g" 1 5) $(seq -f "word_repeat%g" 1 5); do fit spectrogram "$c"; done
for v in gpt2_current gpt2_previous whisper_untrained whisper whisper_past whisper_past_500ms; do for c in real $(seq -f "matched%g" 1 5); do fit "$v" "$c"; done; done
for v in spectrogram whisper_past; do for c in real $(seq -f "matched%g" 1 5); do PLANTED_SIGNAL_STRATA=phoneme fit "$v" "$c" 0.04; done; done

python permutation_test.py && python permutation_distribution.py && python simulation_false_positives.py
python cohort_contribution.py && python other_shuffles.py && python model_free_electrodes.py
python deep_models.py && python whisper_past_only.py
python anatomy.py && python effect_size.py && python prediction_accuracy.py
python shuffle_properties.py && python alignment_quality.py
python tables.py
