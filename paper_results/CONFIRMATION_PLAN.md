# Fixed confirmation plan

The roster was selected from participant IDs, excluding all seven previously tested subjects, with RNG seed 37. No label distributions or model outcomes informed its selection. Test subjects are S023, S015, S025, and S029. They may have appeared in earlier training partitions, so this is a fresh evaluation roster, not a completely untouched dataset or proof of independence from pretraining.

Train and validation caps remain 200 valid windows per subject. Testing uses up to 1000 valid windows per participant. The loader consumes the original sampling RNG before independently selecting expanded test windows, preserving training and validation selection. A regression test checks this invariant. Expanded test windows are non-overlapping, but temporally dependent; do not treat all windows as independent subjects.

Architecture rank 8, learning rate 0.001, 600-update cap, patience 4, distillation coefficient 0.1 and temperature 2 are fixed before this confirmation. Three model seeds are 7, 17 and 27. Validation uses only 30, 15 and 10 Hz. Evaluate 6 Hz as the primary severe-shift condition and 12 Hz as secondary.

Five methods isolate early versus late adaptation, the effect of clean-to-degraded online distillation at both locations, and reverse distillation at the late location. Forward and reverse both use detached teacher logits and supervised cross-entropy on both views. Inference uses only one model and one view.

Report all methods and conditions, seedwise paired gains, per-subject and per-class results, fixed twelve-label F1, supported-label F1, support thresholds 10 and 20, accuracy and NLL. The primary interaction is (late distillation minus late CE) minus (early distillation minus early CE) at 6 Hz. Comparison with reverse distillation tests direction, not a separate teacher architecture. Do not change the configuration to rescue an adverse confirmation result.
