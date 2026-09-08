# Microbiome task corrections (2026-09-08)

MGnify tasks now retain curated biological group identities rather than splitting
individual profiles independently. The grouping map records public source
accessions, provenance and the rule for each task. Profiles belonging to the same
participant, source sample or applicable household are grouped; identical numeric
profiles also join their biological groups. Cached data must contain the current
grouping version and map digest before they can be reused.

`gut-cirrhosis` now uses MetAML's retained Qin cohort
(`Quin_gut_liver_cirrhosis`): 118 cases and 114 controls, all stool, with 232 distinct
participant identities. The previous task mixed cohorts and body sites and allowed
repeated participants in training and testing. Its scores are superseded.

The raw cohort retains all 288,347 binary markers. The >=10% prevalence filter is
fitted only on the training rows, after any training-size subsampling and before
the feature cap, then applied unchanged to test rows. Participant identifiers are
retained for grouped splits. Prepared-cache versioning prevents old matrices from
bypassing the corrected preparation.

Results produced with the previous microbiome splits require recomputation. Do
not compare old and new cirrhosis scores as an isolated leakage intervention:
cohort composition and class balance also changed. The corrected cirrhosis task
evaluates within-cohort generalization, not transfer to an external cohort.

These corrections concern train/test separation; they do not establish absence
of foundation-model pretraining overlap.
