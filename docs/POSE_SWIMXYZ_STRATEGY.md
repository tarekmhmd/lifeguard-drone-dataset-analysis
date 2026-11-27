# SwimXYZ at scale? Dataset strategy assessment for pose estimation (HPE)

Date: 2026-11-27. Planning only: nothing downloaded, trained or changed.
Legend: **[V]** read on the source page (Zenodo records, project page, arXiv paper, model card) ·
**[A]** assumption, must be checked · **[R]** recommendation.

## 1. Verified facts about SwimXYZ

| Record | Content [V] | Size [V] |
| --- | --- | --- |
| [8399376](https://zenodo.org/records/8399376) | Backstroke / Breaststroke / Butterfly / Freestyle `_labels.zip` (1.6, 1.7, 1.7, 1.7 GB) + `smpl_swimming_motions.zip` (57.9 MB) | 6.7 GB |
| [8399837](https://zenodo.org/records/8399837) Backstroke part 1 | front + aerial views | 39.2 GB |
| [8401680](https://zenodo.org/records/8401680) Backstroke part 2 | all 3 side views | 35.3 GB |
| [8401898](https://zenodo.org/records/8401898) Breaststroke part 1 | front + aerial | 39.3 GB |
| [8401923](https://zenodo.org/records/8401923) Breaststroke part 2 | all 3 side views | 33.4 GB |
| [8401954](https://zenodo.org/records/8401954) Butterfly part 1 | front + aerial | 39.6 GB |
| [8401974](https://zenodo.org/records/8401974) Butterfly part 2 | all 3 side views | 33.9 GB |
| [8402009](https://zenodo.org/records/8402009) Freestyle part 1 | front + aerial | 40.3 GB |
| [8402031](https://zenodo.org/records/8402031) Freestyle part 2 | all 3 side views | 36.0 GB |

Totals (my arithmetic): side views (4 x part 2) **138.6 GB**; front + aerial (4 x part 1) **158.4 GB**; all videos
**297.0 GB**; everything **303.7 GB**. All CC BY 4.0, open, MD5 listed on each record.

Other facts [V]: 11,520 videos, 5 s each at 60 fps, 3.4 M frames; 5 views (front, aerial, side underwater, side
above-water, side water-level); 2D and 3D joints in **COCO17, COCO25 and a 48-joint "Base"** skeleton, in global and
pelvis-centred coordinates; 240 SMPL motions, generated with GANimator; 4 competitive strokes only; the authors
list as missing: subject/swimsuit/environment diversity, dives and turns. Real-image validation was qualitative only.
The existing ViTPose-L SwimXYZ checkpoint was fine-tuned on **25,000 freestyle frames (5,000 per view), 5 epochs**,
and scores PCK@0.05 0.982 on synthetic validation (0.989 side underwater).

Not stated anywhere I could read [A]: per-view video counts (2,304 per view if evenly split), video resolution,
the internal folder structure and file format of the label zips, whether 2D labels are per view, extracted sizes.

## 2. The key distinction: two different learning problems

| Problem | What it learns | Can SwimXYZ help? |
| --- | --- | --- |
| **Pose estimation** (image → joints) | where shoulders, wrists, head are in a real pool image | **Partly.** It teaches swimming body shapes and views; it cannot teach real water, real people, refraction, bubbles, lane ropes, several swimmers, or our cameras. |
| **Distress analysis** (joints over time → risk evidence) | what dangerous behaviour looks like | **Almost not at all.** SwimXYZ contains only 4 competitive strokes. It has no treading water, vertical struggling, head-back posture, arm pressing down sideways, floating, sinking or submersion. It can only describe **normal** swimming. |

Normal synthetic swimming is not pre-drowning or drowning data and is never labelled or used as such.

## 3. Comparison of approaches

| # | Approach | Strength | Limitation | Domain gap |
| --- | --- | --- | --- | --- |
| 1 | Only our 450 real frames | exactly our cameras and water; the only data that can **validate** | 300 training frames are few for a ViT-L; no distress coverage beyond what the frames show | none (it is the target) |
| 2 | SwimXYZ annotations only | small download (6.7 GB) | **cannot train an image model**: labels without the rendered frames have no pixels. Useful only for motion models (normal-swimming keypoint sequences) | n/a for images |
| 3 | SwimXYZ annotations + selected videos | large, perfectly labelled, all strokes and views | synthetic look; already exploited by the current checkpoint (synthetic val 0.98); horizontal strokes only | **large**, unmeasured |
| 4 | SwimXYZ + corrected real frames | synthetic gives coverage, real anchors the domain | real share must be protected; annotation effort | reduced, must be measured |
| 5 | **Staged hybrid (recommended)** | spends storage only if the real baseline shows where synthetic data helps | slower start | measured at every step |

Why "bigger" is not automatically better here [R]:

- The checkpoint has already seen SwimXYZ. On SwimXYZ it is near ceiling, so more SwimXYZ frames mainly raise
  synthetic scores. Our earlier detector lesson applies: more **real** videos beat a bigger model or more of the
  same data.
- **Risk specific to a lifeguard [A]:** heavy fine-tuning on horizontal competitive strokes can make the model
  worse on upright, tilted and struggling bodies, which COCO pre-training covers (people standing). Those are
  exactly the postures that matter. This must be measured on a real "vertical / unusual posture" subset before
  and after any synthetic fine-tuning.

## 4. Which parts of SwimXYZ, if any

- **Views [R]:** side underwater, side water-level and **aerial** match us. The `datasets (1)` frames are
  underwater side views; the drone and the ddt model are overhead. Front view (end of lane, head-on) has low
  priority. Note: aerial sits in the part 1 zips together with front, so it costs 158 GB to get it as whole zips.
- **Joints [R]:** COCO17 is provided directly [V], so no mapping is needed, but the joint definitions (synthetic
  skeleton vs. annotator convention, e.g. hip position) must be compared on overlay [A]. COCO25 adds feet (big
  toe, small toe, heel) [A from the OpenPose definition]; not useful under water now. The 48-joint "Base" skeleton
  has no use for us now.
- **Strokes [R]:** keep all four if any videos are used. Variety helps more than volume, and backstroke and
  breaststroke contain the most upright-ish phases.
- **Frames [R]:** 60 fps over 5 s is highly redundant; 1 frame in 10 to 20 keeps the variety.
- **Annotations alone [R]:** worth having for the distress side: thousands of clean normal-swimming 2D keypoint
  sequences, to describe "normal motion" for the temporal model (with noise added to match predicted keypoints).

## 5. Download options

| Option | Files | Download [V] | On Drive (my estimate [A]) | Training value | Needed? |
| --- | --- | --- | --- | --- | --- |
| A. Annotations only | 4 `_labels.zip` + SMPL zip | 6.7 GB | 7 GB zipped; extracted size unknown, budget 20-35 GB | no image training; normal-motion sequences; schema check | **yes, but later**, and start with one stroke (1.7 GB) |
| B. Side views | Backstroke/Breaststroke/Butterfly/Freestyle part 2 | 138.6 GB | 15-40 GB if only sampled frames + labels are kept and zips are deleted after extraction | real gain only if the real baseline fails on swimming strokes in side views | **conditional** |
| B+. Side + aerial | B + the 4 part 1 zips | 297 GB | 30-60 GB sampled | adds overhead/drone view | **conditional** on the drone-view baseline |
| C. Everything | all 9 records | 303.7 GB | about 300 GB kept, about 600 GB while extracting | marginal over B+ | **no** |

Possible way to avoid downloading whole zips [A, must be tested]: Zenodo files may support HTTP range requests.
If they do, a ZIP's file list can be read remotely (a few MB) and single videos extracted without the 35-40 GB
zip. A first test costs almost nothing and tells us the real folder structure.

Colab local disk is limited (often about 100-200 GB [A]); a 40 GB zip must be extracted, sampled and deleted
one at a time. How much free space the Team Drive has is unknown to me: please check.

## 6. Recommended strategy

**Staged, evidence-gated hybrid. Real data decides; synthetic data is added only where the real baseline shows a
gap it can close.**

1. **Real validation first (no download).** Finish the 150 `real_val` frames, stratified on purpose: upright /
   treading, horizontal strokes, unusual postures, splash and occlusion, several swimmers, underwater vs. overhead.
   Tag each frame with these attributes (pose context, not drowning labels).
2. **Zero-shot baseline.** Existing SwimXYZ checkpoint vs. COCO ViTPose-L vs. YOLO11-pose, per attribute subset.
   This shows whether the SwimXYZ checkpoint beats COCO on our data, and where it fails.
3. **Cheap SwimXYZ inspection.** Test remote ZIP listing; download `Freestyle_labels.zip` (1.7 GB) only; check
   structure, COCO17 overlay against a few frames, per-view labels.
4. **Decide on videos from step 2's failures:**
   - fails mostly on strokes in side / overhead views → option B (or B+), sampled frames;
   - fails mostly on upright / unusual postures, multiple swimmers, real water → more synthetic does not help;
     annotate more **real** frames (active learning: annotate where the model is least confident).
5. **Fine-tune** from the SwimXYZ checkpoint (not from scratch), A100:
   - each batch has a fixed real share (for example 30-50%, real oversampled), so synthetic data cannot dominate;
   - lower learning rate, few epochs, early stopping on **real** validation only;
   - compare against step 2 on every subset, especially upright/unusual postures (forgetting check);
   - real test set: a second, later set from different footage, used once.
6. **Grow the real set** to about 1,000-2,000 corrected frames over time, which matters more than synthetic
   volume [R].

Keeping evaluation honest: real val/test never enter training; splits by video or frame-index block; synthetic
validation scores are reported but never used to choose a model.

## 7. The missing distress data (separate track)

Pose gives evidence such as verticality, head height relative to shoulders and the waterline, arm pattern
(lateral pressing vs. stroke recovery), absence of leg kick, and posture change rate. Turning that into risk
evidence needs **real temporal sequences**, which no public dataset I found provides.

Recommended collection [R]:

- **Normal sequences** from our own pool: the 4 strokes, treading water, floating on the back, resting at the
  wall, playing, diving in. Many false alarms come from these.
- **Simulated distress by trained swimmers** (lifeguards / swim club) under a written safety protocol, with a
  lifeguard in the water and consent: vertical struggling, head tilted back, arm pressing down, passive floating
  face-down, slow sinking. Label these as **"simulated"**, never as real drowning.
- **Head-submersion sequences** with timing, matching the existing submersion timers.
- **Real incident footage** only if lawfully obtained; labelled by named human reviewers; ambiguous clips get
  "uncertain", never "drowning".
- Labels per **time interval** (event start/end), by two reviewers, with agreement measured.

Use in the system: pose features enter the temporal behaviour layer as additional columns, with risk weight 0 until
validated on real sessions (unchanged rule). Synthetic SwimXYZ sequences can describe "normal swimming" only (a
reference distribution), so treading water and floating must come from real data, or they would look abnormal.
Pose alone never raises ALARM.

## 8. Remaining gaps and open questions

- SwimXYZ label format and per-view structure: unverified until one label zip is opened.
- Free Team Drive space and Colab disk size: unknown.
- Whether Zenodo serves byte ranges (would avoid whole-zip downloads): untested.
- Domain-gap size: unknown until the zero-shot baseline on corrected real frames exists.
- No real distress sequences exist; collection needs a safety protocol and consent.
