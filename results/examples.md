# Failure examples (Hateful Memes test split, text_plus_clip model)

Thresholds: t_clear = 0.2071, t_flag = 0.6315. Captions are the dataset's overlaid text; images are not reproduced here.

## Missed harmful memes (auto-cleared as Safe)

| id | caption | p_text | p_text_jigsaw | p_image | p_clip | weapon_max_conf | p_harmful |
|---|---|---|---|---|---|---|---|
| 7193 | things i love to hunt | 0.026 | 0.018 | 0.072 | 0.078 | 0.514 | 0.020 |
| 10234 | when you've decided to start a life of crime | 0.025 | 0.103 | 0.593 | 0.095 | 0.000 | 0.023 |
| 64735 | this movie sucks i am thinking of another shooting | 0.044 | 0.884 | 0.002 | 0.114 | 0.000 | 0.028 |
| 80713 | rare picture of child birth | 0.033 | 0.000 | 0.034 | 0.101 | 0.117 | 0.033 |
| 92783 | what's better than winning a gold medal in this contest walking | 0.045 | 0.001 | 0.936 | 0.100 | 0.263 | 0.035 |

Why it fails: every caption here is harmless on its own ("things i love to hunt", "rare picture of child birth"), so
the caption models score them near zero, and the hateful meaning only appears when the words are read against the
picture. CLIP catches some of these pairings but not these. In the first row the weapon detector does fire (0.51),
yet the fusion model learned from training memes that weapons rarely predict hate, so it gives that signal little
weight. These are the misses that argue for a stronger joint image-text model rather than more text features.

## False alarms (auto-flagged Harmful)

| id | caption | p_text | p_text_jigsaw | p_image | p_clip | weapon_max_conf | p_harmful |
|---|---|---|---|---|---|---|---|
| 63719 | hey grandma what are you doing ? im sewing a scarf for your trans brother ajay | 0.979 | 0.138 | 0.917 | 0.926 | 0.121 | 0.972 |
| 40316 | when dishwasher stops working take it back | 0.538 | 0.001 | 0.001 | 0.911 | 0.000 | 0.905 |
| 16042 | women use chemicals to remove polish everyday and no one bats an eye i use chemicals to re | 0.955 | 0.054 | 0.999 | 0.792 | 0.000 | 0.894 |
| 40723 | how to pick up a girl from auschwitz | 0.709 | 0.081 | 0.013 | 0.888 | 0.000 | 0.890 |
| 97853 | ahmed? im pregnant!!! | 0.575 | 0.014 | 0.867 | 0.878 | 0.000 | 0.866 |

Why it fails: most false alarms mention an identity group or a sensitive event (trans, Auschwitz, a Muslim name),
which the models have learned to associate with hate. This is the known identity-term bias of toxicity models, and
Hateful Memes adds "benign confounders" on purpose: the same caption paired with a harmless image is labelled not
hateful. Some of these labels are also arguable, which caps how far any model can go on this benchmark.
