# Provenance

`llm_labeler.py` records that an earlier draft labeled products with a paid
language-model API. The client, the key handling, and the prompts are not in
this folder. Running the file stops immediately.

`llm_labels_convert.py` in the parent directory reads a label file you already
have. It does not contact an API. The label file itself is not distributed
with this code. Protocol P3 and Exercise 4 are not reported. The saved label
file does not store the sampling parameters, so the query sample is not
reconstructed.

Paid client libraries are not installed by `requirements.txt`.
