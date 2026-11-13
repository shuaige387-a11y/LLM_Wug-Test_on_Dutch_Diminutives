The runs of 2026-11-07 did not record the model revision (`config._commit_hash` is not set in
transformers 5). `MODEL_REVISIONS.csv` lists each repo's commit hash as read from
`https://huggingface.co/api/models/<repo>` on 2026-11-08. Every repo's last modification predates
the runs (2026-11-07 21:44 to 23:45 UTC), so these are the revisions that were downloaded.
To reproduce exactly, pass `revision=<sha>` to `from_pretrained`.
