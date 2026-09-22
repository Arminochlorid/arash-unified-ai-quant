# Security

This repository does not require exchange credentials and must never contain
API keys, secrets, wallet material, account exports or personally identifiable
trading records.

Before every public release:

1. scan the complete Git history for credentials;
2. verify that `.env`, data, model and output files are ignored;
3. revoke any credential that was ever committed, even if it was later deleted;
4. keep live broker integration outside this research repository.

Security defects should be reported privately to the repository owner before
public disclosure.
