# Release audit

**Release candidate:** GeoDSSOP v0.1.0

**Audit date:** 2026-08-25

**Status:** pending final Git/GitHub verification

This file is completed by the release workflow after tests, file inventory,
credential scanning, large-file checks, Git push, and remote release-asset
verification. No pass claim should be inferred from an unchecked item.

- [ ] Python source compiles.
- [ ] Unit tests pass in the isolated WQQ validation environment.
- [ ] 1UBQ end-to-end prediction matches the frozen expected CSV.
- [ ] Public weights equal original checkpoint tensors exactly.
- [ ] Public result tables parse and required numerical values are present.
- [ ] No credential, private host, or private absolute path is tracked.
- [ ] No unexpectedly large Git object is tracked.
- [ ] PDB/ESM/data license boundary is documented.
- [ ] README commands and relative links are checked.
- [ ] Private GitHub repository contents are verified after push.
- [ ] Model release assets are downloaded and SHA-verified from GitHub.

The final audit record will include the Git commit, tag, repository URL, test
counts, and release-asset identities.
