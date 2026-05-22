<!--
Thanks for opening a PR.

A few notes before you write:

- The PR title becomes the squash-merged commit on `main`, so use the Conventional Commits format:
    type: short imperative description
  Example: feat: add EasyComm serial handshake
  Types: feat, fix, docs, refactor, chore

- See CONTRIBUTING.md for the full convention reference.
-->

## Summary

<!-- One or two sentences: what does this PR change, and why? -->

## Related issues

<!-- Link any related issues or discussions. Use `Closes #123` to auto-close on merge. -->

## Notes for reviewers

<!-- Anything worth flagging: tricky tradeoffs, things you're unsure about, areas that need a careful look. Delete this section if there's nothing to add. -->

## Checklist

- [ ] PR title follows the `type: description` convention
- [ ] Branch name uses one of the agreed prefixes (`feature/`, `fix/`, `docs/`, `refactor/`, `chore/`)
- [ ] `uv run ruff check` and `uv run ruff format` pass
- [ ] `uv run pytest` passes (or this PR is documentation-only)
- [ ] New code in `core/` does not import from `ui/`
- [ ] New behavior has unit tests in `tests/unit/`, or there's a note in "Notes for reviewers" explaining why not
- [ ] CHANGELOG.md updated under `[Unreleased]` if this is a user-visible change
