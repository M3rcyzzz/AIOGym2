# Website release retention

Effective 2026-10-22, following the user's instruction to stop retaining
historical versions. This supersedes earlier instructions to preserve every
website release. It applies to future publication of `https://m3rcyzzz.club/aio/`.

- Keep one deployed website version: the current live release.
- A new candidate may coexist temporarily with the live release while it is
  staged and verified. Switch the live link only after verification succeeds.
- After the new live release is verified, remove superseded website deployment
  directories inside `/home/ubuntu/aio-releases/`. Do not create another backup
  of those retired deployment directories.
- Do not add new `*-preserved.html` copies or whole-site history snapshots to
  future builds. Historical research results can remain part of the current
  website; they are distinct from snapshots of older website versions.
- Before removing a directory, resolve the live link and each target; require
  every target to be a managed release strictly inside the release root and
  different from the live release. Never remove unrelated directories, the
  active release, or an unverified path. Record the exact removed targets and
  resulting available space.
- Keep the current release if a new publication fails. Retain failure evidence
  locally, without retaining a new remote historical website version.
- Keep the existing capacity check. Do not lower its reserve to force a deploy.

This is a forward publication rule, not an instruction to erase research
history. Training artifacts, scientific reports, original controls, failed-run
evidence and sealed evaluation records retain their existing policies. This
policy change itself performs no deletion of existing website releases.

For the already prepared economic-accounting publication, preserve its original
build and rejection evidence. Apply this policy in a new publication-only
revision if needed to remove newly generated website snapshots and their links;
do not rerun the completed study. Verify its new manifest and delta before
publication. No frozen experiment source, numerical threshold or benchmark is
changed by this retention policy.
