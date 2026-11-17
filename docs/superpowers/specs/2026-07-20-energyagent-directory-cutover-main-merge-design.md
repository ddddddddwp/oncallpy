# EnergyAgent Directory Cutover and Main Merge Design

**Date:** 2026-07-20  
**Repository:** `/root/code/oncallpy`, becoming `/root/code/EnergyAgent`  
**Target branch:** `main`

## Goal

Complete the local project identity change to **EnergyAgent**, preserve existing
external callers during the transition, integrate the reviewed local project
code into `main`, and leave the complete local stack healthy after the merge.

## Current State

- Product text, FastAPI metadata, frontend naming, package metadata, and the
  canonical stack manager already use `EnergyAgent`.
- The Python import root remains `app`; changing it would be an import-only
  migration with no product benefit.
- The repository is currently checked out on
  `checkpoint/energyagent-pre-migration` with tracked and untracked migration
  work in the working tree.
- Local `main` is an ancestor of the current branch, so the committed branch can
  be integrated with a fast-forward-only merge.
- `feat/energyagent-foundation` contains two commits whose patches are already
  represented by equivalent commits on the current branch. It must not be
  merged a second time.
- The requested destination `/root/code/EnergyAgent` does not currently exist.
- Running EnergyAgent processes and CodeGraph were started from the legacy path.
  No active cross-project script was found with a hard-coded
  `/root/code/oncallpy` path, but the old path remains a supported local entry.

## Chosen Approach

Use a compatibility-preserving physical cutover:

1. Move the repository directory from `/root/code/oncallpy` to
   `/root/code/EnergyAgent`.
2. Create the relative compatibility link
   `/root/code/oncallpy -> EnergyAgent`.
3. Keep the internal Python package named `app`.
4. Keep existing compatibility modules and the legacy
   `scripts/oncallpy_stack.py` forwarding entry point.
5. Commit only source, tests, maintained configuration, and documentation that
   belong to the product.
6. Verify the committed branch, fast-forward local `main`, and verify again from
   the canonical directory.

The relative link avoids embedding a second absolute path and allows existing
shell commands, CodeGraph configuration, Git linked-worktree metadata, and local
automation to continue resolving the repository during the compatibility
period.

## Naming Boundaries

The canonical identities after cutover are:

| Surface | Canonical identity |
|---|---|
| Product and UI | `EnergyAgent` |
| Local repository directory | `/root/code/EnergyAgent` |
| Python distribution | `energy-agent` |
| Python import package | `app` |
| Canonical stack command | `scripts/energyagent_stack.py` |
| Legacy local path | `/root/code/oncallpy` compatibility link |
| Legacy stack command | `scripts/oncallpy_stack.py` forwarding shim |

Historical design records may retain old paths when the path is part of the
recorded historical context. Current README instructions, maintained commands,
and active operational documentation use the canonical directory.

The GitHub repository `ddddddddwp/oncallpy` and the configured `origin` URL are
outside this cutover. They are not renamed or pushed by this change.

## Working-Tree Integration Policy

The current working tree contains user-owned migration work. It is preserved and
classified before committing:

- Application, domain, test, Go source, maintained configuration, and current
  documentation changes are reviewed and committed in logical batches.
- Existing compatibility shims under `app.services.*`,
  `app.core.milvus_client`, and the legacy stack command are retained.
- The current and archived OpenSpec copies are retained because they serve
  different lifecycle roles.
- `runtime-logs/`, coverage/cache output, compiled binaries, local credentials,
  and other generated runtime artifacts are not committed.
- The compiled `alarmmock/alarmmock` file is removed from the index and ignored;
  its Go source and tests remain.
- The ignored `.env` remains local and keeps the working Redis credentials. No
  credential value is printed or committed.
- Static-analysis cleanup is limited to proven unused imports, redundant aliases,
  and one duplicate definition of each duplicated time helper. Public behavior
  and compatibility imports do not change.

No unrelated refactor is performed while integrating the working tree.

## Git Integration

The integration sequence is local and non-destructive:

1. Record the pre-cutover branch, commit, worktree list, and status.
2. Commit the approved migration and cleanup batches on
   `checkpoint/energyagent-pre-migration`.
3. Require a clean tracked working tree except for explicitly ignored local
   runtime state.
4. Run the pre-merge verification suite.
5. Switch the primary checkout to `main`.
6. Merge with `git merge --ff-only checkpoint/energyagent-pre-migration`.
7. Run the same verification on `main` from `/root/code/EnergyAgent`.

The checkpoint branch is retained as a named rollback reference. The linked
`feat/energyagent-foundation` worktree and branch are not deleted. No remote push
or remote repository rename is performed.

## Directory Cutover

The physical move is performed only after the destination absence and repository
identity have been rechecked. The operation briefly stops EnergyAgent-owned
processes that may reload files during the move, then:

```text
/root/code/oncallpy
  -> move to /root/code/EnergyAgent

/root/code/oncallpy
  -> relative symbolic link to EnergyAgent
```

Both paths must resolve to the same Git top-level directory after the move.
Existing Git linked worktrees must remain queryable. EnergyAgent services are
then restarted through the canonical stack command from the canonical path.

## Failure Handling and Rollback

- If the destination appears before the move, stop without overwriting it.
- If the directory moves but link creation or Git validation fails, remove only
  the newly created compatibility link when present and move the directory back
  to `/root/code/oncallpy`.
- If a working-tree batch fails its targeted tests, do not commit that batch
  until the failure is corrected.
- If pre-merge verification fails, do not move `main`.
- If the fast-forward precondition no longer holds, stop and inspect the new
  branch relationship; do not create an implicit merge commit.
- If post-merge verification fails, keep the checkpoint branch and move `main`
  back only through an explicit, non-destructive branch operation after recording
  the failing state. Do not use `git reset --hard`.
- Runtime credentials and existing user changes are never discarded during
  rollback.

## Verification

Verification is run before and after the `main` fast-forward:

1. Git and path checks:
   - both old and new paths resolve to the same repository;
   - linked worktrees remain registered and accessible;
   - `main` fast-forward relationship is proven immediately before merging;
   - generated artifacts and secrets are absent from the commit.
2. Static checks:
   - Ruff unused-import and redefinition checks pass;
   - Python compilation/import smoke checks pass;
   - shell and Go tooling checks pass for maintained startup and alarm tools.
3. Automated tests:
   - targeted branding, configuration, compatibility, stack, Milvus, alert, and
     on-call tests pass;
   - the complete Python test suite passes;
   - Go tests for `alarmcheck` and `alarmmock` pass.
4. Runtime checks:
   - the existing EnergyAgent stack command exits successfully;
   - ports 6379, 8000, 8003, 8004, 8888, 9900, 12345, 19090, and 19530 are
     reachable as applicable;
   - EnergyAgent `/health` reports Redis, Milvus, and configuration healthy;
   - the alarmmock ingest smoke test succeeds;
   - Mem0 and OnCall MCP do not report connection-refused failures in the tested
     flow.

## Acceptance Criteria

The work is complete when:

1. `/root/code/EnergyAgent` is the canonical physical repository directory.
2. `/root/code/oncallpy` remains a working compatibility link.
3. Product and maintained operational surfaces identify the project as
   `EnergyAgent`.
4. Existing external imports and old startup commands continue working.
5. Proven generated, unused, and duplicate items are removed without deleting
   compatibility layers or historical specifications.
6. Local `main` contains the committed migration through a fast-forward-only
   integration.
7. The checkpoint branch remains available for rollback.
8. Pre-merge and post-merge verification pass, including the local stack and
   alarmmock smoke path.
9. No remote push, remote rename, credential disclosure, or unrelated worktree
   deletion occurs.
