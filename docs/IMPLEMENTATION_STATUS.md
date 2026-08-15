# VisionDoctor implementation status

Updated 2026-08-12. The frozen simulation, safety, deterministic-QA, and human-approval
boundaries remain unchanged.

## Canonical VisionProject milestone

Implemented:

- a persistent `VisionProject` domain object, separate from individual incidents and diagnosis
  conversations;
- commit-pinned discovery of arbitrary Git layouts without requiring `src/`, `config/`, ROS, a
  robot, depth input, or a prescribed repository template;
- canonical component, asset, relation, runtime, validation, ambiguity, and project-knowledge
  models with graph consistency checks;
- deterministic recognition of Python/C++, ROS 2, Docker, OpenCV, ONNX Runtime, TensorRT,
  MoveIt, Gazebo, models, calibration/configuration, visual data, recordings, robot descriptions,
  deployment assets, runner candidates, and test candidates;
- an explicit capability boundary between broad, read-only project understanding and automatic
  execution: the general repair loop currently accepts only Python runner/test-runner scripts
  implementing the VisionDoctor JSON contract, while ROS 2/Gazebo/MoveIt use their dedicated
  RGB-D pose adapter and simulation gate;
- source-backed semantic understanding in which the real model must use bounded tools to inspect
  a canonical component and read a real file at the exact Git `HEAD` before submitting;
- ordered component-inspection and pinned-file-read evidence stages, followed by a hard eight-call
  accepted exploration budget and at most three terminal-only correction attempts;
- a bounded semantic delta of core component roles, relations, and unresolved choices rather than
  a model rewrite of the deterministic inventory;
- acceptance of evidence paths only when they belong to the canonical graph or were actually read
  from the pinned commit, plus field-specific protocol errors for safe correction;
- same-commit semantic re-runs rebuild the deterministic base and replace the prior model overlay
  while retaining valid human confirmations and incident history;
- customer-facing Chinese project summaries reject internal development vocabulary before they
  can be stored or displayed;
- fail-closed rejection of invented component IDs, graph nodes, evidence paths, relations, and
  premature model understanding, with no rule-generated semantic fallback;
- explicit user confirmation for ambiguous calibration, runtime, test, and data-production
  relationships before repair can start;
- confirmation carry-forward across a new commit only when the selected option remains present;
- an optional, bounded `visiondoctor.yaml`/`.yml`/`.json` ingestion aid whose declared paths must
  exist in the same pinned commit and whose mappings can split scanner-level components safely;
- project-aware Diagnosis Agent graph inspection and impact tracing, Project context for the Patch
  Agent, and patch allowlists derived from actual project source/configuration/test paths;
- lightweight project-list and full-project API routes, plus workbench grouping of multiple
  diagnosis conversations under one long-lived project.

## Conversation-first product milestone

Implemented:

- persistent, listable diagnosis conversations as the only workbench entry point;
- multiple independent conversations with messages, image/file attachments, source state,
  captured scenes, run history, and human feedback;
- repository connection by folder path; the model selects a supported revision comparison and
  deterministic code verifies the selection, so users never label a faulty revision;
- user-language source readiness and activity progress instead of protocol fields, audit hashes,
  internal tool names, and state constants;
- model-backed repair of user-facing responses that expose internal vocabulary or identifiers;
- two concurrent local run workers backed by atomic SQLite job claims;
- non-blocking workbench controls for the official Gazebo Qt GUI, repeated MoveIt UR5e fixed
  motion in the visible session, RGB-D acquisition, and simulation shutdown;
- one-click insertion of a Gazebo capture into the selected conversation without showing paths;
- rejection feedback returns to the same conversation and can start a new diagnosis run;
- patch validation rejects test-only changes and packages that shadow an existing module.

## Product Agent milestone

Implemented:

- strict OpenAI-compatible HTTP model gateway for DeepSeek-compatible APIs;
- allowlisted project `.env` loading with process-environment precedence and no secret audit
  content;
- append-only model metadata audit containing hashes, latency, usage, finish reason, and tool
  names, but not prompts, responses, or credentials;
- real Diagnosis and Patch Agent tool loops;
- commit-pinned read-only repository tools for list, search, file read, and baseline/faulty diff;
- explicit untrusted-data/prompt-injection boundary for repository, issue, log, and tool text;
- structured `submit_diagnosis` and `submit_patch` terminal tools;
- generalized root-cause text and `generated` candidate type;
- model-submitted complete file changes, converted by trusted code into valid Git unified
  diffs with pre-execution path/file/line limits;
- up to three new model attempts with deterministic rejection feedback;
- duplicate rejected-patch detection;
- fail-closed behavior for missing configuration, HTTP failure, malformed model response,
  absent tool calls, invalid terminal payload, exhausted loop, and unavailable Docker;
- no production import of the Demo patch builders and no trusted known-patch hash list.

Production construction has no deterministic Agent fallback. Protocol doubles live only in
the test directory and must be explicitly injected.

## External product intake milestone

Implemented:

- `visiondoctor run --incident <json>` for caller-supplied local Git repositories and cases;
- `POST /api/v1/runs` for durable asynchronous external Incident jobs;
- SQLite queue persistence of run kind and immutable Incident request JSON;
- arbitrary baseline/faulty commits, case sets, acceptance values, and patch scopes validated
  through the strict Incident schema;
- configurable protected `runner.py` and `test_runner.py` repository adapter paths;
- no generated repository, dataset, fault, diagnosis, or patch in the external run path;
- removal of the Demo-only 50-case check from Core;
- Docker-only execution for external model-generated patches, with no local fallback;
- explicit dataset or Gazebo choice for product runs, with no automatic backend fallback.

The repository execution adapter is intentionally a protocol: external repositories provide
the two protected scripts and standardized JSON input/output. The Agent does not receive
repository-specific hard-coded logic.

## Agent/QA data isolation

Implemented:

- separate `evidence_manifest.json` and `qa_reference.json` files;
- schema-level `reference_path` on every case;
- rejection when evidence and reference paths are identical;
- rejection when Agent evidence contains `reference_t_base_object`;
- Agent tools have no filesystem path or API for QA references;
- QA-only references are stored under `trusted_qa/` and hash-verified on every validation;
- Gazebo capture writes runtime truth only to its separate QA reference file.

## Existing deterministic and simulation foundation

Still active:

- strict transforms in metres/radians, quaternion `xyzw`, and `T_target_source` convention;
- deterministic RGB-D marker pose estimation through the `rgbd_pose` task adapter;
- task-level validator plugins for `detection`, `ocr`, and `segmentation`, with explicit image
  input/output contracts and task-specific deterministic metrics;
- a legacy `structured_output` transport for existing typed-JSON datasets, no longer presented as
  proof of visual-task semantics;
- read-only, hash-pinned Diagnosis tools for actual image observation, metadata inspection, and
  bounded point-cloud summaries without QA-reference access;
- baseline/faulty reproduction and isolated Git worktrees;
- deterministic unit, geometry, RGB, depth, TCP, latency, policy, provenance, and integrity
  gates;
- hardened no-network candidate Docker sandbox;
- SQLite history/artifact/approval persistence and recoverable asynchronous queue;
- optional real ROS 2 Jazzy + Gazebo + MoveIt 2 + UR5e fixed-motion release gate;
- official Gazebo Qt GUI through Docker Desktop WSLg;
- human approval creates an unmerged candidate branch and PR materials;
- no automatic merge.

## Intentionally not claimed

- No claim is made that one successful fault proves broad empirical generalization. External
  intake is now generic, but a multi-family unseen-bug evaluation set is still required.
- Authentication, RBAC, managed secret storage, and a remote Git hosting PR adapter remain
  production-hardening work.
- The SQLite workers provide single-service restart recovery and two local execution slots, not
  distributed leases or mid-step checkpointing.
- Gazebo RGB-D and UR5e validation still run as separately isolated simulator contracts;
  their case/target identity is now aligned, but a single shared live Gazebo session remains a
  future integration improvement.

## Commands

```powershell
$env:VISIONDOCTOR_LLM_API_KEY = "<secret>"
$env:VISIONDOCTOR_LLM_BASE_URL = "https://api.deepseek.com"
$env:VISIONDOCTOR_LLM_MODEL = "deepseek-v4-flash"

.\.venv\Scripts\visiondoctor.exe model-check
.\.venv\Scripts\visiondoctor.exe run --incident C:\path\incident.json --sandbox docker
.\.venv\Scripts\visiondoctor.exe serve --host 127.0.0.1 --port 8000

.\.venv\Scripts\python.exe -m ruff check src tests ros
.\.venv\Scripts\python.exe -m compileall -q src tests ros
.\.venv\Scripts\python.exe -m pytest --cov=visiondoctor --cov-report=term-missing
.\.venv\Scripts\python.exe -m pip check
```

## Latest verification

- real provider handshake: `deepseek-v4-flash` completed the required tool call;
- real project-understanding run on the current repository's saved Git version: 39 components,
  2 assets, and 44 scanner relations were discovered without a repository layout template;
- the final model run used 7 accepted bounded read-only calls, inspected a canonical component,
  read 5 actual files at commit `42e57a1735b6e74127d0a2499e2af22d55d8fdf5`, then submitted one
  valid semantic delta;
- the accepted delta added source-grounded interpretations for 3 components and 4 relations while
  retaining 7 unresolved choices for human confirmation; repeated same-commit runs remained at
  48 total relations and 7 pending choices instead of accumulating model output;
- the workbench displays the accepted 385-character Chinese project description and no rejected
  development-process vocabulary; no scanner summary or fallback was used;
- real end-to-end model run: 5 HTTP model calls, diagnosis and patch terminal tools used;
- generated candidate SHA-256 differed from the legacy Demo fixture patch;
- generated candidate: 50/50 cases, every mandatory gate passed;
- final state: `AWAITING_HUMAN_APPROVAL`, automatic merge disabled;
- conversation-first real run: direct edit to the existing pose implementation plus a
  non-commuting regression test; Gazebo scene error 1.02 mm and robot TCP error 1.10 mm;
- an initially passing module-shadowing workaround was manually rejected and is now blocked by
  policy, demonstrating that passing metrics do not override the no-workaround requirement.
- Agent evidence reference leakage check: false;
- automated suite: 116 tests passed in 173.19 seconds;
  Ruff (`src`, `tests`, and `ros`), compileall, pip check, and `git diff --check` passed.
