# External incident contract

`visiondoctor run` never creates a repository, dataset, fault, diagnosis, or patch. The caller
supplies an `Incident` JSON that points to an existing local Git repository and separately
stored evidence/reference manifests.

## Repository contract

The repository must contain both the known-good and faulty commits. At each commit it exposes
two protected adapter scripts:

- `runner.py` reads one JSON object from standard input and writes the documented execution
  result JSON to standard output;
- `test_runner.py` runs repository tests and returns non-zero on failure.

Alternate repository-relative Python script paths can be selected with
`metadata.execution_contract.runner_script` and `test_runner_script`. Both paths must end in
`.py`; shell, C++, ROS launch, and other runtime entrypoints may be discovered as project evidence
but are not executed by this general backend. The runner receives bounded case JSON on standard
input and must return the documented JSON result on standard output. Candidate patches are never
allowed to change either adapter. ROS 2/Gazebo/MoveIt execution is provided only through its
dedicated adapter and simulation gate.

## Case separation

Every `case_set` item has two different paths:

- `manifest_path`: Agent-visible RGB/depth paths, intrinsics, measured frames, and timestamps;
- `reference_path`: QA-only trusted target pose and provenance.

The evidence manifest must not contain `reference_t_base_object`. VisionDoctor rejects the
incident if evidence and reference resolve to the same file, and QA hash-verifies the trusted
reference on every validation.

## Minimal shape

```json
{
  "incident_id": "INC-customer-001",
  "title": "Target pose regression",
  "description": "Observed behavior and reproduction notes",
  "repository": {
    "path": "C:/absolute/path/to/repository",
    "branch": "main",
    "access_mode": "local"
  },
  "baseline_commit": "0123456789abcdef",
  "faulty_commit": "fedcba9876543210",
  "case_set": [
    {
      "case_id": "case-001",
      "manifest_path": "C:/absolute/path/case-001/evidence_manifest.json",
      "reference_path": "C:/absolute/path/case-001/qa_reference.json"
    }
  ],
  "acceptance_criteria": {
    "unit_tests_required": true,
    "translation_rmse_m": 0.005,
    "mean_rotation_error_rad": 0.01745,
    "reprojection_error_px": 5.0,
    "scene_pass_rate": 0.98,
    "fixed_motion_required": true,
    "tcp_error_m": 0.005,
    "latency_growth_ratio": 0.1,
    "policy_checks_required": true
  },
  "allowed_patch_scope": {
    "allowed_globs": ["src/**/*.py", "config/*.yaml", "tests/test_*.py"],
    "forbidden_globs": [
      "acceptance/**",
      "references/**",
      "dataset/**",
      "gazebo/**",
      "moveit/**",
      "qa/**",
      "security/**"
    ],
    "max_files": 3,
    "max_changed_lines": 100,
    "allow_test_changes": true,
    "forbid_test_removal": true,
    "forbid_test_skips": true
  },
  "metadata": {
    "simulation_only": true,
    "raw_log": "Captured application log",
    "presentation_case_id": "case-001",
    "execution_contract": {
      "runner_script": "runner.py",
      "test_runner_script": "test_runner.py"
    }
  }
}
```

Use the generated OpenAPI schema at `/docs` for the authoritative request validation rules.
