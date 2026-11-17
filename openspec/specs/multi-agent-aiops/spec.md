## ADDED Requirements

### Requirement: Multi-agent AIOps orchestration

The system SHALL provide an AIOps diagnosis workflow composed of specialized agent nodes for planning, evidence collection, diagnosis, and report generation.

#### Scenario: Diagnosis runs through specialist agents

- **WHEN** a client starts an AIOps diagnosis
- **THEN** the system SHALL execute a multi-agent workflow that creates a plan, collects evidence, produces a diagnosis, and generates a final report

#### Scenario: Agent failures are captured

- **WHEN** an agent node or tool call fails during diagnosis
- **THEN** the system SHALL capture the failure in workflow state and continue to produce an honest error-aware output when possible

### Requirement: Compatible AIOps streaming events

The system SHALL preserve the existing `/api/aiops` streaming event types consumed by the frontend.

#### Scenario: Frontend receives compatible events

- **WHEN** the frontend calls `/api/aiops`
- **THEN** the streamed events SHALL use compatible event types including `plan`, `step_complete`, `status`, `report`, `complete`, or `error`

#### Scenario: Final report remains Markdown

- **WHEN** the diagnosis completes successfully
- **THEN** the final report SHALL be returned as Markdown text in the streamed report or completion payload

### Requirement: Structured diagnosis evidence

The system SHALL keep structured evidence in the multi-agent workflow state.

#### Scenario: Evidence is available to diagnosis and report agents

- **WHEN** knowledge, log, or metrics evidence is collected
- **THEN** the workflow SHALL store that evidence in named state fields that downstream diagnosis and report agents can read

#### Scenario: Report is based on collected evidence

- **WHEN** the report agent generates the final report
- **THEN** it SHALL base the report on collected evidence, diagnosis, recommendations, and recorded errors instead of inventing unavailable data
