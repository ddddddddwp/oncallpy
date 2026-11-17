## ADDED Requirements

### Requirement: Configurable local log directory

The system SHALL support a configurable local runtime log directory.

#### Scenario: Default log directory is used

- **WHEN** no `LOG_DIR` value is configured
- **THEN** application and local service logs SHALL be written under `runtime-logs/`

#### Scenario: Environment log directory is used

- **WHEN** `LOG_DIR` is configured
- **THEN** application and local service logs SHALL be written under that directory

### Requirement: Local stack script creates service log directories

The local stack restart script SHALL create the configured log directory before starting application services.

#### Scenario: Nested log directory does not exist

- **WHEN** a service is started with a log path under a missing directory
- **THEN** the script SHALL create the directory and start the service with stdout/stderr redirected to that log file
