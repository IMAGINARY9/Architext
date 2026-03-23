# Baseline Rubric and Measurement Protocol

## Environment Freeze Checklist

- Fresh repository clone or clean working copy
- Python virtual environment created and activated
- Dependencies installed from `requirements.txt`
- Default `.env` values set for local test path
- Stable network and machine profile for timed tasks

## Severity Rubric

- Critical: Blocks task completion or causes unsafe/incorrect outcomes for most users.
- High: Causes repeated failure or major confusion for a core flow.
- Medium: Slows users materially but does not block completion.
- Low: Minor friction or polish issue.

## Core Metrics

- Task completion rate
- Time on task (seconds)
- Error count per task
- Wrong-endpoint attempts per session
- Moderator intervention count
- Confidence score (1-5)
- Satisfaction score (1-5)

## Session-Level Pass Criteria

- First-run flow completion by at least 85% of participants
- Median time to first successful query at or below 15 minutes
- Median wrong-endpoint attempts at or below 1

## Evidence Standards

- Every finding includes at least one direct quote or observed behavior
- Every quantitative claim links to raw session notes
- Recommendations map to a measurable KPI

## Logging Schema

Use this schema for every participant and test run:

```markdown
- Participant ID:
- Persona:
- Test ID:
- Task ID:
- Completed (Y/N):
- Time (s):
- Errors:
- Wrong endpoint attempts:
- Intervention required (Y/N):
- Confidence (1-5):
- Satisfaction (1-5):
- Notes:
- Quote(s):
- Candidate fix:
```
