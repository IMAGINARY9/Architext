# Operator Moderated Session Script (60 Minutes)

## 1. Introduction and Consent (5m)

Moderator prompts:
- "Thank you for joining. We are testing the product, not you."
- "Please think aloud as you work."
- "Do we have consent to record the session for internal analysis?"

## 2. Baseline Questions (10m)

- What tools do you currently use for codebase understanding?
- How familiar are you with API-first tools and Swagger?
- What do you expect from a first successful run?

## 3. Task Scenarios (35m)

### Task 1: Start the Server

Prompt:
- "Using project docs only, start Architext locally."

Success criteria:
- Server starts and participant can reach `/docs`

Observation focus:
- Command selection confidence
- Confusion about environment activation

Metrics:
- Completion
- Time
- Errors

### Task 2: Preview Index Plan

Prompt:
- "Preview indexing for `./src` before running full index."

Success criteria:
- Participant executes `/index/preview` correctly

Observation focus:
- Request payload comprehension
- Understanding of why preview happens first

Metrics:
- Completion
- Time
- Wrong payload attempts

### Task 3: Index and Monitor Progress

Prompt:
- "Run index with custom storage and monitor until completion."

Success criteria:
- Index request and status polling pattern are both correct

Observation focus:
- Mental model for async tasks
- Error recovery when request fails

Metrics:
- Completion
- Time
- Wrong-endpoint attempts

### Task 4: Query for Architecture Insight

Prompt:
- "Ask a question about architecture and explain source evidence."

Success criteria:
- Successful query and accurate interpretation of sources

Observation focus:
- Output trust and evidence understanding

Metrics:
- Completion
- Time
- Misinterpretation count

### Task 5: Run One Analysis Task

Prompt:
- "Run structure or health analysis and summarize the finding."

Success criteria:
- Correct task endpoint usage and useful interpretation

Observation focus:
- Task discoverability
- Clarity of task output

Metrics:
- Completion
- Time
- Confusion events

## 4. Debrief (10m)

- What was easiest?
- What was most confusing?
- What would you change first?
- Confidence score (1-5)
- Satisfaction score (1-5)

## 5. Facilitator Notes

- Log direct quotes for all Critical/High findings.
- Capture precise step where failure first occurred.
- Record whether participant self-recovered or needed intervention.
