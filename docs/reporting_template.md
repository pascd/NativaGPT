# Security Report Template

Use this template when reporting a security vulnerability per [`SECURITY.md`](../SECURITY.md). Send it privately (do not open a public GitHub issue) to the contact listed there.

```
## Summary
A one- or two-sentence description of the vulnerability.

## Affected version / commit
e.g. v0.1.0, or a specific commit hash.

## Component
Which part of NativaGPT is affected (e.g. LLMPromptHandler, the REST API,
an MCP server, CommandExecution)?

## Steps to reproduce
1. ...
2. ...
3. ...

Include a minimal reproduction if possible (config snippet, request
payload, command run).

## Impact
What can an attacker do with this? (e.g. arbitrary command execution,
credential leakage, denial of service.)

## Suggested fix / mitigation
Optional - if you have a proposed fix or workaround.

## Reporter contact (optional)
How you'd like to be credited/contacted about the fix, if at all.
```
