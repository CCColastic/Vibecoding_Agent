# mini_agent

This context describes the identities and conversation boundaries used by the
agent application.

## Language

**Owner**:
The local identity that owns one or more Sessions.
_Avoid_: User account, Agent owner

**Session**:
A durable, isolated conversation belonging to one Owner.
_Avoid_: Runtime, Window

**ActiveConversation**:
An in-memory view of one new or resumed Session used for consecutive user turns.
_Avoid_: Session cache, Runtime

**Turn**:
One user message and all assistant and tool messages produced while answering it.
_Avoid_: Step, Run

**Run**:
The temporary execution of the Agent Runtime for one Turn.
_Avoid_: Session, Conversation

**Run ID**:
The stable identity of one execution attempt, shared by its messages and Trace.
Submitting another question or retrying a failed Run creates a new identity.
_Avoid_: Turn ID, Session ID, Tool call ID

**Trace**:
The ordered record of one Run's input, assistant outputs, tool activity, and
outcome. It is independent of the Session's Effective History.
_Avoid_: Chat history, Summary

**Effective History**:
The information retained for continuing a Session: an optional Rolling Summary
and the Turns that have not been summarized. It is not a complete transcript.
_Avoid_: Archive, Full history

**Rolling Summary**:
A condensed account of earlier Turns that preserves goals, constraints, key
results, and unfinished work. A newer summary supersedes the previous one.
_Avoid_: Turn, System instruction
