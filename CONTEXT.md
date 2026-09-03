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
