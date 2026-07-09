# NapMem Comparison

## Why It Belongs Beside HOLA

HOLA studies an exact bounded cache beside a compressive recurrent state: memory is internal to the model's computation.

NapMem studies external text memory as a tool/action interface: memory is outside the model and the agent learns how to navigate it.

These are complementary, not competing:

| Dimension | HOLA | NapMem |
|---|---|---|
| Memory substrate | key-value cache | text/JSONL/Markdown memory bank |
| Read policy | attention over selected cache entries | tool-call navigation over pyramid levels |
| Granularity | token/fact-level exact recall | raw message, record, topic, profile |
| Training target | recall through model architecture | query-time tool-use policy |
| Local question | what should an exact cache retain? | which abstraction level should the agent inspect? |

## Shared Story

Both projects argue for **learned read policy**. HOLA learns/uses a salience score for exact cache reads; NapMem learns a tool policy for textual memory reads. A useful portfolio figure can show a spectrum:

`recurrent state → HOLA exact cache → NapMem text pyramid → AutoMem learned write policy`

## Shared Risk: Read When Needed

The NapMem-VN repro now has a concrete reward finding: a usage bonus can penalize
correctly skipping memory. In a mixed prompted batch, the navigator answered both
memory and non-memory questions correctly, skipped memory for the non-memory
questions, and still scored lower under `F+C+U` than under `F+C`.

That lesson also matters for HOLA. A learned cache read policy should not optimize
for cache access by itself; it should optimize calibrated access. The portfolio
claim becomes stronger if both projects track:

- answer accuracy
- memory/cache access rate
- unnecessary access rate on non-memory tasks
- cost-aware reward without an unconditional usage bonus

## Experiment Bridge

Use the same synthetic facts with distractors:

1. HOLA: can the cache retrieve the buried fact?
2. NapMem: can the tool navigator choose record/topic/raw evidence?
3. AutoMem+NapMem: can the system both write the fact into the right layer and later read the right layer?
4. Cross-project: does either learned read policy over-access memory when the answer does not require it?
