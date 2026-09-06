---
name: didactic-explanation
description: "Explain concepts, symbols and failures from first principles so a newcomer understands; required for any why/what-does-this-mean answer, wiki page, or rationale comment."
---

# didactic-explanation

Goal
- Explain any concept, symbol, or failure so a reader who has never seen this codebase
  understands it — building from first principles, not from vocabulary they lack.

Trigger
- Any time you explain what something *means*: a config field, an invariant, a bug, a
  design decision, a test failure, a wiki page, or a code comment answering "why".
- Also when the user says "I don't understand", "be more didactic", "what does X mean",
  or asks the same question twice — the second ask means the first answer failed.

Rules

- RULE: WHY_BEFORE_WHAT
  DO: Open with the problem the thing exists to solve. ("A spiking neuron can't act on a
      single snapshot — it needs a sequence to charge up in.")
  AVOID: Never open with a definition, a signature, or a type. A definition given before
      the motivation is memorised, not understood.

- RULE: CONCRETE_NUMBERS
  DO: Use one real example carried all the way through, with actual values from the
      project (256 features, T=16, 32 rows).
  AVOID: Avoid abstract placeholders (foo, X, "some value") and avoid switching examples
      mid-explanation.

- RULE: SHOW_THE_SHAPE
  DO: Draw it. An ASCII table, a row layout, a spike raster, a mermaid flow — whatever
      makes the structure visible.
  AVOID: Do not describe a data layout in prose alone. If it has rows, columns, or an
      order, the reader must be able to SEE it.

- RULE: CONTRAST_TABLE
  DO: When two things are confusable, put them side by side in a table with a column that
      answers what each one *answers* ("how long" vs "how many").
  AVOID: Never explain only the one being asked about; the confusion lives in the pair.

- RULE: SAME_INPUT_DIFFERENT_MEANING
  DO: Where a setting changes interpretation rather than value, show the SAME input under
      both settings and state the consequence of each.
  AVOID: Do not say a setting is "important" without demonstrating what breaks.

- RULE: NAME_THE_FAILURE
  DO: State what goes wrong if it is misunderstood, and whether it fails loudly or
      silently. Silent failures deserve the most words.
  AVOID: Never leave the reader thinking a mistake would be obvious when it would not be.

- RULE: PLAIN_WORDS_FIRST
  DO: Use the ordinary word, then introduce the technical term once the idea has landed.
      ("a little movie instead of a single photo" → then "sequence" → then "BPTT unroll").
  AVOID: Avoid jargon chains (time-major, unroll, straight-through estimator) before the
      underlying idea exists in the reader's head.

- RULE: LAYERED_DEPTH
  DO: Structure so the reader can stop early: intuition first, mechanics second, code
      third, edge cases last. Use headings so the layers are skippable.
  AVOID: Do not front-load the hardest paragraph.

- RULE: ADMIT_THE_TRAP
  DO: If the design itself is confusing (near-identical names, inverted sign, misleading
      label), say so plainly and propose the fix.
  AVOID: Never defend a confusing design by explaining it harder. If you had to re-check
      it while writing, the reader will too — say that.

- RULE: NO_FALSE_FLUENCY
  DO: Mark what is measured vs. reasoned. ("Verified by test X" vs "from reading the code,
      unverified".)
  AVOID: Never let a smooth explanation imply evidence you do not have.

Checklist before sending an explanation
- [ ] Does it open with the problem, not the definition?
- [ ] Is there one concrete example with real numbers, carried throughout?
- [ ] Is the structure drawn, not just described?
- [ ] Are confusable things contrasted in a table?
- [ ] Is the failure mode named, and its loud/silent nature stated?
- [ ] Could a reader stop after the first section and still be correct, just less complete?

Worked reference
- `.wiki/Concepts/Time-Steps.md` is the canonical example of this format: motivation →
  spike-raster picture → flat-tensor problem → the one line of code → same-tensor-two-
  meanings table → why the default was banned → pitfalls.
