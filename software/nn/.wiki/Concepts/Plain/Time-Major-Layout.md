# Time-Major Layout — Plain Language

## What is it?

When you feed a sequence of data to a spiking neural network, you have a 3D block of numbers: **time steps × batch samples × features**. The time-major layout is just a specific way of flattening that 3D block into a 2D table so it fits the library's 2D Tensor format.

Think of it like a stack of photos: instead of grouping all photos from the same album together (batch-first), you group all photos taken at the same moment together (time-first).

## The rule in one sentence

> All time-step 0 samples come first, then all time-step 1 samples, and so on.

If you have T=5 time steps and B=4 samples, the table has 5×4 = 20 rows ordered like this:

```
row 0:  time=0, sample=0
row 1:  time=0, sample=1
row 2:  time=0, sample=2
row 3:  time=0, sample=3
row 4:  time=1, sample=0
...
row 19: time=4, sample=3
```

## The golden rule

The number of rows must be exactly divisible by the number of time steps. If not, the code throws an error with the message `"LifBPTT: Input rows must be divisible by time_steps"`.

## Why does this matter?

The spiking neuron layer (`LifBPTT`) keeps a hidden memory (the membrane potential). It needs to process all samples at time=0 together, update its state, then process all samples at time=1, and so on. If the rows were in the wrong order, the memory would be wrong and the neuron would fire at the wrong times.

## Common mistake

If you accidentally use batch-first order (all samples for time=0, time=1... grouped by sample instead of by time), the network will train but give strange results — and it won't throw an error, it will just silently learn the wrong thing.

## Quick checklist

- [ ] `input.rows() % time_steps == 0`
- [ ] Rows ordered: all t=0 rows first, then all t=1 rows, etc.
- [ ] Call `reset_state()` between independent audio clips or EEG epochs

## See Also

- [Time-Major Layout — Technical](../Time-Major-Layout.md)
- [Membrane Dynamics — Plain](./Membrane-Dynamics.md)
- [SNN and Surrogate Gradients — Plain](./SNN-and-Surrogate-Gradients.md)
