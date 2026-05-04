- User prompt: 
Make sure that these tests do not test thing like "the output is non zero" because this is too generic, the tests must verify that, for a given deterministc input value, a deterministic result must be generated.

- Agent thinking so far: 
The user wants me to replace generic "nonzero" tests with specific deterministic known-value tests. Let me audit the new tests I added to identify which ones are too generic:

LeakyTest.VthreshGradNonzero — tests != 0 → too generic

LeakyTest.VthreshGradSign — tests < 0 → borderline, but at least directional

LeakyTest.AllParamsUpdateAfterAdamStep — tests NE (changed) → needs known values

LeakyBPTTTest.AllParamsUpdateAfterAdamStep — also just checking for changes, not actual values

LeakyIntegratorTest.KnownValueForward — this one's solid with actual known values

LeakyIntegratorTest.VthreshGradAlwaysZero — exact equality check that validates a design constraint

LeakyIntegratorTest.RCParamsUpdateAfterAdamStep — another generic change detector

TdBNTest.AllParamsUpdateAfterAdamStep — same issue with just verifying updates happened

MaxPoolTest.MaxPool1dBackwardRoutesToArgmax — checking exact routing with 0 and 1 values, that's good

MaxPoolTest.MaxPool2dBackwardRoutesToArgmax — same solid routing verification

LSTMGateTest.BiasGradNonzero — just checking something's positive, too vague

SimpleResNetTest.KnownValueDepth0 — validates shape and finiteness but missing the actual known value part

SimpleResNetTest.BackwardGradNonzero — another generic positivity check
I'm also noticing I added ResNetBlockTest.ForwardDoesNotCrash which only checks shape, another one that's too loose.

Looking at what I've already added, ResNetBlockTest.OutputNonzero and the crash test are both too generic. I should focus on replacing my new tests with deterministic ones using computed expected values. For the Leaky test, I need to work through the backward pass analytically — with x = 2.0 and V_th = 0.5, I can calculate the surrogate gradient and then the exact gradient with respect to V_th.

I need to check the actual ExponentialSurrogate implementation to get the precise formula before I can compute the expected gradient value.

Now I'm thinking through which tests need to be fixed. Some tests like the threshold gradient and bias gradient ones can express exact surrogate values instead of just checking for nonzero results. For the Adam step tests, I need to compute the actual expected parameter values after one update cycle using the learning rate and moment estimates.

Working through the Adam math with the given hyperparameters, I can see that after the first step, the parameter update simplifies to approximately 0.01 times the sign of the gradient. For the voltage threshold gradient in the Leaky neuron, I need to work out what the gradient sum actually equals.

Actually, I think there's a cleaner way to approach the training loop tests. Instead of trying to verify the exact Adam updates, I can use a deterministic setup: compute the expected gradient analytically, then use SGD with a learning rate of 1.0 so the parameter change is just the negative gradient. This makes it straightforward to verify that the new parameter value matches what I calculated.

Now I'm reviewing which tests are too loose with their assertions. Several of them just check that parameters changed or gradients are nonzero, but they should instead verify exact expected values. Tests like the Vthresh gradient, Adam update steps across different modules, and bias gradients all need to be tightened up to compute and assert against the precise expected values rather than just checking for any change.

The ones that are already solid either test exact numerical values, verify specific gradient properties like always being zero, or route gradients to the correct positions. A few are borderline—like the sign check on Vthresh gradient—but at least they're directional.

Now I'm planning how to replace the loose tests with deterministic ones that compute the exact expected values upfront.

For the exponential surrogate test, I need to check what formula it actually uses in the codebase. Looking at the LSTM implementation, the exponential surrogate is the default, so I should read through that to understand the exact computation.

The LSTM bias gradient test is trickier since the layer has random initialization even with a seed, making it hard to compute the expected gradient analytically. I think the better approach is to construct a simpler scenario with known fixed weights and compute the gradient by hand, or use all-zero inputs and weights to get a predictable result.

For the training loop tests, the real issue is that "NE" (not equal) checks are too loose — they just verify something changed rather than checking for a specific expected output. The user wants deterministic tests: given input X, expect output Y. I should focus on either using SGD with learning rate 1.0 so I can compute the exact parameter update analytically, or verify the gradient value itself is correct before checking if it's applied properly.

Now I'm working through a concrete example with the LeakyImpl neuron: dt=1, R=1, C=1 giving beta≈0.36788, V_th=0.5, input x=2.0. The membrane potential starts at 0, updates to 2.0, and since 2.0 > 0.5 we get a spike. To compute the exact gradient through the surrogate, I need to check the ExponentialSurrogate formula in the actual code.

My plan is to read that file, calculate the precise gradient values, then rewrite the test cases with exact expected values for V_th, R, C gradients across the different test classes.

For the training loop tests, I'm realizing the cleanest approach is to isolate what I'm testing: compute the exact gradient analytically with zeroed non-tested parameters and minimal input, then verify the gradient matches that expected value directly rather than checking if it changed after an optimizer step. This avoids the complexity of Adam's moment estimates and keeps the test focused on gradient correctness.

Now I'm looking at specific test cases to make deterministic. For LeakyTest, I need to check the ExponentialSurrogate formula. For LSTMGateTest.BiasGradNonzero, I can use a minimal setup with input_size=1, hidden_size=1, single timestep, and zero weights to compute the exact bias gradient. I'm planning to replace the current non-deterministic tests with exact-value versions: VthreshGradSign becomes VthreshGradExact using a simple 1-neuron case, and I'll do the same for other gradient tests. I need to finish mapping out the test replacements, then pivot to the core strategy: instead of relying on Adam's opaque updates, I'll set exact gradient values manually and verify against known expected parameter changes. For the Leaky neuron tests specifically, I'm working through the math with concrete values—setting dt=1, R=1, C=1, V_th=0.5 with reset_zero=true, then tracing two steps where the first has no gradient and the second does, so I can compute the exact membrane voltage and gradient contributions.

Now I need to check the ExponentialSurrogate implementation to get the precise gradient formula, then I can calculate the exact expected values for the test assertions.

Need to check ExponentialSurrogate formula to compute exact expected values.

Read ExponentialSurrogate.hpp
Read LeakyIntegrator.hpp (lines 81-180)
Now I have all the formulas. Let me compute exact values for each test.

ExponentialSurrogate formula:

surrogate(v, V_th) = (1/sharpness) * exp(-|v - V_th| / sharpness)
Default sharpness = 1.0.

LeakyTest exact values
Two-step setup:

dt=1, R=1, C=1, V_th=0.5, sharpness=1.0

Step 1 (no grad): x_warm = 0.3

v_mem_t_minus_1 = 0
v_mem = exp(-1)*0 + 0.3 = 0.3
no spike (0.3 < 0.5)
v_mem stays at 0.3
Step 2 (with grad=1.0, single neuron (1,1)):

v_mem_t_minus_1 = 0.3
beta = exp(-1/(1*1)) = exp(-1) ≈ 0.367879441
v_mem = exp(-1) * 0.3 + x_val = 0.367879441 * 0.3 + 2.0 = 0.110
363832

v_mem_pre_spike = 2.110363832
spike fires (2.110363832 > 0.5)
Now I'm computing the voltage threshold gradient using the surrogate derivative. The absolute difference between the membrane voltage and threshold is about 1.61, and applying the exponential surrogate gives roughly 0.19.

Let me recalculate this more carefully by breaking down the exponential into parts. I'm getting approximately 0.1998 when I compute exp(-1.610363832) step by step, but I need to maintain enough precision for the gradient calculation.

Now I'm working through the symbolic form of the surrogate function and its gradient with respect to the threshold. The key is that v_pre exceeds the threshold, so the absolute value simplifies, and I can express the gradient as a negative exponential term. Computing the numerical values: 0.3 times e^(-1) gives about 0.11036, which adds to 1.5 to give roughly 1.61036, and then I need the exponential of the negative of that.

Computing the gradient with respect to beta...

Now I'm working through the chain rule for the resistance and capacitance gradients. Since both R and C appear symmetrically in the exponential decay term, their derivatives with respect to beta are identical when R=C=1, which means the loss gradients for both parameters end up being equal: 0.3 times the surrogate gradient times exp(-1).

Setting up the test values in code to verify this calculation... Now I'm computing the gradient flow through the beta parameter using the chain rule—the gradient with respect to R and C both work out to 4.0 times exp(-1) based on how beta depends on those resistance and capacitance values. I'm moving on to verify the exact gradients for a test case with non-uniform batch dimensions and gate values, starting by setting up the forward pass with the given threshold voltage and time step scaling.

For the single time step, I'm calculating the batch statistics—the mean across the two samples for each feature is [2.0, 4.0, 6.0], and then the variance for each feature works out to 1.0, 4.0, and 9.0 respectively by taking the squared deviations from the mean.

Now I'm normalizing each sample by subtracting the mean and dividing by the standard deviation (square root of variance plus epsilon), which gives me -1.0 for the first sample across all features and 1.0 for the second sample due to symmetry. Then I'm applying the learned scale and shift parameters—since gamma is all ones and beta is all zeros, the scaled output equals the normalized values.

For the gradient with respect to gamma, I'm summing the element-wise products of the output gradients and normalized inputs across the batch, then multiplying by the scale factor. This gives me -1.0 for each feature. But I should double-check my understanding of how the threshold-dependent batch norm works here. Now I'm working through the specific gradient calculations for d_gamma and d_beta with the given values, getting -1.0 and 3.0 respectively. But I need to verify whether the backward implementation actually follows this formula or if there's something different about how it handles the normalized cache and scaling.

Rather than manually computing all these exact values by hand, which is error-prone, I'll take a more practical approach: for each test case, I'll compute the expected gradient analytically using the same mathematical formulas right in the C++ test itself, then assert that the computed gradient matches using approximate equality. This way the test remains deterministic without hardcoding magic numbers.

Writing test assertion code... Now I'm mapping out which tests need to be renamed and consolidated—the key insight is that instead of checking for nonzero gradients or specific signs, I should compute the exact expected values mathematically and verify them precisely. I'm planning to merge some redundant tests and rewrite all of them to use deterministic assertions based on the actual math rather than loose heuristics.

Let me work through the exact calculations for each test case, starting with the leaky neuron tests where I'll set up specific parameters and trace through the forward and backward passes step by step to derive what the gradients should be. 0.3*exp(-1) + 2.0

spike = 1 (v_mem_pre_spike > 0.5)
surrogate = exp(-|v_mem_pre_spike - 0.5|) = exp(-(v_mem_pre_spike - 0.5)) [since v > V_th]
grad_v_pre = 1.0 * surrogate
V_th gradient:

dL/dVth = -sum(grad_v_pre) = -surrogate
R gradient:

d_beta_dR = beta * 1/(1 * 1^2) = beta = exp(-1)
dL_dbeta = grad_v_pre * v_mem_t_minus_1 = surrogate * 0.3
dL_dR = surrogate * 0.3 * exp(-1)
C gradient follows the same pattern as R since they're both 1, giving dL_dC = surrogate * 0.3 * exp(-1). Now I'm setting up a test that computes these gradients numerically to verify the surrogate derivative is working correctly. Looking at the gap analysis, the gradient tests for LeakyBPTT already exist in the test suite—the only missing piece is the training loop test. Since the backward pass uses full BPTT which is complex to verify by hand, I'll focus on checking that all parameters receive nonzero gradients and actually update during training.

Rather than relying on Adam's moment estimates which would make expected values hard to compute, I could use vanilla SGD with a learning rate of 1.0, which simplifies the math to new_param = old_param - grad. Or even simpler: I can just run forward and backward passes separately to verify the exact gradient values directly, without needing to test the optimizer step itself.

So I'll remove Adam from the training loop tests entirely and instead assert that the computed gradients match expected values exactly. This isolates what I'm actually testing—the gradient computation—from optimizer behavior, which gets tested elsewhere.

Setting up the test...

Computing exact gradient values... Writing gradient computation... I'm working through the forward pass for a batch normalization test with specific parameters. Let me compute the mean and variance for each feature across the batch, then trace through the normalization and scaling steps to verify the gradient calculations. Now I need to work through the backward pass for this threshold-dependent batch norm layer. The forward pass applies scaling by V_th/sqrt(T) after the standard batch norm transformation, so I need to account for that in the gradients. Let me trace through how the gradients flow back through both the scaling factor and the normalization step. Finishing the gradient calculations for the remaining features, then I need to verify whether the backward pass actually uses the normalized cache and whether it normalizes by batch size. Let me check the TdBN backward implementation to understand the exact computation, and then set up a simple test case with minimal dimensions to validate my understanding. I'm working through a specific LSTM configuration with minimal parameters to make the math tractable—setting the input and weight matrices to zero, then carefully choosing bias values to control the gate activations. With forget bias set to 1 and others at 0, I can trace through exactly how the cell state and hidden state evolve, which will give me concrete gradient values to verify against the implementation. Continuing with the gradient calculations for the remaining gates and computing the bias gradients, which gives me the pre-activation derivatives that feed into the final bias gradient vector.

Now I'm realizing the bias gradient is exactly [0, 0, 0, 0.25] for the input, forget, output, and candidate gates respectively. The key constraint is that the LSTM layer initializes the forget gate bias to 1.0, so if I set the weight matrices to zero and explicitly configure the bias as [0, 1, 0, 0], I can verify the computation exactly. W_, U_: random normal (seed 42)

b_: all zeros except b_[H:2H] = 1.0 (forget gate)
If I use input_size=1, hidden_size=1:

b_ initialized to [[0], [1], [0], [0]] (i=0, f=1, o=0, g=0)
W_ is random (seed 42, stddev=0.05)
U_ is random (seed 43, stddev=0.05)
With x=[[0.0]] (1×1):

pre = xW^T + hU^T + b = 0W^T + 0U^T + b = b = [[0], [1], [0], [0]] So the same as my calculation above! Good.
With the default initialization and x=[[0.0]], the gradient with respect to b is [[0], [0], [0], 0.25]]. Since the input is zero, the W_ term vanishes regardless of its random values, and with zero initial hidden state, the pre-activation just becomes b. Let me verify this for T=1 with zero input: the LSTM computes pre-activation as the sum of input-weighted, hidden-weighted, and bias terms. Now I'm working through the backward pass, computing gradients for each gate. The output gradient flows back through the tanh and sigmoid functions to get contributions for the output, input, and forget gates, with the cell state gradient being the key intermediate that distributes to all three. I'm computing the gradient for the bias term by transposing the pre-activation gradients, which gives me [[0], [0], [0], [0.25]] — exactly what I'd expect for this single timestep with batch size 1. Now I need to verify that the LSTM implementation actually accumulates these bias gradients correctly across time steps during backpropagation.

Actually, let me step back and focus on what matters: I should write tests with known deterministic inputs and outputs, then use approximate equality checks to handle any floating-point precision issues. The key insight from the user's feedback is that tests need to verify deterministic behavior for given inputs, so I should identify which of my existing tests actually meet that criterion and which ones need to be rewritten or removed.

Looking at the problematic tests, I need to rename several of them to be more explicit about what they're checking. Tests like VthreshGradNonzero should become VthreshGradExact with actual expected values, and tests that just check for updates after Adam steps should be replaced with tests that verify the exact gradient values being computed. I'm going through each test class—LeakyTest, LeakyBPTTTest, LeakyIntegratorTest, TdBNTest, LSTMGateTest, SimpleResNetTest—and deciding whether to rename them for exactness, merge related tests, or replace them entirely with gradient verification tests.

For SimpleResNet specifically, since the layer weights aren't exposed publicly, I can't set them to known values from outside the test. Instead, I'll use finite differences to verify the backward pass—it's deterministic and exact, and there's already a helper function in the file for this.

Writing test code... Writing the finite-diff test... The issue is that the lambda needs a fresh net instance for each call, but I can't easily deep-copy the network without a copy constructor. I think the simpler approach is to drop the finite-diff test for SimpleResNet altogether since the forward pass and shape validation already cover the essential behavior, and the individual Linear layers are tested separately anyway. So my plan is to consolidate the Leaky tests and refactor which tests actually need to stay. AllParamsUpdateAfterAdamStep into:

LeakyTest.AllParamsGradExact — computes exact R, C, V_th gradient analytically
LeakyBPTT: Replace AllParamsUpdateAfterAdamStep with:

Check exact V_th gradient (since it's simpler in BPTT — at least the V_th grad must be negative/nonzero)
Actually for BPTT, the full exact gradient computation involves backward BPTT loop with recurrent terms. Too complex for inline math.

Alternative: use T=1 (single step) for LeakyBPTT which reduces to essentially LeakyImpl single-step behavior. With T=1, the BPTT backward is equivalent to the single-step backward.

With T=1, B=1, F=1: same as LeakyImpl.

LeakyIntegrator: Replace RCParamsUpdateAfterAdamStep with exact gradient values.

TdBN: Replace AllParamsUpdateAfterAdamStep with exact d_gamma and d_beta values.

LSTM: Replace BiasGradNonzero with exact gradient values for the simple case.

SimpleResNet: I could keep the known value at depth 0 and add exact output by zeroing weights, then replace the backward check with finite difference validation using linear composition. Actually, since it's just composing Linear and ReLU with residual connections, let me reconsider the approach entirely.

The user expects every layer to have tests with given inputs, hyperparameters, and expected outputs. For SimpleResNet, I can use depth=0 with all zero weights to get zero output, then verify backward gradients by computing them through the residual block itself.

Actually, let me use a simpler approach: set depth=0 and manually configure the weights to known values like identity matrices, so I can trace through the forward and backward passes with predictable results.

With a basic two-layer network (fc_in → ReLU → fc_out) where all weights are identity and biases are zero, feeding in [1.0, 0.5] should pass through unchanged to [1.0, 0.5]. Now I'm working through the backward pass: the gradient flows back through fc_out (identity), then ReLU (both activations positive so gradient passes through), and finally into fc_in to verify the weight gradients are computed correctly.

The issue is that Kaiming initialization gives random weights by default, so I need to manually override them after construction by accessing the model's parameters in order: fc_in.weight, fc_in.bias, fc_out.weight, fc_out.bias.

Now I'm setting these parameters to known values so the test becomes fully deterministic, then I'll verify the backward pass computation for the batch norm layer matches my expectations.

Need to read TdBN backward before computing exact expected values:

Read ThresholdDependentBatchNorm.hpp (lines 81-230)
Now I have the full TdBN backward. Let me compute the exact expected values.

TdBN backward formula:


d_gamma[f] += (go[b,f] * tdbn_scale) * x_norm[b,f]   (for each b)
d_beta[f] += go[b,f] * tdbn_scale                      (for each b)
where tdbn_scale = V_th / sqrt(T)

With T=1, V_th=1.0 → tdbn_scale = 1.0

Now I'm working through a concrete example with F=3 features and B=2 batch samples. I have the input values, output gradients, and the normalized activations computed during the forward pass for each feature dimension.

For each feature, I'm calculating the gradient with respect to gamma by multiplying the output gradient by the normalized input and summing across the batch. The gradient for beta is simpler—just the sum of output gradients across the batch. Working through the math, I get d_gamma = [-1.0, -1.0, -1.0] and d_beta = [3.0, 3.0, 3.0].

Now I'm moving on to verify the LSTM bias gradient. I need to trace through the forward pass with hidden_size=1, input_size=1, and a single timestep, starting from zero hidden and cell states. The bias vector is organized by gates—input, forget, output, and cell candidate—so I need to confirm the exact row ordering before computing the gradients. Now I'm working through the backward pass for this LSTM step, computing the gradients for the gates and cell state given that the output gradient is 1.0 and all previous states were zero. Now I'm computing the pre-activation gradients by applying the sigmoid/tanh derivative chain rule to each gate, which gives me zeros for the input, forget, and output gates since their upstream gradients are zero, while the candidate gate gradient depends on the input gate value. I need to verify that the LSTMLayer implementation computes the bias gradient correctly—specifically whether it accumulates db_ by transposing and summing the gradient matrix as the BPTT equations dictate, which for my single timestep and batch should give me [[0.0], [0.0], [0.0], [0.25]]. I'm trying to figure out whether the backward pass sets the gradient directly on b_ or accumulates it in a separate db_ member. Looking at the params() method, it returns pointers to W_, U_, and b_, so the optimizer will read b_.grad() during the step. I need to check the backward implementation to see if it calls b_.set_grad(db_) or handles the gradient differently, and the existing tests suggest b_ itself needs to hold the gradient value.

My approach is to write out the test cases with expected analytical values and run them to see if they pass—if not, I'll debug from there. I'm planning to remove the old gradient sign tests and replace them with a more comprehensive test suite. Writing test code...

For the LeakyBPTT case with T=1, I need to think through how the input dimensions work—with a single timestep and batch size 1, the input shape becomes (1, 1), so there's just one forward pass to consider. The backward pass through BPTT should produce the same gradients as the single-step Leaky implementation when T=1, so I'm setting up matching parameters and verifying that behavior. two-step protocol in LeakyBPTT:

Step 1 (no grad): LeakyBPTT.forward([[0.3]], false) But LeakyBPTT requires T=1, so it processes 1 time step. v_mem becomes 0.3.
Step 2 (with grad): LeakyBPTT.forward([[2.0]], true) Now v_mem_t_minus_1 = 0.3. After decay+input: v_mem_pre = exp(-1)*0.3 + 2.0. Spike!
The backward in BPTT for T=1 is essentially:

For t=0 (the only time step):
v_pre = v_mem_history[0] = betav_prev + input = beta0.3 + 2.0
surrogate = exp(-|v_pre - V_th|)
grad_v_pre = grad_out * surrogate
For the V_th gradient, the full BPTT formula includes both a direct spike term and a recurrent reset-path term, but with T=1 there's no next time step, so the recurrent component vanishes and I'm left with just dL/dVth = -surrogate, matching the single-step case. The R gradient uses v_post_history rather than v_pre for the backward pass. eta is the state BEFORE t=0

When I run two sequential forward passes—the first without gradients ending at v_mem=0.3, then the second with gradients starting from that same state—I need to be careful about what v_post_history actually stores. It's the post-reset state after computing v_pre, not the pre-reset value. This is getting tangled, so let me just test with T=1 and compare against the analytical single-step Leaky result—if they match, the implementation is correct. The real question is what v_prev represents in the BPTT d_beta calculation.

Looking at the backward pass more carefully, I think the safest approach is to warm up with requires_grad=false to initialize v_mem, then run a single forward step with requires_grad=true and verify the gradients match the analytical formula. The key subtlety is that v_post_history[0] stores the post-reset state at t=0, and the backward formula for d_beta uses v_post_history rather than the pre-reset membrane potential.

Now I'm tracing through what happens in the second forward call: at t=0, the forward computes v_pre from the previous state (0.3), generates a spike, applies the hard reset to get v_post=0, and stores that in v_post_history[0]. In the backward pass, I need to figure out whether the gradient with respect to beta uses the current v_post_history or references the previous timestep's state.

The key issue is that BPTT iterates backward through time, so at the last step it needs the "previous post state" from before that timestep. But v_post_history only contains states from the current forward call. For a single-step forward (T=1), the state before t=0 is actually the persistent v_mem from the previous forward call (0.3), not something in the current history. I need to check how LeakyBPTT handles this—it likely stores v_mem at the start of forward and uses that as the initial state for the backward pass. I need to understand what state the backward pass uses at t=0—specifically whether it accesses the v_mem value from before the forward pass started. The v_post_history stores the post-reset states for each time step, but the gradient computation at t=0 requires knowing the previous membrane potential. Let me look at the actual backward implementation in LeakyBPTT to see how it handles this initial state, rather than continuing to reason through the theory. the following setup that I KNOW will give a deterministic result:

T=1, B=1, F=1
Single warm-up call: forward([[0.3]], false) to set v_mem=0.3
Then: forward([[2.0]], true), backward([[1.0]])
Expected gradients: same as LeakyImpl single-step case (since T=1 BPTT degenerates to single-step)
BUT: the issue is whether BPTT at T=1 with warm-up produces the same d_beta term as LeakyImpl single-step.

In LeakyImpl: dL_dbeta = grad_v_pre * v_mem_t_minus_1 where v_mem_t_minus_1 is captured before decay.
In LeakyBPTT: the d_beta calculation might use the post-activation history from the warm-up call OR it might only reference the initial state from the current forward call.

The core issue is that BPTT at T=1 doesn't have a clear "previous state" within its own history—it either needs to reference the warm-up state or rely on the persistent v_mem. Rather than untangle this without seeing the backward implementation, I'll switch to a cleaner test: T=2 with the first step producing a nonzero post-activation value and the second step spiking. This way the "previous post state" for the second timestep is well-defined from within the same forward call. At t=1, the membrane potential reaches the spike threshold with the exponential decay factored in, triggering a hard reset to zero. Now I'm working through the backward pass: the gradient with respect to the pre-synaptic voltage uses the surrogate gradient function, and I need to accumulate the contribution to the beta gradient by multiplying through the previous post-synaptic state. For t=0, I'm computing the surrogate gradient at the sub-threshold voltage and continuing the backpropagation chain.

Once I have the total beta gradient, I can derive the gradients for the resistance and capacitance parameters by scaling through the exponential decay factor. For the threshold voltage gradient, I need to account for both the direct term from the spike at t=1 and any recurrent contributions through the reset pathway.

Actually, this full BPTT derivation is getting unwieldy with all the recurrent terms. Let me switch to a simpler approach and just verify the LeakyBPTT implementation directly. with gradient inputs set to zero at t=0 and nonzero at t=1, so that t=0 contributes nothing to the V_th gradient, and only t=1 contributes.

Even then I need to work through the full BPTT formula.

Actually, the cleanest approach for LeakyBPTT is setting T=1 like single-step Leaky, using a fresh v_mem with no warm-up to avoid complications with the previous state when computing dL_dbeta. But with fresh v_mem = 0, that gives dL_dbeta = 0, which zeros out the R and C gradients too—same problem.

Unless I use T=2 with no warm-up: at t=1, the previous post state is v_post[0] from the current call, which is nonzero if t=0 didn't spike. Let me trace through with T=2: at t=0 with input 0.3 and v_mem_init=0, I get v_pre[0]=0.3 (no spike) and v_post[0]=0.3; at t=1 with input 3.0, v_pre[1] becomes beta*0.3+3.0, which spikes, giving v_post[1]=0. Then dL_dbeta sums the gradient contributions, where the t=1 term multiplies grad_v_pre[1] by the nonzero v_post[0].

This matches the warm-up approach and is fully deterministic from the initial state since v_mem starts at 0 for a fresh layer. Now I need to understand the BPTT formula for the V_th gradient by reading through the backward pass carefully.

Need the BPTT backward code for exact V_th computation:

Read LeakyBPTT.hpp (lines 201-400)
Now I have the full BPTT backward formula. Let me trace through it for T=2, B=1, F=1.

BPTT backward setup (T=2, B=1, F=1, dt=1, R=1, C=1, V_th=0.5, reset_zero=true):

Constants:

beta = exp(-1)

d_beta_dR = beta * 1 / (1 * 1 * 1) = beta * 1 = exp(-1) [since tau = RC = 11 = 1 > 1e-12, and raw_R > kMin]

d_beta_dC = exp(-1) as well

Now I'm setting up the input and gradient tensors. The input has two timesteps with values 0.3 and 3.0, and the gradient output is 1.0 at both timesteps. Let me trace through the forward pass starting from zero membrane potential.

At t=0, the pre-synaptic voltage is just the input 0.3, which doesn't exceed the spike threshold of 0.5, so no spike occurs and the membrane potential stays at 0.3. At t=1, the membrane potential decays and accumulates the new input, giving approximately 3.09, which crosses the threshold and triggers a spike, resetting the membrane potential to zero.

Now I'm working through the backward pass, starting from t=1 and moving back to t=0. I'm initializing the gradient accumulators and extracting the stored values from the forward pass to compute how changes in the threshold, resistance, and capacitance affect the loss.

At t=1, I'm calculating the exponential surrogate function based on the membrane potential, determining whether a spike occurred, and then computing the gradients with respect to the threshold using the chain rule—accounting for the reset behavior where the post-spike voltage depends on both the pre-spike voltage and the threshold. Now I'm working through the t=0 case, retrieving the membrane potential from history and computing the surrogate gradient using the exponential function with the threshold of 0.5, which gives me exp(-0.2) for this timestep. Finishing the threshold gradient accumulation and computing the final input gradient, which combines the surrograte derivative and next state contributions. The gradient with respect to the resistance parameter ends up being zero since the previous voltage was zero at the initial timestep. Now I'm consolidating all the accumulated gradients into their final forms, particularly the threshold gradient which involves the exponential decay terms and the surrograte function evaluations.

Now I can see the specific values: the surrograte at t=1 depends on the membrane voltage exceeding threshold, and both the resistance and capacitance gradients scale by the surrograte and the exponential decay factor. I'm implementing this in C++...

Writing gradient computation code... I'm working through the gradient calculations for the LSTM layer, verifying the derivatives for the threshold voltage, resistance, and capacitance parameters. Now I need to derive the bias gradient in the same way. Now I'm working through the backward pass, starting with the output gradient and computing how it flows back through each gate—the forget gate contribution is next. Now I'm checking how the bias gradient accumulation works in the LSTM backward pass—specifically whether db_ gets reset before accumulating or if it properly handles the gradient flow. Let me trace through the backward method implementation to verify the gradient is being set correctly on the bias parameter.

Actually, rather than getting bogged down in the implementation details, I should just write the test with the expected values and run it to see if it passes. If the gradient doesn't match, I can debug from there. With the setup I have—x=0, h_prev=0, and the specific threshold and resistance values—the gradient should be deterministic regardless of the random weights.

Now I'm ready to write out all the exact-value tests, replacing the generic ones with precise expected gradients for each parameter.

For the LeakyTest, I'll use a two-step warm-up: first pass x=0.3 without gradients to initialize the membrane potential, then on the second step with gradients enabled, I'll use x=2.0 and work through the exponential decay, surrogate gradient, and backprop chain to compute the exact gradient contributions for each parameter.

Now I'm working through the exact gradient calculations for the AllParamsGradExact test case with two timesteps and the given input sequence, computing the surrogate losses and their derivatives with respect to the threshold and resistance parameters. Looking at the gradient values for the batch normalization test with those specific dimensions and inputs, I see gamma and beta gradients both have consistent values across the batch. Then for the LSTM gate test, the bias gradient shows mostly zeros except for the last element. I'm also planning to set up a ResNet test that uses the params() method to establish identity weights and verify the forward pass produces the expected output. Now I need to write out all these test cases, replacing what I had before.

For the SimpleResNet with known weights, I'm trying to trace through how params() orders the tensors. With depth=0, the model contains an input linear layer, ReLU activation, and output linear layer all wrapped in a Sequential container. When I call params() on the Sequential, it concatenates the parameter pointers from each layer in order. The linear layers each return their weight and bias pointers, while ReLU contributes nothing. So for a SimpleResNet with those dimensions, the first parameter should be the input layer's weight matrix.

Now I'm working through a concrete example with specific weight values to verify the forward pass. I'm setting all weights to ones and biases to zeros, then tracing what happens when I pass in a simple input vector. I need to check whether the linear layer's matrix multiplication produces the shape I expect. So weight transpose gives me (2,3), and multiplying the input (1,2) by that yields (1,3) with all values of 2. After ReLU, those stay at 2 since they're positive. Then the second layer multiplies this (1,3) tensor by weights of shape (3,2) to get (1,2) output of [[6, 6]]. For the backward pass, I'm computing the gradient with respect to the fc_out input using the output gradient.

Now I'm tracing through the backward computation — the gradient flows back through the weight multiplication, then through ReLU (which preserves the gradient since all activations were positive), and finally back to the fc_in input, which should give [[6, 6]]. This is straightforward to verify. Let me write out the complete test suite with these deterministic values.

I should double-check the parameter ordering in SimpleResNet to make sure params[0] actually refers to fc_in.weight and not something else. Looking at how LinearImpl exposes its parameters...

Verifying parameter order... I'm setting the weight and bias tensors using fill() and setZero() methods, then tracing through the forward pass with ones and zeros to verify the output matches the expected result of [[6,6]], and now working through the backward pass to compute gradients. I'm verifying the matrix dimensions work out correctly for the backward pass through fc_in, and the gradient computation matches the expected output. Now I need to write out all the test cases, though I'm still figuring out some details about how the Linear layer implementation handles certain edge cases. I'm tracing through the backward pass for SimpleResNet to verify the gradient chain works correctly. Starting with grad_out=[[1,1]] at fc_out, it propagates backward through ReLU (which passes it through since all values are positive) and into fc_in, ultimately producing [[6,6]] as expected.

Now I'm writing out the complete test suite for LSTMGateTest, starting with BiasGradExact...

Writing test code... I'm setting up a two-step test protocol where the first step warms up the leaky neuron without computing gradients to establish a non-zero membrane voltage, then the second step applies a gradient signal with a larger input to trigger spiking and verify that the resistance and capacitance parameters receive the expected non-zero gradients based on the exponential decay and surrogate gradient calculations. I'm setting up a test case with the leaky integrate-and-fire layer, initializing it with specific parameters and then running a forward pass with a warmup input followed by the actual input, then computing the backward pass to verify the gradient calculations match my expected values for the threshold, resistance, and capacitance parameters. Writing test assertions... Setting up a test case with the LeakyBPTT layer, initializing input values and gradient outputs, then computing the backward pass to verify the gradient calculations match the expected surrogacy-based derivatives. Finishing the gradient assertions for resistance and capacitance, then moving into a new test case for the leaky integrator that validates exact RC gradients with zero voltage threshold gradient across a two-step forward pass. Now I'm running the forward pass again with gradient tracking enabled, then performing backpropagation with a gradient of ones to compute how the loss changes with respect to the resistance and capacitance parameters. I'm verifying that the computed gradients match the expected values based on the chain rule calculations.

Moving on to a new test case for the temporal batch normalization layer, where I'm setting up a scenario with specific dimensions and input values to check that the gamma and beta parameter gradients are computed exactly. Setting up test tensors...

Now I'm verifying the gradient values for gamma and beta parameters match the expected results within a small tolerance, then moving on to test the max pooling backward pass and a simple ResNet implementation with known values. Setting up the test parameters...

I'm initializing the input tensor with values of 1.0 for both dimensions, then running the forward pass through the network. After that, I'm verifying the output dimensions match expectations and checking that the computed values are close to 6.0 with a small tolerance.

For the backward pass, I'm creating a gradient tensor of all ones and running backpropagation to compute the input gradients. The test then validates that the gradient dimensions are correct and that the computed gradient values match the expected 6.0 with the same tolerance.

Now I need to trace through the parameter ordering in the network to understand how the weights and biases are organized. The network has an input layer mapping from 2 to 3 dimensions, a ReLU activation, and an output layer mapping from 3 to 2 dimensions. Each linear layer stores its weight matrix with shape (output_size, input_size) and a bias vector with shape (output_size, 1). via Sequential → [fc_in.weight(3×2), fc_in.bias(3×1), fc_out.weight(2×3), fc_out.bias(2×1)]

Hmm, but looking at LinearImpl constructor: LinearImpl(int in_features, int out_features). And params are {weight, bias} where weight is (out_features, in_features).

For fc_in = LinearImpl(2, 3): weight is (3, 2), bias is (3, 1). ✓
For fc_out = LinearImpl(3, 2): weight is (2, 3), bias is (2, 1). ✓

So:

params[0] = fc_in.weight (3×2)
params[1] = fc_in.bias (3×1)
params[2] = fc_out.weight (2×3)
params[3] = fc_out.bias (2×1)
Now I'm initializing all parameters to 1.0, then zeroing out the bias terms so the network starts with identity-like behavior through the weights and no bias offsets.

Now tracing through the forward pass with input [[1.0, 1.0]]: the first layer multiplies this by the weight matrix and adds bias (which is zero), giving [[2, 2, 2]] after the matrix multiplication. ReLU keeps this unchanged since all values are positive. The second layer then takes [[2, 2, 2]] and multiplies by its weight matrix.

T, producing the expected output [[6, 6]].

For the backward pass with gradient [[1, 1]], I'm working backwards through the output layer to compute how gradients flow back through the network. The output layer's backward pass multiplies the incoming gradient by the weight matrix, giving [[2, 2, 2]]. ReLU backward preserves this since all activations were positive. Now I'm computing the input layer's backward pass with this gradient.

Let me verify the weight gradient computation in the LinearImpl test to make sure I understand the outer product mechanism correctly.

Now I'm checking the LSTM bias gradient calculation step by step—working through the derivative chain for the cell state and output gate to get the exact dc value. Continuing with the output gate and cell candidate gradients, then computing the bias gradients by stacking all the pre-activation derivatives, which gives me the gradient values for each of the four gates in the LSTM cell.

Now I'm verifying these gradients are correct by running the test—if the implementation matches, it should pass with these exact values. I'm also planning to clean up the test suite by replacing the old finite-difference checks with these precise known-value tests, and consolidating redundant backward tests since the new version already covers gradient computation. I need to finish replacing the last test case, then decide whether to keep the Adam include. Since I added it but the original file didn't have it, I should remove it if none of the remaining tests actually use Adam.

Now I'm working through the math for the LeakyBPTT gradient check at the initial timestep, verifying how the gradient flows back through the threshold potential term. Verifying the gradient formula at t=0 where surr0 equals exp(-0.2) and dvpost_dVth0 accounts for the voltage difference from reset, confirming the expected_dVth calculation matches what I implemented. Now checking what grad_from_next actually contains at the initial timestep.

After processing t=1 backward, grad_next_state[0,0] gets populated with surr1, which is the gradient contribution from the next timestep. Everything aligns with the math, so I'm ready to finalize the implementation.

One more verification needed on the forward BPTT computation: at t=1, the pre-spike voltage combines the decayed membrane potential from t=0 (which is 0.3) with the current input, giving v_pre[t=1] = 0.3 * exp(-1) + 3.0.

The stored post-spike history at t=0 is 0.3 (no spike occurred since 0.3 < threshold), and this gets used as v_prev_post in the backward step at t=1. When t=0, v_prev_post is initialized to 0 as expected.

The math checks out completely. Now I'll write the full test code.

Actually, I need to verify the LSTM backward pass formula for the bias gradient accumulation. The backward step should be accumulating gradients into dW_, dU_, and db_, which the optimizer then reads through the param_ptrs_ vector pointing to W_, U_, and b_. I need to confirm whether the backward pass is directly accumulating into these gradient tensors or if there's a separate mechanism for setting gradients. applies step to W_, U_, b_ by reading their .grad() member.

The backward likely sets b_.grad() somehow. Let me check: either via b_.set_grad(db_) or by directly accumulating into b_.grad_ref().

If the LSTM backward accumulates into db_ and then doesn't call set_grad, then b_.grad() would be empty/zero and the test would fail.

I need to verify this. But rather than reading more code, let me just run the test and see if b_.grad() gives the right answer. If it fails, I'll debug.

Actually, let me look at what the LSTMGateTest.ForgetGateBiasInitOne test does — it accesses lstm.b_.at(r, 0) directly. And the other new tests in my version access lstm.b_.grad(). If b_ gradient is stored in db_ not b_.grad(), I'd need to access lstm.db_ instead.

Let me check whether b_.has_grad() returns true or if db_ is a separate member, then I'll look at the backward implementation to understand how gradients are actually stored.

Read LSTMLayer.hpp (lines 149-268)
Let me read the current state of the test file to see what's there before making replacements.

Read fundamental_mechanisms_gtest.cpp (lines 1260-1559)
Read fundamental_mechanisms_gtest.cpp (lines 1558-1677)
Now I have the full picture of what's there. I need to replace the generic tests with exact deterministic value tests. Let me carefully compute the exact expected values for each test:

1. LSTMGateTest.BiasGradNonzero → LSTMGateTest.BiasGradExact
With H=1, D=1, T=1, x=[[0.0]], grad_out=[[1.0]]:

LSTM forward (forget gate bias = 1, others = 0):

b_ is shape (4H, 1) = (4, 1)

pre = W_x + U_h_prev + b_ = b_ (since x=0, h_prev=0)

pre = [[0], [1], [0], [0]]

i = sigmoid(0) = 0.5

f = sigmoid(1) ≈ 0.7311

o = sigmoid(0) = 0.5

g = tanh(0) = 0

c = fc_prev + ig = 0 (since c_prev=0)

tanh_c = tanh(0) = 0

h = o * tanh_c = 0

Now working through the backward pass with the output gradient of 1.0. The cell state gradient comes out to 0.5, while the gradients for the forget, input, and output gates are all zero since they're being multiplied by zero values. I'm computing the gradient for the candidate cell state next.

Computing the gradients through each gate's activation function. The input, forget, and output gates all have zero gradients flowing back, but the candidate cell state gradient is 0.25. Now I'm stacking these into the bias gradient vector.

Let me verify the gate ordering against the actual LSTM implementation to make sure I have the right positions—the test file confirms the forget gate bias is initialized to 1.0 at positions [H:2H], so the order is input, forget, output, candidate. I'm verifying the gate ordering in the LSTM implementation by checking the test cases—specifically looking at how ForgetBiasInit confirms that the forget gate bias is initialized to 1 at positions [H:2H], which tells me the gate order is [input, forget, output, cell]. I need to figure out the gate ordering by examining the actual LSTM source code, since the test expectations depend on whether it's [i,f,o,g] or [i,f,g,o]. The existing tests like ForgetBiasInit checking b_.at(1, 0) for the forget gate give me clues about the layout, and InputZeroBlocksUpdate suggests how the gates interact, but I need to see the source directly to confirm.

Need to check LSTM gate ordering before writing exact grad test.

Bash Find gate ordering in LSTM forward
IN
grep -n "gate\|pre\|sigmoid\|tanh\|stacked\|\[i\|\[f\|\[o\|\[g\|4\*H\|4H" /home/ensismoebius/Repos/doutorado/software/nn/include/nn/layers/lstm/LSTMLayer.hpp | head -60

OUT
9: *   pre_t  = x_t W^T + H_{t-1} U^T + b^T          // (B, 4H)
10: *   i_t    = sigma(pre_t[0:H])                      // input gate
11: *   f_t    = sigma(pre_t[H:2H])                     // forget gate (bias init=1 [3])
12: *   o_t    = sigma(pre_t[2H:3H])                    // output gate
13: *   g_t    = tanh(pre_t[3H:4H])                     // cell candidate
15: *   H_t    = o_t ⊙ tanh(C_t)                       // hidden state
17: * Gate ordering [i|f|o|g] follows Greff et al. (2015) [2] convention.
18: * Weights stored as stacked matrices in the same order.
23: *   [3] Jozefowicz et al., ICML 2015 (forget-gate bias initialisation).
47:    Tensor h_prev; // H_{t-1} (B, H)
48:    Tensor c_prev; // C_{t-1} (B, H)
49:    Tensor i;      // σ(pre_i) (B, H)
50:    Tensor f;      // σ(pre_f) (B, H)
51:    Tensor o;      // σ(pre_o) (B, H)
52:    Tensor g;      // tanh(pre_g) (B, H)
53:    Tensor tanh_c; // tanh(C_t) (B, H)
155:        Tensor b_T = b_.transpose(); // (1, 4H) — computed once outside loop
157:        auto sigmoid_tensor = [](const Tensor& x) -> Tensor
162:        auto tanh_tensor = [&](const Tensor& x) -> Tensor
166:            return sigmoid_tensor(x * 2.0f) * two - ones;
182:            Tensor pre =
185:            Tensor i_g = sigmoid_tensor(pre.block(0, 0, B, hidden_size_));
186:            Tensor f_g = sigmoid_tensor(pre.block(0, 1 * hidden_size_, B, hidden_size_));
187:            Tensor o_g = sigmoid_tensor(pre.block(0, 2 * hidden_size_, B, hidden_size_));
188:            Tensor g_g = tanh_tensor(pre.block(0, 3 * hidden_size_, B, hidden_size_));
191:            Tensor tc = tanh_tensor(c_new);
299:        auto sigmoid_grad_from_output = [](const Tensor& y) -> Tensor
304:        auto tanh_grad_from_output = [](const Tensor& y) -> Tensor
310:        // Opt: pre-allocate dpre once; overwrite each timestep via setBlock.
311:        Tensor dpre(static_cast<nn::Index>(B), 4 * static_cast<nn::Index>(H));
321:            Tensor do_gate = dh * step.tanh_c;
322:            Tensor dtanh_c = dh * step.o;
323:            Tensor dc = (dtanh_c * tanh_grad_from_output(step.tanh_c)).add(dc_next);
325:            Tensor di_gate = dc * step.g;
326:            Tensor df_gate = dc * step.c_prev;
327:            Tensor dg_gate = dc * step.i;
330:            Tensor dpre_i = di_gate * sigmoid_grad_from_output(step.i);
331:            Tensor dpre_f = df_gate * sigmoid_grad_from_output(step.f);
332:            Tensor dpre_o = do_gate * sigmoid_grad_from_output(step.o);
333:            Tensor dpre_g = dg_gate * tanh_grad_from_output(step.g);
336:            dpre.setBlock(0, 0 * H, dpre_i);
337:            dpre.setBlock(0, 1 * H, dpre_f);
338:            dpre.setBlock(0, 2 * H, dpre_o);
339:            dpre.setBlock(0, 3 * H, dpre_g);
341:            Tensor dpre_T = dpre.transpose(); // (4H, B)
342:            dW.add_inplace(dpre_T.matmul(step.x));
343:            dU.add_inplace(dpre_T.matmul(step.h_prev));
344:            // Opt: sum over batch before accumulating to (4H, 1).
345:            db.add_inplace(dpre_T.rowwise_sum());
348:            dx_all.set_time_slice(static_cast<nn::Index>(t), dpre.matmul(W_));
349:            dh_next = dpre.matmul(U_);
Now I have the gate ordering [i|f|o|g]:

b_[0,0] = input gate bias (initialized to 0)
b_[1,0] = forget gate bias (initialized to 1)
b_[2,0] = output gate bias (initialized to 0)
b_[3,0] = cell gate bias (initialized to 0)
And backward code:

dpre_i at positions [0H : 1H]

dpre_f at positions [1H : 2H]

dpre_o at positions [2H : 3H]

dpre_g at positions [3H : 4H]

For the test case with H=1, D=1, T=1, x=[[0.0]], grad_out=[[1.0]], the pre-activation values are just the biases since the input and hidden state are zero, giving me the initial gate values to work with.

Now I'm computing the sigmoid activations for each gate using those pre-activation values...

Actually, let me continue with the forward pass. The output gate gives 0.5, and the cell candidate is 0 from tanh. The cell state stays at 0 since both the forget gate contribution and input gate contribution multiply to zero. Then in the backward pass, I'm propagating the gradient back through the output gate and cell state, finding that the gradient through the cell candidate is 0.5 but the overall cell gradient becomes 0 due to the tanh derivative at zero.

Now I'm computing gradients for the individual gates. The input gate gradient is 0 since the cell gradient is 0, same with the forget gate. But the candidate gate gradient is 0.25 from multiplying the cell gradient by the input gate value. Then I'm backpropagating through the gate pre-activations using their respective derivatives.

For the bias accumulation, I'm summing the pre-activation gradients across the batch dimension and adding them to the bias gradient vector, which gives me [0, 0, 0, 0.25] for the four gates. Right, so the forget gate bias is initialized to 1, and with the gate ordering [i,f,o,g], the bias vector b_ is shaped (4,1) for H=1, D=1. That means b_[1,0] = 1 for the forget gate while the others are 0. I need to verify how the LSTMLayer constructor actually shapes b_ before finalizing my test case. I'm tracing through the backward pass where dpre_T gets transposed to (4, 1), and each row's sum becomes just that single element, giving me the gradient for the bias as [[0], [0], [0], [0.25]]. Then this gets assigned to the bias gradient.

Now I need to verify the gradient values for the LeakyTest.AllParamsGradExact case with the layer initialized at dt=1, dh=1, dw=1, and alpha=0.5.

Computing beta as the exponential decay factor, I get approximately 0.36788. For the two-step process, the warm-up step feeds in 0.3 which stays below the threshold of 0.5, so no spike occurs and the membrane voltage carries forward to 0.3. But I need to clarify exactly how the Leaky implementation handles the membrane update and reset logic.

Looking at the forward pass more carefully: the membrane voltage updates as beta times the previous voltage plus the input, then checks if it exceeds threshold. With the warm-up input of 0.3, this gives a sub-threshold voltage of 0.3 with no spike, so the internal state remains at 0.3 going into step 2. When the second step applies an input of 2.0, the membrane voltage becomes approximately 2.11, which triggers the surrogate gradient calculation using the exponential function based on the distance from threshold.

Now I'm working through the backward pass: the gradient with respect to threshold is straightforward at -0.1997, but for the resistance and capacitance parameters, I need to compute how the decay constant beta changes with respect to these circuit elements using the chain rule. Now I'm working through the gradient computation for the membrane potential with respect to the previous timestep's state, then deriving how changes in the resistance R affect the voltage through the decay constant beta. I need to carefully track that beta itself depends on R, so I'm computing the derivative of beta with respect to R to get the full chain rule. The warm-up step ran without gradient tracking to set the initial state to 0.3, then the forward pass with gradients enabled uses that as the starting point. Computing the gradients for both R and C with these values shows they're equal in this scenario since the decay constant and capacitance have symmetric roles in the equations.

Now I'm working through the numerical values: with beta around 0.368, the membrane voltage comes out to about 2.11, which gives a surrogacy value of roughly exp(-1.61).

The gradient with respect to the threshold is approximately -0.2, while the gradients for both R and C parameters are about 0.022 each. I need to double-check the test setup though—it mentions 1 neuron with 4 features, so each feature should be getting its own gradient computation.

Looking at the layer implementation, R, C, and V_th are scalar parameters shared across all features, which means their gradients accumulate across the 4 feature columns. With 4 independent features each contributing to the parameter updates, I need to recalculate how these gradients sum up across the batch. step 2 (x=2.0 for all): v_pre[f] = beta * 0.3 + 2.0 for all f

Since all features get the same v_mem_prev (0.3) and the same input (2.0), they all have:

v_pre = beta * 0.3 + 2.0
surr = exp(-(v_pre - 0.5))
Same gradient values
V_th gradient accumulates over all features:

dL/dVth = -sum_f(grad_out[f] * surr[f]) = -(1.0 * surr + 1.0 * surr + 1.0 * surr + 1.0 * surr) = -4 * surr
R and C gradients both accumulate to 4 * surr * beta * 0.3 across the four features. With F=4, I can compute the expected values: dVth ≈ -0.79880, while dR and dC both come out to roughly 0.088136. Now I'm working through the exact formula using beta = exp(-1.0f) and v_pre = beta * 0.3f to verify these calculations.

Actually, I should simplify this test to use just 1 feature instead of 4 to make the math cleaner. For the exact gradient test, I'll set up a single neuron with one input feature, warm it up with x=0.3 to get v_mem=0.3, then step with x=2.0 and go=1.0. This gives me beta = exp(-1.0f), v_pre = beta * 0.3f + 2.0f, and surr = exp(-(v_pre - 0.5f)).

Now I'm verifying the gradient formulas: expected_dVth = -surr for the single feature, expected_dR = surr * beta * 0.3f since d_beta_dR = beta and v_mem_prev = 0.3, and expected_dC = surr * beta * 1.0f because d_beta_dC = beta * dt/(R*C^2) which simplifies to beta when R=C=dt=1.

Let me check the actual Leaky backward implementation to make sure these match what the code is computing. The gradient with respect to beta is the sum of grad_out weighted by the surrogate gradient and the previous membrane voltage, so with one feature that's just surr.

Now I can work out the expected gradients: dVth should be around -0.2, while dR and dC should both be roughly 0.022 based on the formula dL/dbeta times beta times dt divided by the squared resistance or capacitance.

For the LeakyBPTT test with T=2, I need to trace through the computation where beta is exp(-1), the first timestep has input 0.3 and the second has 3.0, both with gradient 1. The expected dVth combines contributions from both timesteps using the surrogates at each step. Working through the forward pass at t=1, where the membrane potential exceeds threshold and triggers a spike, then applying the reset mechanism to compute the post-spike voltage. I need to trace through the exact reset mechanism in the Leaky implementation to get the backward pass right. The BPTT formula shows that dvpost_dVth depends on whether it's a hard reset with v_post = v_pre - V_th * spike or the simpler v_post = 0 when spiking, so I should check the actual Leaky.hpp code to confirm which one applies here. Setting up test constants...

Now I need to verify the BPTT gradient accumulation formula. The key insight is that dL_dVth_sum combines the direct gradient term from the current timestep with the backpropagated gradient from future states, scaled by the decay factor and the post-threshold derivative. At the final timestep there's no future contribution, but at earlier timesteps the gradient flows backward through the recurrent connection. I'm trying to work through how the voltage potential relates to the threshold during backpropagation, but the math is getting tangled with different reset mechanisms. Let me just look at the actual backward pass implementation in the code to get the precise formulas rather than deriving them from scratch.

Need the exact BPTT backward formulas before computing expected values.

Read LeakyBPTT.hpp (lines 241-370)
Now I have the exact backward code for LeakyBPTT. Let me re-derive the expected values for the LeakyBPTTTest.AllParamsGradExact test.

Setup: T=2, B=1, F=1, V_th=0.5, R=1, C=1, dt=1, reset_zero=true, reset_potential=0.

beta = exp(-1/1) = exp(-1)

d_beta_dR = beta * 1 / (1 * 1^2) = beta

d_beta_dC = beta * 1 / (1 * 1^2) = beta

Now I'm working through the forward pass with the time-major layout where the input x = [[0.3], [3.0]] represents T=2 timesteps with B=1 batch and F=1 feature, so x[t=0] = 0.3 and x[t=1] = 3.0.

At t=0: the membrane potential starts at 0, so v_pre = beta * 0 + 0.3 = 0.3, which doesn't exceed the threshold of 0.5, so no spike occurs. With reset_zero enabled, v_post equals v_pre since there's no spike, giving v_post_history[0, 0] = 0.3 and v_mem = 0.3.

At t=1: I'm computing v_pre = beta * 0.3 + 3.0 to continue the recurrent dynamics.

Now I need to clarify what's being stored in each history buffer. Looking at the code, v_post_history stores the post-spike values (after reset), while v_mem_history stores the pre-spike membrane potentials that get fed back into the next timestep. Looking at the forward pass step by step: at t=0, the membrane potential stays at 0.3 since it's below threshold and no spike occurs, so both the pre and post values are 0.3. Then at t=1, the potential jumps to beta times the previous post value plus the new input of 3.0, which crosses the threshold and triggers a spike.

Now I'm working through the backward pass, starting from t=1 and moving back to t=0. The gradient flows from the output with an initial value of 1.0 at each timestep, and I'm computing how the gradients propagate through the spiking mechanism using the surrogate gradient approach. Now I'm computing the gradient contributions for the threshold parameter and tracking how the loss propagates back through the recurrent connections, then moving on to evaluate the timestep t=0 case. Now I'm finishing the gradient calculation for the voltage threshold, accumulating the contributions from both the output loss and the recurrent gradient, then extracting the final input gradient and confirming the backpropagation through time is complete. I'm verifying the derivative formula for beta with respect to R, which checks out to beta given the parameter values. Now I need to look at how the Leaky backward implementation computes the R and C gradients to make sure they're using the same approach. I need to verify the TdBN backward pass with the scale factor of V_th / sqrt(T), which equals 1.0 in this test case, then trace through how the gradients accumulate for gamma and beta given the input dimensions and cached normalized values. I'm working through the normalization for each feature—computing the mean and standard deviation across the batch, then normalizing each sample. For feature 1, I get mean 4 and std 2, giving normalized values of -1 and 1. Now I'm doing the same calculation for feature 2 to complete the normalized output matrix.

Moving into the backward pass, I'm computing the gradients with respect to gamma using the chain rule—multiplying the upstream gradients by the normalized inputs and summing across the batch.

Then I'm calculating the gradients for beta by simply summing the upstream gradients across all samples, which gives me 3.0 for each feature.

Now I'm second-guessing myself on the normalization approach—I need to verify whether TdBN uses sample or population standard deviation, and whether it normalizes across the batch at each timestep or uses some global statistics.

Let me check the actual implementation to see the exact formula being used. I need to check the TdBN forward implementation to see whether it normalizes using population or sample standard deviation, since that determines whether the normalized values are ±1 or ±0.707.

Need to verify TdBN normalization formula before writing exact expected values.

Read ThresholdDependentBatchNorm.hpp (lines 51-200)
Now I can see the exact TdBN forward and backward code.

The normalization uses population variance: var_sum / B (not var_sum / (B-1)).

For the TdBN test setup: T=1, B=2, F=3 (so total_rows = T*B = 2)
x = [[1, 2, 3], [3, 6, 9]]

For feature 0: values = [1, 3]

mean = (1+3)/2 = 2

var_sum = (1-2)^2 + (3-2)^2 = 1 + 1 = 2

var = 2/2 = 1

inv_std = 1/sqrt(1 + 1e-5) ≈ 1.0 (approximately)

x_norm[0,0] = (1-2)/sqrt(1+eps) ≈ -1.0

x_norm[1,0] = (3-2)/sqrt(1+eps) ≈ +1.0

For feature 1: values = [2, 6], mean = 4, and the variance sum is 8 from the squared deviations.

For feature 2: values = [3, 9], mean = 6, variance is 9, and the inverse standard deviation is approximately 1/3. After normalization, both features end up with the same pattern where the first sample maps to -1.0 and the second to +1.0.

Now I'm working through the backward pass with the threshold and temperature parameters. The gradient with respect to gamma comes out to -1 across all features since the scaled normalized values weighted by the output gradients sum to that. For beta, I'm computing the sum of the scaled output gradients.

But I need to be more precise about the epsilon term. When I account for the small epsilon value in the standard deviation calculation, the normalized values are slightly less than ±1, which means the gamma gradient becomes approximately -0.999995 instead of exactly -1.

For the test assertions, I should use a tolerance around 1e-3 or 1e-4 to account for these floating-point precision differences. I'm verifying the gradient calculations are within tolerance, and the beta gradient comes out to exactly 3.0 since it's not affected by the normalized input. Now I need to check the SimpleResNet test case with unit weights to trace through the forward and backward passes. [[6,6]]
Let me verify:

fc_in: weight is (H×D) = (3×2), bias is (H×1) = (3×1)
With all weights=1, biases=0:
output = input @ weight.T + bias
= [[1,1]] @ [[1,1],[1,1],[1,1]] + [[0,0,0]]
= [[11+11, 11+11, 11+11]] = [[2, 2, 2]]
ReLU(2,2,2) = (2,2,2)
fc_out: weight is (O×H) = (2×3), bias is (O×1) = (2×1)
= [[2,2,2]] @ [[1,1],[1,1],[1,1]] + [[0,0]]
= [[2+2+2, 2+2+2]] = [[6, 6]]
Now working through the backward pass with gradient [[1, 1]]: the output layer's gradient flows back through its weight matrix to produce [[2,2,2]], which passes unchanged through ReLU since all activations were positive, then the input layer's backward computation multiplies this by its weight matrix to get the final gradient [[6,6]].

The forward and backward values match what I calculated, so I need to check the params() order in SimpleResNet.hpp to set the weights and biases correctly.

Looking at the parameter shapes: fc_in.weight is (3×2), fc_in.bias should be (3×1), fc_out.weight is (2×3), and fc_out.bias is (1×2). I'm a bit uncertain about the exact bias shape convention in LinearImpl, so let me iterate through the params to figure out the actual dimensions rather than guessing. I'm working through the matrix multiplication to verify the output values—with the input as [[1,1]], a weight matrix of all 1s shaped (3×2), and its transpose (2×3), I need to carefully trace through the dimensions to confirm the computation.

Now I'm checking the backward pass to verify the gradient calculations. For the linear layer's backward step, I need to compute dx using the output gradient and weight matrix, then trace how that gradient flows back through the ReLU layer.

Let me look at the SimpleResNet implementation to understand how the parameters are structured and confirm the backward pass chains correctly through the sequential layers.

Bash Check SimpleResNet params structure
IN
grep -n "params\|fc_in\|fc_out\|Sequential\|Linear\|weight\|bias" /home/ensismoebius/Repos/doutorado/software/nn/include/nn/layers/residual/SimpleResNet.hpp | head -60

OUT
16:#include "nn/layers/base/Sequential.hpp"
17:#include "nn/layers/dense/Linear.hpp"
24:// It uses `ResidualBlockImpl<Backend>` (Linear/ReLU/Linear + skip) stacked `depth` times.
27:// - Internally it builds a `SequentialImpl<Backend>` and delegates `forward/backward/params/train`.
28:// - Weight initialization uses `kaimingSNNInitializer()` on the Linear layers.
29://   This is applied only to known Linear modules (fc_in/fc_out and residual block
39:        // Build model: input -> Linear -> ReLU -> ResidualBlocks -> Linear(output)
40:        auto fc_in = std::make_shared<LinearImpl<Backend>>(input_dim, hidden_dim);
41:        layers_.push_back(fc_in);
50:        auto fc_out = std::make_shared<LinearImpl<Backend>>(hidden_dim, output_dim);
51:        layers_.push_back(fc_out);
53:        model_ = std::make_unique<SequentialImpl<Backend>>(layers_);
55:        // Initialize weights
56:        kaimingSNNInitializer(fc_in);
57:        kaimingSNNInitializer(fc_out);
78:    auto params() -> std::span<Tensor*> override
80:        return model_->params();
90:    std::unique_ptr<SequentialImpl<Backend>> model_;
Bash Check Sequential params ordering
IN
grep -n "params\|param_ptrs\|weight\|bias\|push_back" /home/ensismoebius/Repos/doutorado/software/nn/include/nn/layers/base/Sequential.hpp | head -40

OUT
13: * @brief A lightweight PyTorch-like container that composes multiple `Module`s.
41:    std::vector<Tensor*> param_ptrs_;
60:        layers.push_back(module);
112:    // Returns all trainable parameters (weights and biases) from all layers
113:    [[nodiscard]] auto params() -> std::span<Tensor*> override
115:        param_ptrs_.clear();
118:            auto layer_params = layer->params();
119:            param_ptrs_.insert(param_ptrs_.end(), layer_params.begin(), layer_params.end());
121:        return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
Good. So params() for SequentialImpl iterates over layers in order and accumulates all params. For depth=0, the layers are: fc_in, ReLU, fc_out.

For Linear layer params, I need to check what order they're returned.

Bash Check Linear params order
IN
grep -n "param_ptrs\|weight\|bias\|Tensor\|push" /home/ensismoebius/Repos/doutorado/software/nn/include/nn/layers/dense/Linear.hpp | head -40

OUT
9:#include "nn/tensor/Tensor.hpp"
29: * - Parameters (`weight`, `bias`) are stored as the active backend Tensor type.
37:    /// Tensor type for the active compute backend.
38:    using Tensor = nn::TensorImpl<Backend>;
43:    Tensor weight;
45:    Tensor bias;
47:    Tensor input_cache;
49:    std::optional<Tensor> input_cache_backend;
52:    std::array<Tensor*, 2> param_ptrs_{{&weight, &bias}};
55:     * @brief Construct the layer and allocate uninitialized weight/bias storage.
66:          weight(Tensor(out_features_, in_features_)),
67:          bias(Tensor(out_features_, 1))
72:    auto debug(const Tensor& input) -> void
77:            << "Weight dims: " << weight.rows() << "x" << weight.cols() << "\n"
78:            << "Bias dims: " << bias.rows() << "x" << bias.cols() << "\n";
94:    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
130:        Tensor input_flat = input.reshape(flat_shape);
139:        Tensor result_flat;
141:                          in.matmul_transposed_add_col_bias(w, b);
144:            Tensor weight_t(weight);
145:            Tensor bias_t(bias);
149:                weight_t.get_backend().set_gpu_resident(true);
150:                bias_t.get_backend().set_gpu_resident(true);
152:            result_flat = Tensor(input_flat.get_backend().matmul_transposed_add_col_bias(
153:                weight_t.get_backend(), bias_t.get_backend()));
160:            result_flat = Tensor(input_flat.get_backend().matmul_transposed(weight.get_backend()));
161:            result_flat.get_backend().add_col_vector_to_rows_inplace(bias.get_backend());
165:            Tensor weight_t(weight);
166:            Tensor bias_t(bias);
167:            result_flat = input_flat.matmul_transposed(weight_t);
168:            result_flat.add_col_vector_to_rows_inplace(bias_t);
193:    auto backward(const Tensor& grad_previous) -> Tensor override
224:        Tensor grad_flat = grad_previous.reshape(flat_grad_shape);
239:        Tensor input_t;
248:                input_t = Tensor(input_cache);
254:            input_t = Tensor(input_cache);
258:        Tensor grad_weight;
259:        Tensor grad_t = grad_flat.transpose();
266:            grad_weight =
267:                Tensor(grad_flat.get_backend().matmul_lhs_transposed(input_t.get_backend()));
Good. Now I know:

Linear.param_ptrs_ = {&weight, &bias} (weight first, then bias)
weight shape: (out_features, in_features)
bias shape: (out_features, 1)
For SimpleResNetImpl(D=2, H=3, O=2, depth=0), layers are:

fc_in (Linear D→H = Linear 2→3): weight (3,2), bias (3,1)

ReLU (no params)

fc_out (Linear H→O = Linear 3→2): weight (2,3), bias (2,1)

The Sequential container orders parameters as [fc_in.weight, fc_in.bias, fc_out.weight, fc_out.bias], so in the test params()[0] through params()[3] map to these in order, with the first two filled with 1.0 and 0.0 respectively, and the last two following the same pattern.

Looking at how Linear computes gradients during backward, it calculates grad_weight and grad_bias from the cached input and output gradient. But for this test I really just need to verify the forward pass produces [[6, 6]] from input [[1,1]] with these weight/bias values, and that backward correctly returns dx = [[6, 6]].

Let me trace through the backward pass: the output gradient [[1,1]] flows back through the final layer's weight matrix to produce [[2,2,2]], then through ReLU (which passes it through unchanged since all activations were positive), then through the first layer's weight matrix to give [[6,6]] as the input gradient. That matches what I expect.

Now I should check the Leaky.hpp implementation to see exactly how it handles the gradient computation.

Read Leaky.hpp (lines 121-250)
You've hit your limit · resets 11am (America/Sao_Paulo)