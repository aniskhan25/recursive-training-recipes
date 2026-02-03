# Recursive Training: When Models Learn From Their Own Predictions (Safely)

**Who this is for.** You know what “training a model” means, but you’re new to ideas like EM, self-training, FixMatch, or Mean Teacher.

**What you’ll learn.** By the end, you’ll be able to explain recursive training in plain language, spot when it will fail, and understand how modern methods keep it stable.

---

## 1. Normal training (the safe baseline)

In ordinary supervised learning:

- You have inputs (data) and correct answers (labels).
- The labels are treated as **truth**.
- The model adjusts its parameters to make fewer mistakes.

Even though training runs for many epochs, this is **not recursive**. The model does *not* treat its own predictions as truth.

**What to remember**
- Normal training = truth comes from labels, not from the model.

---

## 2. Why recursion is tempting

In real projects, labels are expensive. You might have:

- a small labeled dataset
- a huge unlabeled dataset

So a natural idea appears:

> If the model is already decent, maybe its predictions can become extra labels.

That idea can work extremely well — or collapse completely.

**What to remember**
- Recursion is tempting because unlabeled data is abundant.

---

## 3. Feedback: the moment recursion begins

Recursive training starts when the model **learns from its own guesses**.

A simple analogy: imagine studying from notes you wrote while you were confused. If the notes are good, you improve faster. If the notes contain mistakes, you can “learn the wrong thing” more confidently.

That’s what feedback does: it creates a **loop**.

**What to remember**
- Feedback can accelerate learning, or amplify early mistakes.

---

## 4. Training state (the one concept that explains everything)

Here’s the key definition:

> **Training state** = what is currently treated as truth during training.

In normal training, the state is mostly fixed (the dataset and labels). In recursive training, the state changes over time.

Examples of what can live in the training state:

- **Hard labels** (risky): “this is class A”
- **Soft beliefs** (safer): “70% A, 30% B”
- **Confidence weights** (safer): “only trust predictions above 0.95”

**What to remember**
- Recursive training happens when the training state depends on the model.

---

## 5. The universal recursive loop

Every recursive training method follows the same loop:

1. Start with a training state.
2. Train a model using that state.
3. Use the model to update the state.
4. Repeat.

A simple diagram:

```
  Training state  --->  Train model  --->  Update beliefs
        ^                                   |
        |-----------------------------------|
```

This can be written mathematically as:

$$
\boxed{
\begin{cases}
\theta^{(t+1)} = \arg\min_{\theta} \mathcal{L}(\theta \mid S^{(t)}) \\
S^{(t+1)} = \Phi(S^{(t)}, f_{\theta^{(t+1)}})
\end{cases}
}
$$

Here:

- $\theta^{(t)}$ are the model parameters at round $t$,
- $S^{(t)}$ is the training state at round $t$,
- $\mathcal{L}(\theta \mid S^{(t)})$ is the loss function minimized given the current state,
- $f_{\theta^{(t+1)}}$ is the model with updated parameters,
- $\Phi$ is the rule that updates the training state using the old state and the new model.

**What to remember**
- Same loop everywhere; methods differ in how they represent and update the state.

---

## 6. EM: recursive training done safely

Expectation–Maximization (EM) is a classic stable form of recursion.

### The problem EM solves
EM is used when the “correct answers” exist but are **hidden** (latent). You can’t directly label each example, but you can still learn by repeatedly refining your beliefs.

### The key idea
EM uses **soft beliefs** represented as a distribution \(q^{(t)}(z)\) over the hidden variables \(z\) at round \(t\). It avoids early commitment.

Instead of “this is class A”, EM keeps uncertainty: “70% A, 30% B”.

EM alternates between two steps:

- **E-step:** update the soft state \(q^{(t+1)}\) given the current model parameters \(\theta^{(t)}\),
- **M-step:** update the model parameters \(\theta^{(t+1)}\) given the updated state \(q^{(t+1)}\).

### Toy example: two clusters with soft beliefs
Suppose you have four points on a line:

- `x = {-1, 0, 4, 6}`

You believe there are two groups, but you don’t know which point belongs where.

Start with two guesses for the group centers (means):

- center1 = 0
- center2 = 5

Now update your beliefs:

- points near 0 (like -1 and 0) get **high** probability of belonging to group 1
- points near 5 (like 4 and 6) get **high** probability of belonging to group 2

Then update the centers using those soft memberships:

- group 1 center moves toward the average of (-1, 0) → roughly **-0.5**
- group 2 center stays near the average of (4, 6) → roughly **5.0**

Repeat this “soft assign → update centers” loop, and it settles quickly.

### Why EM is stable
- it keeps uncertainty in the state
- mistakes don’t dominate early
- updates are gradual

**What to remember**
- EM is recursion with uncertainty built into the state.

---

## 7. Self-training: recursion done recklessly (and why it can fail)

Self-training uses the same loop, but often with **hard pseudo-labels**:

$$
\hat{y}_i^{(t)} = \arg\max_{y} f_{\theta^{(t)}}(x_i)
$$

where $\hat{y}_i^{(t)}$ is the pseudo-label for input $x_i$ at round $t$, and $f_{\theta^{(t)}}(x_i)$ is the model’s predicted probability distribution over classes.

Replacing soft distributions with this hard argmax removes uncertainty—once a class is chosen, the model treats it as absolute truth, which can be risky if the choice is wrong.

The problem: if early predictions are wrong, the model can train itself to be wrong.

### Toy example: a threshold classifier that reinforces a mistake
Labeled data:

- `x = -2` is class 0
- `x =  2` is class 1

Unlabeled data:

- `x = {-1, 0, 1}`

Imagine the model starts with a **bad** decision threshold (too far left), so it predicts class 1 too easily. It might label `x = -1` as class 1 (a mistake).

Now that mistake enters the training state as a “label”.
When you retrain, the model learns that `-1` should indeed be class 1, reinforcing the error.

After a few rounds, the model can become confidently wrong — not because it is “stupid”, but because the loop is amplifying its own early mistakes.

**What to remember**
- Self-training is recursion where guesses are treated as truth.

---

## 8. Will the loop converge or explode? (stability in one idea)

Let $e_t$ be “how wrong the training state is” at round $t$ (for example: the fraction of wrong pseudo-labels).

Recursive training can be summarized as:

$$
e_{t+1} = g(e_t)
$$

If errors shrink, the loop is stable.
If errors grow, the loop is unstable.

More precisely, the loop is **stable** if near zero error:

$$
e_{t+1} < e_t \quad \text{when} \quad e_t \approx 0
$$

You don’t need calculus here — just the intuition that errors must get smaller each round to avoid runaway mistakes.

A helpful mental picture:

- **below the diagonal** ($e_{t+1} < e_t$) = safe (errors shrink)
- **above the diagonal** ($e_{t+1} > e_t$) = risky (errors grow)

**What to remember**
- Recursive training succeeds only if the loop shrinks mistakes faster than it reinforces them.

---

## 9. Stabilizers: how modern methods tame feedback

Modern methods add stabilizers to make the loop behave more like EM (more cautious).

Here’s the simplest way to understand stabilizers: they change the loop by **softening, filtering, slowing, or anchoring**.

| Stabilizer | What it changes in the loop | Intuition |
|---|---|---|
| Soft labels | soften beliefs | don’t fully commit |
| Confidence threshold | filter bad feedback | only trust high-confidence guesses |
| EMA teacher | slow feedback | listen to a calmer self |
| Consistency loss | anchor behavior | don’t change your mind under small changes |
| Class-balance constraints | avoid collapse | prevent “everything is one class” |

**What to remember**
- Stabilizers exist to reduce error amplification in the feedback loop.

---

## 10. Modern examples: FixMatch and Mean Teacher

Both are modern, practical versions of “self-training done carefully”.

### FixMatch (selective trust)
Use these four questions:

1) **State?** High-confidence pseudo-labels.  
2) **Who makes labels?** The current model, using a weakly augmented view.  
3) **Stabilizers?** Confidence threshold + consistency under strong augmentation.  
4) **What can go wrong?** If the threshold is too strict early on, almost no unlabeled data is used.

Simple summary:
- FixMatch controls **what enters the loop**.

### Mean Teacher (calmer self)
1) **State?** Teacher predictions (usually soft).  
2) **Who makes labels?** A teacher model that is an exponential moving average (EMA) of the student.  
3) **Stabilizers?** EMA (temporal smoothing) + consistency loss.  
4) **What can go wrong?** If the teacher is too slow, learning can be sluggish.

Simple summary:
- Mean Teacher controls **how fast the loop evolves**.

**What to remember**
- FixMatch filters the state. Mean Teacher smooths the state over time.

---

## 11. A simple learning pattern: Teacher + Threshold

A useful pattern you’ll see often:

- Use an EMA teacher to generate pseudo-labels (more stable).
- Apply a confidence threshold (ignore uncertain guesses).
- Train the student to match those targets under augmentation.

You can think of it as:

> “Only trust a calm version of yourself, and only when it’s confident.”

**What to remember**
- Combining smoothing + filtering makes recursion far more stable.

---

## 12. Final checklist

When you see a recursive training method, ask:

1. What is the training state?
2. Does it include uncertainty (soft beliefs or weights)?
3. Who generates the “labels” used for learning?
4. How are low-quality guesses prevented from entering the state?
5. What stops collapse (e.g., predicting one class for everything)?

### If you remember only 3 things…
1) Recursive training is a feedback loop: model outputs affect future training.  
2) EM is stable because it keeps uncertainty; naive self-training can amplify mistakes.  
3) Modern methods succeed by adding stabilizers (filter, soften, slow, anchor).
 
---