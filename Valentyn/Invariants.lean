import Init

set_option autoImplicit false

namespace Valentyn

/-- A mathematical certificate contains evidence for its stated proposition.
This models a proof gate; it does not prove the Python verifier correct. -/
structure Certificate (statement : Prop) where
  evidence : statement

def accept {statement : Prop} (certificate : Certificate statement) : statement :=
  certificate.evidence

theorem acceptance_sound {statement : Prop} (certificate : Certificate statement) :
    statement := accept certificate

/-- A candidate replaces the incumbent only on a strict score improvement. -/
def promote (incumbent candidate : Nat) : Nat :=
  if incumbent < candidate then candidate else incumbent

theorem promotion_never_decreases (incumbent candidate : Nat) :
    incumbent ≤ promote incumbent candidate := by
  unfold promote
  split
  next h => exact Nat.le_of_lt h
  next => exact Nat.le_refl _

theorem promotion_is_observed (incumbent candidate : Nat) :
    promote incumbent candidate = incumbent ∨ promote incumbent candidate = candidate := by
  unfold promote
  split
  next => exact Or.inr rfl
  next => exact Or.inl rfl

inductive Derives : List Prop → Prop → Prop where
  | assumption {context : List Prop} {p : Prop} : p ∈ context → Derives context p
  | truth {context : List Prop} : Derives context True
  | intro {context : List Prop} {p q : Prop} :
      Derives (p :: context) q → Derives context (p → q)
  | apply {context : List Prop} {p q : Prop} :
      Derives context (p → q) → Derives context p → Derives context q
  | pair {context : List Prop} {p q : Prop} :
      Derives context p → Derives context q → Derives context (p ∧ q)
  | left {context : List Prop} {p q : Prop} : Derives context (p ∧ q) → Derives context p
  | right {context : List Prop} {p q : Prop} : Derives context (p ∧ q) → Derives context q
  | inl {context : List Prop} {p q : Prop} : Derives context p → Derives context (p ∨ q)
  | inr {context : List Prop} {p q : Prop} : Derives context q → Derives context (p ∨ q)
  | cases {context : List Prop} {p q r : Prop} :
      Derives context (p ∨ q) → Derives (p :: context) r →
      Derives (q :: context) r → Derives context r
  | absurd {context : List Prop} {p : Prop} : Derives context False → Derives context p

/-- Soundness of the inference rules used by the symbolic search engine.
The implementation and its renderer are additionally checked by Lean per result. -/
theorem derivation_sound {context : List Prop} {p : Prop}
    (derivation : Derives context p) : (∀ q, q ∈ context → q) → p := by
  induction derivation with
  | assumption member => intro env; exact env _ member
  | truth => intro _; exact True.intro
  | intro _ ih =>
      intro env hp
      apply ih
      intro q member
      cases member with
      | head => exact hp
      | tail _ member => exact env q member
  | apply _ _ ihf ihp => intro env; exact ihf env (ihp env)
  | pair _ _ ihp ihq => intro env; exact ⟨ihp env, ihq env⟩
  | left _ ih => intro env; exact (ih env).1
  | right _ ih => intro env; exact (ih env).2
  | inl _ ih => intro env; exact Or.inl (ih env)
  | inr _ ih => intro env; exact Or.inr (ih env)
  | cases _ _ _ ihd ihl ihr =>
      intro env
      cases ihd env with
      | inl hp =>
          apply ihl
          intro q member
          cases member with
          | head => exact hp
          | tail _ member => exact env q member
      | inr hp =>
          apply ihr
          intro q member
          cases member with
          | head => exact hp
          | tail _ member => exact env q member
  | absurd _ ih => intro env; exact False.elim (ih env)

end Valentyn
