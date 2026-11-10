import Init

set_option autoImplicit false

namespace Valentyn

theorem conjunction_commutes (p q : Prop) : p ∧ q → q ∧ p := by
  intro h
  exact ⟨h.2, h.1⟩

theorem implication_composes (p q r : Prop) :
    (p → q) → (q → r) → p → r := by
  intro hpq hqr hp
  exact hqr (hpq hp)

theorem disjunction_distributes (p q r : Prop) :
    p ∧ (q ∨ r) → (p ∧ q) ∨ (p ∧ r) := by
  intro h
  cases h.2 with
  | inl hq => exact Or.inl ⟨h.1, hq⟩
  | inr hr => exact Or.inr ⟨h.1, hr⟩

theorem no_finite_maximum : ∀ n : Nat, ∃ m : Nat, n < m := by
  intro n
  exact ⟨n + 1, Nat.lt_succ_self n⟩

theorem sum_of_odds_step (n s : Nat) (h : s = n * n) :
    s + (2 * n + 1) = (n + 1) * (n + 1) := by
  calc
    s + (2 * n + 1) = n * n + (2 * n + 1) :=
      congrArg (fun x => x + (2 * n + 1)) h
    _ = n * n + ((n + n) + 1) :=
      congrArg (fun x => n * n + (x + 1)) (Nat.two_mul n)
    _ = n * n + (n + (n + 1)) :=
      congrArg (fun x => n * n + x) (Nat.add_assoc n n 1)
    _ = (n * n + n) + (n + 1) :=
      (Nat.add_assoc (n * n) n (n + 1)).symm
    _ = (n + 1) * n + (n + 1) :=
      congrArg (fun x => x + (n + 1)) (Nat.succ_mul n n).symm
    _ = (n + 1) * (n + 1) := (Nat.mul_succ (n + 1) n).symm

end Valentyn
