import Mathlib.Logic.Basic
import Mathlib.Data.Set.Basic

universe u

-- ==========================================================
-- 1. Abstract universe of rule systems
-- ==========================================================
variable {R : Type u}                  -- rules / systems
variable {Outcome : Type u}            -- possible outcomes of each rule
variable {traj : R → Set Outcome}      -- trajectories of each rule
variable [∀ r, Nonempty (traj r)]      -- each rule has at least one outcome

-- ==========================================================
-- 2. Logical phases (solid, liquid, plasma)
-- ==========================================================
inductive Phase
  | solid   -- fixed / immutable
  | liquid  -- bounded mutable
  | plasma  -- unbounded mutable

open Phase

variable {phase_of : R → Phase}

-- ==========================================================
-- 3. Axiom of Trinity: existence and uniqueness of canonical outcome
-- ==========================================================
axiom AT :
  ∀ r : R, ∃! o : Outcome,
    (phase_of r = solid   → o ∈ traj r) ∧
    (phase_of r = liquid  → o ∈ traj r) ∧
    (phase_of r = plasma  → o ∈ traj r)

-- Canonical outcome extractor (using classical choice)
noncomputable def canonical (r : R) : Outcome :=
  Classical.choose (AT r)

lemma canonical_spec (r : R) :
    (phase_of r = solid   → canonical r ∈ traj r) ∧
    (phase_of r = liquid  → canonical r ∈ traj r) ∧
    (phase_of r = plasma  → canonical r ∈ traj r) :=
  (Classical.choose_spec (AT r)).1

-- ==========================================================
-- 4. Observer / determinacy
-- ==========================================================
inductive Observer
  | O1 | O2
  deriving DecidableEq

open Observer

variable (outcome_winner : Outcome → Observer)

def Determined (r : R) : Prop :=
  outcome_winner (canonical r) = O1 ∨ outcome_winner (canonical r) = O2
  -- (trivially true if outcome_winner is total, but kept general)

-- ==========================================================
-- 5. Gödel-style incompleteness: no finite or bounded set of rules is perfect
-- ==========================================================
axiom incompleteness :
  ∀ (subset_rules : Set R), subset_rules.Nonempty → ∃ r ∈ subset_rules, phase_of r ≠ solid

-- ==========================================================
-- 6. Metamorphosis theorem
-- ==========================================================
theorem metamorphosis_inevitable
  (subset_rules : Set R)
  (h_nonempty : subset_rules.Nonempty)
  (observer : Observer) :
  ∃ r ∈ subset_rules, phase_of r ≠ solid :=
begin
  exact incompleteness subset_rules h_nonempty,
end

-- Alternative statement without explicit nonempty hypothesis
theorem metamorphosis_inevitable' :
  ∀ (subset_rules : Set R) (observer : Observer),
    subset_rules.Nonempty →
    ∃ r ∈ subset_rules, phase_of r ≠ solid := by
  intros subset_rules _ h
  exact incompleteness subset_rules h

-- ==========================================================
-- Interpretation:
-- Any non-empty collection of rule systems chosen by any observer
-- must contain at least one system that is not fully solid (immutable).
-- In other words: perfect, unchanging solidity is never globally achievable
-- within any bounded context — metamorphosis (change) is inevitable.
-- ==========================================================
