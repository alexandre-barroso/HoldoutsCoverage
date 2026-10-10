import Std

/- Exact finite counts for an explicitly stated information set. No stochastic
   independence, learned-model performance, data parsing or novelty is proved.
   Labels live in one type; every prediction is therefore an admissible label.
   Each index has one label and visibility flag. List multiplicity is retained. -/
namespace SupportAudit

theorem fixedpoint_exclusion {Input : Type} (normalize : Input → Input)
    (rank : Input → Nat) (x : Input) (selectionCut trainCut : Nat)
    (fixed : normalize x = x) (selected : rank x < selectionCut)
    (ordered : selectionCut ≤ trainCut) : rank (normalize x) < trainCut := by
  rw [fixed]
  omega

theorem selected_max_exclusion {Input : Type} (normalize : Input → Input)
    (rank : Input → Nat) (x : Input) (realizedMax trainCut : Nat)
    (fixed : normalize x = x) (selected : rank x ≤ realizedMax)
    (below : realizedMax < trainCut) : rank (normalize x) < trainCut := by
  rw [fixed]
  omega

variable {Index Label : Type} [DecidableEq Label]

def loss (prediction label : Label) : Int := if prediction = label then 0 else 1
def delta (a b label : Label) : Int := loss a label - loss b label
def disagreement (a b : Label) : Int := if a = b then 0 else 1

theorem point_bounds (a b label : Label) :
    -disagreement a b ≤ delta a b label ∧ delta a b label ≤ disagreement a b := by
  by_cases ab : a = b
  · subst b; simp [delta, disagreement]
  · have ba : b ≠ a := Ne.symm ab
    by_cases ay : a = label
    · subst label; simp [delta, loss, disagreement, ab, ba]
    · by_cases byy : b = label
      · subst label; simp [delta, loss, disagreement, ab]
      · simp [delta, loss, disagreement, ab, ay, byy]

theorem point_lower (a b : Label) : delta a b a = -disagreement a b := by
  by_cases ab : a = b
  · subst b; simp [delta, loss, disagreement]
  · have ba : b ≠ a := Ne.symm ab
    simp [delta, loss, disagreement, ab, ba]

theorem point_upper (a b : Label) : delta a b b = disagreement a b := by
  by_cases ab : a = b
  · subst b; simp [delta, loss, disagreement]
  · simp [delta, loss, disagreement, ab]

def total (term : Index → Int) : List Index → Int
  | [] => 0
  | i :: rest => term i + total term rest

def difference (a b labels : Index → Label) (items : List Index) : Int :=
  total (fun i => delta (a i) (b i) (labels i)) items

def known (a b labels : Index → Label) (visible : Index → Bool) (items : List Index) : Int :=
  total (fun i => if visible i then delta (a i) (b i) (labels i) else 0) items

def radius (a b : Index → Label) (visible : Index → Bool) (items : List Index) : Int :=
  total (fun i => if visible i then 0 else disagreement (a i) (b i)) items

def agrees (visible : Index → Bool) (observed completed : Index → Label) : Prop :=
  ∀ i, visible i = true → completed i = observed i

def completeWith (prediction observed : Index → Label) (visible : Index → Bool) : Index → Label :=
  fun i => if visible i then observed i else prediction i

omit [DecidableEq Label] in
theorem completion_agrees (prediction observed : Index → Label) (visible : Index → Bool) :
    agrees visible observed (completeWith prediction observed visible) := by
  intro i h
  simp [completeWith, h]

theorem known_congr (a b observed completed : Index → Label)
    (visible : Index → Bool) (items : List Index) (h : agrees visible observed completed) :
    known a b completed visible items = known a b observed visible items := by
  induction items with
  | nil => rfl
  | cons i rest ih =>
    simp only [known, total] at ih ⊢
    rw [ih]
    cases hv : visible i with
    | false => simp
    | true => simp [h i hv]

theorem comparison_interval (a b labels : Index → Label) (visible : Index → Bool)
    (items : List Index) :
    known a b labels visible items - radius a b visible items ≤ difference a b labels items ∧
    difference a b labels items ≤ known a b labels visible items + radius a b visible items := by
  induction items with
  | nil => simp [known, radius, difference, total]
  | cons i rest ih =>
    have hp := point_bounds (a i) (b i) (labels i)
    simp only [known, radius, difference, total] at ih ⊢
    cases hv : visible i with
    | false => simp only [Bool.false_eq_true, ↓reduceIte] at ⊢; omega
    | true => simp only [↓reduceIte] at ⊢; omega

theorem lower_completion_attains (a b labels : Index → Label) (visible : Index → Bool)
    (items : List Index) :
    difference a b (completeWith a labels visible) items =
      known a b labels visible items - radius a b visible items := by
  induction items with
  | nil => rfl
  | cons i rest ih =>
    simp only [difference, total, known, radius] at ih ⊢
    rw [ih]
    cases hv : visible i with
    | false => simp [completeWith, hv, point_lower]; omega
    | true => simp [completeWith, hv]; omega

theorem upper_completion_attains (a b labels : Index → Label) (visible : Index → Bool)
    (items : List Index) :
    difference a b (completeWith b labels visible) items =
      known a b labels visible items + radius a b visible items := by
  induction items with
  | nil => rfl
  | cons i rest ih =>
    simp only [difference, total, known, radius] at ih ⊢
    rw [ih]
    cases hv : visible i with
    | false => simp [completeWith, hv, point_upper]; omega
    | true => simp [completeWith, hv]; omega

theorem common_lower_completion (a labels : Index → Label) (visible : Index → Bool)
    (items : List Index) :
    ∃ completed : Index → Label, agrees visible labels completed ∧
      ∀ b : Index → Label, difference a b completed items =
        known a b labels visible items - radius a b visible items := by
  exact ⟨completeWith a labels visible, completion_agrees a labels visible,
    fun b => lower_completion_attains a b labels visible items⟩

theorem possible_weak_winner_iff (a labels : Index → Label) (visible : Index → Bool)
    (items : List Index) (panel : List (Index → Label)) :
    (∃ completed : Index → Label, agrees visible labels completed ∧
      ∀ b ∈ panel, difference a b completed items ≤ 0) ↔
    (∀ b ∈ panel, known a b labels visible items - radius a b visible items ≤ 0) := by
  constructor
  · rintro ⟨completed, hagrees, hwin⟩ b hb
    have hbound := (comparison_interval a b completed visible items).1
    rw [known_congr a b labels completed visible items hagrees] at hbound
    exact Int.le_trans hbound (hwin b hb)
  · intro h
    refine ⟨completeWith a labels visible, completion_agrees a labels visible, ?_⟩
    intro b hb
    rw [lower_completion_attains]
    exact h b hb

theorem robust_weak_winner_iff (a labels : Index → Label) (visible : Index → Bool)
    (items : List Index) (panel : List (Index → Label)) :
    (∀ completed : Index → Label, agrees visible labels completed →
      ∀ b ∈ panel, difference a b completed items ≤ 0) ↔
    (∀ b ∈ panel, known a b labels visible items + radius a b visible items ≤ 0) := by
  constructor
  · intro h b hb
    have hwin := h (completeWith b labels visible) (completion_agrees b labels visible) b hb
    rw [upper_completion_attains] at hwin
    exact hwin
  · intro h completed hagrees b hb
    have hbound := (comparison_interval a b completed visible items).2
    rw [known_congr a b labels completed visible items hagrees] at hbound
    exact Int.le_trans hbound (h b hb)

#print axioms fixedpoint_exclusion
#print axioms selected_max_exclusion
#print axioms comparison_interval
#print axioms lower_completion_attains
#print axioms upper_completion_attains
#print axioms common_lower_completion
#print axioms possible_weak_winner_iff
#print axioms robust_weak_winner_iff

end SupportAudit
