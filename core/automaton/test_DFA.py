from dfa import scLTL_to_DFA

formula = (
    "((!unsafe) & (!wet | ((!goal) U carpet))) "
    "U (goal & (!unsafe))"
)

dfa = scLTL_to_DFA(formula)

print("States:", dfa.states)
print("Initial state:", dfa.initial_state)
print("Accepting states:", dfa.accepting_states)
print("Terminal states:", dfa.terminal_states)

print("Atomic propositions:", dfa.atomic_propositions)

print("Transition table:")
print(dfa.transitions)