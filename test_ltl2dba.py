from ltlf2dfa.parser.ltlf import LTLfParser

# Define LTLf specification
#specification = "( ((!unsafe) & (!wet | ((!goal) U carpet))) ) U (goal &  (!unsafe))"
specification = "F(goal) & G(!unsafe) & G(wet -> ((!goal) U carpet))"

# Parse the formula
parser = LTLfParser()
formula = parser(specification)

# Generate DFA
dfa = formula.to_dfa()

# Save DFA to DOT file
with open("automaton.dot", "w") as f:
    f.write(dfa)

print(dfa)