You are analysing one decision made by a web computer using agent.

TASK THE USER ASKED FOR:
{{task}}

OBSERVED STATE:
{{state}}

PROPOSED ACTION:
{{proposed_action}}

Identify the observed facts that the decision depends on.

WHAT A DEPENDENCY IS

A dependency is a fact that the reasoning had to READ in order to land on this
action rather than a different one. It is not only the facts that would flip the
answer if edited on their own.

Most tasks of this kind name a target with two parts:

1. A REQUIREMENT that decides which options are even eligible
   ("completed", "in stock", "with free wifi", "holds at least 12 pairs").
2. A COMPARISON that picks the winner among the eligible options
   ("most recent", "cheapest", "highest-rated", "closest").

Work through these steps IN ORDER and show your working in the output.

STEP 1. List EVERY option named anywhere in the state, in the order it appears.
        An option is an option even if it obviously cannot win.
STEP 2. For EVERY option from step 1, without exception, name the fact id that
        gives its REQUIREMENT attribute and say whether it passes. Every option
        in step 1 must appear here with a fact id.
STEP 3. For EVERY option that passed step 2, name the fact id that gives its
        COMPARISON attribute.
STEP 4. The dependency list is every fact id from step 2 plus every fact id
        from step 3. Nothing else.

Specifically EXCLUDE from step 3:

* The comparison attribute of an option that failed the requirement. Once an
  option is ruled out, its price or date is never consulted.
* Attributes the task never mentions.
* Facts that only make the action technically possible to execute, such as a
  button being present or a page being open.

WORKED EXAMPLE

TASK: Book the highest-rated hotel that has free wifi
STATE:
g1: Hotel Alba has free wifi
g2: Hotel Alba is rated 4.2
g3: Hotel Brook does not have free wifi
g4: Hotel Brook is rated 4.8
g5: Hotel Cedar has free wifi
g6: Hotel Cedar is rated 4.6
g7: Hotel Cedar has a pool
ACTION: Book Hotel Cedar

{
"options": ["Hotel Alba", "Hotel Brook", "Hotel Cedar"],
"requirement_check": ["Alba g1 pass", "Brook g3 fail", "Cedar g5 pass"],
"comparison_reads": ["Alba g2", "Cedar g6"],
"dependencies": ["g1", "g2", "g3", "g5", "g6"],
"justification": "Wifi read for all three hotels; rating read only for the two eligible ones. Brook's rating g4 is never compared, and the pool g7 is irrelevant."
}

Return exactly one JSON object in that format, with all five keys.
Do not output any text before or after the JSON object.
