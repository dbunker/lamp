d5(o3).
d5(o1).
d3(o5).
d4(o3).
d3(o0).

d4(X) :- d3(X).
d1(X) :- d5(X), not d4(X).
d2(X) :- d4(X), d5(X).