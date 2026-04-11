d5(o1).
d3(o5).
d4(o5).
d5(o5).
d3(o3).

d0(X) :- d3(X), d4(X).
d2(X) :- d5(X), d4(X).
d1(X) :- d5(X), d3(X).