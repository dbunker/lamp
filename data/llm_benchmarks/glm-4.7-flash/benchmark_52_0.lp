d5(o4).
d4(o4).
d5(o2).
d3(o5).
d3(o9).
d3(o2).

d2(X) :- d5(X), not d4(X).
d0(X) :- d5(X), not d4(X).
d1(X) :- d4(X).