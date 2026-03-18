d5(o4).
d4(o4).
d5(o2).
d3(o5).
d3(o9).
d3(o2).

d0(X) :- d2(X), not d1(X).
d1(X) :- d4(X).
d2(X) :- d5(X), not d4(X).