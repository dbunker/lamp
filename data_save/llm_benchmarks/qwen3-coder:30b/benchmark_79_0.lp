d2(o0).
d1(o0).
d2(o1).
d3(o1).

d2(X) :- d3(X).
d2(X) :- d2(X), not d0(X).
d0(X) :- d1(X).