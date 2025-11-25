d0(o1).
d2(o4).
d3(o0).
d0(o2).

d1(X) :- d3(X), not d2(X).
d1(X) :- d3(X), not d2(X).