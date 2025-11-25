d3(o4).
d2(o0).
d0(o4).
d3(o2).

d1(X) :- d3(X), not d2(X).
d1(X) :- d0(X), not d2(X).