d3(o4).
d0(o1).
d1(o0).
d0(o2).
d2(o2).

d4(X) :- d0(X), not d3(X).
d0(X) :- d3(X).