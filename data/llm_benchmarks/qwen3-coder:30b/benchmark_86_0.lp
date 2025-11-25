d2(o3).
d4(o2).
d3(o1).
d1(o1).

d1(X) :- d4(X), not d0(X).
d1(X) :- d3(X), not d0(X).
d0(X) :- d2(X).