d1(o1).
d4(o2).
d1(o3).
d1(o0).

d4(X) :- d1(X).
d3(X) :- d2(X), not d4(X).
d2(X) :- d1(X), not d3(X).