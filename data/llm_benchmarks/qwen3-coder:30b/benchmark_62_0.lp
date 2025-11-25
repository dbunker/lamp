d3(o4).
d3(o3).
d0(o1).
d2(o1).
d1(o2).

d3(X) :- d0(X), not d1(X), not d2(X).
d2(X) :- d0(X), not d3(X), not d1(X).
d1(X) :- d0(X), not d3(X), not d2(X).