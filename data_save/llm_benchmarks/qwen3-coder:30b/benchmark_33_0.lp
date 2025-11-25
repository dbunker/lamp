d3(o0).
d3(o4).
d0(o3).
d2(o1).

d2(X) :- not d1(X), d3(X).
d1(X) :- d2(X), not d0(X).
d2(X) :- d0(X), not d1(X).