d1(o3).
d3(o1).
d0(o0).
d2(o0).
d1(o2).

d2(X) :- not d0(X), d1(X), d3(X).
d1(X) :- d2(X), d0(X), not d3(X).
d0(X) :- not d2(X), d1(X), d3(X).