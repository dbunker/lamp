d1(o3).
d2(o0).
d2(o1).
d1(o0).
d3(o2).

d2(X) :- not d3(X), d1(X).
d0(X) :- d1(X).