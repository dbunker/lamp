d3(o0).
d2(o2).
d1(o3).
d2(o1).

d0(X) :- not d2(X), d1(X).
d1(X) :- d2(X).