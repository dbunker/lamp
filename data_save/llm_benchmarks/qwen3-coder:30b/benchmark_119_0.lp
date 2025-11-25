d4(o2).
d1(o0).
d0(o2).
d2(o1).

d1(X) :- not d2(X), d0(X).
d3(X) :- d2(X), not d4(X).