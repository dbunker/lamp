d2(o1).
d2(o4).
d4(o2).
d0(o0).

d1(X) :- not d3(X), d4(X).
d3(X) :- d1(X), not d4(X).
d1(X) :- d2(X), not d4(X).