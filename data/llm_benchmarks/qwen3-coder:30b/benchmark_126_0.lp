d3(o0).
d3(o1).
d4(o0).
d1(o0).
d0(o2).

d3(X) :- d0(X).
d2(X) :- d3(X), not d0(X).
d1(X) :- d4(X).
d0(X) :- not d2(X), d1(X).