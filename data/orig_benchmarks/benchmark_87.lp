d0(o1).
d4(o0).
d2(o1).
d1(o1).

d0(X) :- d1(X), d3(X), d4(X).
d3(X) :- d1(X), d0(X), d2(X).
d1(X) :- d2(X), d3(X), d4(X).