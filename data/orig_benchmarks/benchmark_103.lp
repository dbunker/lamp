d3(o3).
d4(o1).
d0(o0).
d1(o0).
d2(o1).

d2(X) :- d1(X), d0(X).
d0(X) :- d3(X), d4(X).
d0(X) :- d1(X), d2(X).