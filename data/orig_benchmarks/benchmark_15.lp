d2(o0).
d1(o0).
d1(o3).
d0(o0).

d3(X) :- d1(X), d2(X), d0(X).
d0(X) :- d1(X), d3(X), d2(X).
d2(X) :- d1(X), d0(X), d3(X).