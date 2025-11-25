d4(o3).
d0(o0).
d4(o0).
d2(o3).
d1(o0).

d3(X) :- d1(X), d0(X), d4(X).
d3(X) :- d1(X), d0(X), d2(X).
d3(X) :- d1(X), d2(X), d4(X).